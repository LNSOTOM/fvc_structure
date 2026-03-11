import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple
from collections import defaultdict

import numpy as np

try:
    import laspy
except ImportError as e:  # pragma: no cover
    raise SystemExit(
        "Missing dependency 'laspy'. Install with: pip install laspy lazrs"
    ) from e


@dataclass
class InstanceMetrics:
    tp: int
    fp: int
    fn: int
    precision: Optional[float]
    recall: Optional[float]
    f1: Optional[float]
    coverage: Optional[float]
    weighted_coverage: Optional[float]


@dataclass(frozen=True)
class OverlapSummary:
    gt_sizes: Dict[int, int]
    pred_sizes: Dict[int, int]
    best_iou_per_gt: Dict[int, float]
    best_for_pred: Dict[int, Tuple[int, float]]  # pred -> (best_gt, best_iou)
    gt_heights: Optional[Dict[int, float]] = None  # gt_id -> (max_z - min_z)
    pred_heights: Optional[Dict[int, float]] = None  # pred_id -> (max_z - min_z)


def _safe_div(num: float, den: float) -> Optional[float]:
    if den == 0:
        return None
    return num / den


def _safe_f1(precision: Optional[float], recall: Optional[float]) -> Optional[float]:
    if precision is None or recall is None:
        return None
    den = precision + recall
    if den == 0:
        # Defined corner case: precision=0 and recall=0 -> F1=0.
        return 0.0
    return 2 * precision * recall / den


def _pair_key(pred_id: np.ndarray, gt_id: np.ndarray) -> np.ndarray:
    return (pred_id.astype(np.int64) << 32) | (gt_id.astype(np.int64) & 0xFFFFFFFF)


def _decode_pair_key(key: int) -> Tuple[int, int]:
    pred_id = (key >> 32) & 0xFFFFFFFF
    gt_id = key & 0xFFFFFFFF
    return int(pred_id), int(gt_id)


def _counts_1d(arr: np.ndarray) -> Dict[int, int]:
    uniq, cnt = np.unique(arr, return_counts=True)
    return {int(u): int(c) for u, c in zip(uniq.tolist(), cnt.tolist())}


def _heights_by_id(ids: np.ndarray, z: np.ndarray) -> Dict[int, float]:
    """Compute per-instance height (max(z)-min(z)) for each unique id.

    Expects ids and z already filtered to valid points.
    """

    if ids.size == 0:
        return {}

    ids = np.asarray(ids).astype(np.int64, copy=False)
    z = np.asarray(z).astype(np.float64, copy=False)

    order = np.argsort(ids, kind="mergesort")
    ids_s = ids[order]
    z_s = z[order]

    change = np.empty(ids_s.shape[0], dtype=bool)
    change[0] = True
    change[1:] = ids_s[1:] != ids_s[:-1]
    start_idx = np.nonzero(change)[0]
    uniq_ids = ids_s[start_idx]

    z_min = np.minimum.reduceat(z_s, start_idx)
    z_max = np.maximum.reduceat(z_s, start_idx)
    heights = (z_max - z_min).astype(float)

    return {int(i): float(h) for i, h in zip(uniq_ids.tolist(), heights.tolist())}


def _matched_pairs_from_overlap(
    summary: OverlapSummary,
    *,
    iou_threshold: float,
    matching: str,
) -> List[Tuple[int, int, float]]:
    """Return list of matched (pred_id, gt_id, iou) pairs under the matching policy."""

    pred_ids_list = sorted(summary.pred_sizes.keys())
    best_for_pred = summary.best_for_pred

    if matching == "per_pred":
        out: List[Tuple[int, int, float]] = []
        for pid in pred_ids_list:
            best = best_for_pred.get(pid)
            if best is None:
                continue
            gid, best_iou = best
            if best_iou >= iou_threshold:
                out.append((pid, gid, float(best_iou)))
        return out

    if matching != "one_to_one":
        raise ValueError("matching must be 'per_pred' or 'one_to_one'")

    candidates: List[Tuple[float, int, int]] = []  # (iou, pid, gid)
    for pid in pred_ids_list:
        best = best_for_pred.get(pid)
        if best is None:
            continue
        gid, best_iou = best
        candidates.append((float(best_iou), pid, gid))
    candidates.sort(reverse=True)

    matched_gt = set()
    matched_pred = set()
    out2: List[Tuple[int, int, float]] = []
    for iou, pid, gid in candidates:
        if iou < iou_threshold:
            break
        if gid in matched_gt or pid in matched_pred:
            continue
        matched_gt.add(gid)
        matched_pred.add(pid)
        out2.append((pid, gid, float(iou)))
    return out2


def compute_instance_metrics_from_ids(
    gt_ids: np.ndarray,
    pred_ids: np.ndarray,
    *,
    iou_threshold: float = 0.5,
    ignore_pred_id: int = 0,
    ignore_gt_id_leq: int = 0,
    chunk_size: int = 5_000_000,
    matching: str = "per_pred",
) -> Tuple[InstanceMetrics, Dict[int, int], Dict[int, int]]:
    """Compute instance metrics from per-point GT/pred instance ids.

    This mirrors SegmentAnyTree's `_compute_eval` intent, but operates on merged LAZ outputs.

    matching:
      - 'per_pred': count TP for each predicted instance whose best IoU >= threshold (no 1:1 constraint).
      - 'one_to_one': enforce one predicted instance per GT instance (greedy by IoU).
    """

    if gt_ids.shape != pred_ids.shape:
        raise ValueError(f"Shape mismatch: gt {gt_ids.shape} vs pred {pred_ids.shape}")

    gt = np.asarray(gt_ids)
    pred = np.asarray(pred_ids)

    # Normalize to integers.
    if np.issubdtype(gt.dtype, np.floating):
        gt = np.nan_to_num(gt, nan=0.0).astype(np.int64)
    else:
        gt = gt.astype(np.int64, copy=False)

    if np.issubdtype(pred.dtype, np.floating):
        pred = np.nan_to_num(pred, nan=0.0).astype(np.int64)
    else:
        pred = pred.astype(np.int64, copy=False)

    summary = compute_overlap_summary_from_ids(
        gt,
        pred,
        ignore_pred_id=ignore_pred_id,
        ignore_gt_id_leq=ignore_gt_id_leq,
        chunk_size=chunk_size,
    )
    metrics = compute_instance_metrics_from_overlap(
        summary,
        iou_threshold=iou_threshold,
        matching=matching,
    )
    return metrics, summary.gt_sizes, summary.pred_sizes


def compute_overlap_summary_from_ids(
    gt_ids: np.ndarray,
    pred_ids: np.ndarray,
    *,
    ignore_pred_id: int = 0,
    ignore_gt_id_leq: int = 0,
    chunk_size: int = 5_000_000,
    z: Optional[np.ndarray] = None,
    compute_heights: bool = False,
) -> OverlapSummary:
    """Compute overlap summary (best IoU per GT and per pred) from per-point ids.

    This is the expensive part (counts intersections). It can be reused to evaluate
    multiple IoU thresholds without re-reading or re-counting.
    """

    if gt_ids.shape != pred_ids.shape:
        raise ValueError(f"Shape mismatch: gt {gt_ids.shape} vs pred {pred_ids.shape}")

    gt = np.asarray(gt_ids).astype(np.int64, copy=False)
    pred = np.asarray(pred_ids).astype(np.int64, copy=False)

    gt_valid = gt > ignore_gt_id_leq
    pred_valid = pred != ignore_pred_id

    gt_sizes = _counts_1d(gt[gt_valid])
    pred_sizes = _counts_1d(pred[pred_valid])

    gt_heights: Optional[Dict[int, float]] = None
    pred_heights: Optional[Dict[int, float]] = None
    if compute_heights:
        if z is None:
            raise ValueError("compute_heights=True requires z array")
        zz = np.asarray(z).astype(np.float64, copy=False)
        if zz.shape != gt.shape:
            raise ValueError(f"Shape mismatch: z {zz.shape} vs gt {gt.shape}")
        gt_heights = _heights_by_id(gt[gt_valid], zz[gt_valid])
        pred_heights = _heights_by_id(pred[pred_valid], zz[pred_valid])

    pair_counts: Dict[int, int] = {}
    n = gt.shape[0]
    for start in range(0, n, chunk_size):
        end = min(n, start + chunk_size)
        m = gt_valid[start:end] & pred_valid[start:end]
        if not np.any(m):
            continue
        g = gt[start:end][m]
        p = pred[start:end][m]
        keys = _pair_key(p, g)
        uniq, cnt = np.unique(keys, return_counts=True)
        for k, c in zip(uniq.tolist(), cnt.tolist()):
            pair_counts[int(k)] = pair_counts.get(int(k), 0) + int(c)

    best_iou_per_gt: Dict[int, float] = {gid: 0.0 for gid in gt_sizes.keys()}
    best_for_pred: Dict[int, Tuple[int, float]] = {}

    for key, inter in pair_counts.items():
        pid, gid = _decode_pair_key(key)
        ps = pred_sizes.get(pid)
        gs = gt_sizes.get(gid)
        if not ps or not gs:
            continue
        union = ps + gs - inter
        if union <= 0:
            continue
        iou = inter / union

        if iou > best_iou_per_gt.get(gid, 0.0):
            best_iou_per_gt[gid] = float(iou)

        prev = best_for_pred.get(pid)
        if prev is None or iou > prev[1]:
            best_for_pred[pid] = (gid, float(iou))

    return OverlapSummary(
        gt_sizes=gt_sizes,
        pred_sizes=pred_sizes,
        best_iou_per_gt=best_iou_per_gt,
        best_for_pred=best_for_pred,
        gt_heights=gt_heights,
        pred_heights=pred_heights,
    )


def compute_instance_metrics_from_overlap(
    summary: OverlapSummary,
    *,
    iou_threshold: float,
    matching: str = "per_pred",
) -> InstanceMetrics:
    """Compute TP/FP/FN + derived metrics from a precomputed overlap summary."""

    gt_sizes = summary.gt_sizes
    pred_sizes = summary.pred_sizes
    best_iou_per_gt = summary.best_iou_per_gt
    best_for_pred = summary.best_for_pred

    if gt_sizes:
        cov = float(np.mean(list(best_iou_per_gt.values())))
        total_pts = sum(gt_sizes.values())
        wcov = (
            sum(best_iou_per_gt[gid] * gt_sizes[gid] for gid in gt_sizes.keys()) / total_pts
            if total_pts > 0
            else None
        )
    else:
        cov = None
        wcov = None

    pred_ids_list = sorted(pred_sizes.keys())

    if matching not in {"per_pred", "one_to_one"}:
        raise ValueError("matching must be 'per_pred' or 'one_to_one'")

    tp = 0
    fp = 0

    if matching == "per_pred":
        for pid in pred_ids_list:
            best = best_for_pred.get(pid)
            if best is None:
                fp += 1
                continue
            _, best_iou = best
            if best_iou >= iou_threshold:
                tp += 1
            else:
                fp += 1
        fn = max(0, len(gt_sizes) - tp) if gt_sizes else 0
    else:
        candidates: List[Tuple[float, int, int]] = []  # (iou, pid, gid)
        for pid in pred_ids_list:
            best = best_for_pred.get(pid)
            if best is None:
                continue
            gid, best_iou = best
            candidates.append((best_iou, pid, gid))
        candidates.sort(reverse=True)

        matched_gt = set()
        matched_pred = set()
        for iou, pid, gid in candidates:
            if iou < iou_threshold:
                break
            if gid in matched_gt or pid in matched_pred:
                continue
            matched_gt.add(gid)
            matched_pred.add(pid)
            tp += 1

        fp = len(pred_ids_list) - tp
        fn = max(0, len(gt_sizes) - tp)

    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    f1 = _safe_f1(precision, recall)

    return InstanceMetrics(
        tp=tp,
        fp=fp,
        fn=fn,
        precision=precision,
        recall=recall,
        f1=f1,
        coverage=cov,
        weighted_coverage=wcov,
    )


def _get_dim(las, candidates: Iterable[str]) -> str:
    dims = set(las.point_format.dimension_names)
    for c in candidates:
        if c in dims:
            return c
    raise KeyError(f"None of these dimensions found in LAZ: {list(candidates)}")


def evaluate_laz_file(
    path: Path,
    *,
    iou_threshold: float,
    matching: str,
    gt_dim: Optional[str] = None,
    pred_dim: Optional[str] = None,
) -> InstanceMetrics:
    las = laspy.read(str(path))

    if gt_dim is None:
        gt_dim = _get_dim(las, ["treeID", "TreeID", "gt_treeID", "gt_instance", "gt_instance_id"])
    if pred_dim is None:
        pred_dim = _get_dim(las, ["PredInstance", "preds_instance_segmentation", "pred_instance", "instance_pred"])

    gt = np.asarray(getattr(las, gt_dim))
    pred = np.asarray(getattr(las, pred_dim))

    summary = compute_overlap_summary_from_ids(gt, pred)
    return compute_instance_metrics_from_overlap(
        summary,
        iou_threshold=iou_threshold,
        matching=matching,
    )


def evaluate_laz_file_overlap(
    path: Path,
    *,
    gt_dim: Optional[str] = None,
    pred_dim: Optional[str] = None,
    compute_heights: bool = False,
) -> OverlapSummary:
    las = laspy.read(str(path))

    if gt_dim is None:
        gt_dim = _get_dim(las, ["treeID", "TreeID", "gt_treeID", "gt_instance", "gt_instance_id"])
    if pred_dim is None:
        pred_dim = _get_dim(las, ["PredInstance", "preds_instance_segmentation", "pred_instance", "instance_pred"])

    gt = np.asarray(getattr(las, gt_dim))
    pred = np.asarray(getattr(las, pred_dim))

    z = None
    if compute_heights:
        z = np.asarray(las.z)

    # Normalize to int64 the same way as compute_instance_metrics_from_ids.
    if np.issubdtype(gt.dtype, np.floating):
        gt = np.nan_to_num(gt, nan=0.0).astype(np.int64)
    else:
        gt = gt.astype(np.int64, copy=False)

    if np.issubdtype(pred.dtype, np.floating):
        pred = np.nan_to_num(pred, nan=0.0).astype(np.int64)
    else:
        pred = pred.astype(np.int64, copy=False)

    return compute_overlap_summary_from_ids(
        gt,
        pred,
        z=z,
        compute_heights=compute_heights,
    )


def main():
    ap = argparse.ArgumentParser(description="Offline instance metrics from merged SegmentAnyTree LAZ outputs")
    ap.add_argument("--input", required=True, help="LAZ file or directory containing LAZ files")
    ap.add_argument("--iou", type=float, default=0.5, help="IoU threshold for TP")
    ap.add_argument(
        "--matching",
        choices=["per_pred", "one_to_one"],
        default="per_pred",
        help="Matching policy for TP/FP/FN",
    )
    ap.add_argument("--gt-dim", default=None, help="Override GT instance dimension name")
    ap.add_argument("--pred-dim", default=None, help="Override predicted instance dimension name")
    ap.add_argument(
        "--out-csv",
        default=None,
        help=(
            "Write CSV outputs. If a directory, writes per_file.csv and per_region.csv inside it. "
            "If a .csv path, writes <stem>_per_file.csv and <stem>_per_region.csv alongside it."
        ),
    )
    ap.add_argument(
        "--no-print",
        action="store_true",
        help="Suppress console tables (useful with --out-csv)",
    )
    ap.add_argument(
        "--height-metrics",
        action="store_true",
        help="Compute extra height residual/RMSE metrics for matched instances (adds extra CSV columns)",
    )
    args = ap.parse_args()

    iou = float(args.iou)

    p = Path(args.input)
    laz_files: List[Path]
    if p.is_dir():
        laz_files = sorted(p.glob("*.laz")) + sorted(p.glob("*.las"))
    else:
        laz_files = [p]

    if not laz_files:
        raise SystemExit(f"No .laz/.las files found under {p}")

    # Compute expensive overlaps once per file.
    overlaps: List[Tuple[str, OverlapSummary]] = []
    for f in laz_files:
        summary = evaluate_laz_file_overlap(
            f,
            gt_dim=args.gt_dim,
            pred_dim=args.pred_dim,
            compute_heights=bool(args.height_metrics),
        )
        overlaps.append((f.name, summary))

    def region_from_filename(name: str) -> str:
        """Infer a region/site prefix from common ForAINetV2 tile naming.

        Examples:
          - NIBIO2_NIBIO2_plot10_... -> NIBIO2
          - SCION_SCION_plot_... -> SCION
          - NIBIO_MLS_MLS_... -> NIBIO_MLS
          - BlueCat_RN_... -> BlueCat

        Heuristic:
          - if first two tokens match: use token0
          - else if tokens 1 and 2 match: use token0_token1
          - else: use token0
        """
        stem = name
        for suf in (".laz", ".las"):
            if stem.lower().endswith(suf):
                stem = stem[: -len(suf)]
        parts = stem.split("_")
        if len(parts) >= 2 and parts[0] == parts[1]:
            return parts[0]
        if len(parts) >= 3 and parts[1] == parts[2]:
            return "_".join(parts[:2])
        return parts[0] if parts else stem

    def fmt(x: Optional[float]) -> str:
        return "--" if x is None else f"{x*100:.2f}"

    def _csv_value(x: Optional[float]) -> str:
        return "" if x is None else f"{x:.10g}"

    def _write_csv_rows(path: Path, header: List[str], rows: List[List[str]]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(header)
            w.writerows(rows)

    def _resolve_csv_paths(out_csv: str) -> Tuple[Path, Path]:
        out_p = Path(out_csv)
        if out_p.suffix.lower() == ".csv":
            stem = out_p.with_suffix("")
            return (
                stem.parent / f"{stem.name}_per_file.csv",
                stem.parent / f"{stem.name}_per_region.csv",
            )
        # Treat as directory
        return out_p / "per_file.csv", out_p / "per_region.csv"

    def mean(xs: List[float]) -> Optional[float]:
        return float(np.mean(xs)) if xs else None

    def run_one_threshold(iou: float) -> None:
        per_file: List[Tuple[str, InstanceMetrics]] = []
        per_file_extra: Dict[str, Dict[str, Optional[float]]] = {}
        for name, summary in overlaps:
            m = compute_instance_metrics_from_overlap(
                summary,
                iou_threshold=float(iou),
                matching=args.matching,
            )
            per_file.append((name, m))

            # Extra, optional metrics inspired by metrics/instance_segmentation_metrics.py
            gt_count = len(summary.gt_sizes)
            pred_count = len(summary.pred_sizes)
            gt_detected = sum(1 for _, biou in summary.best_iou_per_gt.items() if biou >= float(iou))
            gt_detection_rate = _safe_div(gt_detected, gt_count)
            pred_commission_rate = _safe_div(m.fp, pred_count)

            height_residual_mean: Optional[float] = None
            height_rmse: Optional[float] = None
            height_mae: Optional[float] = None
            height_n: int = 0

            if args.height_metrics and summary.gt_heights is not None and summary.pred_heights is not None:
                pairs = _matched_pairs_from_overlap(summary, iou_threshold=float(iou), matching=args.matching)
                residuals: List[float] = []
                for pid, gid, _ in pairs:
                    gh = summary.gt_heights.get(gid) if summary.gt_heights else None
                    ph = summary.pred_heights.get(pid) if summary.pred_heights else None
                    if gh is None or ph is None:
                        continue
                    residuals.append(float(gh) - float(ph))

                if residuals:
                    rr = np.asarray(residuals, dtype=float)
                    height_n = int(rr.size)
                    height_residual_mean = float(np.mean(rr))
                    height_rmse = float(np.sqrt(np.mean(rr * rr)))
                    height_mae = float(np.mean(np.abs(rr)))

            per_file_extra[name] = {
                "gt_instances": float(gt_count),
                "pred_instances": float(pred_count),
                "gt_detected": float(gt_detected),
                "gt_detection_rate": gt_detection_rate,
                "pred_commission_rate": pred_commission_rate,
                "height_residual_mean": height_residual_mean,
                "height_rmse": height_rmse,
                "height_mae": height_mae,
                "height_n": float(height_n),
            }

        by_region = defaultdict(list)
        for name, m in per_file:
            by_region[region_from_filename(name)].append(m)

        region_rows = []
        for region in sorted(by_region.keys()):
            ms = by_region[region]
            tp = sum(m.tp for m in ms)
            fp = sum(m.fp for m in ms)
            fn = sum(m.fn for m in ms)

            micro_p = _safe_div(tp, tp + fp)
            micro_r = _safe_div(tp, tp + fn)
            micro_f1 = _safe_f1(micro_p, micro_r)

            macro_f1 = mean([m.f1 for m in ms if m.f1 is not None])
            macro_cov = mean([m.coverage for m in ms if m.coverage is not None])
            macro_wcov = mean([m.weighted_coverage for m in ms if m.weighted_coverage is not None])

            region_rows.append(
                {
                    "region": region,
                    "files": len(ms),
                    "tp": tp,
                    "fp": fp,
                    "fn": fn,
                    "micro_precision": micro_p,
                    "micro_recall": micro_r,
                    "micro_f1": micro_f1,
                    "macro_f1": macro_f1,
                    "macro_coverage": macro_cov,
                    "macro_weighted_coverage": macro_wcov,
                }
            )

        if not args.no_print:
            print("Offline Instance Metrics (from LAZ)")
            print(f"Input: {p}")
            print(f"IoU threshold: {float(iou)}, matching: {args.matching}")
            print("=" * 120)
            print(
                f"{'File':<50}{'TP':>8}{'FP':>8}{'FN':>8}{'Prec%':>10}{'Rec%':>10}{'F1%':>10}{'Cov%':>10}{'WCov%':>10}"
            )
            print("-" * 120)

            for name, m in per_file:
                print(
                    f"{name:<50}"
                    f"{m.tp:>8}{m.fp:>8}{m.fn:>8}"
                    f"{fmt(m.precision):>10}{fmt(m.recall):>10}{fmt(m.f1):>10}{fmt(m.coverage):>10}{fmt(m.weighted_coverage):>10}"
                )

            print("\nRegion Summary")
            print("=" * 120)
            print(
                f"{'Region':<18}{'Files':>7}{'TP':>8}{'FP':>8}{'FN':>8}"
                f"{'MicroP%':>10}{'MicroR%':>10}{'MicroF1%':>11}"
                f"{'MacroF1%':>11}{'MacroCov%':>12}{'MacroWCov%':>13}"
            )
            print("-" * 120)

            for r in region_rows:
                print(
                    f"{r['region']:<18}{r['files']:>7}{r['tp']:>8}{r['fp']:>8}{r['fn']:>8}"
                    f"{fmt(r['micro_precision']):>10}{fmt(r['micro_recall']):>10}{fmt(r['micro_f1']):>11}"
                    f"{fmt(r['macro_f1']):>11}{fmt(r['macro_coverage']):>12}{fmt(r['macro_weighted_coverage']):>13}"
                )

        precs = [m.precision for _, m in per_file if m.precision is not None]
        recs = [m.recall for _, m in per_file if m.recall is not None]
        f1s = [m.f1 for _, m in per_file if m.f1 is not None]
        covs = [m.coverage for _, m in per_file if m.coverage is not None]
        wcovs = [m.weighted_coverage for _, m in per_file if m.weighted_coverage is not None]

        if not args.no_print:
            print("-" * 120)
            print(
                f"{'Avg':<50}"
                f"{'':>24}"
                f"{fmt(mean(precs)):>10}{fmt(mean(recs)):>10}{fmt(mean(f1s)):>10}{fmt(mean(covs)):>10}{fmt(mean(wcovs)):>10}"
            )

        if args.out_csv:
            per_file_csv, per_region_csv = _resolve_csv_paths(args.out_csv)

            per_file_rows: List[List[str]] = []
            for name, m in per_file:
                extra = per_file_extra.get(name, {})
                per_file_rows.append(
                    [
                        name,
                        region_from_filename(name),
                        str(m.tp),
                        str(m.fp),
                        str(m.fn),
                        _csv_value(m.precision),
                        _csv_value(m.recall),
                        _csv_value(m.f1),
                        _csv_value(m.coverage),
                        _csv_value(m.weighted_coverage),
                        str(int(extra.get("gt_instances", 0.0) or 0)),
                        str(int(extra.get("pred_instances", 0.0) or 0)),
                        str(int(extra.get("gt_detected", 0.0) or 0)),
                        _csv_value(extra.get("gt_detection_rate")),
                        _csv_value(extra.get("pred_commission_rate")),
                        _csv_value(extra.get("height_residual_mean")),
                        _csv_value(extra.get("height_rmse")),
                        _csv_value(extra.get("height_mae")),
                        str(int(extra.get("height_n", 0.0) or 0)),
                    ]
                )

            _write_csv_rows(
                per_file_csv,
                [
                    "file",
                    "region",
                    "tp",
                    "fp",
                    "fn",
                    "precision",
                    "recall",
                    "f1",
                    "coverage",
                    "weighted_coverage",
                    "gt_instances",
                    "pred_instances",
                    "gt_detected",
                    "gt_detection_rate",
                    "pred_commission_rate",
                    "height_residual_mean",
                    "height_rmse",
                    "height_mae",
                    "height_n",
                ],
                per_file_rows,
            )

            per_region_rows: List[List[str]] = []
            for r in region_rows:
                per_region_rows.append(
                    [
                        r["region"],
                        str(r["files"]),
                        str(r["tp"]),
                        str(r["fp"]),
                        str(r["fn"]),
                        _csv_value(r["micro_precision"]),
                        _csv_value(r["micro_recall"]),
                        _csv_value(r["micro_f1"]),
                        _csv_value(r["macro_f1"]),
                        _csv_value(r["macro_coverage"]),
                        _csv_value(r["macro_weighted_coverage"]),
                    ]
                )

            _write_csv_rows(
                per_region_csv,
                [
                    "region",
                    "files",
                    "tp",
                    "fp",
                    "fn",
                    "micro_precision",
                    "micro_recall",
                    "micro_f1",
                    "macro_f1",
                    "macro_coverage",
                    "macro_weighted_coverage",
                ],
                per_region_rows,
            )

            print(f"Wrote CSV: {per_file_csv}")
            print(f"Wrote CSV: {per_region_csv}")

    run_one_threshold(iou)


if __name__ == "__main__":
    main()
