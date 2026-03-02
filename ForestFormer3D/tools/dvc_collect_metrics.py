#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


_TIMESTAMP_JSON_RE = re.compile(r"^\d{8}_\d{6}\.json$")
_TIMESTAMP_LOG_RE = re.compile(r"^\d{8}_\d{6}\.log$")


def _safe_float(text: str) -> float | None:
    try:
        return float(text)
    except Exception:
        return None


def _safe_int(text: str) -> int | None:
    try:
        return int(text)
    except Exception:
        return None


def _find_latest_timestamp_json(work_dir: Path) -> Path:
    if not work_dir.exists():
        raise FileNotFoundError(f"work_dir does not exist: {work_dir}")

    candidates: list[Path] = []
    for json_path in work_dir.rglob("*.json"):
        if _TIMESTAMP_JSON_RE.match(json_path.name):
            candidates.append(json_path)

    if not candidates:
        raise FileNotFoundError(
            "No timestamped metrics JSON found under work_dir. "
            "Expected files like YYYYMMDD_HHMMSS.json. "
            f"work_dir={work_dir}"
        )

    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]


def _find_matching_log(work_dir: Path, metrics_json: Path) -> Path | None:
    """Try to locate the corresponding timestamped log for a metrics JSON."""
    stem = metrics_json.stem
    direct = metrics_json.with_suffix('.log')
    if direct.exists() and _TIMESTAMP_LOG_RE.match(direct.name):
        return direct

    # Fallback: search under work_dir for a log with same stem
    for log_path in work_dir.rglob('*.log'):
        if log_path.stem == stem and _TIMESTAMP_LOG_RE.match(log_path.name):
            return log_path
    return None


def _parse_metrics_from_log(log_path: Path) -> dict:
    """Parse extra metrics that are printed only to the MMEngine log."""
    text = log_path.read_text(errors='ignore')

    per_class_sem_iou: dict[str, float] = {}
    per_class_sem_iou_by_id: dict[str, float] = {}
    binary_sem_iou: dict[str, float] = {}
    binary_sem_iou_by_id: dict[str, float] = {}

    sem_name_by_id: dict[str, str] = {}
    binary_name_by_id: dict[str, str] = {}

    sem_counts: dict[str, dict[str, int]] = {}
    sem_counts_by_id: dict[str, dict[str, int]] = {}
    binary_counts: dict[str, dict[str, int]] = {}
    binary_counts_by_id: dict[str, dict[str, int]] = {}

    # Example:
    #   Class 0 (hummock): IoU=0.4951, GT=14352, Pred=10737, TP=8308
    class_re = re.compile(
        r"Class\s+(?P<id>\d+)\s+\((?P<name>[^)]+)\):\s+IoU=(?P<iou>[0-9.]+)"
        r"(?:,\s*GT=(?P<gt>\d+),\s*Pred=(?P<pred>\d+),\s*TP=(?P<tp>\d+))?"
    )
    # Example:
    #   Binary class 1 (non-tree): IoU=0.3578, GT=18254, Pred=31988, TP=13239
    binary_re = re.compile(
        r"Binary\s+class\s+(?P<id>\d+)\s+\((?P<name>[^)]+)\):\s+IoU=(?P<iou>[0-9.]+)"
        r"(?:,\s*GT=(?P<gt>\d+),\s*Pred=(?P<pred>\d+),\s*TP=(?P<tp>\d+))?"
    )

    for match in class_re.finditer(text):
        class_id = match.group('id')
        name = match.group('name').strip()
        iou = _safe_float(match.group('iou'))
        if iou is None:
            continue
        per_class_sem_iou[name] = iou
        per_class_sem_iou_by_id[class_id] = iou
        sem_name_by_id[class_id] = name

        gt = match.groupdict().get('gt')
        pred = match.groupdict().get('pred')
        tp = match.groupdict().get('tp')
        if gt is not None and pred is not None and tp is not None:
            gt_i = _safe_int(gt)
            pred_i = _safe_int(pred)
            tp_i = _safe_int(tp)
            if gt_i is not None and pred_i is not None and tp_i is not None:
                sem_counts[name] = {'gt': gt_i, 'pred': pred_i, 'tp': tp_i}
                sem_counts_by_id[class_id] = {'gt': gt_i, 'pred': pred_i, 'tp': tp_i}

    for match in binary_re.finditer(text):
        class_id = match.group('id')
        name = match.group('name').strip()
        iou = _safe_float(match.group('iou'))
        if iou is None:
            continue
        binary_sem_iou[name] = iou
        binary_sem_iou_by_id[class_id] = iou
        binary_name_by_id[class_id] = name

        gt = match.groupdict().get('gt')
        pred = match.groupdict().get('pred')
        tp = match.groupdict().get('tp')
        if gt is not None and pred is not None and tp is not None:
            gt_i = _safe_int(gt)
            pred_i = _safe_int(pred)
            tp_i = _safe_int(tp)
            if gt_i is not None and pred_i is not None and tp_i is not None:
                binary_counts[name] = {'gt': gt_i, 'pred': pred_i, 'tp': tp_i}
                binary_counts_by_id[class_id] = {'gt': gt_i, 'pred': pred_i, 'tp': tp_i}

    # Example:
    #   Total GT instances: 272
    #   Total predicted instances: 86
    #   True positives (IoU≥0.5): 18
    #   False positives: 68
    instance_summary: dict[str, int] = {}
    instance_patterns = {
        'total_gt_instances': re.compile(r"Total GT instances:\s+(?P<v>\d+)"),
        'total_pred_instances': re.compile(r"Total predicted instances:\s+(?P<v>\d+)"),
        'true_positives_iou_ge_0_5': re.compile(r"True positives \(IoU.?=.?0\.5\):\s+(?P<v>\d+)"),
        'false_positives': re.compile(r"False positives:\s+(?P<v>\d+)"),
    }
    for key, rx in instance_patterns.items():
        m = rx.search(text)
        if m:
            v = _safe_int(m.group('v'))
            if v is not None:
                instance_summary[key] = v

    # Example:
    #   Prec=20.93  Rec=6.62  F1=10.06  Cov(MWCov)=26.76
    ind_tree_pct: dict[str, float] = {}
    m = re.search(
        r"Prec=(?P<prec>[0-9.]+)\s+Rec=(?P<rec>[0-9.]+)\s+F1=(?P<f1>[0-9.]+)\s+Cov\(MWCov\)=(?P<cov>[0-9.]+)",
        text,
    )
    if m:
        for out_key, group_key in (
            ('precision_pct', 'prec'),
            ('recall_pct', 'rec'),
            ('f1_pct', 'f1'),
            ('mwcov_pct', 'cov'),
        ):
            val = _safe_float(m.group(group_key))
            if val is not None:
                ind_tree_pct[out_key] = val

    # Short, DVC-friendly keys (less verbose than per_class_sem_iou.*)
    sem_iou: dict[str, float] = dict(per_class_sem_iou)
    sem_iou_pct: dict[str, float] = {k: v * 100.0 for k, v in per_class_sem_iou.items()}
    binary_iou: dict[str, float] = dict(binary_sem_iou)
    binary_iou_pct: dict[str, float] = {k: v * 100.0 for k, v in binary_sem_iou.items()}

    # Ordered groups so DVC columns follow the MMEngine print order.
    # DVC sorts keys lexicographically, so we prefix with 01/02/03.
    ordered_semantic: dict[str, dict] = {}
    ordered_semantic_iou: dict[str, float] = {}
    ordered_semantic_iou_pct: dict[str, float] = {}
    for class_id_str, iou in sorted(per_class_sem_iou_by_id.items(), key=lambda kv: int(kv[0])):
        label = sem_name_by_id.get(class_id_str, f"class_{class_id_str}")
        prefix = f"{int(class_id_str)+1:02d}"
        key = f"{prefix}_{label}"
        ordered_semantic_iou[key] = iou
        ordered_semantic_iou_pct[key] = iou * 100.0
        entry: dict[str, object] = {
            'iou': iou,
            'iou_pct': iou * 100.0,
        }
        if label in sem_counts:
            entry.update(sem_counts[label])
        elif class_id_str in sem_counts_by_id:
            entry.update(sem_counts_by_id[class_id_str])
        ordered_semantic[key] = entry

    ordered_binary: dict[str, dict] = {}
    ordered_binary_iou: dict[str, float] = {}
    ordered_binary_iou_pct: dict[str, float] = {}
    for class_id_str, iou in sorted(binary_sem_iou_by_id.items(), key=lambda kv: int(kv[0])):
        label = binary_name_by_id.get(class_id_str, f"binary_{class_id_str}")
        prefix = f"{int(class_id_str):02d}"  # binary ids are 1/2 in your logs
        key = f"{prefix}_{label}"
        ordered_binary_iou[key] = iou
        ordered_binary_iou_pct[key] = iou * 100.0
        entry: dict[str, object] = {
            'iou': iou,
            'iou_pct': iou * 100.0,
        }
        if label in binary_counts:
            entry.update(binary_counts[label])
        elif class_id_str in binary_counts_by_id:
            entry.update(binary_counts_by_id[class_id_str])
        ordered_binary[key] = entry

    return {
        **({'per_class_sem_iou': per_class_sem_iou} if per_class_sem_iou else {}),
        **({'per_class_sem_iou_by_id': per_class_sem_iou_by_id} if per_class_sem_iou_by_id else {}),
        **({'binary_sem_iou': binary_sem_iou} if binary_sem_iou else {}),
        **({'binary_sem_iou_by_id': binary_sem_iou_by_id} if binary_sem_iou_by_id else {}),
        **({'sem_counts': sem_counts} if sem_counts else {}),
        **({'sem_counts_by_id': sem_counts_by_id} if sem_counts_by_id else {}),
        **({'binary_counts': binary_counts} if binary_counts else {}),
        **({'binary_counts_by_id': binary_counts_by_id} if binary_counts_by_id else {}),
        **({'sem_iou': sem_iou} if sem_iou else {}),
        **({'sem_iou_pct': sem_iou_pct} if sem_iou_pct else {}),
        **({'binary_iou': binary_iou} if binary_iou else {}),
        **({'binary_iou_pct': binary_iou_pct} if binary_iou_pct else {}),
        **({'ordered_semantic': ordered_semantic} if ordered_semantic else {}),
        **({'ordered_semantic_iou': ordered_semantic_iou} if ordered_semantic_iou else {}),
        **({'ordered_semantic_iou_pct': ordered_semantic_iou_pct} if ordered_semantic_iou_pct else {}),
        **({'ordered_binary': ordered_binary} if ordered_binary else {}),
        **({'ordered_binary_iou': ordered_binary_iou} if ordered_binary_iou else {}),
        **({'ordered_binary_iou_pct': ordered_binary_iou_pct} if ordered_binary_iou_pct else {}),
        **({'instance_summary': instance_summary} if instance_summary else {}),
        **({'individual_tree_seg_pct': ind_tree_pct} if ind_tree_pct else {}),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Collect the latest MMEngine timestamped metrics JSON from a work_dir "
            "and write it to a stable output path for DVC."
        )
    )
    parser.add_argument(
        "--work-dir",
        required=True,
        help=(
            "Directory passed to tools/test.py via --work-dir (searched recursively)."
        ),
    )
    parser.add_argument(
        "--out",
        required=True,
        help="Output JSON path to write (stable filename for DVC metrics).",
    )
    args = parser.parse_args()

    work_dir = Path(args.work_dir)
    out_path = Path(args.out)

    latest_metrics = _find_latest_timestamp_json(work_dir)
    raw = json.loads(latest_metrics.read_text())

    data: dict = {}
    # Keep a compact summary so `dvc exp show` remains readable.
    keep_scalar_keys = {
        'mIoU',
        'mIoU_binary',
        'mPQ',
        'mSQ',
        'mRQ',
        'mPrecision',
        'mRecall',
        'F1',
        'mMWCov',
        'mMUCov',
        'time',
        'data_time',
    }
    for key in keep_scalar_keys:
        if key in raw:
            data[key] = raw[key]

    log_path = _find_matching_log(work_dir, latest_metrics)
    if log_path is not None:
        extra = _parse_metrics_from_log(log_path)
        # Keep only the compact/structured extras (avoid verbose duplicates).
        keep_extra_keys = {
            'sem_iou',
            'sem_iou_pct',
            'binary_iou',
            'binary_iou_pct',
            'ordered_semantic_iou_pct',
            'ordered_binary_iou_pct',
            'sem_counts',
            'binary_counts',
            'instance_summary',
            'individual_tree_seg_pct',
        }
        for key in keep_extra_keys:
            if key in extra:
                data[key] = extra[key]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")

    print(f"Collected metrics: {latest_metrics} -> {out_path}")


if __name__ == "__main__":
    main()
