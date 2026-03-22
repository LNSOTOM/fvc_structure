"""Plot offline instance metrics exported by `instance_metrics_from_laz.py`.

Inputs:
  - per-file CSV (per tile)
  - per-region CSV (aggregated)

Outputs (by default under --out-dir):
  - region_metrics.png : 2x2 bar charts of MicroF1, MacroF1, MacroCov, MacroWCov
  - file_scatter_f1_cov.png : scatter of per-file F1 vs Coverage, colored by region

Example:
  python plot_instance_metrics_from_csv.py \
    --per-file-csv outputs/metrics_20260310_iou0p5_per_file.csv \
    --per-region-csv outputs/metrics_20260310_iou0p5_per_region.csv \
    --out-dir outputs/plots_iou0p5
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np


def _to_float(x: str) -> float:
    x = (x or "").strip()
    if not x:
        return float("nan")
    try:
        return float(x)
    except ValueError:
        return float("nan")


@dataclass(frozen=True)
class FileRow:
    file: str
    region: str
    tp: int
    fp: int
    fn: int
    precision: float
    recall: float
    f1: float
    coverage: float
    weighted_coverage: float


@dataclass(frozen=True)
class RegionRow:
    region: str
    files: int
    tp: int
    fp: int
    fn: int
    micro_precision: float
    micro_recall: float
    micro_f1: float
    macro_f1: float
    macro_coverage: float
    macro_weighted_coverage: float


def _read_csv_dicts(path: Path) -> List[Dict[str, str]]:
    with path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        return [dict(r) for r in reader]


def read_per_file_csv(path: Path) -> List[FileRow]:
    rows = _read_csv_dicts(path)
    out: List[FileRow] = []
    for r in rows:
        out.append(
            FileRow(
                file=r["file"],
                region=r.get("region", ""),
                tp=int(r.get("tp", "0") or 0),
                fp=int(r.get("fp", "0") or 0),
                fn=int(r.get("fn", "0") or 0),
                precision=_to_float(r.get("precision", "")),
                recall=_to_float(r.get("recall", "")),
                f1=_to_float(r.get("f1", "")),
                coverage=_to_float(r.get("coverage", "")),
                weighted_coverage=_to_float(r.get("weighted_coverage", "")),
            )
        )
    return out


def read_per_region_csv(path: Path) -> List[RegionRow]:
    rows = _read_csv_dicts(path)
    out: List[RegionRow] = []
    for r in rows:
        out.append(
            RegionRow(
                region=r["region"],
                files=int(r.get("files", "0") or 0),
                tp=int(r.get("tp", "0") or 0),
                fp=int(r.get("fp", "0") or 0),
                fn=int(r.get("fn", "0") or 0),
                micro_precision=_to_float(r.get("micro_precision", "")),
                micro_recall=_to_float(r.get("micro_recall", "")),
                micro_f1=_to_float(r.get("micro_f1", "")),
                macro_f1=_to_float(r.get("macro_f1", "")),
                macro_coverage=_to_float(r.get("macro_coverage", "")),
                macro_weighted_coverage=_to_float(r.get("macro_weighted_coverage", "")),
            )
        )
    return out


def _nan_to_num_for_plot(x: np.ndarray) -> np.ndarray:
    # Keep NaNs as NaNs for bar charts (matplotlib will skip), but cap outside [0,1].
    y = x.astype(float)
    y[(y < 0) | (y > 1)] = np.nan
    return y


def _safe_import_matplotlib():
    try:
        import matplotlib.pyplot as plt  # noqa: WPS433

        return plt
    except Exception as e:  # pragma: no cover
        raise SystemExit(
            "Missing matplotlib. Install with: pip install matplotlib"
        ) from e


def plot_region_bars(regions: List[RegionRow], out_path: Path, *, sort_by: str = "macro_f1") -> None:
    plt = _safe_import_matplotlib()

    # Sorting
    sort_key = {
        "micro_f1": lambda r: r.micro_f1,
        "macro_f1": lambda r: r.macro_f1,
        "macro_coverage": lambda r: r.macro_coverage,
        "macro_weighted_coverage": lambda r: r.macro_weighted_coverage,
    }.get(sort_by)
    if sort_key is None:
        raise ValueError(f"Unknown sort_by={sort_by}")

    regions_sorted = sorted(regions, key=sort_key, reverse=True)
    names = [r.region for r in regions_sorted]

    micro_f1 = _nan_to_num_for_plot(np.array([r.micro_f1 for r in regions_sorted]))
    macro_f1 = _nan_to_num_for_plot(np.array([r.macro_f1 for r in regions_sorted]))
    macro_cov = _nan_to_num_for_plot(np.array([r.macro_coverage for r in regions_sorted]))
    macro_wcov = _nan_to_num_for_plot(np.array([r.macro_weighted_coverage for r in regions_sorted]))

    plt.rcParams.update(
        {
            "figure.dpi": 140,
            "savefig.dpi": 200,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )

    fig, axs = plt.subplots(2, 2, figsize=(14, 9), sharex=True)
    axs = axs.ravel()

    def bar(ax, values, title):
        ax.bar(names, values * 100.0)
        ax.set_title(title)
        ax.set_ylabel("%")
        ax.set_ylim(0, 100)
        ax.tick_params(axis="x", rotation=45, labelsize=9)

    bar(axs[0], micro_f1, "Micro F1 (by region)")
    bar(axs[1], macro_f1, "Macro F1 (mean of files)")
    bar(axs[2], macro_cov, "Macro Coverage (mean of files)")
    bar(axs[3], macro_wcov, "Macro Weighted Coverage (mean of files)")

    fig.suptitle("Offline Instance Metrics (IoU threshold as in CSV)", fontsize=14, y=0.98)
    fig.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_file_scatter(rows: List[FileRow], out_path: Path) -> None:
    plt = _safe_import_matplotlib()

    # Map region -> color index
    regions = sorted({r.region for r in rows})
    region_to_idx = {r: i for i, r in enumerate(regions)}

    x = np.array([r.coverage for r in rows], dtype=float)
    y = np.array([r.f1 for r in rows], dtype=float)
    c = np.array([region_to_idx.get(r.region, 0) for r in rows], dtype=int)

    x = _nan_to_num_for_plot(x)
    y = _nan_to_num_for_plot(y)

    plt.rcParams.update(
        {
            "figure.dpi": 140,
            "savefig.dpi": 200,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )

    fig, ax = plt.subplots(figsize=(10, 7))
    sc = ax.scatter(x * 100.0, y * 100.0, c=c, cmap="tab10", s=60, alpha=0.9, edgecolors="none")

    ax.set_xlabel("Coverage (%)")
    ax.set_ylabel("F1 (%)")
    ax.set_title("Per-file F1 vs Coverage (colored by region)")
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)

    # Legend (region colors)
    handles = []
    labels = []
    for region, idx in region_to_idx.items():
        handles.append(plt.Line2D([0], [0], marker="o", linestyle="", color=plt.get_cmap("tab10")(idx % 10), markersize=8))
        labels.append(region)
    ax.legend(handles, labels, title="Region", loc="lower right", frameon=True)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description="Plot offline instance metrics from exported CSV")
    ap.add_argument(
        "--per-file-csv",
        default="outputs/metrics_20260310_iou0p5_per_file.csv",
        help="Path to per-file CSV written by instance_metrics_from_laz.py",
    )
    ap.add_argument(
        "--per-region-csv",
        default="outputs/metrics_20260310_iou0p5_per_region.csv",
        help="Path to per-region CSV written by instance_metrics_from_laz.py",
    )
    ap.add_argument(
        "--out-dir",
        default="outputs/plots",
        help="Output directory for plots",
    )
    ap.add_argument(
        "--region-sort-by",
        default="macro_f1",
        choices=["micro_f1", "macro_f1", "macro_coverage", "macro_weighted_coverage"],
        help="How to sort regions in bar charts",
    )
    args = ap.parse_args()

    per_file_path = Path(args.per_file_csv)
    per_region_path = Path(args.per_region_csv)
    out_dir = Path(args.out_dir)

    if not per_file_path.exists():
        raise SystemExit(f"Missing per-file CSV: {per_file_path}")
    if not per_region_path.exists():
        raise SystemExit(f"Missing per-region CSV: {per_region_path}")

    file_rows = read_per_file_csv(per_file_path)
    region_rows = read_per_region_csv(per_region_path)

    plot_region_bars(region_rows, out_dir / "region_metrics.png", sort_by=args.region_sort_by)
    plot_file_scatter(file_rows, out_dir / "file_scatter_f1_cov.png")

    print(f"Saved: {out_dir / 'region_metrics.png'}")
    print(f"Saved: {out_dir / 'file_scatter_f1_cov.png'}")


if __name__ == "__main__":
    main()
