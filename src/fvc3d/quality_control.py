"""Quality checks on transferred labels (the paper's Table 5 and Table 6 metrics)."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .io import CLASS_NAMES, STRUCTURE_CODES, LabelRaster
from .label_transfer import pixel_indices
from .refinement import DEFAULT_HEIGHT_RANGES


def majority_per_pixel(labels: LabelRaster, x, y, classes) -> tuple[np.ndarray, np.ndarray]:
    """Majority class of the points in each occupied pixel (ties go to the lower code).

    Returns flat pixel ids and their majority class.
    """
    classes = np.asarray(classes)
    keep = np.isin(classes, STRUCTURE_CODES)
    rows, cols, inside = pixel_indices(labels, np.asarray(x)[keep], np.asarray(y)[keep])
    cls = classes[keep][inside].astype(np.int64)
    if len(cls) == 0:
        return np.array([], dtype=np.int64), np.array([], dtype=np.int16)
    cells = rows[inside] * labels.shape[1] + cols[inside]
    pair, counts = np.unique(cells * 256 + cls, return_counts=True)
    pair_cell, pair_cls = pair // 256, pair % 256
    order = np.lexsort((-counts, pair_cell))
    first = np.ones(len(order), dtype=bool)
    first[1:] = pair_cell[order][1:] != pair_cell[order][:-1]
    return pair_cell[order][first], pair_cls[order][first].astype(np.int16)


def label_agreement(labels: LabelRaster, reference: tuple, refined: tuple) -> dict:
    """Overall accuracy and per-class IoU between two labelled clouds, per pixel.

    ``reference`` and ``refined`` are ``(x, y, classes)`` tuples, usually the
    labels draped from the raster and the final transferred labels. Both are
    reduced to one majority class per pixel and compared on pixels both occupy.
    """
    ref_cells, ref_cls = majority_per_pixel(labels, *reference)
    new_cells, new_cls = majority_per_pixel(labels, *refined)
    common, i_ref, i_new = np.intersect1d(ref_cells, new_cells, return_indices=True)
    out = {
        "overall_accuracy_pct": float("nan"),
        "common_pixels": int(len(common)),
        "reference_pixels": int(len(ref_cells)),
        "refined_pixels": int(len(new_cells)),
        "iou": {CLASS_NAMES[c]: float("nan") for c in STRUCTURE_CODES},
    }
    if len(common) == 0:
        return out
    a, b = ref_cls[i_ref], new_cls[i_new]
    out["overall_accuracy_pct"] = float(np.mean(a == b) * 100.0)
    for code in STRUCTURE_CODES:
        tp = int(np.sum((a == code) & (b == code)))
        denom = tp + int(np.sum((a != code) & (b == code))) + int(np.sum((a == code) & (b != code)))
        out["iou"][CLASS_NAMES[code]] = tp / denom if denom else float("nan")
    return out


def height_plausibility(z_norm, classes, ranges: dict | None = None) -> list[dict]:
    """Per class: point count, median and IQR of ``Z_norm``, and % outside the expected range."""
    ranges = ranges or DEFAULT_HEIGHT_RANGES
    z_norm = np.asarray(z_norm, dtype=np.float32)
    classes = np.asarray(classes)
    rows = []
    for code in STRUCTURE_CODES:
        values = z_norm[classes == code]
        lo, hi = ranges[code]
        row = {
            "class": CLASS_NAMES[code],
            "code": code,
            "points": int(len(values)),
            "expected_min_m": float(lo),
            "expected_max_m": float(hi),
            "median_m": float("nan"),
            "iqr_m": float("nan"),
            "outside_range_pct": float("nan"),
        }
        if len(values):
            q1, q3 = np.percentile(values, [25, 75])
            row["median_m"] = float(np.median(values))
            row["iqr_m"] = float(q3 - q1)
            row["outside_range_pct"] = float(((values < lo) | (values > hi)).mean() * 100.0)
        rows.append(row)
    return rows


@dataclass
class QualityReport:
    label_agreement: dict
    height_plausibility: list[dict]
    warnings: list[str] = field(default_factory=list)

    def summary(self) -> str:
        la = self.label_agreement
        lines = [
            f"Label agreement with raster: OA {la['overall_accuracy_pct']:.1f}% "
            f"over {la['common_pixels']:,} pixels",
            "  IoU " + ", ".join(f"{k} {v:.2f}" for k, v in la["iou"].items() if np.isfinite(v)),
            "Height plausibility (median Z_norm, % outside expected range):",
        ]
        for row in self.height_plausibility:
            if row["points"]:
                lines.append(
                    f"  {row['class']:<13} n={row['points']:>9,}  median {row['median_m']:.2f} m  "
                    f"outside {row['outside_range_pct']:.1f}%"
                )
        lines += [f"  warning: {w}" for w in self.warnings]
        return "\n".join(lines)


def quality_report(
    labels: LabelRaster,
    draped: tuple,
    final: tuple,
    z_norm_final,
    *,
    min_accuracy_pct: float = 70.0,
) -> QualityReport:
    """Build a :class:`QualityReport`.

    ``draped`` and ``final`` are ``(x, y, classes)``; ``z_norm_final`` aligns
    with ``final``. A warning flags accuracy below ``min_accuracy_pct``.

    Height plausibility is reported but not warned on: every point under a
    labelled pixel keeps the label, including ground under canopy, so a large
    share of low mallee and shrub points is expected (about 75% of mallee
    points lie below 1.5 m on the paper's tile 109).
    """
    agreement = label_agreement(labels, draped, final)
    heights = height_plausibility(z_norm_final, final[2])
    warnings = []
    oa = agreement["overall_accuracy_pct"]
    if np.isfinite(oa) and oa < min_accuracy_pct:
        warnings.append(f"overall accuracy {oa:.1f}% is below {min_accuracy_pct:.0f}%")
    return QualityReport(agreement, heights, warnings)
