"""Internal consistency diagnostics, as defined in the paper (Section 3, Validation).

* :func:`voxel_label_agreement` — Step 2 voxel labels vs Step 5 consolidated
  labels at common occupied voxels: overall accuracy and class-wise IoU.
* :func:`height_plausibility` — per refined class, the median ``Z_norm`` and
  the share of points outside the configured class height range.
* :func:`crown_retention` — final non-zero treeIDs as a percentage of the
  watershed-derived crowns.

These check that the workflow is internally consistent; they are not an
independent ground-truth accuracy assessment. :func:`label_agreement` (2D
raster vs 3D labels, per pixel) is an extra check, not one of the paper's.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .classes import ClassScheme
from .io import LabelRaster
from .label_transfer import pixel_indices


def _majority(keys: np.ndarray, classes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Majority class per key (ties go to the lower class code)."""
    if len(keys) == 0:
        return np.array([], dtype=np.int64), np.array([], dtype=np.int16)
    pair, counts = np.unique(keys * 256 + classes.astype(np.int64), return_counts=True)
    pair_key, pair_cls = pair // 256, pair % 256
    order = np.lexsort((-counts, pair_key))
    first = np.ones(len(order), dtype=bool)
    first[1:] = pair_key[order][1:] != pair_key[order][:-1]
    return pair_key[order][first], pair_cls[order][first].astype(np.int16)


def _agreement(cells_a, cls_a, cells_b, cls_b, scheme: ClassScheme) -> dict:
    common, i_a, i_b = np.intersect1d(cells_a, cells_b, return_indices=True)
    out = {
        "overall_accuracy_pct": float("nan"),
        "common": int(len(common)),
        "iou": {scheme.name(c): float("nan") for c in scheme.codes},
    }
    if len(common) == 0:
        return out
    a, b = cls_a[i_a], cls_b[i_b]
    out["overall_accuracy_pct"] = float(np.mean(a == b) * 100.0)
    for code in scheme.codes:
        tp = int(np.sum((a == code) & (b == code)))
        denom = tp + int(np.sum((a != code) & (b == code))) + int(np.sum((a == code) & (b != code)))
        out["iou"][scheme.name(code)] = tp / denom if denom else float("nan")
    return out


def voxel_label_agreement(xyz, step2, step5, voxel_xy: float = 0.10, voxel_z: float = 0.10, classes=None) -> dict:
    """Overall accuracy and class-wise IoU between Step 2 and Step 5 labels.

    Points are grouped into voxels of the Stage 2 size; each voxel takes the
    majority label of its labelled points at each step, and the two are
    compared at voxels labelled in both (the paper's Eq. 4). Labels must be
    in the scheme's own classes (before any output relabelling).
    """
    scheme = ClassScheme.from_any(classes)
    xyz = np.asarray(xyz, dtype=float)
    vox = np.floor((xyz - xyz.min(axis=0)) / np.array([voxel_xy, voxel_xy, voxel_z])).astype(np.int64)
    dims = vox.max(axis=0) + 1
    keys = (vox[:, 0] * dims[1] + vox[:, 1]) * dims[2] + vox[:, 2]
    result = []
    for labels in (np.asarray(step2), np.asarray(step5)):
        keep = np.isin(labels, scheme.codes)
        result.extend(_majority(keys[keep], labels[keep]))
    out = _agreement(*result, scheme)
    out["common_voxels"] = out.pop("common")
    # As in the paper, classes that Stage 5 removed entirely (e.g. woody debris,
    # merged into mallee or dropped) are reported as absent, not as IoU 0.
    before, after = set(result[1].tolist()), set(result[3].tolist())
    out["absent_after_consolidation"] = [scheme.name(c) for c in scheme.codes if c in before and c not in after]
    for name in out["absent_after_consolidation"]:
        out["iou"][name] = float("nan")
    return out


def height_plausibility(z_norm, classes, ranges: dict | None = None, scheme=None) -> list[dict]:
    """Per refined class: point count, median and IQR of ``Z_norm``, and % outside the configured range.

    ``ranges`` defaults to the scheme's configured height ranges (default scheme: the paper's).
    """
    scheme = ClassScheme.from_any(scheme)
    ranges = ranges or scheme.height_ranges
    z_norm = np.asarray(z_norm, dtype=np.float32)
    classes = np.asarray(classes)
    rows = []
    for code in scheme.codes:
        values = z_norm[classes == code]
        lo, hi = ranges[code]
        row = {
            "class": scheme.name(code),
            "code": code,
            "points": int(len(values)),
            "configured_min_m": float(lo),
            "configured_max_m": float(hi),
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


def crown_retention(watershed_crowns: int, final_treeids: int) -> dict:
    """Final non-zero treeIDs as a percentage of watershed crowns (the paper's Eq. 5)."""
    pct = 100.0 * final_treeids / watershed_crowns if watershed_crowns else float("nan")
    return {"watershed_crowns": int(watershed_crowns), "final_treeids": int(final_treeids), "retention_pct": pct}


def label_agreement(labels: LabelRaster, reference: tuple, refined: tuple, classes=None) -> dict:
    """Extra check (not in the paper): 2D raster labels vs 3D labels, one majority class per pixel.

    ``reference`` and ``refined`` are ``(x, y, classes)`` tuples.
    """
    scheme = ClassScheme.from_any(classes)
    cells = []
    for x, y, cls in (reference, refined):
        cls = np.asarray(cls)
        keep = np.isin(cls, scheme.codes)
        rows, cols, inside = pixel_indices(labels, np.asarray(x)[keep], np.asarray(y)[keep])
        cells.extend(_majority(rows[inside] * labels.shape[1] + cols[inside], cls[keep][inside]))
    out = _agreement(*cells, scheme)
    out["common_pixels"] = out.pop("common")
    return out


@dataclass
class QualityReport:
    voxel_agreement: dict
    height_plausibility: list[dict]
    crown_retention: dict | None = None
    crown_name: str = "crown"
    warnings: list[str] = field(default_factory=list)

    def summary(self) -> str:
        va = self.voxel_agreement
        lines = [
            f"Voxel-level label agreement (Step 2 vs Step 5): OA {va['overall_accuracy_pct']:.1f}% "
            f"over {va['common_voxels']:,} voxels",
            "  IoU " + ", ".join(f"{k} {v:.2f}" for k, v in va["iou"].items() if np.isfinite(v)),
            *(["  absent after consolidation: " + ", ".join(va["absent_after_consolidation"])]
              if va.get("absent_after_consolidation") else []),
            "Height plausibility of refined classes (median Z_norm, % outside configured range):",
        ]
        for row in self.height_plausibility:
            if row["points"]:
                lines.append(
                    f"  {row['class']:<13} n={row['points']:>9,}  median {row['median_m']:.2f} m  "
                    f"outside {row['outside_range_pct']:.1f}%"
                )
        cr = self.crown_retention
        if cr is not None:
            lines.append(
                f"{self.crown_name.capitalize()} crown-instance retention: {cr['final_treeids']} treeIDs / "
                f"{cr['watershed_crowns']} watershed crowns = {cr['retention_pct']:.1f}%"
            )
        lines += [f"  warning: {w}" for w in self.warnings]
        return "\n".join(lines)


def quality_report(
    xyz,
    step2,
    step5,
    z_norm,
    *,
    voxel_xy: float = 0.10,
    voxel_z: float = 0.10,
    watershed_crowns: int | None = None,
    final_treeids: int | None = None,
    classes=None,
    min_accuracy_pct: float = 70.0,
) -> QualityReport:
    """The paper's three diagnostics for one tile.

    ``step2`` and ``step5`` are per-point labels (in the scheme's own classes)
    aligned with ``xyz`` and ``z_norm``. Crown retention is reported when the
    scheme has a crown class and both crown counts are given. A warning flags
    voxel agreement below ``min_accuracy_pct``.
    """
    scheme = ClassScheme.from_any(classes)
    step5 = np.asarray(step5)
    agreement = voxel_label_agreement(xyz, step2, step5, voxel_xy, voxel_z, classes=scheme)
    labelled = step5 != 0
    heights = height_plausibility(np.asarray(z_norm)[labelled], step5[labelled], scheme=scheme)
    retention = None
    if scheme.crown_class is not None and watershed_crowns is not None and final_treeids is not None:
        retention = crown_retention(watershed_crowns, final_treeids)
    warnings = []
    oa = agreement["overall_accuracy_pct"]
    if np.isfinite(oa) and oa < min_accuracy_pct:
        warnings.append(f"voxel-level agreement {oa:.1f}% is below {min_accuracy_pct:.0f}%")
    crown_name = scheme.name(scheme.crown_class) if scheme.crown_class is not None else "crown"
    return QualityReport(agreement, heights, retention, crown_name, warnings)
