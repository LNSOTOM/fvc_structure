"""Checks that labels and LiDAR are aligned before labels are transferred.

This module does not co-register data. Label transfer assumes the LiDAR is
already registered to the imagery the labels were drawn on (Phase 2 GeFolki
plus ICP in the original workflow). :func:`check_alignment` verifies that
assumption:

* both inputs have the same projected CRS in metres;
* the point cloud and the labelled area overlap;
* there is no residual horizontal offset: the LiDAR canopy (points with
  ``Z_norm`` at or above ``canopy_height``) is phase-correlated with the
  mallee pixels of the label raster, giving the XY shift between them.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .io import LabelRaster, PointCloud
from .label_transfer import pixel_indices
from .refinement import MALLEE


class AlignmentError(ValueError):
    """Raised when labels and point cloud fail the alignment checks."""


@dataclass
class AlignmentReport:
    crs_match: bool
    crs_projected_metres: bool
    points_on_labels_pct: float
    labelled_pixels_with_points_pct: float
    shift_x: float | None = None
    shift_y: float | None = None
    canopy_iou: float | None = None
    canopy_iou_after_shift: float | None = None
    max_shift: float = 0.50
    problems: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def shift(self) -> float | None:
        """Horizontal offset (m) of the LiDAR canopy relative to the labels."""
        if self.shift_x is None:
            return None
        return float(np.hypot(self.shift_x, self.shift_y))

    @property
    def ok(self) -> bool:
        return not self.problems

    def summary(self) -> str:
        lines = [f"Alignment: {'OK' if self.ok else 'FAILED'}"]
        lines.append(f"  CRS match: {self.crs_match}; projected metres: {self.crs_projected_metres}")
        lines.append(f"  points over the label raster: {self.points_on_labels_pct:.1f}%")
        lines.append(f"  labelled pixels with LiDAR points: {self.labelled_pixels_with_points_pct:.1f}%")
        if self.shift is not None:
            lines.append(
                f"  canopy offset: dx={self.shift_x:+.3f} m, dy={self.shift_y:+.3f} m "
                f"(|d|={self.shift:.3f} m, limit {self.max_shift:.2f} m)"
            )
            lines.append(
                f"  canopy IoU: {self.canopy_iou:.2f} as given, {self.canopy_iou_after_shift:.2f} after shift"
            )
        lines += [f"  problem: {p}" for p in self.problems]
        lines += [f"  note: {n}" for n in self.notes]
        return "\n".join(lines)


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    union = np.count_nonzero(a | b)
    return float(np.count_nonzero(a & b) / union) if union else float("nan")


def _shift_mask(mask: np.ndarray, rows: int, cols: int) -> np.ndarray:
    out = np.zeros_like(mask)
    h, w = mask.shape
    src_r = slice(max(0, -rows), min(h, h - rows))
    dst_r = slice(max(0, rows), min(h, h + rows))
    src_c = slice(max(0, -cols), min(w, w - cols))
    dst_c = slice(max(0, cols), min(w, w + cols))
    out[dst_r, dst_c] = mask[src_r, src_c]
    return out


def estimate_canopy_shift(
    labels: LabelRaster,
    x, y, z_norm,
    canopy_height: float = 1.0,
    min_pixels: int = 200,
) -> tuple[float, float, float, float] | None:
    """Estimate the XY offset (m) of LiDAR canopy relative to mallee labels.

    Returns ``(dx, dy, iou_before, iou_after)`` where (dx, dy) is how far the
    LiDAR is displaced from the labels (east, north), or ``None`` when either
    mask has fewer than ``min_pixels`` pixels.
    """
    from skimage.registration import phase_cross_correlation

    label_mask = labels.data == MALLEE
    rows, cols, inside = pixel_indices(labels, x, y)
    tall = inside & (np.asarray(z_norm) >= canopy_height)
    lidar_mask = np.zeros(labels.shape, dtype=bool)
    lidar_mask[rows[tall], cols[tall]] = True
    if np.count_nonzero(label_mask) < min_pixels or np.count_nonzero(lidar_mask) < min_pixels:
        return None

    shift, _, _ = phase_cross_correlation(
        label_mask.astype(float), lidar_mask.astype(float), upsample_factor=10, normalization=None
    )
    # `shift` moves the LiDAR mask onto the labels; the LiDAR offset is its negative.
    d_row, d_col = -float(shift[0]), -float(shift[1])
    res_x, res_y = labels.resolution
    dx = d_col * res_x * np.sign(labels.transform.a)
    dy = d_row * res_y * np.sign(labels.transform.e)
    iou_before = _iou(label_mask, lidar_mask)
    iou_after = _iou(label_mask, _shift_mask(lidar_mask, int(round(-d_row)), int(round(-d_col))))
    return float(dx), float(dy), iou_before, iou_after


def check_alignment(
    labels: LabelRaster,
    cloud: PointCloud,
    z_norm: np.ndarray | None = None,
    max_shift: float = 0.50,
    canopy_height: float = 1.0,
) -> AlignmentReport:
    """Check that ``labels`` and ``cloud`` line up; never raises.

    ``max_shift`` (m) is the largest canopy offset accepted. Pass ``z_norm``
    if the cloud has no ``Z_norm`` dimension, to enable the offset check.
    """
    problems, notes = [], []

    crs_match = labels.crs is not None and cloud.crs is not None and labels.crs.equals(
        cloud.crs, ignore_axis_order=True
    )
    if labels.crs is None or cloud.crs is None:
        problems.append("labels and point cloud must both carry a CRS")
    elif not crs_match:
        problems.append(
            f"CRS differ (labels {labels.crs.to_string()}, point cloud {cloud.crs.to_string()}); "
            "reproject one of them first"
        )
    projected = bool(cloud.crs is not None and cloud.crs.is_projected and all(
        ax.unit_name in ("metre", "meter") for ax in cloud.crs.axis_info[:2]
    ))
    if cloud.crs is not None and not projected:
        problems.append("point cloud CRS must be projected, in metres")

    rows, cols, inside = pixel_indices(labels, cloud.x, cloud.y)
    on_raster = 100.0 * np.count_nonzero(inside) / len(cloud) if len(cloud) else 0.0
    labelled = labels.data > 0
    hit = np.zeros(labels.shape, dtype=bool)
    hit[rows[inside], cols[inside]] = True
    covered = 100.0 * np.count_nonzero(hit & labelled) / max(1, np.count_nonzero(labelled))
    if np.count_nonzero(inside) == 0:
        problems.append("point cloud does not overlap the label raster")
    elif covered < 50.0:
        notes.append(f"only {covered:.0f}% of labelled pixels contain LiDAR points")

    report = AlignmentReport(
        crs_match=crs_match,
        crs_projected_metres=projected,
        points_on_labels_pct=on_raster,
        labelled_pixels_with_points_pct=covered,
        max_shift=max_shift,
        problems=problems,
        notes=notes,
    )

    zn = z_norm if z_norm is not None else cloud.z_norm
    if problems:
        return report
    if zn is None:
        notes.append("canopy offset not measured: no Z_norm")
        return report

    est = estimate_canopy_shift(labels, cloud.x, cloud.y, zn, canopy_height=canopy_height)
    if est is None:
        notes.append("canopy offset not measured: too little mallee canopy in labels or LiDAR")
        return report
    report.shift_x, report.shift_y, report.canopy_iou, report.canopy_iou_after_shift = est
    if report.shift > max_shift:
        problems.append(
            f"LiDAR canopy is offset {report.shift:.2f} m from the labels "
            f"(dx={report.shift_x:+.2f}, dy={report.shift_y:+.2f}); limit is {max_shift:.2f} m. "
            "Co-register the LiDAR to the imagery first, or raise max_shift"
        )
    return report
