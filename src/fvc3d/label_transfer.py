"""Moving 2D labels onto 3D points.

Three steps, each usable on its own:

* :func:`drape_labels` gives every LiDAR point the label of the raster pixel it
  falls in (what ``maskclip`` did with PDAL ``filters.colorization``).
* :func:`height_above_ground` computes ``Z_norm`` from ground points
  (the ``groundnorm normalize`` IDW interpolation).
* :func:`voxel_transfer` moves labels between clouds by majority vote in a
  shared 3D voxel grid (Stage 2 of the method).
"""

from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

from .io import STRUCTURE_CODES, LabelRaster, PointCloud


def pixel_indices(labels: LabelRaster, x, y) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Row/column of the pixel containing each (x, y), and whether it is inside the raster."""
    inv = ~labels.transform
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    col_f = inv.a * x + inv.b * y + inv.c
    row_f = inv.d * x + inv.e * y + inv.f
    rows = np.floor(row_f).astype(np.int64)
    cols = np.floor(col_f).astype(np.int64)
    n_rows, n_cols = labels.shape
    inside = (rows >= 0) & (rows < n_rows) & (cols >= 0) & (cols < n_cols)
    return rows, cols, inside


def drape_labels(labels: LabelRaster, x, y) -> np.ndarray:
    """Label of the pixel under each point (0 outside the raster or where unlabelled)."""
    rows, cols, inside = pixel_indices(labels, x, y)
    out = np.zeros(len(rows), dtype=np.uint8)
    out[inside] = labels.data[rows[inside], cols[inside]]
    return out


def height_above_ground(x, y, z, ground_xyz: np.ndarray, k: int = 3) -> np.ndarray:
    """Height above ground by inverse-distance weighting of the ``k`` XY-nearest ground points.

    Negative heights are clipped to 0. Matches ``groundnorm.normalize_las_z``.
    """
    ground_xyz = np.asarray(ground_xyz, dtype=float)
    if len(ground_xyz) < k:
        raise ValueError(f"Need at least {k} ground points to compute Z_norm, got {len(ground_xyz)}")
    tree = cKDTree(ground_xyz[:, :2])
    dist, idx = tree.query(np.column_stack((x, y)), k=k, workers=-1)
    if dist.ndim == 1:
        dist, idx = dist[:, None], idx[:, None]
    weights = 1.0 / np.maximum(dist, 1e-6)
    weights /= weights.sum(axis=1, keepdims=True)
    ground_z = np.sum(ground_xyz[idx, 2] * weights, axis=1)
    return np.maximum(np.asarray(z, dtype=float) - ground_z, 0.0).astype(np.float32).astype(float)


def resolve_z_norm(cloud: PointCloud, ground: PointCloud | None = None, k: int = 3) -> tuple[np.ndarray, str]:
    """Return ``Z_norm`` for ``cloud`` and a note on where it came from.

    In order of preference: an existing ``Z_norm`` dimension; ``ground`` (a
    ground-only cloud, e.g. from CSF); points of ``cloud`` already classified
    as ground (class 2).
    """
    if cloud.z_norm is not None and ground is None:
        return cloud.z_norm, "Z_norm dimension in point cloud"
    if ground is not None:
        ground_xyz = np.column_stack((ground.x, ground.y, ground.z))
        source = f"ground points from {ground.path.name if ground.path else 'ground cloud'}"
    else:
        is_ground = cloud.classification == 2 if cloud.classification is not None else np.zeros(len(cloud), bool)
        if np.count_nonzero(is_ground) < k:
            raise ValueError(
                "The point cloud has no Z_norm dimension and no ground points (class 2). "
                "Classify ground first (e.g. `groundnorm csf in.las ground.las`, or PDAL "
                "filters.csf) and pass it as ground=..., or provide a cloud with Z_norm."
            )
        ground_xyz = np.column_stack((cloud.x[is_ground], cloud.y[is_ground], cloud.z[is_ground]))
        source = "class-2 ground points in point cloud"
    return height_above_ground(cloud.x, cloud.y, cloud.z, ground_xyz, k=k), source


def voxel_transfer(
    target_xyz: np.ndarray,
    source_xyz: np.ndarray,
    source_classes: np.ndarray,
    voxel_xy: float = 0.10,
    voxel_z: float = 0.10,
) -> tuple[np.ndarray, dict]:
    """Give each target point the majority source label of its voxel.

    The grid origin is the minimum corner of both clouds. Ties go to the lower
    class code. Source labels outside 64–68 are ignored; target points in a
    voxel with no labelled source points get 0.

    Returns the per-target-point labels and a stats dict.
    """
    target_xyz = np.asarray(target_xyz, dtype=float)
    source_xyz = np.asarray(source_xyz, dtype=float)
    source_classes = np.asarray(source_classes)

    origin = np.minimum(target_xyz.min(axis=0), source_xyz.min(axis=0))
    size = np.array([voxel_xy, voxel_xy, voxel_z], dtype=float)
    vox_t = np.floor((target_xyz - origin) / size).astype(np.int64)
    vox_s = np.floor((source_xyz - origin) / size).astype(np.int64)
    dims = np.maximum(vox_t.max(axis=0), vox_s.max(axis=0)) + 1

    def encode(v):
        return (v[:, 0] * dims[1] + v[:, 1]) * dims[2] + v[:, 2]

    valid = np.isin(source_classes, STRUCTURE_CODES)
    out = np.zeros(len(target_xyz), dtype=np.uint8)
    n_voxels = 0
    if np.any(valid):
        keys = encode(vox_s[valid])
        cls = source_classes[valid].astype(np.int64)
        # Count (voxel, class) pairs, sorted by voxel then class.
        pair = keys * 256 + cls
        uniq_pair, counts = np.unique(pair, return_counts=True)
        pair_vox, pair_cls = uniq_pair // 256, uniq_pair % 256
        # Within each voxel take the highest count; stable sort keeps the lower class on ties.
        order = np.lexsort((-counts, pair_vox))
        first = np.ones(len(order), dtype=bool)
        first[1:] = pair_vox[order][1:] != pair_vox[order][:-1]
        voxel_keys = pair_vox[order][first]
        voxel_cls = pair_cls[order][first]
        n_voxels = len(voxel_keys)

        t_keys = encode(vox_t)
        pos = np.searchsorted(voxel_keys, t_keys)
        pos = np.clip(pos, 0, n_voxels - 1)
        hit = voxel_keys[pos] == t_keys
        out[hit] = voxel_cls[pos[hit]]

    n_labeled = int(np.count_nonzero(out))
    stats = {
        "voxel_size_xy": float(voxel_xy),
        "voxel_size_z": float(voxel_z),
        "nonzero_structure_voxels": int(n_voxels),
        "labeled_points": n_labeled,
        "total_points": int(len(out)),
        "labeled_pct": 100.0 * n_labeled / len(out) if len(out) else 0.0,
    }
    return out, stats
