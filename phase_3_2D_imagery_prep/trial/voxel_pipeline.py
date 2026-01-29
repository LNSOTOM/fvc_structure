"""Shared helpers for LiDAR → mask-aligned voxel processing."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import laspy
import numpy as np
import rasterio
from rasterio.transform import rowcol


@dataclass
class LidarPoints:
    coords: np.ndarray  # [N, 3] -> X, Y, Z_norm
    z_source: str
    total_points: int


@dataclass
class MaskSample:
    coords: np.ndarray
    labels: np.ndarray
    cols: np.ndarray
    rows: np.ndarray
    transform: rasterio.Affine
    bounds: rasterio.coords.BoundingBox
    mask_shape: tuple[int, int]
    pixel_size: float
    nodata: float | int | None
    kept_fraction: float
    drop_stats: dict[str, int]


@dataclass
class VoxelProducts:
    occupancy: np.ndarray
    point_counts: np.ndarray
    dominant_labels: np.ndarray
    metadata: dict


def select_z_dimension(
    las: laspy.LasData,
    override: str | None = None,
    priority: Sequence[str] = ("z_norm", "hag", "heightaboveground", "hag_delaunay"),
) -> tuple[str, np.ndarray]:
    extras = las.point_format.extra_dimensions or []
    if override:
        override_lower = override.lower()
        for extra in extras:
            if extra.name.lower() == override_lower:
                return extra.name, las[extra.name].astype(np.float64)
        raise ValueError(f"Requested Z dimension '{override}' not found in LAS file")

    for extra in extras:
        name_lower = extra.name.lower()
        if any(key in name_lower for key in priority):
            return extra.name, las[extra.name].astype(np.float64)

    for extra in extras:
        values = las[extra.name].astype(np.float64)
        if np.nanmax(values) > 0.0 and np.nanmax(values) <= 40 and np.nanmin(values) >= -5:
            return extra.name, values

    return "las.z", las.z.astype(np.float64)


def load_lidar_points(path: Path, z_override: str | None = None) -> LidarPoints:
    if not path.exists():
        raise FileNotFoundError(f"LAS/LAZ file not found: {path}")
    las = laspy.read(path)
    z_name, z_norm = select_z_dimension(las, override=z_override)
    coords = np.column_stack((las.x, las.y, z_norm))
    return LidarPoints(coords=coords, z_source=z_name, total_points=las.header.point_count)


def sample_mask(
    points: LidarPoints,
    mask_path: Path,
    ignore_labels: Iterable[float],
    exclude_classes: Iterable[float],
) -> MaskSample:
    if not mask_path.exists():
        raise FileNotFoundError(f"Mask raster not found: {mask_path}")

    ignore_set = {float(v) for v in ignore_labels}
    exclude_set = {float(v) for v in exclude_classes}

    with rasterio.open(mask_path) as src:
        mask = src.read(1)
        transform = src.transform
        bounds = src.bounds
        nodata = src.nodata
        pixel_size = float(abs(transform.a))
        if not np.isclose(pixel_size, abs(transform.e)):
            raise ValueError("Mask pixels are not square; cannot build cubic voxels")

        inside = (
            (points.coords[:, 0] >= bounds.left)
            & (points.coords[:, 0] <= bounds.right)
            & (points.coords[:, 1] >= bounds.bottom)
            & (points.coords[:, 1] <= bounds.top)
        )
        points_inside = points.coords[inside]
        if points_inside.size == 0:
            raise ValueError("No LiDAR points fall within mask bounds")

        rows, cols = rowcol(transform, points_inside[:, 0], points_inside[:, 1], op=float)
        rows = rows.astype(int)
        cols = cols.astype(int)

        valid = (
            (rows >= 0)
            & (rows < mask.shape[0])
            & (cols >= 0)
            & (cols < mask.shape[1])
        )

        points_valid = points_inside[valid]
        rows_valid = rows[valid]
        cols_valid = cols[valid]
        labels = mask[rows_valid, cols_valid].astype(mask.dtype)

    drop_stats = {
        "outside_mask_extent": int(points.coords.shape[0] - points_inside.shape[0]),
        "off_edge_pixels": int(points_inside.shape[0] - points_valid.shape[0]),
    }

    if nodata is not None:
        ignore_set.add(float(nodata))

    if np.issubdtype(labels.dtype, np.floating):
        finite = np.isfinite(labels)
    else:
        finite = np.ones(labels.shape, dtype=bool)
    keep = finite

    if ignore_set:
        keep &= ~np.isin(labels, list(ignore_set))
    if exclude_set:
        keep &= ~np.isin(labels, list(exclude_set))

    drop_stats["excluded_labels"] = int(labels.size - np.count_nonzero(keep))

    points_kept = points_valid[keep]
    rows_kept = rows_valid[keep]
    cols_kept = cols_valid[keep]
    labels_kept = labels[keep]
    if labels_kept.size == 0:
        raise ValueError("All points were filtered out by ignore/exclude label rules")

    kept_fraction = labels_kept.size / points.coords.shape[0]

    if not np.issubdtype(labels_kept.dtype, np.integer):
        labels_kept = labels_kept.astype(np.int32)

    return MaskSample(
        coords=points_kept,
        labels=labels_kept,
        cols=cols_kept.astype(np.int64),
        rows=rows_kept.astype(np.int64),
        transform=transform,
        bounds=bounds,
        mask_shape=mask.shape,
        pixel_size=pixel_size,
        nodata=nodata,
        kept_fraction=kept_fraction,
        drop_stats=drop_stats,
    )


def voxelise(sample: MaskSample, voxel_size: float | None = None) -> VoxelProducts:
    vs_xy = voxel_size if voxel_size else sample.pixel_size
    vs_z = vs_xy

    z_min = float(sample.coords[:, 2].min())
    z_max = float(sample.coords[:, 2].max())
    nz = int(np.ceil((z_max - z_min) / vs_z)) + 1

    ix = sample.cols
    iy = sample.rows
    iz = np.floor((sample.coords[:, 2] - z_min) / vs_z).astype(np.int64)
    iz = np.clip(iz, 0, nz - 1)

    nx = sample.mask_shape[1]
    ny = sample.mask_shape[0]
    grid_shape = (nx, ny, nz)

    count_vol = np.zeros(grid_shape, dtype=np.uint32)
    np.add.at(count_vol, (ix, iy, iz), 1)
    occupancy = (count_vol > 0).astype(np.uint8)

    flat_index = np.ravel_multi_index((ix, iy, iz), grid_shape, mode="raise")
    uniq_vox, inverse = np.unique(flat_index, return_inverse=True)

    uniq_labels = np.unique(sample.labels)
    label_lookup = np.searchsorted(uniq_labels, sample.labels)
    votes = np.zeros((uniq_vox.size, uniq_labels.size), dtype=np.uint32)
    np.add.at(votes, (inverse, label_lookup), 1)
    dom_idx = votes.argmax(axis=1)
    dominant_flat = uniq_labels[dom_idx].astype(np.int32)

    dominant_labels = np.full(np.prod(grid_shape), -1, dtype=np.int32)
    dominant_labels[uniq_vox] = dominant_flat
    dominant_labels = dominant_labels.reshape(grid_shape)
    dominant_labels[count_vol == 0] = -1

    meta = {
        "grid_shape": {
            "nx": int(nx),
            "ny": int(ny),
            "nz": int(nz),
        },
        "voxel_size_m": {
            "x": float(vs_xy),
            "y": float(vs_xy),
            "z": float(vs_z),
        },
        "origin": {
            "x": float(sample.transform.c),
            "y": float(sample.transform.f),
            "z": z_min,
        },
        "z_range": {
            "min": z_min,
            "max": z_max,
        },
        "occupied_voxels": int(occupancy.sum()),
        "total_points_used": int(sample.labels.size),
        "label_set": [int(v) for v in uniq_labels],
    }

    return VoxelProducts(
        occupancy=occupancy,
        point_counts=count_vol,
        dominant_labels=dominant_labels,
        metadata=meta,
    )
