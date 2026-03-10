#!/usr/bin/env python3
"""Derive rangeland structural classes from ForestFormer3D predictions.

Path A workflow (no retraining / no relabeling):
- Run the pretrained model to produce per-scene PLY outputs that include
    `semantic_pred` (0=ground, 1=wood, 2=leaf).
- Derive mallee as a *subset* of vegetation predictions (default: leaf only)
    using height-above-ground and spatial clustering filters.

This is intentionally heuristic.

Expected input PLY (written by `ForAINetV2OneFormer3D_XAwarequery.save_ply_withscore`):
- x, y, z (float)
- semantic_pred (int)
- instance_pred (int)
- score (float)
Optionally may contain semantic_gt/instance_gt.

Output:
- Writes a new PLY per input with extra vertex properties:
    - h_agl: estimated height above local ground (float32)
    - mallee_pred: 0/1 (derived)
    - woody_canopy_pred: 0/1 (derived umbrella: tree/mallee canopy)
    - canopy_cluster_id: cluster id for canopy points, else -1

Optional:
- If `--full-rangeland` is enabled, also writes:
    - rangeland_pred: int label (see `RANGELAND_LABELS`)

Label space (only when `--full-rangeland` is enabled):
0 ground
1 forb
2 hummock
3 shrub
4 mallee (canopy-forming woody vegetation)

Note: "woody canopy" is always provided as a binary field.
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from glob import glob
from typing import Dict, Iterable, List, Tuple

import numpy as np


RANGELAND_LABELS: Dict[int, str] = {
    0: "ground",
    1: "forb",
    2: "hummock",
    3: "shrub",
    4: "mallee",
}


@dataclass(frozen=True)
class Thresholds:
    grid_size: float
    h_canopy: float
    forb_max_h: float
    hummock_max_h: float
    shrub_max_h: float

    cluster_eps: float
    cluster_min_points: int
    min_canopy_points: int

    mallee_h_min: float
    mallee_h_max: float


def _read_ply_vertices(path: str) -> np.ndarray:
    from plyfile import PlyData

    ply = PlyData.read(path)
    if "vertex" not in ply:
        raise ValueError(f"PLY has no vertex element: {path}")
    return ply["vertex"].data


def _write_ply_vertices(path: str, vertex: np.ndarray) -> None:
    from plyfile import PlyData, PlyElement

    os.makedirs(os.path.dirname(path), exist_ok=True)
    el = PlyElement.describe(vertex, "vertex")
    PlyData([el], text=True).write(path)


def _estimate_ground_grid(points_xyz: np.ndarray, is_ground: np.ndarray, grid_size: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Estimate ground height per XY grid cell using min ground z.

    Returns:
      - cell_key per point (int64)
      - cell_to_zmin (dict-like arrays: unique_keys, zmin)
      - zmin_per_point (float32)
    """
    if points_xyz.ndim != 2 or points_xyz.shape[1] != 3:
        raise ValueError("points_xyz must be (N,3)")

    xy = points_xyz[:, :2]
    grid_xy = np.floor(xy / grid_size).astype(np.int64)
    # Hash 2D cell into 1D key.
    # Use large prime to reduce collisions.
    cell_key = grid_xy[:, 0] * np.int64(1_000_003) + grid_xy[:, 1]

    ground_keys = cell_key[is_ground]
    if ground_keys.size == 0:
        # Fallback: treat global min z as "ground".
        z0 = float(np.min(points_xyz[:, 2]))
        return cell_key, np.array([0], dtype=np.int64), np.full((points_xyz.shape[0],), z0, dtype=np.float32)

    ground_z = points_xyz[is_ground, 2]

    # Compute per-cell zmin from ground points.
    unique_keys, inv = np.unique(ground_keys, return_inverse=True)
    zmin = np.full((unique_keys.shape[0],), np.inf, dtype=np.float32)
    # vectorized min reduction
    np.minimum.at(zmin, inv, ground_z.astype(np.float32))

    # For each point, lookup the cell zmin; if missing, use global ground min.
    global_ground_min = float(np.min(ground_z))

    # Map all point keys to indices in unique_keys via searchsorted.
    sorter = np.argsort(unique_keys)
    keys_sorted = unique_keys[sorter]
    idx = np.searchsorted(keys_sorted, cell_key)
    in_bounds = (idx >= 0) & (idx < keys_sorted.shape[0])
    matched = in_bounds & (keys_sorted[idx] == cell_key)

    zmin_per_point = np.full((points_xyz.shape[0],), global_ground_min, dtype=np.float32)
    # only update where matched
    zmin_per_point[matched] = zmin[sorter[idx[matched]]]

    return cell_key, unique_keys, zmin_per_point


def _cluster_radius(points_xy: np.ndarray, mask: np.ndarray, eps: float, min_points: int) -> np.ndarray:
    """Simple radius-based clustering using cKDTree + BFS.

    Returns cluster_id per point, -1 for non-masked points or noise.
    """
    cluster_id = np.full((points_xy.shape[0],), -1, dtype=np.int32)
    idxs = np.flatnonzero(mask)
    if idxs.size == 0:
        return cluster_id

    try:
        from scipy.spatial import cKDTree
    except Exception as e:
        raise RuntimeError(
            "scipy is required for clustering (scipy.spatial.cKDTree). "
            "Install scipy or disable clustering by setting cluster_min_points very high."
        ) from e

    tree = cKDTree(points_xy[idxs])
    visited = np.zeros((idxs.size,), dtype=bool)

    next_cluster = 0
    for i in range(idxs.size):
        if visited[i]:
            continue
        # seed neighborhood
        neigh = tree.query_ball_point(points_xy[idxs[i]], r=eps)
        if len(neigh) < min_points:
            visited[i] = True
            continue

        # BFS expand
        queue = list(neigh)
        visited[queue] = True
        members = []
        while queue:
            j = queue.pop()
            members.append(j)
            neigh2 = tree.query_ball_point(points_xy[idxs[j]], r=eps)
            if len(neigh2) >= min_points:
                for k in neigh2:
                    if not visited[k]:
                        visited[k] = True
                        queue.append(k)

        # assign
        cluster_id[idxs[np.array(members, dtype=np.int64)]] = next_cluster
        next_cluster += 1

    return cluster_id


def derive_labels(
    xyz: np.ndarray,
    semantic_pred: np.ndarray,
    thresholds: Thresholds,
    mallee_source: str,
    full_rangeland: bool,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]:
    """Return (h_agl, mallee_pred, woody_canopy_pred, canopy_cluster_id, rangeland_pred_or_none)."""

    is_ground = semantic_pred == 0
    is_veg = (semantic_pred == 1) | (semantic_pred == 2)
    if mallee_source == "leaf":
        mallee_base = semantic_pred == 2
    elif mallee_source == "veg":
        mallee_base = is_veg
    else:
        raise ValueError("mallee_source must be one of: leaf, veg")

    _, _, z_ground = _estimate_ground_grid(xyz, is_ground=is_ground, grid_size=thresholds.grid_size)
    h_agl = (xyz[:, 2].astype(np.float32) - z_ground).astype(np.float32)

    # Canopy points: tall vegetation.
    is_canopy = mallee_base & (h_agl >= thresholds.h_canopy)

    canopy_cluster_id = _cluster_radius(
        points_xy=xyz[:, :2].astype(np.float32),
        mask=is_canopy,
        eps=thresholds.cluster_eps,
        min_points=thresholds.cluster_min_points,
    )

    # Mark "valid canopy" clusters by size (in points).
    woody_canopy_pred = np.zeros((xyz.shape[0],), dtype=np.uint8)
    if np.any(canopy_cluster_id >= 0):
        ids, counts = np.unique(canopy_cluster_id[canopy_cluster_id >= 0], return_counts=True)
        valid = ids[counts >= thresholds.min_canopy_points]
        if valid.size > 0:
            woody_canopy_pred[np.isin(canopy_cluster_id, valid)] = 1

    # Decide mallee as canopy points within plausible height band and in a valid canopy cluster.
    mallee_mask = (woody_canopy_pred == 1) & (h_agl >= thresholds.mallee_h_min) & (h_agl <= thresholds.mallee_h_max)
    mallee_pred = mallee_mask.astype(np.uint8)

    rangeland_pred = None
    if full_rangeland:
        # Heuristic non-woody vegetation classes by height bands (only for points not in mallee canopy).
        rangeland_pred = np.full((xyz.shape[0],), 0, dtype=np.int16)  # default ground

        # Start with ground
        rangeland_pred[~is_ground] = 0

        # Vegetation not in mallee canopy
        non_mallee_veg = is_veg & (~mallee_mask)

        forb = non_mallee_veg & (h_agl < thresholds.forb_max_h)
        hummock = non_mallee_veg & (h_agl >= thresholds.forb_max_h) & (h_agl < thresholds.hummock_max_h)
        shrub = non_mallee_veg & (h_agl >= thresholds.hummock_max_h) & (h_agl < thresholds.shrub_max_h)

        rangeland_pred[forb] = 1
        rangeland_pred[hummock] = 2
        rangeland_pred[shrub] = 3
        rangeland_pred[mallee_mask] = 4

    return h_agl, mallee_pred, woody_canopy_pred, canopy_cluster_id, rangeland_pred


def _build_vertex_with_extra_fields(vertices: np.ndarray, extra_fields: Dict[str, np.ndarray]) -> np.ndarray:
    """Return a new structured array with extra fields appended."""
    if vertices.dtype.fields is None:
        raise ValueError("Expected structured vertex array")

    new_dtype = list(vertices.dtype.descr)
    for name, arr in extra_fields.items():
        if arr.shape[0] != vertices.shape[0]:
            raise ValueError(f"Field {name} length mismatch")
        new_dtype.append((name, arr.dtype.str))

    out = np.empty(vertices.shape[0], dtype=new_dtype)
    for name in vertices.dtype.names:
        out[name] = vertices[name]
    for name, arr in extra_fields.items():
        out[name] = arr
    return out


def _iter_input_files(in_path: str) -> List[str]:
    if os.path.isdir(in_path):
        files = sorted(glob(os.path.join(in_path, "*.ply")))
        return files
    if os.path.isfile(in_path) and in_path.lower().endswith(".ply"):
        return [in_path]
    raise FileNotFoundError(f"No PLY files found at: {in_path}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--in-path", required=True, help="Input PLY file or a directory containing PLYs")
    p.add_argument("--out-dir", required=True, help="Output directory for derived PLYs")

    p.add_argument("--mallee-source", choices=["leaf", "veg"], default="leaf",
                   help="Which semantic points are eligible to become mallee canopy")

    p.add_argument(
        "--full-rangeland",
        action="store_true",
        help="Also derive forb/hummock/shrub labels using height bands (heuristic).",
    )

    p.add_argument("--grid-size", type=float, default=1.0, help="Ground grid cell size (meters)")
    p.add_argument("--h-canopy", type=float, default=2.0, help="Min height (AGL) to be considered canopy")

    p.add_argument("--cluster-eps", type=float, default=0.8, help="XY clustering radius for canopy points")
    p.add_argument("--cluster-min-points", type=int, default=20, help="Min neighbors to expand a cluster")
    p.add_argument("--min-canopy-points", type=int, default=200, help="Min points to accept a canopy cluster")

    p.add_argument("--forb-max-h", type=float, default=0.3)
    p.add_argument("--hummock-max-h", type=float, default=0.7)
    p.add_argument("--shrub-max-h", type=float, default=2.0)

    p.add_argument("--mallee-h-min", type=float, default=2.0)
    p.add_argument("--mallee-h-max", type=float, default=10.0)

    return p.parse_args()


def main() -> None:
    args = parse_args()

    thresholds = Thresholds(
        grid_size=args.grid_size,
        h_canopy=args.h_canopy,
        forb_max_h=args.forb_max_h,
        hummock_max_h=args.hummock_max_h,
        shrub_max_h=args.shrub_max_h,
        cluster_eps=args.cluster_eps,
        cluster_min_points=args.cluster_min_points,
        min_canopy_points=args.min_canopy_points,
        mallee_h_min=args.mallee_h_min,
        mallee_h_max=args.mallee_h_max,
    )

    in_files = _iter_input_files(args.in_path)
    os.makedirs(args.out_dir, exist_ok=True)

    for path in in_files:
        v = _read_ply_vertices(path)
        required = {"x", "y", "z", "semantic_pred"}
        missing = required.difference(set(v.dtype.names or []))
        if missing:
            raise ValueError(f"Missing required fields {sorted(missing)} in {path}")

        xyz = np.vstack([v["x"], v["y"], v["z"]]).T.astype(np.float32)
        semantic_pred = v["semantic_pred"].astype(np.int64)

        h_agl, mallee_pred, woody_canopy_pred, canopy_cluster_id, rangeland_pred = derive_labels(
            xyz=xyz,
            semantic_pred=semantic_pred,
            thresholds=thresholds,
            mallee_source=args.mallee_source,
            full_rangeland=bool(args.full_rangeland),
        )

        extra = {
            "h_agl": h_agl.astype(np.float32),
            "mallee_pred": mallee_pred.astype(np.uint8),
            "woody_canopy_pred": woody_canopy_pred.astype(np.uint8),
            "canopy_cluster_id": canopy_cluster_id.astype(np.int32),
        }
        if rangeland_pred is not None:
            extra["rangeland_pred"] = rangeland_pred.astype(np.int16)

        out_v = _build_vertex_with_extra_fields(v, extra)

        base = os.path.basename(path)
        out_path = os.path.join(args.out_dir, base)
        _write_ply_vertices(out_path, out_v)

        # quick summary
        canopy_pts = int(np.sum(woody_canopy_pred == 1))
        mallee_pts = int(np.sum(mallee_pred == 1))
        msg = f"Wrote {out_path} | canopy_pts={canopy_pts} | mallee_pts={mallee_pts}"
        if rangeland_pred is not None:
            counts = {RANGELAND_LABELS[i]: int(np.sum(rangeland_pred == i)) for i in sorted(RANGELAND_LABELS)}
            msg += f" | counts={counts}"
        print(msg)


if __name__ == "__main__":
    main()
