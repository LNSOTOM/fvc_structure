#%%
"""
3D fusion of Z_norm LiDAR tiles with vegetation-instance tracking.

This variant extends the mallee-only pipeline by assigning IDs to all
above-ground vegetation classes (treeID, forbID, hummockID, shrubID) so
that downstream tooling can reason about per-object instances across
multiple structural classes.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Dict, Mapping, Optional, Tuple

import laspy
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import ndimage
from scipy.ndimage import center_of_mass
from skimage.feature import peak_local_max
from skimage.measure import label as skimage_label
from skimage.measure import regionprops
from skimage.segmentation import watershed

matplotlib.use("Agg")

# -----------------------------------------------------------
# Global constants
# -----------------------------------------------------------

CLASS_NAMES: Dict[int, str] = {
    64: "Forb",
    65: "Woody Debris",
    66: "Hummock",
    67: "Shrub",
    68: "Mallee",
    0: "Unmatched",
}

HEIGHT_RANGES: Dict[int, Tuple[float, float]] = {
    64: (0.00, 0.40),   # Forb
    65: (0.05, 5.00),   # Woody debris / standing wood
    66: (0.00, 0.80),   # Hummock
    67: (0.40, 3.00),   # Shrub
    68: (1.50, 8.00),   # Mallee
}

DEFAULT_R_BIN = 0.10
DEFAULT_THETA_BIN_DEG = 10.0
DEFAULT_MIN_CONF = 0.3
DEFAULT_MIN_FLIP_RATIO = 1.5
DEFAULT_GROUND_TH = 0.01

# -----------------------------------------------------------
# Default configuration
# -----------------------------------------------------------

DEFAULT_CONFIG: Dict[str, float] = {
    "voxel_size_xy": 0.10,
    "voxel_size_z": 0.10,
    "ground_thresh": 0.15,
    "r_bin": DEFAULT_R_BIN,
    "theta_bin_deg": DEFAULT_THETA_BIN_DEG,
    "min_conf": DEFAULT_MIN_CONF,
    "min_flip_ratio": DEFAULT_MIN_FLIP_RATIO,
    "max_fill_height": 2.0,
    "reuse_existing_cartesian": True,
}

# -----------------------------------------------------------
# Helpers
# -----------------------------------------------------------

def _extra_dims(header: laspy.LasHeader) -> list:
    """Return extra byte params regardless of laspy version."""
    extras = getattr(header, "extra_dimensions", None)
    if extras is None:
        extras = getattr(header, "extra_dims", None)
    if extras is None and hasattr(header, "point_format"):
        extras = getattr(header.point_format, "extra_dimensions", None)
    return extras or []


def ensure_las14_pf6(las_in: laspy.LasData) -> laspy.LasData:
    """Ensure LAS 1.4 / point format 6 while preserving extra dims."""
    if las_in.header.version == "1.4" and las_in.header.point_format.id >= 6:
        return las_in

    hdr = laspy.LasHeader(version="1.4", point_format=6)
    hdr.scales = las_in.header.scales
    hdr.offsets = las_in.header.offsets

    out = laspy.LasData(hdr)
    out.points = laspy.ScaleAwarePointRecord.zeros(
        point_count=len(las_in.points),
        point_format=hdr.point_format,
        scales=hdr.scales,
        offsets=hdr.offsets,
    )

    out.X = las_in.X
    out.Y = las_in.Y
    out.Z = las_in.Z

    src_dims = set(las_in.point_format.dimension_names)
    dst_dims = set(out.point_format.dimension_names)
    for dim in (src_dims & dst_dims):
        if dim not in {"X", "Y", "Z"}:
            out[dim] = las_in[dim]

    for eb in _extra_dims(las_in.header):
        if eb.name not in out.point_format.dimension_names:
            out.add_extra_dim(eb)
        out[eb.name] = las_in[eb.name]

    return out


def soft_height_prior(z_norm: float, cls: int) -> float:
    """Soft prior to downweight unlikely class/height combinations."""
    if cls not in HEIGHT_RANGES:
        return 1.0
    zmin, zmax = HEIGHT_RANGES[cls]
    return 1.0 if zmin <= z_norm <= zmax else 0.1


def compute_class_height_thresholds(
    las_s: laspy.LasData,
    labeled_grid: np.ndarray,
    centroids: np.ndarray,
    min_x: float,
    min_y: float,
    grid_res: float,
    percentile_low: float = 15.0,
    percentile_high: float = 85.0,
) -> Dict[int, Tuple[float, float]]:
    """
    Compute data-driven height thresholds from structure LAS using majority class voting.
    For mallee (68), uses per-tree statistics from watershed-detected crowns.
    
    Parameters
    ----------
    las_s : LasData
        Structure-labelled LAS with reference classifications.
    labeled_grid : array
        Watershed crown labels from get_mallee_centroids_watershed.
    centroids : array
        Crown centroids from watershed.
    min_x, min_y : float
        Grid origin coordinates.
    grid_res : float
        Grid resolution.
    percentile_low : float
        Lower percentile for threshold.
    percentile_high : float
        Upper percentile for threshold.
    
    Returns
    -------
    Dict mapping class ID to (low_threshold, high_threshold) tuples.
    """
    thresholds = {}
    
    # Get structure data
    s_cls = las_s.classification
    s_z = las_s.z
    
    # Compute tree IDs for structure points
    s_r, s_th, s_tree_ids = calculate_local_polar(
        las_s.x, las_s.y, labeled_grid, centroids, min_x, min_y, grid_res
    )
    
    # Compute stats for non-tree classes (64, 66, 67) from structure labels
    for cls in [64, 66, 67]:
        mask = s_cls == cls
        if not np.any(mask):
            thresholds[cls] = HEIGHT_RANGES.get(cls, (0.0, 10.0))
            continue
        
        heights = s_z[mask]
        if len(heights) < 20:
            thresholds[cls] = HEIGHT_RANGES.get(cls, (0.0, 10.0))
            continue
        
        # Remove outliers beyond 5-95 percentile range
        p5, p95 = np.percentile(heights, [5, 95])
        clean_heights = heights[(heights >= p5) & (heights <= p95)]
        
        if len(clean_heights) < 10:
            clean_heights = heights
        
        p_low = np.percentile(clean_heights, percentile_low)
        p_high = np.percentile(clean_heights, percentile_high)
        
        p_low = max(0.0, p_low)
        p_high = min(10.0, p_high)
        
        thresholds[cls] = (p_low, p_high)
        print(f"  Class {cls} ({CLASS_NAMES[cls]}): {p_low:.2f}m - {p_high:.2f}m (n={len(clean_heights):,})")
    
    # Mallee (68): use per-tree median heights for robust estimate
    mask_68 = s_cls == 68
    if np.any(mask_68):
        # Compute per-tree medians using structure labels + tree IDs
        tree_medians = []
        unique_trees = np.unique(s_tree_ids[(s_tree_ids > 0) & mask_68])
        
        if len(unique_trees) >= 3:
            for tree_id in unique_trees:
                tree_mask = (s_tree_ids == tree_id) & mask_68
                if np.any(tree_mask):
                    tree_heights = s_z[tree_mask]
                    if len(tree_heights) >= 5:
                        tree_medians.append(np.median(tree_heights))
            
            if len(tree_medians) >= 3:
                # Use median of tree medians for robust estimation
                tree_medians = np.array(tree_medians)
                p_low = np.percentile(tree_medians, percentile_low)
                p_high = np.percentile(tree_medians, percentile_high)
                print(f"  Class 68 (Mallee): {p_low:.2f}m - {p_high:.2f}m (n_trees={len(tree_medians)}, median={np.median(tree_medians):.2f}m)")
            else:
                # Fallback: use all mallee points
                mallee_heights = s_z[mask_68]
                p5, p95 = np.percentile(mallee_heights, [5, 95])
                clean_heights = mallee_heights[(mallee_heights >= p5) & (mallee_heights <= p95)]
                p_low = np.percentile(clean_heights, percentile_low)
                p_high = np.percentile(clean_heights, percentile_high)
                print(f"  Class 68 (Mallee): {p_low:.2f}m - {p_high:.2f}m (n={len(clean_heights):,}, insufficient trees)")
        else:
            # No trees detected, use all mallee points
            mallee_heights = s_z[mask_68]
            p5, p95 = np.percentile(mallee_heights, [5, 95])
            clean_heights = mallee_heights[(mallee_heights >= p5) & (mallee_heights <= p95)]
            p_low = np.percentile(clean_heights, percentile_low)
            p_high = np.percentile(clean_heights, percentile_high)
            print(f"  Class 68 (Mallee): {p_low:.2f}m - {p_high:.2f}m (n={len(clean_heights):,}, no tree detection)")
        
        p_low = max(0.5, p_low)  # Mallee should be at least 0.5m
        p_high = min(10.0, p_high)
        thresholds[68] = (p_low, p_high)
    else:
        thresholds[68] = HEIGHT_RANGES.get(68, (1.5, 8.0))
        print(f"  Class 68 (Mallee): using defaults {thresholds[68][0]:.2f}m - {thresholds[68][1]:.2f}m (no data)")
    
    return thresholds


# -----------------------------------------------------------
# Crown-centric helpers (shared with mallee pipeline)
# -----------------------------------------------------------

def get_mallee_centroids_watershed(
    las_s: laspy.LasData,
    grid_res: float = 0.10,
    min_tree_dist: float = 0.75,
    smoothing_sigma: float = 0.5,
) -> Tuple[np.ndarray, np.ndarray, float, float]:
    x = las_s.x
    y = las_s.y
    z = las_s.z
    cls = las_s.classification

    min_x, max_x = np.min(x), np.max(x)
    min_y, max_y = np.min(y), np.max(y)

    w = int(np.ceil((max_x - min_x) / grid_res)) + 1
    h = int(np.ceil((max_y - min_y) / grid_res)) + 1

    mask_68 = cls == 68
    if not np.any(mask_68):
        return np.zeros((h, w), dtype=int), np.array([]), min_x, min_y

    mx = ((x[mask_68] - min_x) / grid_res).astype(int)
    my = ((y[mask_68] - min_y) / grid_res).astype(int)
    mz = z[mask_68]

    mx = np.clip(mx, 0, w - 1)
    my = np.clip(my, 0, h - 1)

    sort_idx = np.argsort(mz)
    chm_grid = np.zeros((h, w), dtype=np.float32)
    chm_grid[my[sort_idx], mx[sort_idx]] = mz[sort_idx]

    binary_mask = chm_grid > 0
    chm_smoothed = ndimage.gaussian_filter(chm_grid, sigma=smoothing_sigma)

    min_dist_px = max(1, int(min_tree_dist / grid_res))
    coordinates = peak_local_max(
        chm_smoothed,
        min_distance=min_dist_px,
        threshold_abs=0.1,
        labels=binary_mask,
    )

    local_maxi = np.zeros_like(chm_smoothed, dtype=bool)
    if len(coordinates) > 0:
        local_maxi[coordinates[:, 0], coordinates[:, 1]] = True

    markers = ndimage.label(local_maxi)[0]
    blob_labels = skimage_label(binary_mask)
    blob_props = regionprops(blob_labels, intensity_image=chm_grid)
    peak_coords = np.argwhere(local_maxi)

    next_marker_id = markers.max() + 1
    for blob in blob_props:
        min_row, min_col, max_row, max_col = blob.bbox
        in_bbox = (
            (peak_coords[:, 0] >= min_row)
            & (peak_coords[:, 0] < max_row)
            & (peak_coords[:, 1] >= min_col)
            & (peak_coords[:, 1] < max_col)
        )
        possible_peaks = peak_coords[in_bbox]
        has_marker = any(blob_labels[py, px] == blob.label for py, px in possible_peaks)
        if not has_marker:
            cy, cx = map(int, blob.weighted_centroid)
            cy = np.clip(cy, 0, h - 1)
            cx = np.clip(cx, 0, w - 1)
            markers[cy, cx] = next_marker_id
            next_marker_id += 1

    labels = watershed(-chm_smoothed, markers, mask=binary_mask)
    num_features = labels.max()

    if num_features == 0:
        return labels, np.array([]), min_x, min_y

    coms = center_of_mass(binary_mask, labels, range(1, num_features + 1))
    centroids = np.zeros((num_features + 1, 2))
    for i, (cy, cx) in enumerate(coms):
        centroids[i + 1] = [min_x + (cx * grid_res), min_y + (cy * grid_res)]

    print(f"🌳 [Watershed] Identified {num_features} crowns.")
    return labels, centroids, min_x, min_y


def plot_mallee_debug(
    labeled_grid: np.ndarray,
    centroids: np.ndarray,
    min_x: float,
    min_y: float,
    grid_res: float,
    tile_id: str,
    out_dir: Path,
) -> None:
    if len(centroids) < 2:
        return

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        fig = plt.figure(figsize=(10, 10))
        masked_grid = np.ma.masked_where(labeled_grid == 0, labeled_grid)
        plt.imshow(masked_grid, origin="lower", cmap="nipy_spectral", interpolation="nearest")

        real_cx = centroids[1:, 0]
        real_cy = centroids[1:, 1]
        grid_cx = (real_cx - min_x) / grid_res
        grid_cy = (real_cy - min_y) / grid_res

        plt.scatter(grid_cx, grid_cy, c="white", edgecolors="black", marker="X", s=50, linewidths=1)
        plt.title(f"Tile {tile_id}: {len(centroids) - 1} crowns")
        plt.savefig(out_dir / f"debug_mallee_{tile_id}.png", dpi=100)
        plt.close(fig)
    except Exception as exc:
        print(f"❌ Plot failed for {tile_id}: {exc}")


def calculate_local_polar(
    x_arr: np.ndarray,
    y_arr: np.ndarray,
    labeled_grid: np.ndarray,
    centroids: np.ndarray,
    min_x: float,
    min_y: float,
    grid_res: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    ix = np.clip(((x_arr - min_x) / grid_res).astype(int), 0, labeled_grid.shape[1] - 1)
    iy = np.clip(((y_arr - min_y) / grid_res).astype(int), 0, labeled_grid.shape[0] - 1)

    obj_ids = labeled_grid[iy, ix]
    r = np.full_like(x_arr, np.nan, dtype=float)
    theta = np.full_like(x_arr, np.nan, dtype=float)

    valid_mask = obj_ids > 0
    if np.any(valid_mask):
        safe_mask = np.zeros_like(valid_mask)
        safe_mask[valid_mask] = obj_ids[valid_mask] < len(centroids)
        if np.any(safe_mask):
            final_ids = obj_ids[safe_mask]
            cx = centroids[final_ids, 0]
            cy = centroids[final_ids, 1]
            dx = x_arr[safe_mask] - cx
            dy = y_arr[safe_mask] - cy
            r[safe_mask] = np.sqrt(dx * dx + dy * dy)
            theta[safe_mask] = np.mod(np.arctan2(dy, dx), 2 * np.pi)

    return r, theta, obj_ids


def polar_bins(r: np.ndarray, theta: np.ndarray, r_bin: float, theta_bin_deg: float) -> Tuple[np.ndarray, np.ndarray]:
    ir = np.floor(r / r_bin).astype(np.int32)
    it = np.floor(theta / np.deg2rad(theta_bin_deg)).astype(np.int32)
    return ir, it


# -----------------------------------------------------------
# Cartesian merge
# -----------------------------------------------------------

def cartesian_merge(
    las_z: laspy.LasData,
    las_s: laspy.LasData,
    voxel_size_xy: float,
    voxel_size_z: float,
) -> np.ndarray:
    x_min = min(las_z.x.min(), las_s.x.min())
    y_min = min(las_z.y.min(), las_s.y.min())
    z_min = min(las_z.z.min(), las_s.z.min())
    origin = np.array([x_min, y_min, z_min], dtype=np.float64)

    def vox(x, y, z):
        return np.stack(
            [
                np.floor((x - origin[0]) / voxel_size_xy).astype(int),
                np.floor((y - origin[1]) / voxel_size_xy).astype(int),
                np.floor((z - origin[2]) / voxel_size_z).astype(int),
            ],
            axis=-1,
        )

    vox_s = vox(las_s.x, las_s.y, las_s.z)
    raw_struct = np.where((las_s.classification >= 64) & (las_s.classification <= 68), las_s.classification, 0)
    uniq_vox_s, inv_s = np.unique(vox_s, axis=0, return_inverse=True)

    voxel_class = np.zeros(len(uniq_vox_s), dtype=np.uint16)
    for k in range(len(uniq_vox_s)):
        vals = raw_struct[inv_s == k]
        vals = vals[vals != 0]
        if vals.size:
            voxel_class[k] = Counter(vals).most_common(1)[0][0]

    voxel_struct_map = {tuple(idx): c for idx, c in zip(uniq_vox_s, voxel_class) if c != 0}
    print(f"✅ Voxels with non-zero structure class: {len(voxel_struct_map):,}")

    vox_z = vox(las_z.x, las_z.y, las_z.z)
    classification = np.zeros(len(las_z.points), dtype=np.uint8)
    for i, v in enumerate(vox_z):
        classification[i] = voxel_struct_map.get(tuple(v), 0)

    return classification


# -----------------------------------------------------------
# Crown-centric refinement (treeID)
# -----------------------------------------------------------

def refine_mallee_with_local_polar(
    las_z: laspy.LasData,
    las_s: laspy.LasData,
    classification: np.ndarray,
    tile_id: str,
    output_dir: Path,
    r_bin: float,
    theta_bin_deg: float,
    min_conf: float,
    min_flip_ratio: float,
    ground_thresh: float,
    grid_res: float = 0.10,
) -> Tuple[np.ndarray, np.ndarray]:
    cls_out = classification.copy()
    instance_ids = np.zeros(len(classification), dtype=np.uint16)

    if not np.any(las_s.classification == 68):
        return cls_out, instance_ids

    labeled_grid, centroids, min_x, min_y = get_mallee_centroids_watershed(
        las_s,
        grid_res=grid_res,
        min_tree_dist=1.5,
        smoothing_sigma=1.0,
    )

    if len(centroids) == 0:
        return cls_out, instance_ids

    plot_mallee_debug(labeled_grid, centroids, min_x, min_y, grid_res, tile_id, output_dir)

    mask_65_66_68 = (
        (las_s.classification == 65)
        | (las_s.classification == 66)
        | (las_s.classification == 68)
    )

    s_r, s_th, s_obj_ids = calculate_local_polar(
        las_s.x[mask_65_66_68],
        las_s.y[mask_65_66_68],
        labeled_grid,
        centroids,
        min_x,
        min_y,
        grid_res,
    )

    valid_s = s_obj_ids > 0
    s_ir, s_it = polar_bins(s_r[valid_s], s_th[valid_s], r_bin, theta_bin_deg)
    df_s = pd.DataFrame(
        {
            "obj_id": s_obj_ids[valid_s],
            "ir": s_ir,
            "it": s_it,
            "cls": las_s.classification[mask_65_66_68][valid_s],
        }
    )

    col_counts = df_s.groupby(["obj_id", "ir", "it", "cls"]).size().unstack(fill_value=0)
    col_probs = col_counts.div(col_counts.sum(axis=1), axis=0)
    prob_lookup = {idx: col_probs.loc[idx] for idx in col_probs.index}

    z_r, z_th, z_obj_ids = calculate_local_polar(
        las_z.x,
        las_z.y,
        labeled_grid,
        centroids,
        min_x,
        min_y,
        grid_res,
    )
    instance_ids = z_obj_ids.astype(np.uint16)

    z_ir, z_it = polar_bins(z_r, z_th, r_bin, theta_bin_deg)
    zn = las_z["Z_norm"].astype(float)

    idx_candidates = np.where((cls_out == 65) | (cls_out == 66) | (cls_out == 68))[0]
    for idx in idx_candidates:
        if zn[idx] <= ground_thresh:
            continue
        obj_id = z_obj_ids[idx]
        if obj_id == 0:
            continue
        base = prob_lookup.get((obj_id, z_ir[idx], z_it[idx]))
        if base is None:
            continue
        classes = base.index.to_numpy()
        probs = base.to_numpy()
        post = np.array([p * soft_height_prior(zn[idx], int(c)) for p, c in zip(probs, classes)])
        if post.sum() <= 0:
            continue
        best_cls = int(classes[np.argmax(post)])
        if best_cls != int(cls_out[idx]):
            cur_val = post[classes == int(cls_out[idx])][0] if int(cls_out[idx]) in classes else 0.0
            if post.max() / (cur_val + 1e-9) >= min_flip_ratio:
                cls_out[idx] = best_cls

    return cls_out, instance_ids


# -----------------------------------------------------------
# Radial gap fill (tile wide)
# -----------------------------------------------------------

def tile_center_xy(las_z: laspy.LasData) -> np.ndarray:
    return np.array([
        0.5 * (las_z.x.min() + las_z.x.max()),
        0.5 * (las_z.y.min() + las_z.y.max()),
    ])


def radial_coords_tile(x: np.ndarray, y: np.ndarray, origin: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    dx = x - origin[0]
    dy = y - origin[1]
    r = np.sqrt(dx * dx + dy * dy)
    theta = np.mod(np.arctan2(dy, dx), 2 * np.pi)
    return r, theta


def fill_missing_classes_radial(
    las_z: laspy.LasData,
    las_s: laspy.LasData,
    classification: np.ndarray,
    r_bin: float,
    theta_bin_deg: float,
    min_conf: float,
    ground_thresh: float,
    max_fill_height: float,
) -> np.ndarray:
    cls_out = classification.copy()
    origin = tile_center_xy(las_z)

    mask_ext = (las_s.classification >= 64) & (las_s.classification <= 68)
    s_r, s_th = radial_coords_tile(las_s.x, las_s.y, origin)
    s_ir, s_it = polar_bins(s_r, s_th, r_bin, theta_bin_deg)

    df = pd.DataFrame(
        {
            "ir": s_ir[mask_ext],
            "it": s_it[mask_ext],
            "cls": las_s.classification[mask_ext],
        }
    )
    col_probs = df.groupby(["ir", "it", "cls"]).size().unstack(fill_value=0)
    col_probs = col_probs.div(col_probs.sum(axis=1), axis=0)
    if min_conf > 0:
        col_probs = col_probs[col_probs.max(axis=1) >= min_conf]
    col_map = {idx: col_probs.loc[idx] for idx in col_probs.index}

    z_r, z_th = radial_coords_tile(las_z.x, las_z.y, origin)
    z_ir, z_it = polar_bins(z_r, z_th, r_bin, theta_bin_deg)
    zn = las_z["Z_norm"].astype(float)

    zero_idx = np.where((cls_out == 0) & (zn > ground_thresh) & (zn <= max_fill_height))[0]
    for idx in zero_idx:
        base = col_map.get((z_ir[idx], z_it[idx]))
        if base is None:
            continue
        classes = base.index.to_numpy()
        probs = base.to_numpy()
        post = np.array([p * soft_height_prior(zn[idx], int(c)) for p, c in zip(probs, classes)])
        if post.sum() > 0:
            cls_out[idx] = int(classes[np.argmax(post)])

    return cls_out


# -----------------------------------------------------------
# ID helpers
# -----------------------------------------------------------

def write_id_subset_las(
    path: Path,
    base_las: laspy.LasData,
    mask: np.ndarray,
    classification_arr: np.ndarray,
    zn_arr: np.ndarray,
    extra_ids: Dict[str, np.ndarray],
) -> None:
    hdr = laspy.LasHeader(version="1.4", point_format=6)
    hdr.scales = base_las.header.scales
    hdr.offsets = base_las.header.offsets

    out = laspy.LasData(hdr)

    if "Z_norm" not in out.point_format.dimension_names:
        out.add_extra_dim(laspy.ExtraBytesParams(name="Z_norm", type=np.float32))
    for name in extra_ids:
        if name not in out.point_format.dimension_names:
            out.add_extra_dim(laspy.ExtraBytesParams(name=name, type=np.uint16))

    out.points = laspy.ScaleAwarePointRecord.zeros(
        point_count=len(classification_arr),
        point_format=out.header.point_format,
        scales=out.header.scales,
        offsets=out.header.offsets,
    )

    out.X = base_las.X[mask]
    out.Y = base_las.Y[mask]
    out.Z = base_las.Z[mask]

    for dim in (
        "intensity",
        "return_number",
        "number_of_returns",
        "scan_direction_flag",
        "edge_of_flight_line",
        "classification_flags",
        "scan_angle_rank",
        "user_data",
        "point_source_id",
        "gps_time",
        "red",
        "green",
        "blue",
        "nir",
    ):
        if dim in out.point_format.dimension_names and dim in base_las.point_format.dimension_names:
            out[dim] = base_las[dim][mask]

    out.classification = classification_arr
    out["Z_norm"] = zn_arr.astype(np.float32)
    for name, arr in extra_ids.items():
        out[name] = arr

    out = ensure_las14_pf6(out)
    path.parent.mkdir(parents=True, exist_ok=True)
    out.write(str(path))


def compute_component_ids(
    x_arr: np.ndarray,
    y_arr: np.ndarray,
    cls_arr: np.ndarray,
    target_cls: int,
    grid_res: float,
) -> np.ndarray:
    ids = np.zeros(len(cls_arr), dtype=np.uint16)
    mask = cls_arr == target_cls
    if not np.any(mask):
        return ids

    x = x_arr[mask]
    y = y_arr[mask]
    min_x, max_x = np.min(x), np.max(x)
    min_y, max_y = np.min(y), np.max(y)

    w = int(np.ceil((max_x - min_x) / grid_res)) + 1
    h = int(np.ceil((max_y - min_y) / grid_res)) + 1

    ix = np.clip(((x - min_x) / grid_res).astype(int), 0, w - 1)
    iy = np.clip(((y - min_y) / grid_res).astype(int), 0, h - 1)

    grid = np.zeros((h, w), dtype=bool)
    grid[iy, ix] = True

    labeled, _ = ndimage.label(grid)
    ids_masked = labeled[iy, ix]
    ids[mask] = ids_masked.astype(np.uint16)
    return ids


# -----------------------------------------------------------
# Per-file fusion
# -----------------------------------------------------------

def merge_znorm_structure_one_file(
    z_norm_las,
    structure_las,
    output_las,
    output_las_no0,
    config: Mapping[str, float],
) -> None:
    z_norm_las = Path(z_norm_las)
    structure_las = Path(structure_las)
    output_las = Path(output_las)
    output_las_no0 = Path(output_las_no0) if output_las_no0 else None

    tile_id = z_norm_las.stem.split(".")[-1]

    las_z = laspy.read(z_norm_las)
    las_s = laspy.read(structure_las)
    print(f"🔎 Tile {tile_id} | Z: {len(las_z.points):,}, S: {len(las_s.points):,}")

    if config.get("reuse_existing_cartesian", True) and np.any((las_z.classification >= 64) & (las_z.classification <= 68)):
        classification = las_z.classification.astype(np.uint8)
        print("✅ Reusing existing classes from Z_norm LAS.")
    else:
        classification = cartesian_merge(
            las_z,
            las_s,
            config["voxel_size_xy"],
            config["voxel_size_z"],
        )

    classification, instance_ids = refine_mallee_with_local_polar(
        las_z,
        las_s,
        classification,
        tile_id=tile_id,
        output_dir=output_las.parent,
        r_bin=config["r_bin"],
        theta_bin_deg=config["theta_bin_deg"],
        min_conf=config["min_conf"],
        min_flip_ratio=config["min_flip_ratio"],
        ground_thresh=config["ground_thresh"],
    )

    classification = fill_missing_classes_radial(
        las_z,
        las_s,
        classification,
        r_bin=config["r_bin"],
        theta_bin_deg=config["theta_bin_deg"],
        min_conf=config["min_conf"],
        ground_thresh=config["ground_thresh"],
        max_fill_height=config["max_fill_height"],
    )

    zn_arr = las_z["Z_norm"].astype(float)
    
    print(f"📊 Computing data-driven height thresholds for tile {tile_id}:")
    
    # Get watershed info from structure LAS for stats computation
    grid_res = config.get("r_bin", 0.10)
    labeled_grid, centroids, min_x, min_y = get_mallee_centroids_watershed(
        las_s,
        grid_res=grid_res,
        min_tree_dist=1.5,
        smoothing_sigma=1.0,
    )
    
    thresholds = compute_class_height_thresholds(
        las_s, labeled_grid, centroids, min_x, min_y, grid_res
    )
    
    # Extract specific thresholds with conservative factors
    hummock_high = thresholds.get(66, (0.0, 0.8))[1]
    shrub_low, shrub_high = thresholds.get(67, (0.4, 3.0))
    mallee_low, mallee_high = thresholds.get(68, (1.5, 8.0))
    
    # Conservative transition thresholds
    hummock_to_shrub = min(hummock_high * 0.9, 0.7)  # Don't exceed 0.7m
    shrub_to_mallee = max(shrub_high * 0.8, mallee_low * 0.85)  # Between shrub high and mallee low
    mallee_min = max(0.6, mallee_low * 0.7)  # At least 0.6m
    mallee_max = min(6.0, mallee_high * 1.2)  # Cap at 6m
    
    print(f"  → Hummock→Shrub: {hummock_to_shrub:.2f}m | Shrub→Mallee: {shrub_to_mallee:.2f}m")
    print(f"  → Mallee range: {mallee_min:.2f}m - {mallee_max:.2f}m")

    # Apply adaptive rules
    # Low mallee -> shrub when below adaptive threshold
    classification[(classification == 68) & (zn_arr < mallee_min)] = 67
    # Hummock -> shrub in transition zone
    classification[(classification == 66) & (zn_arr >= hummock_to_shrub)] = 67
    # Shrub -> mallee when tall enough
    classification[(classification == 67) & (zn_arr >= shrub_to_mallee)] = 68
    # Very tall mallee -> woody debris (standing dead wood)
    classification[(classification == 68) & (zn_arr >= mallee_max)] = 65

    MERGE_65_TO_68_ZNORM = 0.5
    mask_65 = classification == 65
    classification[mask_65 & (zn_arr > MERGE_65_TO_68_ZNORM)] = 68
    keep_mask_base = ~(mask_65 & (zn_arr < MERGE_65_TO_68_ZNORM))

    classification_kept = classification[keep_mask_base]
    zn_kept = zn_arr[keep_mask_base]
    ids_kept = instance_ids[keep_mask_base]
    classification_kept[classification_kept == 65] = 68

    grid_res = config.get("r_bin", 0.10)
    forb_ids = compute_component_ids(las_z.x[keep_mask_base], las_z.y[keep_mask_base], classification_kept, 64, grid_res)
    hummock_ids = compute_component_ids(las_z.x[keep_mask_base], las_z.y[keep_mask_base], classification_kept, 66, grid_res)
    shrub_ids = compute_component_ids(las_z.x[keep_mask_base], las_z.y[keep_mask_base], classification_kept, 67, grid_res)

    las_out = laspy.LasData(las_z.header)
    las_out.points = las_z.points[keep_mask_base].copy()
    las_out.classification = classification_kept

    if "Z_norm" not in las_out.point_format.dimension_names:
        las_out.add_extra_dim(laspy.ExtraBytesParams(name="Z_norm", type=np.float32))
    las_out["Z_norm"] = zn_kept.astype(np.float32)

    if "treeID" not in las_out.point_format.dimension_names:
        las_out.add_extra_dim(laspy.ExtraBytesParams(name="treeID", type=np.uint16))
    las_out["treeID"] = ids_kept

    for name, array in {
        "forbID": forb_ids,
        "hummockID": hummock_ids,
        "shrubID": shrub_ids,
    }.items():
        if name not in las_out.point_format.dimension_names:
            las_out.add_extra_dim(laspy.ExtraBytesParams(name=name, type=np.uint16))
        las_out[name] = array

    las_out = ensure_las14_pf6(las_out)
    output_las.parent.mkdir(parents=True, exist_ok=True)
    las_out.write(str(output_las))

    if output_las_no0:
        HIGH_Z_KEEP = 1.3
        keep = (classification_kept != 0) | ((classification_kept == 0) & (zn_kept >= HIGH_Z_KEEP))

        final_cls = classification_kept[keep]
        final_zn = zn_kept[keep]
        final_ids = ids_kept[keep]
        final_forb_ids = forb_ids[keep]
        final_hummock_ids = hummock_ids[keep]
        final_shrub_ids = shrub_ids[keep]

        promote = (final_cls == 0) & (final_zn >= HIGH_Z_KEEP)
        final_cls[promote] = 65
        final_cls[final_cls == 65] = 68

        las_no0 = laspy.LasData(las_out.header)
        las_no0.points = las_out.points[keep].copy()
        las_no0.classification = final_cls
        for name, array in {
            "treeID": final_ids,
            "forbID": final_forb_ids,
            "hummockID": final_hummock_ids,
            "shrubID": final_shrub_ids,
        }.items():
            las_no0[name] = array
        las_no0 = ensure_las14_pf6(las_no0)
        las_no0.write(str(output_las_no0))

    treeid_only_path = output_las.parent / f"{output_las.stem}_treeID_only.las"
    vegids_path = output_las.parent / f"{output_las.stem}_vegIDs.las"

    write_id_subset_las(
        treeid_only_path,
        las_z,
        keep_mask_base,
        classification_kept,
        zn_kept,
        {"treeID": ids_kept},
    )
    write_id_subset_las(
        vegids_path,
        las_z,
        keep_mask_base,
        classification_kept,
        zn_kept,
        {
            "treeID": ids_kept,
            "forbID": forb_ids,
            "hummockID": hummock_ids,
            "shrubID": shrub_ids,
        },
    )

    print(f"✅ Wrote: {output_las.name} (TreeIDs + veg IDs saved)")


# -----------------------------------------------------------
# Folder-level runner
# -----------------------------------------------------------

def merge_folder(
    z_dir,
    s_dir,
    o_full,
    o_no0,
    config: Optional[Mapping[str, float]] = None,
) -> None:
    cfg = dict(DEFAULT_CONFIG)
    if config is not None:
        cfg.update(config)

    z_norm_dir, structure_dir, out_dir_full, out_dir_no0 = map(Path, [z_dir, s_dir, o_full, o_no0])
    out_dir_full.mkdir(parents=True, exist_ok=True)
    out_dir_no0.mkdir(parents=True, exist_ok=True)

    z_files = sorted(z_norm_dir.glob("z_norm_csf_tiles_multispectral.*.las"))
    print(f"📁 Processing {len(z_files)} tiles from {z_norm_dir} ...")

    for zf in z_files:
        tile_id = zf.stem.split(".")[-1]
        struct_file = structure_dir / f"mask_tiles_multispectral.{tile_id}.las"
        if not struct_file.exists():
            print(f"⚠️ Missing structure file for tile {tile_id}")
            continue

        merge_znorm_structure_one_file(
            zf,
            struct_file,
            out_dir_full / f"z_norm_struct_{tile_id}_obj_polar.las",
            out_dir_no0 / f"z_norm_struct_{tile_id}_obj_polar_no0.las",
            cfg,
        )


# -----------------------------------------------------------
# Main
# -----------------------------------------------------------

if __name__ == "__main__":
    #site1
    # z_dir = "/data/calperumResearch/site1_1_DD0001/lidar/clipped/raster/z_norm/csf/"
    # s_dir = "/data/calperumResearch/site1_1_DD0001/lidar/clipped/mask/coreg/icp/"
    # o_full = "/data/calperumResearch/site1_1_DD0001/lidar/clipped/3d_structure_vegid"
    # o_no0 = "/data/calperumResearch/site1_1_DD0001/lidar/clipped/3d_structure_mask_vegid"
    
    #site11
    z_dir = "/data/calperumResearch/site2_1_DD0011/lidar/clipped/raster/z_norm/csf/"
    s_dir = "/data/calperumResearch/site2_1_DD0011/lidar/clipped/mask/coreg/icp/"
    o_full = "/data/calperumResearch/site2_1_DD0011/lidar/clipped/3d_structure_vegid"
    o_no0 = "/data/calperumResearch/site2_1_DD0011/lidar/clipped/3d_structure_mask_vegid"


    merge_folder(z_dir, s_dir, o_full, o_no0, config={})
#%%
