#!/usr/bin/env python3
#%%
"""
Process a single LAS tile pair through the mallee 3D fusion pipeline.

Usage
-----
    python process_single_tile_malle.py <z_norm_las> <structure_las> <output_dir>

Example
-------
    python process_single_tile_malle.py \\
        /data/.../z_norm_csf_tiles_multispectral.12345.las \\
        /data/.../mask_tiles_multispectral.12345.las \\
        /data/.../output/

This will create:
  - output/z_norm_struct_12345_obj_polar.las (full output with treeID)
  - output/z_norm_struct_12345_obj_polar_no0.las (filtered, class-0 removed)
  - output/debug_mallee_12345.png (crown detection visualization)
"""

from __future__ import annotations

import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from pathlib import Path
from collections import Counter
from typing import Dict, Tuple, Mapping

import laspy
import numpy as np
import pandas as pd
from scipy.ndimage import center_of_mass
from skimage.feature import peak_local_max
from skimage.segmentation import watershed
from skimage.measure import label as skimage_label
from skimage.measure import regionprops
from scipy import ndimage

# -----------------------------------------------------------
# Global constants
# -----------------------------------------------------------

CLASS_NAMES: Dict[int, str] = {
    64: "Forb",
    65: "Woody Debris",
    66: "Hummock",
    67: "Shrub",
    68: "Mallee",
    0:  "Unmatched",
}

HEIGHT_RANGES: Dict[int, Tuple[float, float]] = {
    64: (0.00, 0.40),
    65: (0.05, 5.00),
    66: (0.00, 0.80),
    67: (0.40, 3.00),
    68: (1.50, 8.00),
}

DEFAULT_CONFIG = {
    "voxel_size_xy": 0.10,
    "voxel_size_z": 0.10,
    "ground_thresh": 0.15,
    "r_bin": 0.10,
    "theta_bin_deg": 10.0,
    "min_conf": 0.3,
    "min_flip_ratio": 1.5,
    "max_fill_height": 2.0,
    "reuse_existing_cartesian": True,
}

# -----------------------------------------------------------
# Helpers
# -----------------------------------------------------------

def ensure_las14_pf6(las_in: laspy.LasData) -> laspy.LasData:
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
    out.X, out.Y, out.Z = las_in.X, las_in.Y, las_in.Z

    for dim in (set(las_in.point_format.dimension_names) & set(out.point_format.dimension_names)):
        if dim not in ("X", "Y", "Z"):
            out[dim] = las_in[dim]

    for eb in las_in.header.extra_dimensions:
        if eb.name not in out.point_format.dimension_names:
            out.add_extra_dim(eb)
        out[eb.name] = las_in[eb.name]

    return out


def soft_height_prior(z_norm: float, cls: int) -> float:
    if cls not in HEIGHT_RANGES:
        return 1.0
    zmin, zmax = HEIGHT_RANGES[cls]
    return 1.0 if (zmin <= z_norm <= zmax) else 0.1


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

    mask_68 = (cls == 68)
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

    min_dist_px = int(min_tree_dist / grid_res)
    min_dist_px = max(1, min_dist_px)

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
            (peak_coords[:, 0] >= min_row) & (peak_coords[:, 0] < max_row) &
            (peak_coords[:, 1] >= min_col) & (peak_coords[:, 1] < max_col)
        )
        possible_peaks = peak_coords[in_bbox]

        has_marker = False
        for py, px in possible_peaks:
            if blob_labels[py, px] == blob.label:
                has_marker = True
                break

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
        real_x = min_x + (cx * grid_res)
        real_y = min_y + (cy * grid_res)
        centroids[i + 1] = [real_x, real_y]

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
        plt.imshow(masked_grid, origin='lower', cmap='nipy_spectral', interpolation='nearest')

        real_cx = centroids[1:, 0]
        real_cy = centroids[1:, 1]
        grid_cx = (real_cx - min_x) / grid_res
        grid_cy = (real_cy - min_y) / grid_res

        plt.scatter(grid_cx, grid_cy, c='white', edgecolors='black',
                    marker='X', s=50, linewidths=1)
        plt.title(f"Tile {tile_id}: {len(centroids) - 1} crowns")
        plt.savefig(out_dir / f"debug_mallee_{tile_id}.png", dpi=100)
        plt.close(fig)
        print(f"📊 Saved debug plot: debug_mallee_{tile_id}.png")
    except Exception as e:
        print(f"❌ Plot failed: {e}")


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

    valid_mask = (obj_ids > 0)
    if np.any(valid_mask):
        safe_mask = np.zeros_like(valid_mask)
        safe_mask[valid_mask] = (obj_ids[valid_mask] < len(centroids))

        if np.any(safe_mask):
            final_ids = obj_ids[safe_mask]
            cx, cy = centroids[final_ids, 0], centroids[final_ids, 1]
            dx, dy = x_arr[safe_mask] - cx, y_arr[safe_mask] - cy

            r[safe_mask] = np.sqrt(dx**2 + dy**2)
            theta[safe_mask] = np.mod(np.arctan2(dy, dx), 2 * np.pi)

    return r, theta, obj_ids


def polar_bins(r: np.ndarray, theta: np.ndarray, r_bin: float, theta_bin_deg: float):
    ir = np.floor(r / r_bin).astype(np.int32)
    it = np.floor(theta / np.deg2rad(theta_bin_deg)).astype(np.int32)
    return ir, it


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
    raw_struct = np.where(
        (las_s.classification >= 64) & (las_s.classification <= 68),
        las_s.classification,
        0,
    )
    uniq_vox_s, inv_s = np.unique(vox_s, axis=0, return_inverse=True)

    voxel_class = np.zeros(len(uniq_vox_s), dtype=np.uint16)
    for k in range(len(uniq_vox_s)):
        vals = raw_struct[inv_s == k]
        vals = vals[vals != 0]
        if vals.size > 0:
            voxel_class[k] = Counter(vals).most_common(1)[0][0]

    voxel_struct_map = {
        tuple(idx): c for idx, c in zip(uniq_vox_s, voxel_class) if c != 0
    }
    print(f"✅ Voxels with non-zero structure class: {len(voxel_struct_map):,}")

    vox_z = vox(las_z.x, las_z.y, las_z.z)
    classification = np.zeros(len(las_z.points), dtype=np.uint8)
    for i, v in enumerate(vox_z):
        classification[i] = voxel_struct_map.get(tuple(v), 0)

    return classification


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
        (las_s.classification == 65) |
        (las_s.classification == 66) |
        (las_s.classification == 68)
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

    valid_s = (s_obj_ids > 0)
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

    idx_candidates = np.where(
        (cls_out == 65) | (cls_out == 66) | (cls_out == 68)
    )[0]

    for idx in idx_candidates:
        if zn[idx] <= ground_thresh:
            continue

        obj_id = z_obj_ids[idx]
        if obj_id == 0:
            continue

        base = prob_lookup.get((obj_id, z_ir[idx], z_it[idx]), None)
        if base is None:
            continue

        classes = base.index.to_numpy()
        probs = base.to_numpy()
        post = np.array(
            [p * soft_height_prior(zn[idx], int(c)) for p, c in zip(probs, classes)]
        )
        if post.sum() <= 0:
            continue

        best_cls = int(classes[np.argmax(post)])
        if best_cls != int(cls_out[idx]):
            cur_val = post[classes == int(cls_out[idx])][0] if int(cls_out[idx]) in classes else 0.0
            if post.max() / (cur_val + 1e-9) >= min_flip_ratio:
                cls_out[idx] = best_cls

    return cls_out, instance_ids


def tile_center_xy(las_z: laspy.LasData) -> np.ndarray:
    return np.array(
        [
            0.5 * (las_z.x.min() + las_z.x.max()),
            0.5 * (las_z.y.min() + las_z.y.max()),
        ]
    )


def radial_coords_tile(
    x: np.ndarray,
    y: np.ndarray,
    origin: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    dx = x - origin[0]
    dy = y - origin[1]
    r = np.sqrt(dx**2 + dy**2)
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

    zero_idx = np.where(
        (cls_out == 0) & (zn > ground_thresh) & (zn <= max_fill_height)
    )[0]

    for idx in zero_idx:
        base = col_map.get((z_ir[idx], z_it[idx]), None)
        if base is None:
            continue

        classes = base.index.to_numpy()
        probs = base.to_numpy()
        post = np.array(
            [p * soft_height_prior(zn[idx], int(c)) for p, c in zip(probs, classes)]
        )
        if post.sum() > 0:
            cls_out[idx] = int(classes[np.argmax(post)])

    return cls_out


def process_single_tile(
    z_norm_las: Path,
    structure_las: Path,
    output_dir: Path,
    config: Mapping[str, float],
) -> None:
    """Process a single LAS tile pair."""
    z_norm_las = Path(z_norm_las)
    structure_las = Path(structure_las)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tile_id = z_norm_las.stem.split('.')[-1]
    print(f"\n{'='*60}")
    print(f"Processing tile: {tile_id}")
    print(f"{'='*60}")

    las_z = laspy.read(z_norm_las)
    las_s = laspy.read(structure_las)
    print(f"🔎 Z_norm points: {len(las_z.points):,}")
    print(f"🔎 Structure points: {len(las_s.points):,}")

    # 1) Cartesian merge
    if config.get("reuse_existing_cartesian", True) and np.any(
        (las_z.classification >= 64) & (las_z.classification <= 68)
    ):
        classification = las_z.classification.astype(np.uint8)
        print("✅ Reusing existing classes from Z_norm LAS.")
    else:
        print("🔄 Running cartesian voxel merge...")
        classification = cartesian_merge(
            las_z,
            las_s,
            config["voxel_size_xy"],
            config["voxel_size_z"],
        )

    # 2) Crown-centric refinement
    print("🔄 Running crown-centric refinement...")
    classification, instance_ids = refine_mallee_with_local_polar(
        las_z,
        las_s,
        classification,
        tile_id=tile_id,
        output_dir=output_dir,
        r_bin=config["r_bin"],
        theta_bin_deg=config["theta_bin_deg"],
        min_conf=config["min_conf"],
        min_flip_ratio=config["min_flip_ratio"],
        ground_thresh=config["ground_thresh"],
    )

    # 3) Tile-wide radial gap fill
    print("🔄 Running radial gap fill...")
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

    # 4) Height-based rules
    print("🔄 Applying height-based consistency rules...")
    zn_arr = las_z["Z_norm"].astype(float)

    classification[(classification == 66) & (zn_arr >= 0.80)] = 68
    classification[(classification == 67) & (zn_arr >= 0.40)] = 65
    classification[(classification == 68) & (zn_arr >= 3.0)] = 65

    MERGE_65_TO_68_ZNORM = 0.5
    mask_65 = classification == 65
    mask_65_high = mask_65 & (zn_arr > MERGE_65_TO_68_ZNORM)
    mask_65_low = mask_65 & (zn_arr < MERGE_65_TO_68_ZNORM)
    classification[mask_65_high] = 68

    keep_mask_base = ~mask_65_low
    classification_kept = classification[keep_mask_base]
    zn_kept = zn_arr[keep_mask_base]
    ids_kept = instance_ids[keep_mask_base]

    classification_kept[classification_kept == 65] = 68

    # Write full LAS output
    output_las = output_dir / f"z_norm_struct_{tile_id}_obj_polar.las"
    las_out = laspy.LasData(las_z.header)
    las_out.points = las_z.points[keep_mask_base].copy()
    las_out.classification = classification_kept

    if "Z_norm" not in las_out.point_format.dimension_names:
        las_out.add_extra_dim(laspy.ExtraBytesParams(name="Z_norm", type=np.float32))
    las_out["Z_norm"] = zn_kept.astype(np.float32)

    if "treeID" not in las_out.point_format.dimension_names:
        las_out.add_extra_dim(laspy.ExtraBytesParams(name="treeID", type=np.uint16))
    las_out["treeID"] = ids_kept

    las_out = ensure_las14_pf6(las_out)
    las_out.write(str(output_las))
    print(f"✅ Wrote full output: {output_las.name}")

    # Write filtered LAS (no class-0)
    output_las_no0 = output_dir / f"z_norm_struct_{tile_id}_obj_polar_no0.las"
    HIGH_Z_KEEP = 1.3
    keep = (classification_kept != 0) | (
        (classification_kept == 0) & (zn_kept >= HIGH_Z_KEEP)
    )

    final_cls = classification_kept[keep]
    final_zn = zn_kept[keep]
    final_ids = ids_kept[keep]

    promote = (final_cls == 0) & (final_zn >= HIGH_Z_KEEP)
    final_cls[promote] = 65
    final_cls[final_cls == 65] = 68

    las_no0 = laspy.LasData(las_out.header)
    las_no0.points = las_out.points[keep].copy()
    las_no0.classification = final_cls
    las_no0["treeID"] = final_ids
    las_no0 = ensure_las14_pf6(las_no0)
    las_no0.write(str(output_las_no0))
    print(f"✅ Wrote filtered output: {output_las_no0.name}")

    # Print summary
    print(f"\n📊 Classification Summary:")
    for cls_id, cls_name in CLASS_NAMES.items():
        count = np.sum(classification_kept == cls_id)
        if count > 0:
            print(f"   {cls_name:15s} ({cls_id:2d}): {count:8,} points")
    
    num_trees = len(np.unique(ids_kept[ids_kept > 0]))
    print(f"\n🌳 Detected trees: {num_trees}")
    print(f"{'='*60}\n")


def main():
    if len(sys.argv) != 4:
        print("Usage: python process_single_tile_malle.py <z_norm_las> <structure_las> <output_dir>")
        print("\nExample:")
        print("  python process_single_tile_malle.py \\")
        print("    /data/calperumResearch/site1_1_DD0001/lidar/clipped/raster/z_norm/csf/z_norm_csf_tiles_multispectral.113.las \\")
        print("    /data/calperumResearch/site1_1_DD0001/lidar/clipped/mask/coreg/icp/mask_tiles_multispectral.113.las \\")
        print("    /data/calperumResearch/site1_1_DD0001/lidar/clipped/3d_structure_malleid/")
        sys.exit(1)

    z_norm_las = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/data/calperumResearch/site1_1_DD0001/lidar/clipped/raster/z_norm/csf/z_norm_csf_tiles_multispectral.113.las")
    structure_las = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("/data/calperumResearch/site1_1_DD0001/lidar/clipped/mask/coreg/icp/mask_tiles_multispectral.113.las")
    output_dir = Path(sys.argv[3]) if len(sys.argv) > 3 else Path("/data/calperumResearch/site1_1_DD0001/lidar/clipped/3d_structure_malleid")

    if not z_norm_las.exists():
        print(f"❌ Error: Z_norm LAS file not found: {z_norm_las}")
        sys.exit(1)

    if not structure_las.exists():
        print(f"❌ Error: Structure LAS file not found: {structure_las}")
        sys.exit(1)

    print("🚀 Starting single tile processing...")
    print(f"   Z_norm LAS: {z_norm_las}")
    print(f"   Structure LAS: {structure_las}")
    print(f"   Output dir: {output_dir}")

    process_single_tile(z_norm_las, structure_las, output_dir, DEFAULT_CONFIG)
    print("🎉 Processing complete!")


if __name__ == "__main__":
    main()
#%%
