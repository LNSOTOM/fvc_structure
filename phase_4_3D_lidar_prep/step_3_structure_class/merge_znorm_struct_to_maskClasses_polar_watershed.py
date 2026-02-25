#%%
"""
Merge Z_norm LAS with structure LAS using a 3-stage fusion with
CROWN-CENTRIC POLAR REFINEMENT + DEBUG VISUALIZATION.

FIXED: Added matplotlib backend 'Agg' to ensure plots save on servers/headless environments.
"""

from __future__ import annotations

import matplotlib
# --- FIX: Set backend to non-interactive 'Agg' BEFORE importing pyplot ---
matplotlib.use('Agg') 
import matplotlib.pyplot as plt

from pathlib import Path
from collections import Counter
from typing import Dict, Tuple, Mapping, Optional

import laspy
import numpy as np
import pandas as pd
from scipy.ndimage import label, center_of_mass
from skimage.feature import peak_local_max
from skimage.segmentation import watershed
from skimage.feature import peak_local_max

from skimage.measure import label as skimage_label
from skimage.measure import regionprops
from scipy import ndimage


# -----------------------------------------------------------
# Global constants
# -----------------------------------------------------------

# Extended class names
CLASS_NAMES: Dict[int, str] = {
    64: "Forb",
    65: "Woody Debris",
    66: "Hummock",
    67: "Shrub",
    68: "Mallee",
    0:  "Unmatched",
}

# Expected Z_norm ranges for each class
HEIGHT_RANGES: Dict[int, Tuple[float, float]] = {
    64: (0.00, 0.40),   # Forb
    65: (0.05, 5.00),   # Woody debris
    66: (0.00, 0.80),   # Hummock
    67: (0.40, 3.00),   # Shrub
    68: (1.50, 8.00),   # Mallee
}

# Default hyperparameters
DEFAULT_R_BIN = 0.10          # m
DEFAULT_THETA_BIN_DEG = 10.0  # degrees
DEFAULT_MIN_CONF = 0.3
DEFAULT_MIN_FLIP_RATIO = 1.5
DEFAULT_GROUND_TH = 0.01      # m


# -----------------------------------------------------------
# LAS 1.4 / Point Format 6 helper
# -----------------------------------------------------------

def ensure_las14_pf6(las_in: laspy.LasData) -> laspy.LasData:
    if las_in.header.version == "1.4" and las_in.header.point_format.id >= 6:
        return las_in

    hdr = laspy.LasHeader(version="1.4", point_format=6)
    hdr.scales = las_in.header.scales
    hdr.offsets = las_in.header.offsets

    out = laspy.LasData(hdr)
    out.points = laspy.ScaleAwarePointRecord.zeros(len(las_in.points), hdr.point_format)

    out.X = las_in.X
    out.Y = las_in.Y
    out.Z = las_in.Z

    src_dims = set(las_in.point_format.dimension_names)
    dst_dims = set(out.point_format.dimension_names)
    for dim in (src_dims & dst_dims):
        if dim not in ("X", "Y", "Z"):
            out[dim] = las_in[dim]

    for eb in las_in.header.extra_dimensions:
        if eb.name not in out.point_format.dimension_names:
            out.add_extra_dim(eb)
        out[eb.name] = las_in[eb.name]

    return out


# -----------------------------------------------------------
# Height prior helper
# -----------------------------------------------------------

def soft_height_prior(z_norm: float, cls: int) -> float:
    if cls not in HEIGHT_RANGES:
        return 1.0
    zmin, zmax = HEIGHT_RANGES[cls]
    return 1.0 if (zmin <= z_norm <= zmax) else 0.1


# -----------------------------------------------------------
# CROWN-CENTRIC HELPERS
# -----------------------------------------------------------

def get_mallee_centroids_watershed(
    las_s: laspy.LasData, 
    grid_res: float = 0.10,
    min_tree_dist: float = 0.75, # REDUCED from 1.5 to 0.75 to catch closer/smaller trees
    smoothing_sigma: float = 0.5 # REDUCED from 1.0 to 0.5 to stop blurring small trees
) -> Tuple[np.ndarray, np.ndarray, float, float]:
    """
    ROBUST TOP-DOWN DETECTOR (Hybrid):
    1. Identifies "Blobs" (Old V1 method) to ensure no trees are lost.
    2. Identifies "Peaks" (New Watershed method) to split touching trees.
    3. Merges them: If a V1 Blob has no Watershed Peak, we force-add a seed.
    """
    x = las_s.x
    y = las_s.y
    z = las_s.z
    cls = las_s.classification
    
    # 1. Setup Grid Limits
    min_x, max_x = np.min(x), np.max(x)
    min_y, max_y = np.min(y), np.max(y)
    
    w = int(np.ceil((max_x - min_x) / grid_res)) + 1
    h = int(np.ceil((max_y - min_y) / grid_res)) + 1
    
    # 2. Create Canopy Height Model (CHM)
    mask_68 = (cls == 68)
    if not np.any(mask_68):
        return np.zeros((h, w), dtype=int), np.array([]), min_x, min_y

    mx = ((x[mask_68] - min_x) / grid_res).astype(int)
    my = ((y[mask_68] - min_y) / grid_res).astype(int)
    mz = z[mask_68]
    
    mx = np.clip(mx, 0, w - 1)
    my = np.clip(my, 0, h - 1)
    
    # CHM: Max Z per pixel
    sort_idx = np.argsort(mz)
    chm_grid = np.zeros((h, w), dtype=np.float32)
    chm_grid[my[sort_idx], mx[sort_idx]] = mz[sort_idx]
    
    # Binary Mask (Where trees exist at all)
    binary_mask = chm_grid > 0

    # 3. Gaussian Smooth (Gentle smoothing)
    chm_smoothed = ndimage.gaussian_filter(chm_grid, sigma=smoothing_sigma)

    # 4. Find Standard Peaks (for splitting)
    min_dist_px = int(min_tree_dist / grid_res)
    # Ensure min_dist_px is at least 1
    min_dist_px = max(1, min_dist_px) 

    # Find peaks as coordinates
    coordinates = peak_local_max(
        chm_smoothed,
        min_distance=min_dist_px,
        threshold_abs=0.1,  # minimum height threshold
        labels=binary_mask
    )
    
    # Create binary marker image from coordinates
    local_maxi = np.zeros_like(chm_smoothed, dtype=bool)
    if len(coordinates) > 0:
        local_maxi[coordinates[:, 0], coordinates[:, 1]] = True
    
    # 5. THE RESCUE STEP (Fix for missing canopies)
    # Find connected components (V1 style)
    # If a component has NO peaks, it means the tree was too small/flat and got deleted.
    # We must add a seed back in for it.
    
    # Create integer labels for markers
    markers = ndimage.label(local_maxi)[0]
    
    # Label the original binary mask (V1 style blobs)
    blob_labels = skimage_label(binary_mask)
    blob_props = regionprops(blob_labels, intensity_image=chm_grid)
    
    # Get coordinates of current peaks
    peak_coords = np.argwhere(local_maxi) # (y, x)
    
    next_marker_id = markers.max() + 1
    
    for blob in blob_props:
        # Check if this blob contains any existing marker
        # We look at the bounding box of the blob to speed up checking
        min_row, min_col, max_row, max_col = blob.bbox
        
        # Check if any peak falls inside this blob's mask
        # (A simplified check is bounding box, but let's be precise)
        has_marker = False
        
        # Optimization: Only check peaks within the bbox
        in_bbox = (
            (peak_coords[:, 0] >= min_row) & (peak_coords[:, 0] < max_row) &
            (peak_coords[:, 1] >= min_col) & (peak_coords[:, 1] < max_col)
        )
        possible_peaks = peak_coords[in_bbox]
        
        for py, px in possible_peaks:
            if blob_labels[py, px] == blob.label:
                has_marker = True
                break
        
        if not has_marker:
            # RESCUE: This blob was missed! Add a marker at its highest point.
            # weighted_centroid returns (row, col)
            cy, cx = map(int, blob.weighted_centroid)
            
            # Ensure coordinates are within bounds
            cy = np.clip(cy, 0, h - 1)
            cx = np.clip(cx, 0, w - 1)
            
            markers[cy, cx] = next_marker_id
            next_marker_id += 1

    # 6. Watershed Segmentation
    # We use the augmented markers (Standard Peaks + Rescued Blobs)
    labels = watershed(-chm_smoothed, markers, mask=binary_mask)
    
    num_features = labels.max()
    
    if num_features == 0:
        return labels, np.array([]), min_x, min_y

    # 7. Calculate Centroids
    coms = center_of_mass(binary_mask, labels, range(1, num_features + 1))
    
    centroids = np.zeros((num_features + 1, 2))
    for i, (cy, cx) in enumerate(coms):
        real_x = min_x + (cx * grid_res)
        real_y = min_y + (cy * grid_res)
        centroids[i+1] = [real_x, real_y]
        
    print(f"🌳 [Robust Watershed] Identified {num_features} crowns (Recalled missing trees).")
    return labels, centroids, min_x, min_y

# -----------------------------------------------------------
# VISUALIZATION FUNCTION
# -----------------------------------------------------------

def plot_mallee_debug(labeled_grid, centroids, min_x, min_y, grid_res, tile_id, out_dir):
    """
    Saves a PNG showing identified trees (colored blobs) and their centroids (X).
    """
    # Centroids array includes index 0 (background), so len < 2 means no trees found.
    if len(centroids) < 2:
        print(f"⚠️ No trees to plot for tile {tile_id}")
        return

    # Ensure output directory exists (just in case)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        fig = plt.figure(figsize=(10, 10))
        
        # Mask background (0) so it is white
        masked_grid = np.ma.masked_where(labeled_grid == 0, labeled_grid)
        
        # Plot grid
        plt.imshow(masked_grid, origin='lower', cmap='nipy_spectral', interpolation='nearest')
        
        # Plot centroids
        real_cx = centroids[1:, 0]
        real_cy = centroids[1:, 1]
        
        grid_cx = (real_cx - min_x) / grid_res
        grid_cy = (real_cy - min_y) / grid_res
        
        plt.scatter(grid_cx, grid_cy, c='white', edgecolors='black', marker='X', s=50, linewidths=1, label='Centroid')
        
        plt.title(f"Tile {tile_id}: {len(centroids)-1} Objects")
        plt.legend()
        plt.tight_layout()
        
        out_path = out_dir / f"debug_mallee_{tile_id}.png"
        plt.savefig(out_path, dpi=100)
        plt.close(fig) # Explicitly close figure to free memory
        print(f"🖼️  Saved debug plot: {out_path}")
        
    except Exception as e:
        print(f"❌ Failed to save plot for {tile_id}: {e}")


def calculate_local_polar(
    x_arr: np.ndarray,
    y_arr: np.ndarray,
    labeled_grid: np.ndarray,
    centroids: np.ndarray,
    min_x: float,
    min_y: float,
    grid_res: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    # Map points to grid indices
    ix = ((x_arr - min_x) / grid_res).astype(int)
    iy = ((y_arr - min_y) / grid_res).astype(int)
    
    h, w = labeled_grid.shape
    ix = np.clip(ix, 0, w - 1)
    iy = np.clip(iy, 0, h - 1)
    
    # Look up Object ID for each point
    obj_ids = labeled_grid[iy, ix]
    
    r = np.full_like(x_arr, np.nan)
    theta = np.full_like(x_arr, np.nan)
    
    valid_mask = (obj_ids > 0)
    
    if np.any(valid_mask):
        valid_ids = obj_ids[valid_mask]
        cx = centroids[valid_ids, 0]
        cy = centroids[valid_ids, 1]
        
        dx = x_arr[valid_mask] - cx
        dy = y_arr[valid_mask] - cy
        
        r[valid_mask] = np.sqrt(dx*dx + dy*dy)
        theta[valid_mask] = np.mod(np.arctan2(dy, dx), 2 * np.pi)
        
    return r, theta, obj_ids


def polar_bins(r, theta, r_bin, theta_bin_deg):
    ir = np.floor(r / r_bin).astype(np.int32)
    it = np.floor(theta / np.deg2rad(theta_bin_deg)).astype(np.int32)
    return ir, it


# -----------------------------------------------------------
# Cartesian voxel merge
# -----------------------------------------------------------

def cartesian_merge(las_z, las_s, voxel_size_xy=0.10, voxel_size_z=0.10):
    print(f"   [cartesian_merge] voxel_size_xy={voxel_size_xy:.3f}, voxel_size_z={voxel_size_z:.3f}")
    x_min = min(las_z.x.min(), las_s.x.min())
    y_min = min(las_z.y.min(), las_s.y.min())
    z_min = min(las_z.z.min(), las_s.z.min())
    origin = np.array([x_min, y_min, z_min], dtype=np.float64)

    def vox(x, y, z):
        ix = np.floor((x - origin[0]) / voxel_size_xy).astype(np.int32)
        iy = np.floor((y - origin[1]) / voxel_size_xy).astype(np.int32)
        iz = np.floor((z - origin[2]) / voxel_size_z).astype(np.int32)
        return np.stack([ix, iy, iz], axis=-1)

    vox_s = vox(las_s.x, las_s.y, las_s.z)
    raw_struct = las_s.classification.astype(np.uint16)
    raw_struct_masked = np.where((raw_struct >= 64) & (raw_struct <= 68), raw_struct, 0)

    uniq_vox_s, inv_s = np.unique(vox_s, axis=0, return_inverse=True)
    voxel_class = np.zeros(len(uniq_vox_s), dtype=np.uint16)

    for k in range(len(uniq_vox_s)):
        vals = raw_struct_masked[inv_s == k]
        vals = vals[vals != 0]
        if vals.size == 0: continue
        voxel_class[k] = Counter(vals).most_common(1)[0][0]

    voxel_struct_map = {tuple(idx): c for idx, c in zip(uniq_vox_s, voxel_class) if c != 0}
    print(f"✅ Voxels with non-zero structure class: {len(voxel_struct_map):,}")

    vox_z = vox(las_z.x, las_z.y, las_z.z)
    classification = np.zeros(len(las_z.points), dtype=np.uint8)

    matched = 0
    for i, v in enumerate(vox_z):
        lbl = voxel_struct_map.get(tuple(v), 0)
        classification[i] = lbl
        if lbl != 0: matched += 1

    print(f"✅ Matched labels via Cartesian voxels: {matched:,} / {len(las_z.points):,}")
    return classification


# -----------------------------------------------------------
# OBJECT-CENTRIC Polar Refinement
# -----------------------------------------------------------

def refine_mallee_with_local_polar(
    las_z: laspy.LasData,
    las_s: laspy.LasData,
    classification: np.ndarray,
    tile_id: str,
    output_dir: Path,
    r_bin: float = DEFAULT_R_BIN,
    theta_bin_deg: float = DEFAULT_THETA_BIN_DEG,
    min_conf: float = DEFAULT_MIN_CONF,
    min_flip_ratio: float = DEFAULT_MIN_FLIP_RATIO,
    ground_thresh: float = DEFAULT_GROUND_TH,
    grid_res: float = 0.10,
) -> np.ndarray:
    """
    Refines classes using polar coordinates relative to the NEAREST MALLEE OBJECT CENTROID.
    """
    cls_out = classification.copy()

    # 1. Identify Mallee Objects
    if not np.any(las_s.classification == 68):
        print("⚪ No mallee (68) in structure LAS → skip polar refinement.")
        return cls_out

    # --- UPDATED TO USE WATERSHED LOGIC ---
    labeled_grid, centroids, min_x, min_y = get_mallee_centroids_watershed(
        las_s, 
        grid_res=grid_res,
        min_tree_dist=1.5,
        smoothing_sigma=1.0
    )
    
    if len(centroids) == 0:
        return cls_out

    # --- VISUALIZATION STEP ---
    plot_mallee_debug(labeled_grid, centroids, min_x, min_y, grid_res, tile_id, output_dir)
    # --------------------------

    # 2. Build Column Probabilities (Structure LAS)
    mask_65_66_68 = (
        (las_s.classification == 65) |
        (las_s.classification == 66) |
        (las_s.classification == 68)
    )
    
    s_points_mask = mask_65_66_68
    
    s_r, s_th, s_obj_ids = calculate_local_polar(
        las_s.x[s_points_mask], las_s.y[s_points_mask],
        labeled_grid, centroids, min_x, min_y, grid_res
    )
    
    valid_s = (s_obj_ids > 0)
    s_ir, s_it = polar_bins(s_r[valid_s], s_th[valid_s], r_bin, theta_bin_deg)
    
    df_s = pd.DataFrame({
        "obj_id": s_obj_ids[valid_s],
        "ir":  s_ir,
        "it":  s_it,
        "cls": las_s.classification[s_points_mask][valid_s],
    })

    col_counts = df_s.groupby(["obj_id", "ir", "it", "cls"]).size().unstack(fill_value=0)
    col_probs = col_counts.div(col_counts.sum(axis=1), axis=0)
    prob_lookup = {idx: col_probs.loc[idx] for idx in col_probs.index}
    
    print(f"📦 Built local polar models for {len(prob_lookup):,} columns.")

    # 3. Apply to Z_norm LAS
    z_r, z_th, z_obj_ids = calculate_local_polar(
        las_z.x, las_z.y,
        labeled_grid, centroids, min_x, min_y, grid_res
    )
    
    z_ir, z_it = polar_bins(z_r, z_th, r_bin, theta_bin_deg)
    zn = las_z["Z_norm"].astype(float)

    flips = Counter()
    idx_candidates = np.where((cls_out == 65) | (cls_out == 66) | (cls_out == 68))[0]

    for idx in idx_candidates:
        if zn[idx] <= ground_thresh: continue
        obj_id = z_obj_ids[idx]
        if obj_id == 0: continue 
        
        key = (obj_id, z_ir[idx], z_it[idx])
        base = prob_lookup.get(key, None)
        if base is None: continue

        classes = base.index.to_numpy()
        probs = base.to_numpy()
        post = np.array([
            p * soft_height_prior(zn[idx], int(c))
            for p, c in zip(probs, classes)
        ])
        if post.sum() <= 0: continue

        best_cls = int(classes[np.argmax(post)])
        current = int(cls_out[idx])
        if best_cls == current: continue

        best_val = post[classes == best_cls][0]
        cur_val = post[classes == current][0]

        if best_val / (cur_val + 1e-9) >= min_flip_ratio:
            flips[f"{current}->{best_cls}"] += 1
            cls_out[idx] = best_cls

    print(f"🔁 Polar Flips (Object-Centric): {dict(flips)}")
    return cls_out


# -----------------------------------------------------------
# Simple Cylindrical Fill (Fallback to Tile Center)
# -----------------------------------------------------------

def tile_center_xy(las_z):
    x0 = 0.5 * (las_z.x.min() + las_z.x.max())
    y0 = 0.5 * (las_z.y.min() + las_z.y.max())
    return np.array([x0, y0], dtype=float)

def cart_to_polar_xy_global(x, y, origin):
    dx = x - origin[0]
    dy = y - origin[1]
    r = np.sqrt(dx * dx + dy * dy)
    theta = np.mod(np.arctan2(dy, dx), 2 * np.pi)
    return r, theta

def fill_missing_with_cyl_height(
    las_z, las_s, classification,
    r_bin=DEFAULT_R_BIN, theta_bin_deg=DEFAULT_THETA_BIN_DEG,
    min_conf=DEFAULT_MIN_CONF, ground_thresh=DEFAULT_GROUND_TH,
    max_fill_height=2.0
):
    cls_out = classification.copy()
    origin = tile_center_xy(las_z)
    
    mask_ext = (las_s.classification >= 64) & (las_s.classification <= 68)
    
    def build_global_probs():
        s_r, s_th = cart_to_polar_xy_global(las_s.x, las_s.y, origin)
        s_ir, s_it = polar_bins(s_r, s_th, r_bin, theta_bin_deg)
        df = pd.DataFrame({"ir": s_ir[mask_ext], "it": s_it[mask_ext], "cls": las_s.classification[mask_ext]})
        col_counts = df.groupby(["ir", "it", "cls"]).size().unstack(fill_value=0)
        col_probs = col_counts.div(col_counts.sum(axis=1), axis=0)
        if min_conf > 0:
            col_probs = col_probs[col_probs.max(axis=1) >= min_conf]
        return {idx: col_probs.loc[idx] for idx in col_probs.index}

    col_map = build_global_probs()
    if not col_map: return cls_out

    z_r, z_th = cart_to_polar_xy_global(las_z.x, las_z.y, origin)
    z_ir, z_it = polar_bins(z_r, z_th, r_bin, theta_bin_deg)
    zn = las_z["Z_norm"].astype(float)

    cond = (cls_out == 0) & (zn > ground_thresh)
    if max_fill_height: cond &= (zn <= max_fill_height)
    
    zero_idx = np.where(cond)[0]
    filled = 0

    for idx in zero_idx:
        key = (z_ir[idx], z_it[idx])
        base = col_map.get(key, None)
        if base is None: continue
        
        classes = base.index.to_numpy()
        probs = base.to_numpy()
        post = np.array([p * soft_height_prior(zn[idx], int(c)) for p, c in zip(probs, classes)])
        
        if post.sum() > 0:
            best_cls = int(classes[np.argmax(post)])
            cls_out[idx] = best_cls
            filled += 1
            
    print(f"🧩 Cylindrical fill (Global): labelled {filled:,} points.")
    return cls_out


# -----------------------------------------------------------
# Pipeline
# -----------------------------------------------------------

def merge_znorm_structure_one_file(
    z_norm_las: str | Path,
    structure_las: str | Path,
    output_las: str | Path,
    output_las_no0: str | Path | None = None,
    voxel_size_xy: float = 0.10,
    voxel_size_z: float = 0.10,
    ground_thresh: float = 0.01,
    r_bin: float = DEFAULT_R_BIN,
    theta_bin_deg: float = DEFAULT_THETA_BIN_DEG,
    min_conf: float = DEFAULT_MIN_CONF,
    min_flip_ratio: float = DEFAULT_MIN_FLIP_RATIO,
    reuse_existing_cartesian: bool = True,
) -> None:
    z_norm_las = Path(z_norm_las)
    structure_las = Path(structure_las)
    output_las = Path(output_las)
    if output_las_no0: output_las_no0 = Path(output_las_no0)

    # Extract tile ID for debug plotting
    tile_id = z_norm_las.stem.split('.')[-1]

    las_z = laspy.read(z_norm_las)
    las_s = laspy.read(structure_las)
    print(f"🔎 Z: {len(las_z.points):,}, S: {len(las_s.points):,}")

    if "Z_norm" not in las_z.point_format.dimension_names:
        raise ValueError("Missing 'Z_norm'")

    # 1. Cartesian Merge
    has_ext_classes = np.any((las_z.classification >= 64) & (las_z.classification <= 68))
    if reuse_existing_cartesian and has_ext_classes:
        classification = las_z.classification.astype(np.uint8)
        print("✅ Reusing existing classes.")
    else:
        classification = cartesian_merge(las_z, las_s, voxel_size_xy, voxel_size_z)

    # 2. Object-Centric Polar Refinement + VISUALIZATION
    classification = refine_mallee_with_local_polar(
        las_z, las_s, classification,
        tile_id=tile_id,              # <-- NEW
        output_dir=output_las.parent, # <-- NEW
        r_bin=r_bin,
        theta_bin_deg=theta_bin_deg,
        min_conf=min_conf,
        min_flip_ratio=min_flip_ratio,
        ground_thresh=ground_thresh,
    )

    # 3. Global Cylindrical Fill
    classification = fill_missing_with_cyl_height(
        las_z, las_s, classification,
        r_bin=r_bin, theta_bin_deg=theta_bin_deg,
        min_conf=min_conf, ground_thresh=ground_thresh,
        max_fill_height=2.0,
    )

    # 4. Height Rules
    zn_arr = las_z["Z_norm"].astype(float)
    HUMMOCK_TO_MALLEE_TH = 2.0
    SHRUB_TO_WOODY_TH    = 2.0
    MALLEE_TO_WOODY_TH   = 3.0

    mask_hummock_tall = (classification == 66) & (zn_arr >= HUMMOCK_TO_MALLEE_TH)
    classification[mask_hummock_tall] = 68

    mask_shrub_tall = (classification == 67) & (zn_arr >= SHRUB_TO_WOODY_TH)
    classification[mask_shrub_tall] = 65

    mask_mallee_tall = (classification == 68) & (zn_arr >= MALLEE_TO_WOODY_TH)
    classification[mask_mallee_tall] = 65

    # Merge woody debris (65) into mallee (68) when tall enough; drop low 65s
    MERGE_65_TO_68_ZNORM = 0.5
    mask_65 = classification == 65
    mask_65_high = mask_65 & (zn_arr > MERGE_65_TO_68_ZNORM)
    mask_65_low = mask_65 & (zn_arr < MERGE_65_TO_68_ZNORM)
    classification[mask_65_high] = 68

    keep_mask_base = ~mask_65_low
    classification_kept = classification[keep_mask_base]
    zn_kept = zn_arr[keep_mask_base]

    # Write Output
    las_out = laspy.LasData(las_z.header)
    las_out.points = las_z.points[keep_mask_base].copy()
    las_out.classification = classification_kept
    
    if "Z_norm" not in las_out.point_format.dimension_names:
        las_out.add_extra_dim(laspy.ExtraBytesParams(name="Z_norm", type=np.float32))
    las_out["Z_norm"] = zn_kept.astype(np.float32)
    
    las_out = ensure_las14_pf6(las_out)
    output_las.parent.mkdir(parents=True, exist_ok=True)
    las_out.write(str(output_las))
    
    if output_las_no0:
        HIGH_Z_KEEP = 1.3
        keep = (classification_kept != 0) | ((classification_kept == 0) & (zn_kept >= HIGH_Z_KEEP))
        # Logic update: if keeping high 0s, promote them to 65
        final_cls = classification_kept[keep]
        final_zn = zn_kept[keep]
        promote = (final_cls == 0) & (final_zn >= HIGH_Z_KEEP)
        final_cls[promote] = 65
        
        las_no0 = laspy.LasData(las_out.header)
        las_no0.points = las_out.points[keep].copy()
        las_no0.classification = final_cls
        las_no0 = ensure_las14_pf6(las_no0)
        las_no0.write(str(output_las_no0))

    print(f"✅ Wrote: {output_las.name}")

# -----------------------------------------------------------
# Batch Helper
# -----------------------------------------------------------

def merge_folder(
    z_norm_dir, structure_dir, out_dir_full, out_dir_no0,
    voxel_size_xy=0.10, voxel_size_z=0.10, ground_thresh=0.15,
    r_bin=DEFAULT_R_BIN, theta_bin_deg=10.0,
    min_conf=DEFAULT_MIN_CONF, min_flip_ratio=DEFAULT_MIN_FLIP_RATIO,
    reuse_existing_cartesian=True
):
    z_norm_dir = Path(z_norm_dir)
    structure_dir = Path(structure_dir)
    out_dir_full = Path(out_dir_full)
    out_dir_no0 = Path(out_dir_no0)
    out_dir_full.mkdir(parents=True, exist_ok=True)
    out_dir_no0.mkdir(parents=True, exist_ok=True)

    z_files = sorted(z_norm_dir.glob("z_norm_csf_tiles_multispectral.*.las"))
    print(f"📁 Processing {len(z_files)} tiles...")

    for zf in z_files:
        stem = zf.stem
        # Assumes format: z_norm_csf_tiles_multispectral.TILEID.las
        try:
            tile_id = stem.split('.')[-1]
        except IndexError:
            print(f"⚠️ Skipping weird filename: {stem}")
            continue

        struct_file = structure_dir / f"mask_tiles_multispectral.{tile_id}.las"
        
        if not struct_file.exists(): 
            print(f"⚠️ Missing struct file for {tile_id}")
            continue
        
        merge_znorm_structure_one_file(
            zf, struct_file,
            out_dir_full / f"z_norm_struct_{tile_id}_obj_polar.las",
            out_dir_no0 / f"z_norm_struct_{tile_id}_obj_polar_no0.las",
            voxel_size_xy, voxel_size_z, ground_thresh,
            r_bin, theta_bin_deg, min_conf, min_flip_ratio,
            reuse_existing_cartesian
        )

# Main block
if __name__ == "__main__":
    #site 1
    z_dir = "/data/calperumResearch/site1_1_DD0001/lidar/clipped/raster/z_norm/csf/"
    s_dir = "/data/calperumResearch/site1_1_DD0001/lidar/clipped/mask/coreg/icp/"
    o_full = "/data/calperumResearch/site1_1_DD0001/lidar/clipped/3d_structure_obj_polar"
    o_no0 = "/data/calperumResearch/site1_1_DD0001/lidar/clipped/3d_structure_mask_obj_polar"
        
    merge_folder(z_dir, s_dir, o_full, o_no0, reuse_existing_cartesian=True)
# %%
