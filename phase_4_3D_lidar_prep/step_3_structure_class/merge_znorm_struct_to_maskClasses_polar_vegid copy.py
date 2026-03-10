#%%
"""
Merge Z_norm LAS with structure LAS using a 3-stage fusion with
CROWN-CENTRIC POLAR REFINEMENT + DEBUG VISUALIZATION.

UPDATED: Now saves 'treeID' based on detected crowns to the LAS file.
"""

from __future__ import annotations

import matplotlib
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
from skimage.measure import label as skimage_label
from skimage.measure import regionprops
from scipy import ndimage

# -----------------------------------------------------------
# Global constants
# -----------------------------------------------------------

CLASS_NAMES: Dict[int, str] = {
    64: "Forb", 65: "Woody Debris", 66: "Hummock", 67: "Shrub", 68: "Mallee", 0: "Unmatched"
}

HEIGHT_RANGES: Dict[int, Tuple[float, float]] = {
    64: (0.00, 0.40),   # Forb
    65: (0.05, 5.00),   # Woody debris/standing wood
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
# Helpers
# -----------------------------------------------------------

def _extra_dims(header) -> list:
    """
    Return extra byte params from a laspy header across versions.
    Older laspy builds used `extra_dims` while newer ones expose
    `extra_dimensions`. Fallback to the point_format entry if present.
    """
    extras = getattr(header, "extra_dimensions", None)
    if extras is None:
        extras = getattr(header, "extra_dims", None)
    if extras is None and hasattr(header, "point_format"):
        extras = getattr(header.point_format, "extra_dimensions", None)
    return extras or []

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
        if dim not in ("X", "Y", "Z"): out[dim] = las_in[dim]
    for eb in _extra_dims(las_in.header):
        if eb.name not in out.point_format.dimension_names: out.add_extra_dim(eb)
        out[eb.name] = las_in[eb.name]
    return out

def soft_height_prior(z_norm: float, cls: int) -> float:
    if cls not in HEIGHT_RANGES: return 1.0
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

def plot_mallee_debug(labeled_grid, centroids, min_x, min_y, grid_res, tile_id, out_dir):
    if len(centroids) < 2: return
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
        plt.scatter(grid_cx, grid_cy, c='white', edgecolors='black', marker='X', s=50, linewidths=1)
        plt.title(f"Tile {tile_id}: {len(centroids)-1} Objects")
        plt.savefig(out_dir / f"debug_mallee_{tile_id}.png", dpi=100)
        plt.close(fig)
    except Exception as e: print(f"❌ Plot failed: {e}")

def calculate_local_polar(x_arr, y_arr, labeled_grid, centroids, min_x, min_y, grid_res):
    ix = np.clip(((x_arr - min_x) / grid_res).astype(int), 0, labeled_grid.shape[1] - 1)
    iy = np.clip(((y_arr - min_y) / grid_res).astype(int), 0, labeled_grid.shape[0] - 1)
    obj_ids = labeled_grid[iy, ix]
    r = np.full_like(x_arr, np.nan)
    theta = np.full_like(x_arr, np.nan)
    valid_mask = (obj_ids > 0)
    if np.any(valid_mask):
        valid_ids = obj_ids[valid_mask]
        valid_ids = valid_ids[valid_ids < len(centroids)] # Safety check
        safe_mask = np.zeros_like(valid_mask)
        safe_mask[valid_mask] = (obj_ids[valid_mask] < len(centroids))
        if np.any(safe_mask):
            final_ids = obj_ids[safe_mask]
            cx, cy = centroids[final_ids, 0], centroids[final_ids, 1]
            dx, dy = x_arr[safe_mask] - cx, y_arr[safe_mask] - cy
            r[safe_mask] = np.sqrt(dx**2 + dy**2)
            theta[safe_mask] = np.mod(np.arctan2(dy, dx), 2 * np.pi)
    return r, theta, obj_ids

def polar_bins(r, theta, r_bin, theta_bin_deg):
    ir = np.floor(r / r_bin).astype(np.int32)
    it = np.floor(theta / np.deg2rad(theta_bin_deg)).astype(np.int32)
    return ir, it

# -----------------------------------------------------------
# Cartesian Merge
# -----------------------------------------------------------

def cartesian_merge(las_z, las_s, voxel_size_xy, voxel_size_z):
    x_min, y_min, z_min = min(las_z.x.min(), las_s.x.min()), min(las_z.y.min(), las_s.y.min()), min(las_z.z.min(), las_s.z.min())
    origin = np.array([x_min, y_min, z_min], dtype=np.float64)
    def vox(x, y, z): return np.stack([np.floor((x-origin[0])/voxel_size_xy).astype(int), np.floor((y-origin[1])/voxel_size_xy).astype(int), np.floor((z-origin[2])/voxel_size_z).astype(int)], axis=-1)
    
    vox_s = vox(las_s.x, las_s.y, las_s.z)
    raw_struct = np.where((las_s.classification >= 64) & (las_s.classification <= 68), las_s.classification, 0)
    uniq_vox_s, inv_s = np.unique(vox_s, axis=0, return_inverse=True)
    
    voxel_class = np.zeros(len(uniq_vox_s), dtype=np.uint16)
    for k in range(len(uniq_vox_s)):
        vals = raw_struct[inv_s == k]
        vals = vals[vals != 0]
        if vals.size > 0: voxel_class[k] = Counter(vals).most_common(1)[0][0]
            
    voxel_struct_map = {tuple(idx): c for idx, c in zip(uniq_vox_s, voxel_class) if c != 0}
    print(f"✅ Voxels with non-zero structure class: {len(voxel_struct_map):,}")
    
    vox_z = vox(las_z.x, las_z.y, las_z.z)
    classification = np.zeros(len(las_z.points), dtype=np.uint8)
    for i, v in enumerate(vox_z): classification[i] = voxel_struct_map.get(tuple(v), 0)
    return classification

# -----------------------------------------------------------
# REFINEMENT (MODIFIED FOR IDs)
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
) -> Tuple[np.ndarray, np.ndarray]:
    """Returns (refined_classification, instance_ids)"""
    cls_out = classification.copy()
    instance_ids = np.zeros(len(classification), dtype=np.uint16) # Initialize IDs

    if not np.any(las_s.classification == 68):
        return cls_out, instance_ids

    labeled_grid, centroids, min_x, min_y = get_mallee_centroids_watershed(
        las_s, grid_res=grid_res, min_tree_dist=1.5, smoothing_sigma=1.0
    )
    
    if len(centroids) == 0: return cls_out, instance_ids

    plot_mallee_debug(labeled_grid, centroids, min_x, min_y, grid_res, tile_id, output_dir)

    # Build Probs
    mask_65_66_68 = ((las_s.classification == 65) | (las_s.classification == 66) | (las_s.classification == 68))
    s_points_mask = mask_65_66_68
    s_r, s_th, s_obj_ids = calculate_local_polar(las_s.x[s_points_mask], las_s.y[s_points_mask], labeled_grid, centroids, min_x, min_y, grid_res)
    valid_s = (s_obj_ids > 0)
    s_ir, s_it = polar_bins(s_r[valid_s], s_th[valid_s], r_bin, theta_bin_deg)
    
    df_s = pd.DataFrame({"obj_id": s_obj_ids[valid_s], "ir": s_ir, "it": s_it, "cls": las_s.classification[s_points_mask][valid_s]})
    col_counts = df_s.groupby(["obj_id", "ir", "it", "cls"]).size().unstack(fill_value=0)
    col_probs = col_counts.div(col_counts.sum(axis=1), axis=0)
    prob_lookup = {idx: col_probs.loc[idx] for idx in col_probs.index}

    # Apply to Target
    z_r, z_th, z_obj_ids = calculate_local_polar(las_z.x, las_z.y, labeled_grid, centroids, min_x, min_y, grid_res)
    
    # --- CAPTURE IDs ---
    instance_ids = z_obj_ids.astype(np.uint16)

    z_ir, z_it = polar_bins(z_r, z_th, r_bin, theta_bin_deg)
    zn = las_z["Z_norm"].astype(float)
    idx_candidates = np.where((cls_out == 65) | (cls_out == 66) | (cls_out == 68))[0]

    for idx in idx_candidates:
        if zn[idx] <= ground_thresh: continue
        obj_id = z_obj_ids[idx]
        if obj_id == 0: continue 
        
        base = prob_lookup.get((obj_id, z_ir[idx], z_it[idx]), None)
        if base is None: continue

        classes = base.index.to_numpy()
        post = np.array([p * soft_height_prior(zn[idx], int(c)) for p, c in zip(base.to_numpy(), classes)])
        if post.sum() <= 0: continue

        best_cls = int(classes[np.argmax(post)])
        if best_cls != int(cls_out[idx]):
            cur_val = post[classes == int(cls_out[idx])][0] if int(cls_out[idx]) in classes else 0
            if post.max() / (cur_val + 1e-9) >= min_flip_ratio:
                cls_out[idx] = best_cls

    return cls_out, instance_ids

# -----------------------------------------------------------
# Global Fill & Pipeline
# -----------------------------------------------------------

def tile_center_xy(las_z): return np.array([0.5*(las_z.x.min()+las_z.x.max()), 0.5*(las_z.y.min()+las_z.y.max())])
def cart_to_polar_xy_global(x, y, origin): return np.sqrt((x-origin[0])**2 + (y-origin[1])**2), np.mod(np.arctan2(y-origin[1], x-origin[0]), 2*np.pi)

def fill_missing_with_cyl_height(las_z, las_s, classification, r_bin, theta_bin_deg, min_conf, ground_thresh, max_fill_height):
    cls_out = classification.copy()
    origin = tile_center_xy(las_z)
    mask_ext = (las_s.classification >= 64) & (las_s.classification <= 68)
    
    s_r, s_th = cart_to_polar_xy_global(las_s.x, las_s.y, origin)
    s_ir, s_it = polar_bins(s_r, s_th, r_bin, theta_bin_deg)
    df = pd.DataFrame({"ir": s_ir[mask_ext], "it": s_it[mask_ext], "cls": las_s.classification[mask_ext]})
    col_probs = df.groupby(["ir", "it", "cls"]).size().unstack(fill_value=0)
    col_probs = col_probs.div(col_probs.sum(axis=1), axis=0)
    if min_conf > 0: col_probs = col_probs[col_probs.max(axis=1) >= min_conf]
    col_map = {idx: col_probs.loc[idx] for idx in col_probs.index}

    z_r, z_th = cart_to_polar_xy_global(las_z.x, las_z.y, origin)
    z_ir, z_it = polar_bins(z_r, z_th, r_bin, theta_bin_deg)
    zn = las_z["Z_norm"].astype(float)
    
    zero_idx = np.where((cls_out == 0) & (zn > ground_thresh) & (zn <= max_fill_height))[0]
    for idx in zero_idx:
        base = col_map.get((z_ir[idx], z_it[idx]), None)
        if base is None: continue
        classes = base.index.to_numpy()
        post = np.array([p * soft_height_prior(zn[idx], int(c)) for p, c in zip(base.to_numpy(), classes)])
        if post.sum() > 0: cls_out[idx] = int(classes[np.argmax(post)])
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
    extra_ids: Dict[str, np.ndarray]
):
    """
    Write a LAS containing the filtered points and supplied ID dimensions.
    Keeps standard dimensions from base_las, adds Z_norm if missing.
    """
    hdr = laspy.LasHeader(version="1.4", point_format=6)
    hdr.scales = base_las.header.scales
    hdr.offsets = base_las.header.offsets

    out = laspy.LasData(hdr)

    # --- REMOVE / COMMENT OUT THIS BLOCK ---
    # for eb in _extra_dims(base_las.header):
    #     if eb.name not in out.point_format.dimension_names:
    #         out.add_extra_dim(eb)
    # ---------------------------------------

    # add required extras
    if "Z_norm" not in out.point_format.dimension_names:
        out.add_extra_dim(laspy.ExtraBytesParams(name="Z_norm", type=np.float32))
    for name in extra_ids.keys():
        if name not in out.point_format.dimension_names:
            out.add_extra_dim(laspy.ExtraBytesParams(name=name, type=np.uint16))

    # allocate points
    out.points = laspy.ScaleAwarePointRecord.zeros(
        point_count=len(classification_arr),
        point_format=out.header.point_format,
        scales=out.header.scales,
        offsets=out.header.offsets,
    )

    # copy coordinates
    out.X = base_las.X[mask]
    out.Y = base_las.Y[mask]
    out.Z = base_las.Z[mask]

    # copy common standard dims when present
    for dim in ("intensity", "return_number", "number_of_returns", "scan_direction_flag",
                "edge_of_flight_line", "classification_flags", "scan_angle_rank",
                "user_data", "point_source_id", "gps_time", "red", "green", "blue", "nir"):
        if dim in out.point_format.dimension_names and dim in base_las.point_format.dimension_names:
            out[dim] = base_las[dim][mask]

    out.classification = classification_arr
    out["Z_norm"] = zn_arr.astype(np.float32)
    for name, arr in extra_ids.items():
        out[name] = arr

    out = ensure_las14_pf6(out)
    path.parent.mkdir(parents=True, exist_ok=True)
    out.write(str(path))



def compute_component_ids(x_arr: np.ndarray, y_arr: np.ndarray, cls_arr: np.ndarray, target_cls: int, grid_res: float = 0.10) -> np.ndarray:
    """
    Computes 2D connected-component IDs for a target class using a simple grid.
    Returns an array of IDs (len = len(cls_arr)), with 0 meaning "no component".
    """
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

    labeled, num = ndimage.label(grid)
    if num == 0:
        return ids

    ids_masked = labeled[iy, ix]
    ids[mask] = ids_masked.astype(np.uint16)
    return ids

def merge_znorm_structure_one_file(z_norm_las, structure_las, output_las, output_las_no0, **kwargs):
    z_norm_las = Path(z_norm_las)
    structure_las = Path(structure_las)
    output_las = Path(output_las)
    if output_las_no0: output_las_no0 = Path(output_las_no0)
    tile_id = z_norm_las.stem.split('.')[-1]

    las_z = laspy.read(z_norm_las)
    las_s = laspy.read(structure_las)
    print(f"🔎 Z: {len(las_z.points):,}, S: {len(las_s.points):,}")

    if kwargs.get('reuse_existing_cartesian', True) and np.any((las_z.classification >= 64) & (las_z.classification <= 68)):
        classification = las_z.classification.astype(np.uint8)
        print("✅ Reusing existing classes.")
    else:
        classification = cartesian_merge(las_z, las_s, kwargs['voxel_size_xy'], kwargs['voxel_size_z'])

    # 2. Refinement (Gets IDs now)
    classification, instance_ids = refine_mallee_with_local_polar(
        las_z, las_s, classification,
        tile_id=tile_id, output_dir=output_las.parent,
        r_bin=kwargs['r_bin'], theta_bin_deg=kwargs['theta_bin_deg'],
        min_conf=kwargs['min_conf'], min_flip_ratio=kwargs['min_flip_ratio'], ground_thresh=kwargs['ground_thresh'],
    )

    # 3. Fill
    classification = fill_missing_with_cyl_height(
        las_z, las_s, classification,
        r_bin=kwargs['r_bin'], theta_bin_deg=kwargs['theta_bin_deg'],
        min_conf=kwargs['min_conf'], ground_thresh=kwargs['ground_thresh'], max_fill_height=2.0,
    )

    # 4. Height Rules
    zn_arr = las_z["Z_norm"].astype(float)
    classification[(classification == 66) & (zn_arr >= 2.0)] = 68
    classification[(classification == 67) & (zn_arr >= 2.0)] = 65
    classification[(classification == 68) & (zn_arr >= 3.0)] = 65

    # Merge woody debris (65) into mallee (68) when tall enough; drop low 65s
    MERGE_65_TO_68_ZNORM = 0.5
    mask_65 = classification == 65
    mask_65_high = mask_65 & (zn_arr > MERGE_65_TO_68_ZNORM)
    mask_65_low = mask_65 & (zn_arr < MERGE_65_TO_68_ZNORM)
    classification[mask_65_high] = 68

    keep_mask_base = ~mask_65_low
    classification_kept = classification[keep_mask_base]
    zn_kept = zn_arr[keep_mask_base]
    ids_kept = instance_ids[keep_mask_base]
    # Ensure no remaining 65 labels in output
    classification_kept[classification_kept == 65] = 68

    # Compute simple 2D component IDs for other classes on kept points
    grid_res = kwargs.get("r_bin", 0.10)
    forb_ids = compute_component_ids(las_z.x[keep_mask_base], las_z.y[keep_mask_base], classification_kept, 64, grid_res=grid_res)
    hummock_ids = compute_component_ids(las_z.x[keep_mask_base], las_z.y[keep_mask_base], classification_kept, 66, grid_res=grid_res)
    shrub_ids = compute_component_ids(las_z.x[keep_mask_base], las_z.y[keep_mask_base], classification_kept, 67, grid_res=grid_res)

    # Write Output
    las_out = laspy.LasData(las_z.header)
    las_out.points = las_z.points[keep_mask_base].copy()
    las_out.classification = classification_kept
    
    if "Z_norm" not in las_out.point_format.dimension_names:
        las_out.add_extra_dim(laspy.ExtraBytesParams(name="Z_norm", type=np.float32))
    las_out["Z_norm"] = zn_kept.astype(np.float32)
    
    # --- SAVE treeID ---
    if "treeID" not in las_out.point_format.dimension_names:
        las_out.add_extra_dim(laspy.ExtraBytesParams(name="treeID", type=np.uint16))
    las_out["treeID"] = ids_kept
    if "forbID" not in las_out.point_format.dimension_names:
        las_out.add_extra_dim(laspy.ExtraBytesParams(name="forbID", type=np.uint16))
    las_out["forbID"] = forb_ids
    if "hummockID" not in las_out.point_format.dimension_names:
        las_out.add_extra_dim(laspy.ExtraBytesParams(name="hummockID", type=np.uint16))
    las_out["hummockID"] = hummock_ids
    if "shrubID" not in las_out.point_format.dimension_names:
        las_out.add_extra_dim(laspy.ExtraBytesParams(name="shrubID", type=np.uint16))
    las_out["shrubID"] = shrub_ids
    
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
         # Ensure no remaining 65 labels in no0 output
        final_cls[final_cls == 65] = 68
        
        las_no0 = laspy.LasData(las_out.header)
        las_no0.points = las_out.points[keep].copy()
        las_no0.classification = final_cls
        las_no0["treeID"] = final_ids # Save IDs to no0 file
        las_no0["forbID"] = final_forb_ids
        las_no0["hummockID"] = final_hummock_ids
        las_no0["shrubID"] = final_shrub_ids
        las_no0 = ensure_las14_pf6(las_no0)
        las_no0.write(str(output_las_no0))

    # Additional ID-only exports
    treeid_only_path = output_las.parent / f"{output_las.stem}_treeID_only.las"
    vegids_path = output_las.parent / f"{output_las.stem}_vegIDs.las"
    write_id_subset_las(
        treeid_only_path,
        las_z,
        keep_mask_base,
        classification_kept,
        zn_kept,
        {"treeID": ids_kept}
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
        }
    )

    print(f"✅ Wrote: {output_las.name} (TreeIDs + veg IDs saved)")

def merge_folder(z_dir, s_dir, o_full, o_no0, voxel_size_xy=0.10, voxel_size_z=0.10, ground_thresh=0.15, r_bin=0.10, theta_bin_deg=10.0, min_conf=0.3, min_flip_ratio=1.5, reuse_existing_cartesian=True):
    z_norm_dir, structure_dir, out_dir_full, out_dir_no0 = map(Path, [z_dir, s_dir, o_full, o_no0])
    out_dir_full.mkdir(parents=True, exist_ok=True)
    out_dir_no0.mkdir(parents=True, exist_ok=True)
    z_files = sorted(z_norm_dir.glob("z_norm_csf_tiles_multispectral.*.las"))
    print(f"📁 Processing {len(z_files)} tiles...")

    for zf in z_files:
        tile_id = zf.stem.split('.')[-1]
        struct_file = structure_dir / f"mask_tiles_multispectral.{tile_id}.las"
        if not struct_file.exists(): 
            print(f"⚠️ Missing struct file for {tile_id}")
            continue
        
        merge_znorm_structure_one_file(
            zf, struct_file,
            out_dir_full / f"z_norm_struct_{tile_id}_obj_polar.las",
            out_dir_no0 / f"z_norm_struct_{tile_id}_obj_polar_no0.las",
            voxel_size_xy=voxel_size_xy, voxel_size_z=voxel_size_z, ground_thresh=ground_thresh,
            r_bin=r_bin, theta_bin_deg=theta_bin_deg, min_conf=min_conf, min_flip_ratio=min_flip_ratio,
            reuse_existing_cartesian=reuse_existing_cartesian
        )



# Main block
if __name__ == "__main__":
    #site 1
    z_dir = "/data/calperumResearch/site1_1_DD0001/lidar/clipped/raster/z_norm/csf/"
    s_dir = "/data/calperumResearch/site1_1_DD0001/lidar/clipped/mask/coreg/icp/"
    o_full = "/data/calperumResearch/site1_1_DD0001/lidar/clipped/3d_structure_vegid"
    o_no0 = "/data/calperumResearch/site1_1_DD0001/lidar/clipped/3d_structure_mask_vegid"
        
    merge_folder(z_dir, s_dir, o_full, o_no0, reuse_existing_cartesian=True)
#%%
