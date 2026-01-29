# structure_fusion_3d.py
# -*- coding: utf-8 -*-
"""
3D fusion of Z_norm LiDAR tiles with 2D structure labels.

Pipeline (matches diagram)
--------------------------
Inputs:
  - Z_norm LAS (geometry + Z_norm heights)
  - Structure LAS (labels 64–68 from 2D masks)

Steps per tile:
  1. Cartesian voxel merge
     -> Shared 3D voxel grid, majority-vote transfer of labels (64–68)
        from structure LAS to Z_norm LAS.

  2. Polar refinement (r, θ, z) + height priors
     -> Detect mallee crowns via CHM + watershed, assign crown IDs.
        In each crown-centric polar bin (r, θ), estimate class
        probabilities from structure LAS + apply soft height priors
        driven by data-driven class height ranges.

  3. Cylindrical / radial fill (class-0 gaps) [OPTIONAL]
     -> In tile-wide radial sectors, estimate class distributions
        from structure labels, combine with height priors to fill
        remaining class-0 points above ground.

  4. Final height sanity rules
     -> Use the same data-driven height ranges to fix obviously
        inconsistent labels (e.g. low mallee -> shrub, tall shrub
        -> mallee / woody debris).

Final Option A (simple treeID logic)
------------------------------------
In the FINAL outputs:

  • Only canopy mallee points get a treeID:
      classification == 68  AND  Z_norm >= mallee_min
  • All other points have treeID = 0.

Stage snapshots may store richer crown IDs for debugging, but the
main z_norm_struct_*_obj_polar.las outputs are Option A.

Outputs
-------
Per tile:
  - Full LAS with refined structure classes (64–68), Z_norm, and treeID.
    treeID > 0 is only set for canopy mallee (Option A).
  - Mask LAS with most class-0 points removed for 3D CNN training.

All outputs are LAS 1.4, point format 6 (PF6).
"""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pathlib import Path
from collections import Counter
from typing import Dict, Tuple, Mapping, Optional

import laspy
import numpy as np
import pandas as pd

from scipy.ndimage import center_of_mass
from scipy import ndimage
from scipy.spatial import cKDTree

from skimage.feature import peak_local_max
from skimage.segmentation import watershed
from skimage.measure import label as skimage_label
from skimage.measure import regionprops


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

# Coarse defaults (fallback if we cannot estimate
# data-driven thresholds for a class)
DEFAULT_HEIGHT_RANGES: Dict[int, Tuple[float, float]] = {
    64: (0.00, 0.40),  # Forb
    65: (0.05, 5.00),  # Woody debris / standing wood
    66: (0.00, 0.80),  # Hummock
    67: (0.40, 3.00),  # Shrub
    68: (1.50, 8.00),  # Mallee
}

DEFAULT_R_BIN = 0.10
DEFAULT_THETA_BIN_DEG = 10.0
DEFAULT_MIN_CONF = 0.3
DEFAULT_MIN_FLIP_RATIO = 1.5
DEFAULT_GROUND_TH = 0.01


# -----------------------------------------------------------
# Default configuration (can be overridden)
# -----------------------------------------------------------

DEFAULT_CONFIG = {
    "voxel_size_xy": 0.10,
    "voxel_size_z": 0.10,
    "ground_thresh": 0.15,
    "r_bin": DEFAULT_R_BIN,
    "theta_bin_deg": DEFAULT_THETA_BIN_DEG,
    "min_conf": DEFAULT_MIN_CONF,
    "min_flip_ratio": DEFAULT_MIN_FLIP_RATIO,
    "max_fill_height": 2.0,
    # Turn radial gap filling on/off
    "enable_radial_fill": False,
    # Save Step 1 (Cartesian voxel majority) LAS outputs here; if None,
    # they will be saved alongside the full outputs.
    "cartesian_out_dir": None,
    # Optional directories for Stage 2 (polar refinement) and
    # Stage 3 (radial fill) snapshots; default to full-output dir.
    "polar_out_dir": None,
    "radial_out_dir": None,
}


# -----------------------------------------------------------
# Helpers
# -----------------------------------------------------------

def ensure_las14_pf6(las_in: laspy.LasData) -> laspy.LasData:
    """
    Ensure LAS 1.4 / point format 6, preserving coordinates and extra dims.
    """
    if las_in.header.version == "1.4" and las_in.header.point_format.id >= 6:
        return las_in

    hdr = laspy.LasHeader(version="1.4", point_format=6)
    hdr.scales = las_in.header.scales
    hdr.offsets = las_in.header.offsets

    out = laspy.LasData(hdr)
    out.points = laspy.ScaleAwarePointRecord.zeros(len(las_in.points), hdr.point_format)
    out.X, out.Y, out.Z = las_in.X, las_in.Y, las_in.Z

    # Copy overlapping standard dimensions
    for dim in (set(las_in.point_format.dimension_names) &
                set(out.point_format.dimension_names)):
        if dim not in ("X", "Y", "Z"):
            out[dim] = las_in[dim]

    # Copy extra dimensions
    for eb in las_in.header.extra_dimensions:
        if eb.name not in out.point_format.dimension_names:
            out.add_extra_dim(eb)
        out[eb.name] = las_in[eb.name]

    return out


def soft_height_prior(
    z_norm: float,
    cls: int,
    height_ranges: Dict[int, Tuple[float, float]],
) -> float:
    """
    Soft prior: downweight classes outside their typical Z_norm range.
    Uses per-tile height_ranges (data-driven), with DEFAULT_HEIGHT_RANGES
    as fallback.
    """
    if cls in height_ranges:
        zmin, zmax = height_ranges[cls]
    else:
        zmin, zmax = DEFAULT_HEIGHT_RANGES.get(cls, (0.0, 10.0))
    return 1.0 if (zmin <= z_norm <= zmax) else 0.1


def compute_class_height_thresholds(
    las_s: laspy.LasData,
    las_z: laspy.LasData,
    percentile_low: float = 15.0,
    percentile_high: float = 85.0,
    max_match_dist: float = 0.30,
) -> Dict[int, Tuple[float, float]]:
    """
    Compute data-driven height ranges per class using:

      • class labels  -> from STRUCTURE LAS (las_s.classification)
      • heights       -> Z_norm from Z_NORM LAS, matched by nearest
                         neighbour in XY.
    """
    if "Z_norm" not in las_z.point_format.dimension_names:
        raise ValueError("Z_norm LAS has no 'Z_norm' extra dimension.")

    zn_z = np.asarray(las_z["Z_norm"], dtype=float)

    # Match each structure point to nearest Z_norm point in XY
    xy_z = np.column_stack((las_z.x, las_z.y))
    xy_s = np.column_stack((las_s.x, las_s.y))

    tree = cKDTree(xy_z)
    dists, idx = tree.query(xy_s, k=1, distance_upper_bound=max_match_dist)

    s_zn = np.full(len(las_s.points), np.nan, dtype=float)
    valid = np.isfinite(dists) & (idx < len(xy_z))
    s_zn[valid] = zn_z[idx[valid]]

    n_valid = int(np.sum(valid))
    print(
        f"  ℹ️ Matched {n_valid:,} / {len(las_s.points):,} structure points to Z_norm "
        f"(≤ {max_match_dist:.2f} m)"
    )
    if n_valid == 0:
        print("  ⚠️ No matches; falling back to DEFAULT_HEIGHT_RANGES.")
        return DEFAULT_HEIGHT_RANGES.copy()

    thresholds: Dict[int, Tuple[float, float]] = {}

    for cls in [64, 65, 66, 67, 68]:
        mask = (las_s.classification == cls) & np.isfinite(s_zn)
        n = int(np.sum(mask))
        if n < 20:
            thresholds[cls] = DEFAULT_HEIGHT_RANGES.get(cls, (0.0, 10.0))
            print(
                f"  ⚠️ Class {cls} ({CLASS_NAMES.get(cls,'?')}): only {n} pts, "
                f"using default {thresholds[cls][0]:.2f}–{thresholds[cls][1]:.2f} m"
            )
            continue

        heights = s_zn[mask]

        # Robust: clip extreme tails first
        p5, p95 = np.percentile(heights, [5, 95])
        heights_clip = heights[(heights >= p5) & (heights <= p95)]
        if len(heights_clip) >= 10:
            heights_use = heights_clip
        else:
            heights_use = heights

        low, high = np.percentile(heights_use, [percentile_low, percentile_high])
        if low > high:
            low, high = high, low

        low = max(0.0, float(low))
        high = float(high)

        thresholds[cls] = (low, high)
        print(
            f"  Class {cls} ({CLASS_NAMES.get(cls,'?')}): "
            f"{low:.2f}–{high:.2f} m (n={len(heights_use):,})"
        )

    return thresholds


# -----------------------------------------------------------
# Crown-centric helpers
# -----------------------------------------------------------

def get_mallee_centroids_watershed(
    las_s: laspy.LasData,
    grid_res: float = 0.10,
    min_tree_dist: float = 0.75,
    smoothing_sigma: float = 0.5,
    fallback_points: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]] = None,
    fallback_min_height: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray, float, float]:
    """
    Detect mallee crowns from a canopy height model (CHM) using a hybrid
    blobs + watershed approach. If structure LAS lacks class-68 points,
    fall back to supplied points (typically Z_norm LAS points above a
    minimum height).
    """
    x_s = las_s.x
    y_s = las_s.y
    z_s = np.asarray(las_s.z, dtype=float)
    cls_s = las_s.classification

    mask_mallee = cls_s == 68
    source = "structure"

    if np.any(mask_mallee):
        use_x = x_s[mask_mallee]
        use_y = y_s[mask_mallee]
        use_h = z_s[mask_mallee]
    else:
        # Fallback to geometry if no mallee in structure LAS
        if fallback_points is None:
            min_x = float(np.min(x_s))
            max_x = float(np.max(x_s))
            min_y = float(np.min(y_s))
            max_y = float(np.max(y_s))
            w = int(np.ceil((max_x - min_x) / grid_res)) + 1
            h = int(np.ceil((max_y - min_y) / grid_res)) + 1
            print(
                "  ⚠️ No mallee crowns in structure LAS and no fallback provided; "
                "skipping watershed."
            )
            return np.zeros((h, w), dtype=int), np.array([]), min_x, min_y

        fb_x, fb_y, fb_h = fallback_points
        max_fb_h = float(np.nanmax(fb_h)) if fb_h.size > 0 else 0.0
        if max_fb_h < fallback_min_height:
            min_x = float(np.min(fb_x))
            max_x = float(np.max(fb_x))
            min_y = float(np.min(fb_y))
            max_y = float(np.max(fb_y))
            w = int(np.ceil((max_x - min_x) / grid_res)) + 1
            h = int(np.ceil((max_y - min_y) / grid_res)) + 1
            print(
                f"  ⚠️ Fallback heights max out at {max_fb_h:.2f}m "
                f"(< {fallback_min_height:.2f}m); skipping watershed."
            )
            return np.zeros((h, w), dtype=int), np.array([]), min_x, min_y

        # progressively relax threshold to ensure enough points
        fb_thresholds = [
            fallback_min_height,
            max(0.8 * fallback_min_height, 0.6),
            max(0.6 * fallback_min_height, 0.5),
            max(0.4 * fallback_min_height, 0.4),
            0.3,
        ]
        chosen_mask = None
        chosen_th = fallback_min_height
        for th in fb_thresholds:
            mask = fb_h >= th
            if np.count_nonzero(mask) >= 100:
                chosen_mask = mask
                chosen_th = th
                break
        if chosen_mask is None:
            mask = fb_h >= 0.1
            if np.count_nonzero(mask) == 0:
                min_x = float(np.min(fb_x))
                max_x = float(np.max(fb_x))
                min_y = float(np.min(fb_y))
                max_y = float(np.max(fb_y))
                w = int(np.ceil((max_x - min_x) / grid_res)) + 1
                h = int(np.ceil((max_y - min_y) / grid_res)) + 1
                print("  ⚠️ Fallback heights missing even at 0.10m; skipping watershed.")
                return np.zeros((h, w), dtype=int), np.array([]), min_x, min_y
            chosen_mask = mask
            chosen_th = 0.1

        use_x = fb_x[chosen_mask]
        use_y = fb_y[chosen_mask]
        use_h = np.asarray(fb_h[chosen_mask], dtype=float)
        source = f"Z_norm fallback (≥ {chosen_th:.2f}m, {len(use_x):,} pts)"

    min_x = float(np.min(use_x))
    max_x = float(np.max(use_x))
    min_y = float(np.min(use_y))
    max_y = float(np.max(use_y))

    w = int(np.ceil((max_x - min_x) / grid_res)) + 1
    h = int(np.ceil((max_y - min_y) / grid_res)) + 1
    w = max(1, w)
    h = max(1, h)

    mx = ((use_x - min_x) / grid_res).astype(int)
    my = ((use_y - min_y) / grid_res).astype(int)
    mz = use_h

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
        print(f"  ⚠️ Watershed failed to find crowns using {source} data.")
        return labels, np.array([]), min_x, min_y

    coms = center_of_mass(binary_mask, labels, range(1, num_features + 1))

    centroids = np.zeros((num_features + 1, 2))
    for i, (cy, cx) in enumerate(coms):
        real_x = min_x + (cx * grid_res)
        real_y = min_y + (cy * grid_res)
        centroids[i + 1] = [real_x, real_y]

    print(f"🌳 [Watershed] Identified {num_features} crowns ({source}).")
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
    """
    Save a debug plot showing crown labels and centroids for a tile.
    """
    if len(centroids) < 2:
        return

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        fig = plt.figure(figsize=(10, 10))
        masked_grid = np.ma.masked_where(labeled_grid == 0, labeled_grid)
        plt.imshow(
            masked_grid,
            origin="lower",
            cmap="nipy_spectral",
            interpolation="nearest",
        )

        real_cx = centroids[1:, 0]
        real_cy = centroids[1:, 1]
        grid_cx = (real_cx - min_x) / grid_res
        grid_cy = (real_cy - min_y) / grid_res

        plt.scatter(
            grid_cx,
            grid_cy,
            c="white",
            edgecolors="black",
            marker="X",
            s=50,
            linewidths=1,
        )
        plt.title(f"Tile {tile_id}: {len(centroids) - 1} crowns")
        plt.savefig(out_dir / f"debug_mallee_{tile_id}.png", dpi=100)
        plt.close(fig)
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
    """
    For each point, find its crown label and compute crown-centric
    polar coordinates (r, θ) around the crown centroid.
    """
    ix = np.clip(((x_arr - min_x) / grid_res).astype(int),
                 0, labeled_grid.shape[1] - 1)
    iy = np.clip(((y_arr - min_y) / grid_res).astype(int),
                 0, labeled_grid.shape[0] - 1)

    obj_ids = labeled_grid[iy, ix]

    r = np.full_like(x_arr, np.nan, dtype=float)
    theta = np.full_like(x_arr, np.nan, dtype=float)

    valid_mask = obj_ids > 0
    if np.any(valid_mask):
        safe_mask = np.zeros_like(valid_mask)
        safe_mask[valid_mask] = obj_ids[valid_mask] < len(centroids)

        if np.any(safe_mask):
            final_ids = obj_ids[safe_mask]
            cx, cy = centroids[final_ids, 0], centroids[final_ids, 1]
            dx, dy = x_arr[safe_mask] - cx, y_arr[safe_mask] - cy

            r[safe_mask] = np.sqrt(dx**2 + dy**2)
            theta[safe_mask] = np.mod(np.arctan2(dy, dx), 2 * np.pi)

    return r, theta, obj_ids


def polar_bins(
    r: np.ndarray,
    theta: np.ndarray,
    r_bin: float,
    theta_bin_deg: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Discretise polar coordinates into integer bin indices (r, θ).
    """
    ir = np.full(r.shape, -1, dtype=np.int32)
    it = np.full(r.shape, -1, dtype=np.int32)
    valid = np.isfinite(r) & np.isfinite(theta)
    if np.any(valid):
        ir[valid] = np.floor(r[valid] / r_bin).astype(np.int32)
        it[valid] = np.floor(theta[valid] / np.deg2rad(theta_bin_deg)).astype(
            np.int32
        )
    return ir, it


# -----------------------------------------------------------
# Cartesian voxel merge
# -----------------------------------------------------------

def cartesian_merge(
    las_z: laspy.LasData,
    las_s: laspy.LasData,
    voxel_size_xy: float,
    voxel_size_z: float,
) -> np.ndarray:
    """
    Transfer structure labels (64–68) from structure LAS to Z_norm LAS
    via a shared 3D voxel grid (majority vote within voxels).
    """
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

    # Structure voxels
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

    # Transfer to Z_norm points
    vox_z = vox(las_z.x, las_z.y, las_z.z)
    classification = np.zeros(len(las_z.points), dtype=np.uint8)
    for i, v in enumerate(vox_z):
        classification[i] = voxel_struct_map.get(tuple(v), 0)

    return classification


def write_fullcloud_stage_output(
    las_template: laspy.LasData,
    classification: np.ndarray,
    output_path: Path,
    zn_values: np.ndarray,
    stage_label: str,
    tree_ids: Optional[np.ndarray] = None,
) -> None:
    """
    Save a stage snapshot (Stage 1, 2, or 3) containing:
      - XYZ from las_template
      - classification
      - Z_norm
      - optional treeID

    Always written as LAS 1.4 PF6.

    NOTE: For final outputs, Option A behaviour is enforced elsewhere.
          Here we just persist what we are given (useful for debugging).
    """

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Convert to LAS 1.4 PF6 FIRST
    las_base = ensure_las14_pf6(las_template)

    n = len(las_base.points)
    cls_arr = np.asarray(classification, dtype=np.uint8)
    zn_arr = np.asarray(zn_values, dtype=np.float32)
    if tree_ids is None:
        ids_arr = np.zeros(n, dtype=np.uint16)
    else:
        ids_arr = np.asarray(tree_ids, dtype=np.uint16)

    if len(cls_arr) != n or len(zn_arr) != n or len(ids_arr) != n:
        raise ValueError("write_fullcloud_stage_output(): array length mismatch.")

    # Create empty LAS with correct header
    las_out = laspy.LasData(las_base.header)

    # Add extra dimensions BEFORE assigning points
    if "Z_norm" not in las_out.point_format.dimension_names:
        las_out.add_extra_dim(laspy.ExtraBytesParams(name="Z_norm", type=np.float32))

    if "treeID" not in las_out.point_format.dimension_names:
        las_out.add_extra_dim(laspy.ExtraBytesParams(name="treeID", type=np.uint16))

    # Allocate full point record
    las_out.points = laspy.ScaleAwarePointRecord.zeros(n, las_out.point_format)

    # Copy XYZ from base
    las_out.X = las_base.X
    las_out.Y = las_base.Y
    las_out.Z = las_base.Z

    # Assign fields
    las_out.classification = cls_arr
    las_out["Z_norm"] = zn_arr
    las_out["treeID"] = ids_arr

    # Write
    las_out.write(str(output_path))
    print(f"   📝 Saved {stage_label}: {output_path.name}")


# -----------------------------------------------------------
# Tile-wide radial / cylindrical gap fill
# -----------------------------------------------------------

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
    """
    Express points in tile-wide radial coordinates (r, θ) around origin.
    """
    dx = x - origin[0]
    dy = y - origin[1]
    r = np.sqrt(dx**2 + dy**2)
    theta = np.mod(np.arctan2(dy, dx), 2 * np.pi)
    return r, theta


def fill_missing_classes_radial(
    las_z: laspy.LasData,
    las_s: laspy.LasData,
    classification: np.ndarray,
    height_ranges: Dict[int, Tuple[float, float]],
    r_bin: float,
    theta_bin_deg: float,
    min_conf: float,
    ground_thresh: float,
    max_fill_height: float,
) -> np.ndarray:
    """
    Fill remaining class-0 points using tile-wide radial sectors.

    For each (r, θ) sector we estimate a class distribution from
    STRUCTURE labels and use it to assign likely classes to
    unlabelled Z_norm points, modulated by the same height prior.
    """
    cls_out = classification.copy()
    origin = tile_center_xy(las_z)

    # Sector stats from structure labels
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

    # Apply to Z_norm points that remain class 0
    z_r, z_th = radial_coords_tile(las_z.x, las_z.y, origin)
    z_ir, z_it = polar_bins(z_r, z_th, r_bin, theta_bin_deg)
    zn = np.asarray(las_z["Z_norm"], dtype=float)

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
            [
                p * soft_height_prior(zn[idx], int(c), height_ranges)
                for p, c in zip(probs, classes)
            ]
        )
        if post.sum() > 0:
            cls_out[idx] = int(classes[np.argmax(post)])

    return cls_out


# -----------------------------------------------------------
# Crown-centric refinement (with treeID)
# -----------------------------------------------------------

def refine_mallee_with_local_polar(
    las_z: laspy.LasData,
    las_s: laspy.LasData,
    classification: np.ndarray,
    height_ranges: Dict[int, Tuple[float, float]],
    tile_id: str,
    output_dir: Path,
    r_bin: float = DEFAULT_R_BIN,
    theta_bin_deg: float = DEFAULT_THETA_BIN_DEG,
    min_conf: float = DEFAULT_MIN_CONF,
    min_flip_ratio: float = DEFAULT_MIN_FLIP_RATIO,
    ground_thresh: float = DEFAULT_GROUND_TH,
    grid_res: float = 0.10,
    labeled_grid: Optional[np.ndarray] = None,
    centroids: Optional[np.ndarray] = None,
    min_x: Optional[float] = None,
    min_y: Optional[float] = None,
    fallback_points: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, float]:
    """
    Crown-centric refinement for woody/hummock/mallee classes.

    Returns
    -------
    cls_out : np.ndarray
        Refined classification array.
    instance_ids : np.ndarray
        Crown labels per point (treeID-like crown IDs).
    labeled_grid, centroids, min_x, min_y
        Watershed outputs (for reuse).

    NOTE: Option A treeID filter (only canopy mallee) is applied later.
    Here we assign crown IDs to all crown-associated points so the
    bookkeeping is easier.
    """
    cls_out = classification.copy()
    instance_ids = np.zeros(len(classification), dtype=np.uint16)

    # Use provided watershed results or compute new ones
    if labeled_grid is None or centroids is None:
        labeled_grid, centroids, min_x, min_y = get_mallee_centroids_watershed(
            las_s,
            grid_res=grid_res,
            min_tree_dist=1.5,
            smoothing_sigma=1.0,
            fallback_points=fallback_points,
        )

    if len(centroids) == 0:
        empty_grid = np.zeros((10, 10), dtype=int)
        empty_cents = np.array([])
        return cls_out, instance_ids, empty_grid, empty_cents, 0.0, 0.0

    plot_mallee_debug(
        labeled_grid,
        centroids,
        min_x,
        min_y,
        grid_res,
        tile_id,
        output_dir,
    )

    # Build local class probabilities in (treeID, r, θ) bins from STRUCTURE LAS
    mask_65_66_68 = np.isin(las_s.classification, [65, 66, 68])
    prob_lookup: Dict = {}

    if np.any(mask_65_66_68):
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
        if np.any(valid_s):
            s_ir, s_it = polar_bins(
                s_r[valid_s], s_th[valid_s], r_bin, theta_bin_deg
            )

            df_s = pd.DataFrame(
                {
                    "obj_id": s_obj_ids[valid_s],
                    "ir": s_ir,
                    "it": s_it,
                    "cls": las_s.classification[mask_65_66_68][valid_s],
                }
            )

            col_counts = (
                df_s.groupby(["obj_id", "ir", "it", "cls"])
                .size()
                .unstack(fill_value=0)
            )
            col_probs = col_counts.div(col_counts.sum(axis=1), axis=0)
            prob_lookup = {idx: col_probs.loc[idx] for idx in col_probs.index}

    # Apply to Z_norm points
    z_r, z_th, z_obj_ids = calculate_local_polar(
        las_z.x,
        las_z.y,
        labeled_grid,
        centroids,
        min_x,
        min_y,
        grid_res,
    )

    # Assign crown IDs (raw, before Option A filter)
    instance_ids = z_obj_ids.astype(np.uint16)

    z_ir, z_it = polar_bins(z_r, z_th, r_bin, theta_bin_deg)
    zn = np.asarray(las_z["Z_norm"], dtype=float)

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
            [
                p * soft_height_prior(zn[idx], int(c), height_ranges)
                for p, c in zip(probs, classes)
            ]
        )
        if post.sum() <= 0:
            continue

        best_cls = int(classes[np.argmax(post)])
        if best_cls != int(cls_out[idx]):
            cur_val = (
                post[classes == int(cls_out[idx])][0]
                if int(cls_out[idx]) in classes
                else 0.0
            )
            if post.max() / (cur_val + 1e-9) >= min_flip_ratio:
                cls_out[idx] = best_cls

    return cls_out, instance_ids, labeled_grid, centroids, min_x, min_y


# -----------------------------------------------------------
# Per-file fusion (main workhorse)
# -----------------------------------------------------------

def merge_znorm_structure_one_file(
    z_norm_las,
    structure_las,
    output_las,
    output_las_no0,
    config: Mapping[str, float],
    output_cartesian_dir: Optional[Path] = None,
    output_polar_dir: Optional[Path] = None,
    output_radial_dir: Optional[Path] = None,
) -> None:
    """
    Run the full 3D fusion pipeline for a single pair of LAS tiles.

    Option A: in the final outputs, only canopy mallee points carry treeID.
    """
    z_norm_las = Path(z_norm_las)
    structure_las = Path(structure_las)
    output_las = Path(output_las)
    if output_las_no0:
        output_las_no0 = Path(output_las_no0)

    tile_id = z_norm_las.stem.split(".")[-1]

    las_z = laspy.read(z_norm_las)
    las_s = laspy.read(structure_las)
    print(f"🔎 Tile {tile_id} | Z: {len(las_z.points):,}, S: {len(las_s.points):,}")

    if "Z_norm" not in las_z.point_format.dimension_names:
        raise ValueError("Z_norm LAS has no 'Z_norm' extra dimension.")
    zn_all = np.asarray(las_z["Z_norm"], dtype=float)

    stage_prefix = f"z_norm_struct_{tile_id}"

    cart_out_dir = Path(output_cartesian_dir or output_las.parent)
    polar_out_dir = Path(output_polar_dir or output_las.parent)
    radial_out_dir = Path(output_radial_dir or output_las.parent)

    cart_out_dir.mkdir(parents=True, exist_ok=True)
    polar_out_dir.mkdir(parents=True, exist_ok=True)
    if config.get("enable_radial_fill", False):
        radial_out_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Cartesian voxel merge (always run and save)
    classification = cartesian_merge(
        las_z,
        las_s,
        config["voxel_size_xy"],
        config["voxel_size_z"],
    )

    stage1_path = cart_out_dir / f"{stage_prefix}_stage1_cartesian.las"
    write_fullcloud_stage_output(
        las_z,
        classification,
        stage1_path,
        zn_all,
        "Stage 1 (Cartesian majority)",
    )

    # Derive per-class height stats from STRUCTURE labels + Z_norm heights
    print(f"📊 Computing data-driven height thresholds for tile {tile_id}:")
    height_ranges = compute_class_height_thresholds(las_s, las_z)

    # Compute watershed once, reuse for refinement
    grid_res = config.get("r_bin", 0.10)
    fallback_points = (
        las_z.x,
        las_z.y,
        zn_all,
    )

    labeled_grid, centroids, min_x, min_y = get_mallee_centroids_watershed(
        las_s,
        grid_res=grid_res,
        min_tree_dist=1.5,
        smoothing_sigma=1.0,
        fallback_points=fallback_points,
    )

    # Step 2: Crown-centric refinement (with provisional treeIDs)
    classification, instance_ids, _, _, _, _ = refine_mallee_with_local_polar(
        las_z,
        las_s,
        classification,
        height_ranges=height_ranges,
        tile_id=tile_id,
        output_dir=output_las.parent,
        r_bin=config["r_bin"],
        theta_bin_deg=config["theta_bin_deg"],
        min_conf=config["min_conf"],
        min_flip_ratio=config["min_flip_ratio"],
        ground_thresh=config["ground_thresh"],
        grid_res=grid_res,
        labeled_grid=labeled_grid,
        centroids=centroids,
        min_x=min_x,
        min_y=min_y,
        fallback_points=fallback_points,
    )

    stage2_path = polar_out_dir / f"{stage_prefix}_stage2_polar.las"
    write_fullcloud_stage_output(
        las_z,
        classification,
        stage2_path,
        zn_all,
        "Stage 2 (Polar refinement)",
        tree_ids=instance_ids,
    )

    # Step 3: Tile-wide radial gap fill (optional)
    if config.get("enable_radial_fill", False):
        classification = fill_missing_classes_radial(
            las_z,
            las_s,
            classification,
            height_ranges=height_ranges,
            r_bin=config["r_bin"],
            theta_bin_deg=config["theta_bin_deg"],
            min_conf=config["min_conf"],
            ground_thresh=config["ground_thresh"],
            max_fill_height=config["max_fill_height"],
        )
        stage3_path = radial_out_dir / f"{stage_prefix}_stage3_radial.las"
        write_fullcloud_stage_output(
            las_z,
            classification,
            stage3_path,
            zn_all,
            "Stage 3 (Radial fill)",
            tree_ids=instance_ids,
        )
    else:
        print("  🔧 Radial gap fill disabled (enable_radial_fill=False)")

    # Step 4: Final height sanity rules (using same height_ranges)
    zn_arr = zn_all

    hummock_low, hummock_high = height_ranges.get(66, (0.0, 0.8))
    shrub_low, shrub_high = height_ranges.get(67, (0.4, 3.0))
    mallee_low, mallee_high = height_ranges.get(68, (1.5, 8.0))

    # Conservative transition thresholds
    hummock_to_shrub = min(hummock_high * 0.9, 0.7)  # hummock too tall -> shrub
    shrub_to_mallee = max(shrub_high * 0.8, mallee_low * 0.85)
    mallee_min = max(0.6, mallee_low * 0.7)
    mallee_max = min(6.0, mallee_high * 1.2)

    print(
        f"  → Hummock→Shrub: {hummock_to_shrub:.2f}m | "
        f"Shrub→Mallee: {shrub_to_mallee:.2f}m"
    )
    print(f"  → Mallee range: {mallee_min:.2f}m - {mallee_max:.2f}m")

    # Low mallee -> shrub when below adaptive threshold
    classification[(classification == 68) & (zn_arr < mallee_min)] = 67

    # Hummock -> shrub in transition zone
    classification[(classification == 66) & (zn_arr >= hummock_to_shrub)] = 67

    # Shrub -> mallee when tall enough
    classification[(classification == 67) & (zn_arr >= shrub_to_mallee)] = 68

    # Very tall mallee -> woody debris (standing dead wood)
    classification[(classification == 68) & (zn_arr >= mallee_max)] = 65

    # Merge woody debris (65) into mallee (68) when tall enough; drop low 65s
    MERGE_65_TO_68_ZNORM = 0.5
    mask_65 = classification == 65
    mask_65_high = mask_65 & (zn_arr > MERGE_65_TO_68_ZNORM)
    mask_65_low = mask_65 & (zn_arr < MERGE_65_TO_68_ZNORM)
    classification[mask_65_high] = 68

    # Base keep mask: drop low woody debris (likely noise)
    keep_mask_base = ~mask_65_low

    classification_kept = classification[keep_mask_base]
    zn_kept = zn_arr[keep_mask_base]

    # Option A:
    # Start from crown labels but **zero out everything that is not canopy mallee**
    ids_kept = instance_ids[keep_mask_base].copy()
    is_mallee_crown = (classification_kept == 68) & (zn_kept >= mallee_min)
    ids_kept[~is_mallee_crown] = 0

    # No remaining 65 labels in main output: merge all to 68
    classification_kept[classification_kept == 65] = 68

    # Debug: print class distribution
    unique_classes, counts = np.unique(classification_kept, return_counts=True)
    print("  📋 Final class distribution:")
    for cls, count in zip(unique_classes, counts):
        pct = 100.0 * count / len(classification_kept)
        cls_name = CLASS_NAMES.get(cls, f"Unknown-{cls}")
        print(f"     Class {cls} ({cls_name}): {count:,} points ({pct:.1f}%)")

    unique_trees = np.unique(ids_kept[ids_kept > 0])
    print(f"  🌳 TreeIDs (mallee canopy only): {len(unique_trees)} unique crowns")

    # Write full LAS output (Output 1 in diagram)
    las_base = ensure_las14_pf6(las_z)
    subset_points = las_base.points[keep_mask_base].copy()

    las_out = laspy.LasData(las_base.header)
    las_out.points = subset_points

    if "treeID" not in las_out.point_format.dimension_names:
        las_out.add_extra_dim(
            laspy.ExtraBytesParams(name="treeID", type=np.uint16)
        )
    if "Z_norm" not in las_out.point_format.dimension_names:
        las_out.add_extra_dim(
            laspy.ExtraBytesParams(name="Z_norm", type=np.float32)
        )
    las_out.classification = np.asarray(classification_kept, dtype=np.uint8)
    las_out["Z_norm"] = np.asarray(zn_kept, dtype=np.float32)
    las_out["treeID"] = np.asarray(ids_kept, dtype=np.uint16)
    output_las.parent.mkdir(parents=True, exist_ok=True)
    las_out.write(str(output_las))

    # Optional filtered LAS with most class-0 removed (Output 2)
    if output_las_no0:
        HIGH_Z_KEEP = 1.3
        keep = (classification_kept != 0) | (
            (classification_kept == 0) & (zn_kept >= HIGH_Z_KEEP)
        )

        final_cls = classification_kept[keep]
        final_zn = zn_kept[keep]
        final_ids = ids_kept[keep]

        # Any high unlabeled points are promoted to woody/mallee
        promote = (final_cls == 0) & (final_zn >= HIGH_Z_KEEP)
        final_cls[promote] = 65
        final_cls[final_cls == 65] = 68  # merge to mallee

        # --- Build no-zero filtering output LAS ---
        las_no0 = laspy.LasData(las_out.header)
        n2 = np.sum(keep)

        if "treeID" not in las_no0.point_format.dimension_names:
            las_no0.add_extra_dim(
                laspy.ExtraBytesParams(name="treeID", type=np.uint16)
            )
        if "Z_norm" not in las_no0.point_format.dimension_names:
            las_no0.add_extra_dim(
                laspy.ExtraBytesParams(name="Z_norm", type=np.float32)
            )

        las_no0.points = laspy.ScaleAwarePointRecord.zeros(n2, las_no0.point_format)

        subset2 = las_out.points[keep]

        las_no0.X = subset2.X
        las_no0.Y = subset2.Y
        las_no0.Z = subset2.Z

        las_no0.classification = final_cls.astype(np.uint8)
        las_no0["Z_norm"] = final_zn.astype(np.float32)
        las_no0["treeID"] = final_ids.astype(np.uint16)

        las_no0.write(str(output_las_no0))
        print(f"   ✔ Filtered LAS written: {output_las_no0.name}")

    print(f"✅ Wrote: {output_las.name} (Option A: treeID for canopy mallee only)")


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
    """
    Run the 3D fusion pipeline for all tiles in a folder.
    """
    cfg = dict(DEFAULT_CONFIG)
    if config is not None:
        cfg.update(config)

    z_norm_dir, structure_dir, out_dir_full, out_dir_no0 = map(
        Path,
        [z_dir, s_dir, o_full, o_no0],
    )
    out_dir_full.mkdir(parents=True, exist_ok=True)
    out_dir_no0.mkdir(parents=True, exist_ok=True)
    out_dir_cart = (
        out_dir_full
        if cfg.get("cartesian_out_dir") is None
        else Path(cfg["cartesian_out_dir"])
    )
    out_dir_cart.mkdir(parents=True, exist_ok=True)

    out_dir_polar = (
        out_dir_full
        if cfg.get("polar_out_dir") is None
        else Path(cfg["polar_out_dir"])
    )
    out_dir_polar.mkdir(parents=True, exist_ok=True)

    radial_dir_setting = cfg.get("radial_out_dir")
    out_dir_radial = Path(radial_dir_setting) if radial_dir_setting else out_dir_full
    if cfg.get("enable_radial_fill", False):
        out_dir_radial.mkdir(parents=True, exist_ok=True)

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
            output_cartesian_dir=out_dir_cart,
            output_polar_dir=out_dir_polar,
            output_radial_dir=out_dir_radial,
        )
