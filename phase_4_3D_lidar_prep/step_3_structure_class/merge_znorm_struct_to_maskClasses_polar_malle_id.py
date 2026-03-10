
#%%
"""
3D fusion of Z_norm LiDAR tiles with 2D structure labels.

Fixes included
--------------
✅ Robust LAS 1.4 PF6 conversion that never crashes on missing/broken extra dims
   (prevents: ValueError: no field of name treeID)

✅ Clean, modular pipeline:
   1) Cartesian voxel majority transfer (structure -> Z_norm)
   2) Crown-centric polar refinement (watershed crowns -> treeID)
   3) Optional tile-wide radial gap fill
   4) Final height sanity rules
   5) Write full + filtered (no0) outputs

Outputs
-------
- Full LAS: refined classes (64–68), Z_norm, treeID (treeID only for mallee crowns)
- No0 LAS: removes most class-0 points for 3D CNN training

LAS output format: LAS 1.4, Point Format 6 (PF6)
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from collections import Counter
from typing import Dict, Tuple, Optional, Mapping, Any

import numpy as np
import pandas as pd
import laspy

from scipy.spatial import cKDTree
from scipy.ndimage import center_of_mass
from scipy import ndimage

from skimage.feature import peak_local_max
from skimage.segmentation import watershed
from skimage.measure import label as skimage_label
from skimage.measure import regionprops

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# =============================================================================
# Constants
# =============================================================================

CLASS_NAMES: Dict[int, str] = {
    64: "Forb",
    65: "Woody Debris",
    66: "Hummock",
    67: "Shrub",
    68: "Mallee",
    0: "Unmatched",
}

DEFAULT_HEIGHT_RANGES: Dict[int, Tuple[float, float]] = {
    64: (0.00, 0.40),
    65: (0.05, 5.00),
    66: (0.00, 0.80),
    67: (0.40, 3.00),
    68: (1.50, 8.00),
}


# =============================================================================
# Config
# =============================================================================

@dataclass
class FusionConfig:
    voxel_size_xy: float = 0.10
    voxel_size_z: float = 0.10

    # Heights
    ground_thresh: float = 0.15  # ignore below this for refinement/fill

    # Polar refinement bins (around crown centroid)
    r_bin: float = 0.10
    theta_bin_deg: float = 10.0

    # Watershed CHM grid resolution (do NOT tie this to r_bin)
    grid_res: float = 0.10

    # Refinement thresholds
    min_conf: float = 0.30
    min_flip_ratio: float = 1.50

    # Radial fill
    enable_radial_fill: bool = False
    max_fill_height: float = 2.0

    # Optional stage snapshots
    cartesian_out_dir: Optional[str] = None
    polar_out_dir: Optional[str] = None
    radial_out_dir: Optional[str] = None


# =============================================================================
# LAS utilities (robust)
# =============================================================================

def _dtype_from_eb_type(eb_type: Any) -> np.dtype:
    try:
        return np.dtype(eb_type)
    except Exception:
        return np.dtype("float32")


def ensure_las14_pf6(las_in: laspy.LasData) -> laspy.LasData:
    """
    Return a LAS 1.4 PF6 version of las_in, preserving:
      - XYZ
      - overlapping standard dims (safe copy)
      - extra dims (safe copy: if unreadable/missing -> fill zeros)

    Robust against "broken" LAS where header.extra_dimensions lists a field
    that is not readable from the packed point record.
    """
    if las_in.header.version == "1.4" and las_in.header.point_format.id >= 6:
        return las_in

    hdr = laspy.LasHeader(version="1.4", point_format=6)
    hdr.scales = las_in.header.scales
    hdr.offsets = las_in.header.offsets

    out = laspy.LasData(hdr)
    out.points = laspy.ScaleAwarePointRecord.zeros(len(las_in.points), hdr.point_format)
    out.X, out.Y, out.Z = las_in.X, las_in.Y, las_in.Z

    in_dims = set(las_in.point_format.dimension_names)
    out_dims = set(out.point_format.dimension_names)

    # Copy overlapping standard dims (safe)
    for dim in (in_dims & out_dims):
        if dim in ("X", "Y", "Z"):
            continue
        try:
            out[dim] = las_in[dim]
        except Exception:
            pass

    # Copy extra dims safely
    for eb in las_in.header.extra_dimensions:
        if eb.name not in out.point_format.dimension_names:
            out.add_extra_dim(eb)

        try:
            out[eb.name] = las_in[eb.name]
        except Exception:
            dt = _dtype_from_eb_type(getattr(eb, "type", np.float32))
            out[eb.name] = np.zeros(len(las_in.points), dtype=dt)

    return out


def ensure_extra_dim(las_obj: laspy.LasData, name: str, dtype) -> None:
    if name not in las_obj.point_format.dimension_names:
        las_obj.add_extra_dim(laspy.ExtraBytesParams(name=name, type=dtype))


def build_pf6_output_from_points(
    base_pf6: laspy.LasData,
    points_subset,
    classification: np.ndarray,
    zn_values: np.ndarray,
    tree_ids: Optional[np.ndarray] = None,
) -> laspy.LasData:
    """
    Build an output LAS (PF6) from a PF6 base header + selected points.
    """
    las_out = laspy.LasData(base_pf6.header)
    las_out.points = points_subset

    n = len(las_out.points)
    cls_arr = np.asarray(classification, dtype=np.uint8)
    zn_arr = np.asarray(zn_values, dtype=np.float32)
    ids_arr = (
        np.asarray(tree_ids, dtype=np.uint16)
        if tree_ids is not None
        else np.zeros(n, dtype=np.uint16)
    )

    if len(cls_arr) != n or len(zn_arr) != n or len(ids_arr) != n:
        raise ValueError("build_pf6_output_from_points(): array length mismatch")

    las_out.classification = cls_arr

    ensure_extra_dim(las_out, "Z_norm", np.float32)
    ensure_extra_dim(las_out, "treeID", np.uint16)

    las_out["Z_norm"] = zn_arr
    las_out["treeID"] = ids_arr

    return las_out


def write_stage_snapshot(
    las_template: laspy.LasData,
    classification: np.ndarray,
    zn_values: np.ndarray,
    out_path: Path,
    label: str,
    tree_ids: Optional[np.ndarray] = None,
) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    base = ensure_las14_pf6(las_template)
    las_out = build_pf6_output_from_points(
        base_pf6=base,
        points_subset=base.points.copy(),
        classification=classification,
        zn_values=zn_values,
        tree_ids=tree_ids,
    )
    las_out.write(str(out_path))
    print(f"   📝 Saved {label}: {out_path.name}")


# =============================================================================
# Probabilistic helpers
# =============================================================================

def soft_height_prior(
    z_norm: float,
    cls: int,
    height_ranges: Dict[int, Tuple[float, float]],
) -> float:
    zmin, zmax = height_ranges.get(cls, DEFAULT_HEIGHT_RANGES.get(cls, (0.0, 10.0)))
    return 1.0 if (zmin <= z_norm <= zmax) else 0.1


def compute_class_height_thresholds(
    las_s: laspy.LasData,
    las_z: laspy.LasData,
    percentile_low: float = 15.0,
    percentile_high: float = 85.0,
    max_match_dist: float = 0.30,
) -> Dict[int, Tuple[float, float]]:
    if "Z_norm" not in las_z.point_format.dimension_names:
        raise ValueError("Z_norm LAS has no 'Z_norm' extra dimension.")

    zn_z = np.asarray(las_z["Z_norm"], dtype=float)

    xy_z = np.column_stack((np.asarray(las_z.x), np.asarray(las_z.y)))
    xy_s = np.column_stack((np.asarray(las_s.x), np.asarray(las_s.y)))

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

    s_cls = np.asarray(las_s.classification)
    for cls in [64, 65, 66, 67, 68]:
        mask = (s_cls == cls) & np.isfinite(s_zn)
        n = int(np.sum(mask))
        if n < 20:
            thresholds[cls] = DEFAULT_HEIGHT_RANGES.get(cls, (0.0, 10.0))
            print(
                f"  ⚠️ Class {cls} ({CLASS_NAMES.get(cls,'?')}): only {n} pts, "
                f"using default {thresholds[cls][0]:.2f}–{thresholds[cls][1]:.2f} m"
            )
            continue

        heights = s_zn[mask]
        p5, p95 = np.percentile(heights, [5, 95])
        heights_clip = heights[(heights >= p5) & (heights <= p95)]
        heights_use = heights_clip if len(heights_clip) >= 10 else heights

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


# =============================================================================
# Watershed + polar coords
# =============================================================================

def get_mallee_centroids_watershed(
    las_s: laspy.LasData,
    grid_res: float = 0.10,
    min_tree_dist: float = 0.75,
    smoothing_sigma: float = 0.5,
    fallback_points: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]] = None,
    fallback_min_height: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray, float, float]:
    x_s = np.asarray(las_s.x)
    y_s = np.asarray(las_s.y)
    h_s = np.asarray(las_s.z, dtype=float)
    cls_s = np.asarray(las_s.classification)

    mask_mallee = cls_s == 68
    source = "structure"

    if np.any(mask_mallee):
        use_x = x_s[mask_mallee]
        use_y = y_s[mask_mallee]
        use_h = h_s[mask_mallee]
    else:
        if fallback_points is None:
            min_x, min_y = float(x_s.min()), float(y_s.min())
            return np.zeros((1, 1), dtype=int), np.array([]), min_x, min_y

        fb_x, fb_y, fb_h = fallback_points
        fb_x = np.asarray(fb_x)
        fb_y = np.asarray(fb_y)
        fb_h = np.asarray(fb_h, dtype=float)

        max_fb_h = float(np.nanmax(fb_h)) if fb_h.size > 0 else 0.0
        if max_fb_h < fallback_min_height:
            min_x, min_y = float(fb_x.min()), float(fb_y.min())
            return np.zeros((1, 1), dtype=int), np.array([]), min_x, min_y

        thresholds = [
            fallback_min_height,
            max(0.8 * fallback_min_height, 0.6),
            max(0.6 * fallback_min_height, 0.5),
            max(0.4 * fallback_min_height, 0.4),
            0.3,
            0.1,
        ]
        chosen_mask = None
        chosen_th = thresholds[0]
        for th in thresholds:
            mask = fb_h >= th
            if np.count_nonzero(mask) >= 100:
                chosen_mask = mask
                chosen_th = th
                break
        if chosen_mask is None:
            min_x, min_y = float(fb_x.min()), float(fb_y.min())
            return np.zeros((1, 1), dtype=int), np.array([]), min_x, min_y

        use_x = fb_x[chosen_mask]
        use_y = fb_y[chosen_mask]
        use_h = fb_h[chosen_mask]
        source = f"Z_norm fallback (≥ {chosen_th:.2f}m, {len(use_x):,} pts)"

    min_x, max_x = float(use_x.min()), float(use_x.max())
    min_y, max_y = float(use_y.min()), float(use_y.max())

    w = max(1, int(np.ceil((max_x - min_x) / grid_res)) + 1)
    h = max(1, int(np.ceil((max_y - min_y) / grid_res)) + 1)

    mx = np.clip(((use_x - min_x) / grid_res).astype(int), 0, w - 1)
    my = np.clip(((use_y - min_y) / grid_res).astype(int), 0, h - 1)

    # CHM: keep max height per cell (sort ascending -> last write is max)
    chm_grid = np.zeros((h, w), dtype=np.float32)
    order = np.argsort(use_h)
    chm_grid[my[order], mx[order]] = use_h[order]

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
    next_marker_id = int(markers.max()) + 1

    # ensure every blob has at least one marker
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
    num = int(labels.max())

    if num == 0:
        print(f"  ⚠️ Watershed found 0 crowns ({source}).")
        return labels, np.array([]), min_x, min_y

    coms = center_of_mass(binary_mask, labels, range(1, num + 1))
    centroids = np.zeros((num + 1, 2), dtype=float)
    for i, (cy, cx) in enumerate(coms):
        centroids[i + 1] = [min_x + (cx * grid_res), min_y + (cy * grid_res)]

    print(f"🌳 [Watershed] Identified {num} crowns ({source}).")
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
    if centroids is None or len(centroids) < 2:
        return
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        fig = plt.figure(figsize=(10, 10))
        masked = np.ma.masked_where(labeled_grid == 0, labeled_grid)
        plt.imshow(masked, origin="lower", cmap="nipy_spectral", interpolation="nearest")

        real_cx = centroids[1:, 0]
        real_cy = centroids[1:, 1]
        grid_cx = (real_cx - min_x) / grid_res
        grid_cy = (real_cy - min_y) / grid_res

        plt.scatter(grid_cx, grid_cy, c="white", edgecolors="black", marker="X", s=50, linewidths=1)
        plt.title(f"Tile {tile_id}: {len(centroids) - 1} crowns")
        plt.savefig(out_dir / f"debug_mallee_{tile_id}.png", dpi=110)
        plt.close(fig)
    except Exception as e:
        print(f"❌ Debug plot failed: {e}")


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

    obj_ids = labeled_grid[iy, ix].astype(int)

    r = np.full(x_arr.shape, np.nan, dtype=float)
    theta = np.full(x_arr.shape, np.nan, dtype=float)

    valid = obj_ids > 0
    if np.any(valid):
        safe = valid & (obj_ids < len(centroids))
        if np.any(safe):
            cxy = centroids[obj_ids[safe]]
            dx = x_arr[safe] - cxy[:, 0]
            dy = y_arr[safe] - cxy[:, 1]
            r[safe] = np.sqrt(dx * dx + dy * dy)
            theta[safe] = np.mod(np.arctan2(dy, dx), 2 * np.pi)

    return r, theta, obj_ids


def polar_bins(
    r: np.ndarray,
    theta: np.ndarray,
    r_bin: float,
    theta_bin_deg: float,
) -> Tuple[np.ndarray, np.ndarray]:
    ir = np.full(r.shape, -1, dtype=np.int32)
    it = np.full(r.shape, -1, dtype=np.int32)
    valid = np.isfinite(r) & np.isfinite(theta)
    if np.any(valid):
        ir[valid] = np.floor(r[valid] / r_bin).astype(np.int32)
        it[valid] = np.floor(theta[valid] / np.deg2rad(theta_bin_deg)).astype(np.int32)
    return ir, it


# =============================================================================
# Step 1: Cartesian voxel merge
# =============================================================================

def cartesian_merge(
    las_z: laspy.LasData,
    las_s: laspy.LasData,
    voxel_size_xy: float,
    voxel_size_z: float,
) -> np.ndarray:
    x_min = float(min(las_z.x.min(), las_s.x.min()))
    y_min = float(min(las_z.y.min(), las_s.y.min()))
    z_min = float(min(las_z.z.min(), las_s.z.min()))
    origin = np.array([x_min, y_min, z_min], dtype=np.float64)

    def vox(x, y, z):
        return np.stack(
            [
                np.floor((x - origin[0]) / voxel_size_xy).astype(np.int32),
                np.floor((y - origin[1]) / voxel_size_xy).astype(np.int32),
                np.floor((z - origin[2]) / voxel_size_z).astype(np.int32),
            ],
            axis=-1,
        )

    vox_s = vox(np.asarray(las_s.x), np.asarray(las_s.y), np.asarray(las_s.z))
    s_cls = np.asarray(las_s.classification)
    raw_struct = np.where((s_cls >= 64) & (s_cls <= 68), s_cls, 0).astype(np.uint16)

    uniq_vox_s, inv = np.unique(vox_s, axis=0, return_inverse=True)
    voxel_class = np.zeros(len(uniq_vox_s), dtype=np.uint16)

    for k in range(len(uniq_vox_s)):
        vals = raw_struct[inv == k]
        vals = vals[vals != 0]
        if vals.size:
            voxel_class[k] = Counter(vals).most_common(1)[0][0]

    voxel_map = {tuple(v): c for v, c in zip(uniq_vox_s, voxel_class) if c != 0}
    print(f"✅ Voxels with non-zero structure class: {len(voxel_map):,}")

    vox_z = vox(np.asarray(las_z.x), np.asarray(las_z.y), np.asarray(las_z.z))
    out_cls = np.zeros(len(las_z.points), dtype=np.uint8)
    for i, v in enumerate(vox_z):
        out_cls[i] = voxel_map.get(tuple(v), 0)

    return out_cls


# =============================================================================
# Step 2: Crown-centric polar refinement (treeID)
# =============================================================================

def refine_mallee_with_local_polar(
    las_z: laspy.LasData,
    las_s: laspy.LasData,
    classification: np.ndarray,
    height_ranges: Dict[int, Tuple[float, float]],
    tile_id: str,
    debug_dir: Path,
    cfg: FusionConfig,
    labeled_grid: Optional[np.ndarray] = None,
    centroids: Optional[np.ndarray] = None,
    min_x: Optional[float] = None,
    min_y: Optional[float] = None,
    fallback_points: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, float]:
    cls_out = classification.copy()
    instance_ids = np.zeros(len(cls_out), dtype=np.uint16)

    if labeled_grid is None or centroids is None or min_x is None or min_y is None:
        labeled_grid, centroids, min_x, min_y = get_mallee_centroids_watershed(
            las_s,
            grid_res=cfg.grid_res,
            min_tree_dist=1.5,
            smoothing_sigma=1.0,
            fallback_points=fallback_points,
        )

    if centroids is None or len(centroids) == 0:
        return cls_out, instance_ids, labeled_grid, np.array([]), float(min_x or 0.0), float(min_y or 0.0)

    plot_mallee_debug(labeled_grid, centroids, float(min_x), float(min_y), cfg.grid_res, tile_id, debug_dir)

    # Build class probabilities for (obj_id, rbin, tbin) from structure points
    s_cls = np.asarray(las_s.classification)
    mask_interest = np.isin(s_cls, [65, 66, 68])
    prob_lookup: Dict[Tuple[int, int, int], pd.Series] = {}

    if np.any(mask_interest):
        s_r, s_th, s_obj = calculate_local_polar(
            np.asarray(las_s.x)[mask_interest],
            np.asarray(las_s.y)[mask_interest],
            labeled_grid,
            centroids,
            float(min_x),
            float(min_y),
            cfg.grid_res,
        )

        valid_s = s_obj > 0
        if np.any(valid_s):
            s_ir, s_it = polar_bins(s_r[valid_s], s_th[valid_s], cfg.r_bin, cfg.theta_bin_deg)
            df_s = pd.DataFrame(
                {
                    "obj": s_obj[valid_s].astype(int),
                    "ir": s_ir.astype(int),
                    "it": s_it.astype(int),
                    "cls": s_cls[mask_interest][valid_s].astype(int),
                }
            )
            counts = df_s.groupby(["obj", "ir", "it", "cls"]).size().unstack(fill_value=0)
            probs = counts.div(counts.sum(axis=1), axis=0)
            prob_lookup = {idx: probs.loc[idx] for idx in probs.index}

    # Apply to Z_norm points
    z_r, z_th, z_obj = calculate_local_polar(
        np.asarray(las_z.x),
        np.asarray(las_z.y),
        labeled_grid,
        centroids,
        float(min_x),
        float(min_y),
        cfg.grid_res,
    )
    instance_ids = z_obj.astype(np.uint16)

    z_ir, z_it = polar_bins(z_r, z_th, cfg.r_bin, cfg.theta_bin_deg)
    zn = np.asarray(las_z["Z_norm"], dtype=float)

    candidates = np.where((cls_out == 65) | (cls_out == 66) | (cls_out == 68))[0]

    for idx in candidates:
        if zn[idx] <= cfg.ground_thresh:
            continue
        obj_id = int(z_obj[idx])
        if obj_id == 0:
            continue

        base = prob_lookup.get((obj_id, int(z_ir[idx]), int(z_it[idx])), None)
        if base is None:
            continue

        classes = base.index.to_numpy(dtype=int)
        probs = base.to_numpy(dtype=float)

        post = np.array(
            [p * soft_height_prior(zn[idx], int(c), height_ranges) for p, c in zip(probs, classes)],
            dtype=float,
        )
        if post.sum() <= 0:
            continue

        best = int(classes[int(np.argmax(post))])
        if best != int(cls_out[idx]):
            cur_val = float(post[classes == int(cls_out[idx])][0]) if int(cls_out[idx]) in classes else 0.0
            if (post.max() / (cur_val + 1e-9)) >= cfg.min_flip_ratio:
                cls_out[idx] = best

    return cls_out, instance_ids, labeled_grid, centroids, float(min_x), float(min_y)


# =============================================================================
# Step 3: Tile-wide radial fill (optional)
# =============================================================================

def tile_center_xy(las_z: laspy.LasData) -> np.ndarray:
    return np.array(
        [
            0.5 * (float(las_z.x.min()) + float(las_z.x.max())),
            0.5 * (float(las_z.y.min()) + float(las_z.y.max())),
        ],
        dtype=float,
    )


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
    height_ranges: Dict[int, Tuple[float, float]],
    cfg: FusionConfig,
) -> np.ndarray:
    cls_out = classification.copy()
    origin = tile_center_xy(las_z)

    s_cls = np.asarray(las_s.classification)
    mask_ext = (s_cls >= 64) & (s_cls <= 68)

    s_r, s_th = radial_coords_tile(np.asarray(las_s.x), np.asarray(las_s.y), origin)
    s_ir, s_it = polar_bins(s_r, s_th, cfg.r_bin, cfg.theta_bin_deg)

    df = pd.DataFrame(
        {
            "ir": s_ir[mask_ext].astype(int),
            "it": s_it[mask_ext].astype(int),
            "cls": s_cls[mask_ext].astype(int),
        }
    )

    probs = df.groupby(["ir", "it", "cls"]).size().unstack(fill_value=0)
    probs = probs.div(probs.sum(axis=1), axis=0)

    if cfg.min_conf > 0 and len(probs) > 0:
        probs = probs[probs.max(axis=1) >= cfg.min_conf]

    col_map = {idx: probs.loc[idx] for idx in probs.index}

    zn = np.asarray(las_z["Z_norm"], dtype=float)

    z_r, z_th = radial_coords_tile(np.asarray(las_z.x), np.asarray(las_z.y), origin)
    z_ir, z_it = polar_bins(z_r, z_th, cfg.r_bin, cfg.theta_bin_deg)

    zero_idx = np.where((cls_out == 0) & (zn > cfg.ground_thresh) & (zn <= cfg.max_fill_height))[0]

    for idx in zero_idx:
        base = col_map.get((int(z_ir[idx]), int(z_it[idx])), None)
        if base is None:
            continue

        classes = base.index.to_numpy(dtype=int)
        probs_arr = base.to_numpy(dtype=float)

        post = np.array(
            [p * soft_height_prior(zn[idx], int(c), height_ranges) for p, c in zip(probs_arr, classes)],
            dtype=float,
        )
        if post.sum() > 0:
            cls_out[idx] = int(classes[int(np.argmax(post))])

    return cls_out


# =============================================================================
# Step 4: Final sanity rules + outputs
# =============================================================================

def apply_final_height_rules(
    classification: np.ndarray,
    zn_arr: np.ndarray,
    height_ranges: Dict[int, Tuple[float, float]],
) -> Tuple[np.ndarray, float, float]:
    cls = classification.copy()

    hummock_low, hummock_high = height_ranges.get(66, (0.0, 0.8))
    shrub_low, shrub_high = height_ranges.get(67, (0.4, 3.0))
    mallee_low, mallee_high = height_ranges.get(68, (1.5, 8.0))

    hummock_to_shrub = min(hummock_high * 0.9, 0.7)
    shrub_to_mallee = max(shrub_high * 0.8, mallee_low * 0.85)
    mallee_min = max(0.3, mallee_low * 0.4)  # Lowered from 0.6/0.7 to include more mallee points with treeID
    mallee_max = min(6.0, mallee_high * 1.2)

    print(
        f"  → Hummock→Shrub: {hummock_to_shrub:.2f}m | "
        f"Shrub→Mallee: {shrub_to_mallee:.2f}m"
    )
    print(f"  → Mallee range: {mallee_min:.2f}m - {mallee_max:.2f}m")

    cls[(cls == 68) & (zn_arr < mallee_min)] = 67
    cls[(cls == 66) & (zn_arr >= hummock_to_shrub)] = 67
    cls[(cls == 67) & (zn_arr >= shrub_to_mallee)] = 68
    cls[(cls == 68) & (zn_arr >= mallee_max)] = 65

    MERGE_65_TO_68_ZNORM = 0.5
    mask_65 = cls == 65
    cls[mask_65 & (zn_arr > MERGE_65_TO_68_ZNORM)] = 68

    return cls, mallee_min, mallee_max


def print_class_stats(classification: np.ndarray) -> None:
    u, c = np.unique(classification, return_counts=True)
    print("  📋 Class distribution:")
    total = float(len(classification)) if len(classification) else 1.0
    for cls, cnt in zip(u, c):
        pct = 100.0 * float(cnt) / total
        print(f"     Class {int(cls)} ({CLASS_NAMES.get(int(cls), 'Unknown')}): {int(cnt):,} pts ({pct:.1f}%)")


# =============================================================================
# Per-tile runner
# =============================================================================

def merge_znorm_structure_one_file(
    z_norm_las: Path,
    structure_las: Path,
    output_las: Path,
    output_las_no0: Optional[Path],
    cfg: FusionConfig,
) -> None:
    z_norm_las = Path(z_norm_las)
    structure_las = Path(structure_las)
    output_las = Path(output_las)
    if output_las_no0 is not None:
        output_las_no0 = Path(output_las_no0)

    tile_id = z_norm_las.stem.split(".")[-1]

    las_z = laspy.read(str(z_norm_las))
    las_s = laspy.read(str(structure_las))

    print(f"🔎 Tile {tile_id} | Z: {len(las_z.points):,}, S: {len(las_s.points):,}")

    if "Z_norm" not in las_z.point_format.dimension_names:
        raise ValueError("Z_norm LAS has no 'Z_norm' extra dimension.")
    zn_all = np.asarray(las_z["Z_norm"], dtype=float)

    stage_prefix = f"z_norm_struct_{tile_id}"

    out_full_dir = output_las.parent
    cart_dir = Path(cfg.cartesian_out_dir) if cfg.cartesian_out_dir else out_full_dir
    polar_dir = Path(cfg.polar_out_dir) if cfg.polar_out_dir else out_full_dir
    radial_dir = Path(cfg.radial_out_dir) if cfg.radial_out_dir else out_full_dir

    cart_dir.mkdir(parents=True, exist_ok=True)
    polar_dir.mkdir(parents=True, exist_ok=True)
    if cfg.enable_radial_fill:
        radial_dir.mkdir(parents=True, exist_ok=True)

    # --- Step 1: Cartesian merge
    cls_stage1 = cartesian_merge(las_z, las_s, cfg.voxel_size_xy, cfg.voxel_size_z)
    write_stage_snapshot(
        las_z,
        cls_stage1,
        zn_all,
        cart_dir / f"{stage_prefix}_stage1_cartesian.las",
        "Stage 1 (Cartesian majority)",
        tree_ids=None,
    )

    # --- Height thresholds
    print(f"📊 Computing data-driven height thresholds for tile {tile_id}:")
    height_ranges = compute_class_height_thresholds(las_s, las_z)

    # --- Watershed once, reuse
    fallback_points = (np.asarray(las_z.x), np.asarray(las_z.y), zn_all)
    labeled_grid, centroids, min_x, min_y = get_mallee_centroids_watershed(
        las_s,
        grid_res=cfg.grid_res,
        min_tree_dist=1.5,
        smoothing_sigma=1.0,
        fallback_points=fallback_points,
    )

    # --- Step 2: Polar refinement + treeID
    cls_stage2, tree_ids, _, _, _, _ = refine_mallee_with_local_polar(
        las_z,
        las_s,
        cls_stage1,
        height_ranges,
        tile_id=tile_id,
        debug_dir=out_full_dir,
        cfg=cfg,
        labeled_grid=labeled_grid,
        centroids=centroids,
        min_x=min_x,
        min_y=min_y,
        fallback_points=fallback_points,
    )
    write_stage_snapshot(
        las_z,
        cls_stage2,
        zn_all,
        polar_dir / f"{stage_prefix}_stage2_polar.las",
        "Stage 2 (Polar refinement)",
        tree_ids=tree_ids,
    )

    cls_work = cls_stage2.copy()

    # --- Step 3: Radial fill (optional)
    if cfg.enable_radial_fill:
        cls_work = fill_missing_classes_radial(las_z, las_s, cls_work, height_ranges, cfg)
        write_stage_snapshot(
            las_z,
            cls_work,
            zn_all,
            radial_dir / f"{stage_prefix}_stage3_radial.las",
            "Stage 3 (Radial fill)",
            tree_ids=tree_ids,
        )
    else:
        print("  🔧 Radial gap fill disabled (enable_radial_fill=False)")

    # --- Step 4: Final sanity rules
    cls_final, mallee_min, _mallee_max = apply_final_height_rules(cls_work, zn_all, height_ranges)

    # Drop low woody debris noise (only low 65; high 65 already merged)
    mask_65_low = (cls_final == 65) & (zn_all < 0.5)
    keep_mask_base = ~mask_65_low

    cls_kept = cls_final[keep_mask_base].copy()
    zn_kept = zn_all[keep_mask_base].copy()

    ids_kept = tree_ids[keep_mask_base].copy()
    is_mallee_crown = (cls_kept == 68) & (zn_kept >= mallee_min)
    ids_kept[~is_mallee_crown] = 0

    # merge any remaining 65 into 68 in final training outputs
    cls_kept[cls_kept == 65] = 68

    print_class_stats(cls_kept)
    unique_trees = np.unique(ids_kept[ids_kept > 0])
    print(f"  🌳 TreeIDs (mallee crowns only): {len(unique_trees)} unique trees")

    # --- Write full output (PF6, robust)
    output_las.parent.mkdir(parents=True, exist_ok=True)
    base_pf6 = ensure_las14_pf6(las_z)
    subset_points = base_pf6.points[keep_mask_base].copy()

    las_out = build_pf6_output_from_points(
        base_pf6=base_pf6,
        points_subset=subset_points,
        classification=cls_kept,
        zn_values=zn_kept,
        tree_ids=ids_kept,
    )
    las_out.write(str(output_las))
    print(f"✅ Wrote: {output_las.name} (treeID for mallee crowns only)")

    # --- Write no0 output (optional)
    if output_las_no0 is not None:
        HIGH_Z_KEEP = 1.3
        keep2 = (cls_kept != 0) | ((cls_kept == 0) & (zn_kept >= HIGH_Z_KEEP))

        final_cls = cls_kept[keep2].copy()
        final_zn = zn_kept[keep2].copy()
        final_ids = ids_kept[keep2].copy()

        # promote tall class-0 to mallee
        promote = (final_cls == 0) & (final_zn >= HIGH_Z_KEEP)
        final_cls[promote] = 68

        las_no0 = build_pf6_output_from_points(
            base_pf6=ensure_las14_pf6(las_out),
            points_subset=ensure_las14_pf6(las_out).points[keep2].copy(),
            classification=final_cls,
            zn_values=final_zn,
            tree_ids=final_ids,
        )

        output_las_no0.parent.mkdir(parents=True, exist_ok=True)
        las_no0.write(str(output_las_no0))
        print(f"   ✔ Filtered LAS written: {output_las_no0.name}")


# =============================================================================
# Folder runner + main
# =============================================================================

def merge_folder(
    z_dir: str,
    s_dir: str,
    o_full: str,
    o_no0: str,
    config: Optional[Mapping] = None,
) -> None:
    cfg = FusionConfig()
    if config:
        for k, v in dict(config).items():
            if hasattr(cfg, k):
                setattr(cfg, k, v)

    print("⚙️ Config:", asdict(cfg))

    z_norm_dir = Path(z_dir)
    structure_dir = Path(s_dir)
    out_dir_full = Path(o_full)
    out_dir_no0 = Path(o_no0)

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
            z_norm_las=zf,
            structure_las=struct_file,
            output_las=out_dir_full / f"z_norm_struct_{tile_id}_obj_polar.las",
            output_las_no0=out_dir_no0 / f"z_norm_struct_{tile_id}_obj_polar_no0.las",
            cfg=cfg,
        )


if __name__ == "__main__":

    # site 1
    z_dir = "/data/calperumResearch/site1_1_DD0001/lidar/clipped/raster/z_norm/csf/"
    s_dir = "/data/calperumResearch/site1_1_DD0001/lidar/clipped/mask/coreg/icp/"
    o_full = "/data/calperumResearch/site1_1_DD0001/lidar/clipped/3d_structure_malleid"
    o_no0 = "/data/calperumResearch/site1_1_DD0001/lidar/clipped/3d_structure_mask_malleid"

    # site 11 (toggle if needed)
    # z_dir = "/data/calperumResearch/site2_1_DD0011/lidar/clipped/raster/z_norm/csf/"
    # s_dir = "/data/calperumResearch/site2_1_DD0011/lidar/clipped/mask/coreg/icp/"
    # o_full = "/data/calperumResearch/site2_1_DD0011/lidar/clipped/3d_structure_malleid"
    # o_no0 = "/data/calperumResearch/site2_1_DD0011/lidar/clipped/3d_structure_mask_malleid"

    config = {
        # "enable_radial_fill": True,
        # "cartesian_out_dir": "/tmp/stage1",
        # "polar_out_dir": "/tmp/stage2",
        # "radial_out_dir": "/tmp/stage3",
        # "grid_res": 0.10,
    }

    merge_folder(z_dir, s_dir, o_full, o_no0, config=config)

#%%
