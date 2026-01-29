"""Watershed-based crown detection with robust fallback logic."""

import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree
from scipy.ndimage import center_of_mass
from skimage.feature import peak_local_max
from skimage.segmentation import watershed
from skimage.measure import label as skimage_label, regionprops
from typing import Tuple, Optional
from pathlib import Path
def auto_watershed_params(
    las_s,
    las_z,
    grid_res: float = 0.10,
    fallback_min_height: float = 1.0,
) -> tuple:
    """Automatically derive (min_tree_dist, smoothing_sigma) for watershed.

    Heuristic:
      - Use XY nearest-neighbour distances among canopy points to estimate
        typical crown spacing, then set `min_tree_dist` to a robust percentile
        of those distances.
      - Set Gaussian `smoothing_sigma` in pixels as a fraction of that spacing.

    Returns:
      (min_tree_dist_meters, smoothing_sigma_pixels)
    """
    # Prefer class-68 points from structure LAS; else fallback to Z_norm geometry
    x_s = las_s.x
    y_s = las_s.y
    cls_s = las_s.classification
    mask_mallee = cls_s == 68

    if np.any(mask_mallee):
        pts = np.column_stack((x_s[mask_mallee], y_s[mask_mallee]))
    else:
        # Fallback: use Z_norm points above a minimum height
        xz = las_z.x
        yz = las_z.y
        zn = np.asarray(las_z["Z_norm"], dtype=float)
        mask = zn >= fallback_min_height
        if np.count_nonzero(mask) < 100:
            # Insufficient signal; return safe defaults
            return 1.5, 1.0
        pts = np.column_stack((xz[mask], yz[mask]))

    if len(pts) < 100:
        return 1.5, 1.0

    # Subsample for efficiency
    max_samples = 50000
    if len(pts) > max_samples:
        idx = np.random.choice(len(pts), size=max_samples, replace=False)
        pts = pts[idx]

    try:
        tree = cKDTree(pts)
        # Query two neighbors (self + nearest other); take the second distance
        dists, _ = tree.query(pts, k=2, workers=-1)
        nn = dists[:, 1]
        # Robust estimate: 60th percentile of NN distances
        est = float(np.percentile(nn, 60))
        # Clamp to reasonable bounds for mallee spacing
        min_tree_dist = float(np.clip(est, 0.8, 3.0))
        # Convert to smoothing sigma (in pixels), gentle smoothing ~ spacing/6
        sigma_px = float((min_tree_dist / grid_res) / 6.0)
        sigma_px = float(np.clip(sigma_px, 0.8, 2.5))
        return min_tree_dist, sigma_px
    except Exception:
        return 1.5, 1.0



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
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        
        fig = plt.figure(figsize=(10, 10))
        masked_grid = np.ma.masked_where(labeled_grid == 0, labeled_grid)
        plt.imshow(
            masked_grid, origin="lower", cmap="nipy_spectral", interpolation="nearest"
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


def get_mallee_centroids_watershed(
    las_s,
    grid_res: float = 0.10,
    min_tree_dist: float = 0.75,
    smoothing_sigma: float = 0.5,
    fallback_points: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]] = None,
    fallback_min_height: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray, float, float]:
    """
    Detect mallee crowns from CHM using hybrid blobs + watershed approach.
    
    If structure LAS has no class-68 points, falls back to using
    Z_norm points above a minimum height threshold with progressive
    threshold relaxation.
    
    Returns:
        labels: Grid of crown IDs (0 = background)
        centroids: Array of (x, y) crown centers, indexed by crown ID
        min_x: Grid origin X
        min_y: Grid origin Y
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
            print("  ⚠️ No mallee crowns in structure LAS and no fallback provided; skipping watershed.")
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
            print(f"  ⚠️ Fallback heights max out at {max_fb_h:.2f}m (< {fallback_min_height:.2f}m); skipping watershed.")
            return np.zeros((h, w), dtype=int), np.array([]), min_x, min_y
        
        # Progressively relax threshold to ensure enough points
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
    
    # Build CHM
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
    
    # Create markers from peaks
    local_maxi = np.zeros_like(chm_smoothed, dtype=bool)
    if len(coordinates) > 0:
        local_maxi[coordinates[:, 0], coordinates[:, 1]] = True
    
    markers = ndimage.label(local_maxi)[0]
    
    # Blob-based marker enhancement (ensure each blob has at least one marker)
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
    
    # Watershed
    labels = watershed(-chm_smoothed, markers, mask=binary_mask)
    num_features = labels.max()
    
    if num_features == 0:
        print(f"  ⚠️ Watershed failed to find crowns using {source} data.")
        return labels, np.array([]), min_x, min_y
    
    # Compute centroids using center of mass
    coms = center_of_mass(binary_mask, labels, range(1, num_features + 1))
    
    centroids = np.zeros((num_features + 1, 2))
    for i, (cy, cx) in enumerate(coms):
        real_x = min_x + (cx * grid_res)
        real_y = min_y + (cy * grid_res)
        centroids[i + 1] = [real_x, real_y]
    
    print(f"🌳 [Watershed] Identified {num_features} crowns ({source}).")
    return labels, centroids, min_x, min_y
