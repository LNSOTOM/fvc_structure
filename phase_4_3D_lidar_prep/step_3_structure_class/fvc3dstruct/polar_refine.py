"""Crown-centric polar coordinate refinement with height priors."""

import numpy as np
import pandas as pd
from collections import Counter
from typing import Dict, Tuple


def calculate_polar_coords(
    x_arr: np.ndarray,
    y_arr: np.ndarray,
    labeled_grid: np.ndarray,
    centroids: np.ndarray,
    grid_min_x: float,
    grid_min_y: float,
    grid_res: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    For each point, compute crown-centric polar coordinates (r, θ).
    
    Returns:
        r: Radial distance from crown center
        theta: Angle in radians [0, 2π)
        crown_ids: Crown label for each point (0 = no crown)
    """
    ix = np.clip(
        ((x_arr - grid_min_x) / grid_res).astype(int),
        0, labeled_grid.shape[1] - 1
    )
    iy = np.clip(
        ((y_arr - grid_min_y) / grid_res).astype(int),
        0, labeled_grid.shape[0] - 1
    )
    
    crown_ids = labeled_grid[iy, ix]
    
    r = np.full_like(x_arr, np.nan, dtype=float)
    theta = np.full_like(x_arr, np.nan, dtype=float)
    
    valid_mask = (crown_ids > 0) & (crown_ids < len(centroids))
    
    if np.any(valid_mask):
        cid = crown_ids[valid_mask]
        cx, cy = centroids[cid, 0], centroids[cid, 1]
        dx = x_arr[valid_mask] - cx
        dy = y_arr[valid_mask] - cy
        
        r[valid_mask] = np.sqrt(dx**2 + dy**2)
        theta[valid_mask] = np.mod(np.arctan2(dy, dx), 2 * np.pi)
    
    return r, theta, crown_ids


def polar_bin_indices(
    r: np.ndarray,
    theta: np.ndarray,
    r_bin: float,
    theta_bin_deg: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Discretize polar coordinates into bin indices."""
    ir = np.full(r.shape, -1, dtype=np.int32)
    it = np.full(r.shape, -1, dtype=np.int32)
    
    valid = np.isfinite(r) & np.isfinite(theta)
    if np.any(valid):
        ir[valid] = np.floor(r[valid] / r_bin).astype(np.int32)
        it[valid] = np.floor(theta[valid] / np.deg2rad(theta_bin_deg)).astype(np.int32)
    
    return ir, it


def soft_height_prior(
    z_norm: float,
    cls: int,
    height_ranges: Dict[int, Tuple[float, float]],
) -> float:
    """
    Soft height prior: downweight classes outside their typical range.
    Returns 1.0 if in range, 0.1 if outside.
    """
    if cls not in height_ranges or cls == 0:
        return 0.1
    
    zmin, zmax = height_ranges[cls]
    return 1.0 if (zmin <= z_norm <= zmax) else 0.1


def refine_with_polar_coords(
    las_z,
    las_s,
    classification: np.ndarray,
    zn_arr: np.ndarray,
    label_grid: np.ndarray,
    centroids: np.ndarray,
    grid_min_x: float,
    grid_min_y: float,
    grid_res: float,
    height_ranges: Dict[int, Tuple[float, float]],
    r_bin: float = 0.10,
    theta_bin_deg: float = 10.0,
    min_conf: float = 0.3,
    min_flip_ratio: float = 1.5,
) -> np.ndarray:
    """
    Refine classification using crown-centric polar bins + height priors.
    
    For each (crown, r_bin, theta_bin):
    - Estimate class probabilities from structure LAS points in that bin
    - Apply height priors based on data-driven ranges
    - Update Z_norm points if confidence is sufficient
    
    Args:
        las_z: Z_norm LAS (target points)
        las_s: Structure LAS (source labels)
        classification: Current classification array
        zn_arr: Z_norm heights
        label_grid: Crown label grid
        centroids: Crown centroids
        grid_min_x, grid_min_y: Grid origin
        grid_res: Grid resolution
        height_ranges: Data-driven height ranges per class
        r_bin: Radial bin size (m)
        theta_bin_deg: Angular bin size (degrees)
        min_conf: Minimum confidence to update label
        min_flip_ratio: Minimum ratio new_conf/old_conf to flip label
    
    Returns:
        Updated classification array
    """
    # Compute polar coordinates for both clouds
    r_z, theta_z, crown_z = calculate_polar_coords(
        las_z.x, las_z.y, label_grid, centroids,
        grid_min_x, grid_min_y, grid_res
    )
    
    r_s, theta_s, crown_s = calculate_polar_coords(
        las_s.x, las_s.y, label_grid, centroids,
        grid_min_x, grid_min_y, grid_res
    )
    
    # Bin polar coordinates
    ir_z, it_z = polar_bin_indices(r_z, theta_z, r_bin, theta_bin_deg)
    ir_s, it_s = polar_bin_indices(r_s, theta_s, r_bin, theta_bin_deg)
    
    # Build polar bin statistics from structure LAS
    bin_stats = {}
    
    struct_cls = las_s.classification
    valid_struct = (crown_s > 0) & (ir_s >= 0) & (it_s >= 0)
    valid_struct &= (struct_cls >= 64) & (struct_cls <= 68)
    
    for i in np.where(valid_struct)[0]:
        key = (int(crown_s[i]), int(ir_s[i]), int(it_s[i]))
        cls = int(struct_cls[i])
        
        if key not in bin_stats:
            bin_stats[key] = Counter()
        bin_stats[key][cls] += 1
    
    print(f"  📊 Built statistics for {len(bin_stats):,} polar bins")
    
    # Refine Z_norm classifications
    classification_out = classification.copy()
    n_updated = 0
    
    valid_z = (crown_z > 0) & (ir_z >= 0) & (it_z >= 0)
    
    for i in np.where(valid_z)[0]:
        key = (int(crown_z[i]), int(ir_z[i]), int(it_z[i]))
        
        if key not in bin_stats:
            continue
        
        counts = bin_stats[key]
        total = sum(counts.values())
        
        if total < 3:  # Require minimum samples
            continue
        
        z_val = zn_arr[i]
        current_cls = classification[i]
        
        # Compute weighted probabilities with height priors
        probs = {}
        for cls, count in counts.items():
            raw_prob = count / total
            height_weight = soft_height_prior(z_val, cls, height_ranges)
            probs[cls] = raw_prob * height_weight
        
        # Normalize
        prob_sum = sum(probs.values())
        if prob_sum > 0:
            probs = {c: p / prob_sum for c, p in probs.items()}
        else:
            continue
        
        # Find best candidate
        if not probs:
            continue
        
        best_cls = max(probs, key=probs.get)
        best_conf = probs[best_cls]
        
        # Update logic
        if current_cls == 0 or current_cls not in [64, 65, 66, 67, 68]:
            # Fill empty/invalid labels if confident enough
            if best_conf >= min_conf:
                classification_out[i] = best_cls
                n_updated += 1
        else:
            # Flip existing label only if new label is significantly better
            current_conf = probs.get(current_cls, 0.0)
            if best_cls != current_cls and best_conf > min_flip_ratio * current_conf:
                classification_out[i] = best_cls
                n_updated += 1
    
    pct_updated = 100.0 * n_updated / len(classification) if len(classification) > 0 else 0.0
    print(f"  ✅ Updated {n_updated:,}/{len(classification):,} labels ({pct_updated:.1f}%)")
    
    return classification_out
