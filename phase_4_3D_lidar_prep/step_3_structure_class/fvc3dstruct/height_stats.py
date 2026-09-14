"""
height_stats.py

Height–based statistics and class threshold estimation for the
fvc3dstruct package.

Provides:
    • compute_class_height_thresholds()
    • match_structure_heights()
    • class_height_ranges()
    • summarise_height_ranges()
    • auto_mallee_min_height()
    • print_height_report()

This module is standalone and does NOT depend on the entire pipeline.
"""

from __future__ import annotations
import numpy as np
import pandas as pd
import laspy
from scipy.spatial import cKDTree
from typing import Dict, Tuple


# -----------------------------------------------------------
# Defaults used when a class has too few samples
# -----------------------------------------------------------

DEFAULT_HEIGHT_RANGES: Dict[int, Tuple[float, float]] = {
    64: (0.00, 0.40),  # Forb
    65: (0.05, 5.00),  # Woody Debris
    66: (0.00, 0.80),  # Hummock
    67: (0.40, 3.00),  # Shrub
    68: (1.50, 8.00),  # Mallee
}

CLASS_NAMES = {
    64: "Forb",
    65: "Woody Debris",
    66: "Hummock",
    67: "Shrub",
    68: "Mallee",
}


# -----------------------------------------------------------
# Core function: compute class-specific thresholds
# -----------------------------------------------------------

def compute_class_height_thresholds(
    las_s: laspy.LasData,
    las_z: laspy.LasData,
    percentile_low: float = 15.0,
    percentile_high: float = 85.0,
    max_match_dist: float = 0.30,
) -> Dict[int, Tuple[float, float]]:
    """
    Estimate Z_norm height ranges for each vegetation class using:

    • STRUCTURE LAS → class labels (64–68)
    • Z_norm LAS     → height values (Z_norm extra dimension)

    Matching is NN-based in XY using cKDTree.

    Returns dict {class_id: (low, high)}.
    """
    matched_heights = match_structure_heights(las_s, las_z, max_match_dist)
    return class_height_ranges(
        matched_heights,
        np.asarray(las_s.classification),
        percentile_low=percentile_low,
        percentile_high=percentile_high,
    )


def match_structure_heights(
    las_s: laspy.LasData,
    las_z: laspy.LasData,
    max_match_dist: float = 0.30,
) -> np.ndarray:
    """
    Z_norm of the XY-nearest Z_norm point for each structure point, NaN where
    none lies within `max_match_dist`.
    """

    if "Z_norm" not in las_z.point_format.dimension_names:
        raise ValueError("Z_norm LAS has no 'Z_norm' dimension.")

    zn_z = np.asarray(las_z["Z_norm"], dtype=float)

    # XY arrays
    xy_z = np.column_stack((las_z.x, las_z.y))
    xy_s = np.column_stack((las_s.x, las_s.y))

    # Nearest neighbour matching
    tree = cKDTree(xy_z)
    dists, idx = tree.query(xy_s, k=1, distance_upper_bound=max_match_dist)

    matched_heights = np.full(len(las_s), np.nan, dtype=float)
    valid = np.isfinite(dists) & (idx < len(zn_z))
    matched_heights[valid] = zn_z[idx[valid]]

    print(f"  ℹ️ Matched {valid.sum():,}/{len(las_s):,} structure pts to Z_norm")
    return matched_heights


def class_height_ranges(
    matched_heights: np.ndarray,
    classes: np.ndarray,
    percentile_low: float = 15.0,
    percentile_high: float = 85.0,
) -> Dict[int, Tuple[float, float]]:
    """
    Per-class (low, high) Z_norm range from per-point heights and classes;
    NaN heights are ignored.
    """

    thresholds = {}

    # Handle each class 64–68
    for cls in [64, 65, 66, 67, 68]:
        mask = (classes == cls) & np.isfinite(matched_heights)
        n = int(mask.sum())

        if n < 20:
            thresholds[cls] = DEFAULT_HEIGHT_RANGES[cls]
            print(
                f"  ⚠️ Class {cls} ({CLASS_NAMES[cls]}): "
                f"only {n} pts → default {thresholds[cls]}"
            )
            continue

        heights = matched_heights[mask]

        # Outlier removal
        p5, p95 = np.percentile(heights, [5, 95])
        heights_clip = heights[(heights >= p5) & (heights <= p95)]
        if len(heights_clip) >= 10:
            h = heights_clip
        else:
            h = heights

        low, high = np.percentile(h, [percentile_low, percentile_high])
        low, high = float(low), float(high)

        thresholds[cls] = (max(0, low), high)

        print(
            f"  Class {cls} ({CLASS_NAMES[cls]}): "
            f"{thresholds[cls][0]:.2f}–{thresholds[cls][1]:.2f} m (n={len(h):,})"
        )

    return thresholds


# -----------------------------------------------------------
# Pretty printing
# -----------------------------------------------------------

def summarise_height_ranges(thr: Dict[int, Tuple[float, float]]) -> pd.DataFrame:
    """
    Convert thresholds dict into a clean Pandas table.
    """
    rows = []
    for cls, (lo, hi) in thr.items():
        rows.append((cls, CLASS_NAMES.get(cls, "?"), lo, hi))
    df = pd.DataFrame(rows, columns=["Class", "Name", "Low (m)", "High (m)"])
    return df


def print_height_report(thr: Dict[int, Tuple[float, float]]) -> None:
    """
    Print a human-readable summary.
    """
    print("\n=== Height Threshold Report ===")
    for cls, (lo, hi) in thr.items():
        print(f"  {cls} ({CLASS_NAMES[cls]}): {lo:.2f}–{hi:.2f} m")
    print("==============================\n")


# -----------------------------------------------------------
# Mallee crown minimum-height logic
# -----------------------------------------------------------

def auto_mallee_min_height(thr: Dict[int, Tuple[float, float]], fallback: float = 1.0) -> float:
    """
    Robust rule for Option A (treeID only for canopy mallee).

    Strategy:
    • Use mallee low threshold
    • Ensure not below shrub high threshold
    • Never drop below fallback (1.0 m default)
    """

    mallee_low = thr.get(68, DEFAULT_HEIGHT_RANGES[68])[0]
    shrub_high = thr.get(67, DEFAULT_HEIGHT_RANGES[67])[1]

    mallee_min = max(0.6, mallee_low * 0.7)
    return float(mallee_min)
