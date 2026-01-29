"""
fvc3dstruct - 3D vegetation structure fusion pipeline.

Comprehensive pipeline for fusing Z_norm LiDAR with 2D structure labels:
- Cartesian voxel merge with majority voting
- Watershed-based mallee crown detection
- Crown-centric polar refinement with height priors
- Data-driven class height thresholds
- TreeID assignment for mallee canopy only
"""

from .pipeline import process_tile
from .voxel_merge import cartesian_voxel_merge
from .watershed import get_mallee_centroids_watershed
from .polar_refine import refine_with_polar_coords
from .height_stats import (
    compute_class_height_thresholds,
    summarise_height_ranges,
    auto_mallee_min_height,
    print_height_report,
)
from .config import DEFAULT_CONFIG
from .io_las import ensure_las14_pf6, read_las, write_las

__version__ = "1.0.0"

__all__ = [
    "process_tile",
    "cartesian_voxel_merge",
    "get_mallee_centroids_watershed",
    "refine_with_polar_coords",
    "compute_class_height_thresholds",
    "summarise_height_ranges",
    "auto_mallee_min_height",
    "print_height_report",
    "DEFAULT_CONFIG",
    "ensure_las14_pf6",
    "read_las",
    "write_las",
]
