"""
LiDAR Preprocessing Package for MINSU3D and ForestFormer3D

Unified preprocessing pipeline for Calperum LiDAR tiles.
"""

from .preprocessing import (
    normalise_intensity,
    get_z_norm,
    remap_classes,
    get_tree_ids,
    write_forestformer_ply,
    process_las,
    batch_process,
)

from .config import PreprocessingConfig

__version__ = "1.0.0"

__all__ = [
    "normalise_intensity",
    "get_z_norm",
    "remap_classes",
    "get_tree_ids",
    "write_forestformer_ply",
    "process_las",
    "batch_process",
    "PreprocessingConfig",
]
