"""Configuration settings for LiDAR preprocessing."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional


@dataclass
class PreprocessingConfig:
    """Configuration for LiDAR preprocessing pipeline."""
    
    # Input/Output paths
    input_dir: Path = Path(
        "/data/calperumResearch/site1_1_DD0001/lidar/clipped/3d_structure_obj_polar_id"
    )
    out_npz_dir: Optional[Path] = Path(
        "/data/calperumResearch/site1_1_DD0001/lidar/clipped/preprocessed/minsu3d_npz/"
    )
    out_ply_dir: Path = Path(
        "/data/calperumResearch/site1_1_DD0001/lidar/clipped/preprocessed/forestformer3d_ply/"
    )

    # Output toggles
    # If False, the pipeline will NOT write MINSU3D NPZ files.
    save_npz: bool = True
    ply_use_local_coords: bool = True
    save_ply_offset: bool = True
    
    # Processing parameters
    voxel_size: float = 0.10  # 10 cm voxels
    ignore_index: int = -100  # PyTorch convention
    
    # Class mapping: LAS classification codes -> model class IDs
    class_remap: Dict[int, int] = field(default_factory=lambda: {
        64: 0,  # Forb
        65: 1,  # Woody debris
        66: 2,  # Hummock
        67: 3,  # Shrub
        68: 4,  # Mallee
        0: -100,  # Unmatched / background (will be set to ignore_index)
    })
    
    def __post_init__(self):
        """Ensure output directories exist."""
        self.input_dir = Path(self.input_dir)
        if self.out_npz_dir is not None:
            self.out_npz_dir = Path(self.out_npz_dir)
        self.out_ply_dir = Path(self.out_ply_dir)

        if self.save_npz:
            if self.out_npz_dir is None:
                raise ValueError("save_npz=True requires out_npz_dir to be set")
            self.out_npz_dir.mkdir(parents=True, exist_ok=True)
        self.out_ply_dir.mkdir(parents=True, exist_ok=True)
        
        # Update class_remap to use the ignore_index
        self.class_remap[0] = self.ignore_index
    
    @property
    def num_classes(self) -> int:
        """Number of valid classes (excluding ignore_index)."""
        return len([v for v in self.class_remap.values() if v != self.ignore_index])
    
    def __repr__(self) -> str:
        """String representation of config."""
        return (
            f"PreprocessingConfig(\n"
            f"  input_dir={self.input_dir},\n"
            f"  out_npz_dir={self.out_npz_dir},\n"
            f"  out_ply_dir={self.out_ply_dir},\n"
            f"  save_npz={self.save_npz},\n"
            f"  ply_use_local_coords={self.ply_use_local_coords},\n"
            f"  save_ply_offset={self.save_ply_offset},\n"
            f"  voxel_size={self.voxel_size},\n"
            f"  num_classes={self.num_classes},\n"
            f"  ignore_index={self.ignore_index}\n"
            f")"
        )
