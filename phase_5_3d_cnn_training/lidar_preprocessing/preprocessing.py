"""Core preprocessing functions for LiDAR data."""

import laspy
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, List, Optional
from pyntcloud import PyntCloud

from .config import PreprocessingConfig


def normalise_intensity(intensity: np.ndarray) -> np.ndarray:
    """
    Normalise intensity values to [0, 1] range.
    
    Handles 8-bit, 16-bit, or arbitrary range data.
    
    Args:
        intensity: Raw intensity values
        
    Returns:
        Normalised intensity in [0, 1]
    """
    intensity = intensity.astype(np.float32)
    if intensity.size == 0:
        return intensity
    
    max_val = float(intensity.max())
    if max_val <= 0:
        return np.zeros_like(intensity, dtype=np.float32)
    
    # Auto-detect bit depth
    if max_val <= 255:
        scale = 255.0
    elif max_val <= 65535:
        scale = 65535.0
    else:
        scale = max_val
    
    return np.clip(intensity / scale, 0.0, 1.0)


def get_z_norm(las, raw_xyz: np.ndarray) -> np.ndarray:
    """
    Extract or compute normalized height (Z_norm).
    
    Priority:
      1) Existing ExtraBytes dimension named 'Z_norm'
      2) Relative height within the tile (z - z_min)
    
    Args:
        las: Laspy LAS object
        raw_xyz: Raw XYZ coordinates (N, 3)
        
    Returns:
        Normalized height values (N,)
    """
    dims = las.point_format.dimension_names
    if "Z_norm" in dims:
        return las["Z_norm"].astype(np.float32)
    
    # Fallback: relative height within tile
    z = raw_xyz[:, 2]
    return (z - z.min()).astype(np.float32)


def remap_classes(raw_labels: np.ndarray, config: PreprocessingConfig) -> np.ndarray:
    """
    Remap LAS classification codes to compact class IDs.
    
    Args:
        raw_labels: Original LAS classification codes
        config: Preprocessing configuration with class_remap
        
    Returns:
        Remapped class labels (N,)
    """
    sem = np.full_like(raw_labels, config.ignore_index, dtype=np.int32)
    for old, new in config.class_remap.items():
        sem[raw_labels == old] = new
    return sem


def get_tree_ids(las, n: int, fname: str) -> np.ndarray:
    """
    Extract tree instance IDs from LAS extra dimensions.
    
    Args:
        las: Laspy LAS object
        n: Number of points
        fname: Filename for warning messages
        
    Returns:
        Tree instance IDs (N,)
    """
    dims = las.point_format.dimension_names
    if "treeID" in dims:
        return las["treeID"].astype(np.int32)
    
    print(f"⚠️ Warning: no treeID in {fname}, using zeros.")
    return np.zeros(n, dtype=np.int32)


def write_forestformer_ply(
    out_path: Path,
    xyz: np.ndarray,
    semantic: np.ndarray,
    inst: np.ndarray,
    z_norm: np.ndarray,
    intensity: np.ndarray,
    config: PreprocessingConfig,
) -> None:
    """
    Write PLY file for ForestFormer3D, excluding ignore-index points.
    
    Args:
        out_path: Output PLY file path
        xyz: XYZ coordinates (N, 3). Use local coordinates for best precision.
        semantic: Semantic class labels (N,)
        inst: Instance IDs (N,)
        z_norm: Normalized heights (N,)
        intensity: Normalized intensity (N,)
        config: Preprocessing configuration
    """
    valid_mask = semantic != config.ignore_index
    n_valid = int(valid_mask.sum())
    
    if n_valid == 0:
        print(f"⚠️ No valid points for {out_path.name}, skipping PLY.")
        return
    
    df = pd.DataFrame({
        "x": xyz[valid_mask, 0],
        "y": xyz[valid_mask, 1],
        "z": xyz[valid_mask, 2],
        "semantic_seg": semantic[valid_mask],
        "treeID": inst[valid_mask],
        "z_norm": z_norm[valid_mask],
        "intensity": intensity[valid_mask],
    })
    cloud = PyntCloud(df)
    cloud.to_file(str(out_path))


def process_las(path: Path, config: PreprocessingConfig) -> Tuple[Optional[Path], Path]:
    """
    Process single LAS file into optional NPZ (MINSU3D) and PLY (ForestFormer3D).
    
    Args:
        path: Path to input LAS file
        config: Preprocessing configuration
        
    Returns:
        Tuple of (npz_path, ply_path). If config.save_npz is False, npz_path is None.
    """
    print(f"📦 Processing {path.name}")
    las = laspy.read(path)
    
    # Extract XYZ coordinates
    raw_xyz = np.vstack([las.x, las.y, las.z]).T
    if raw_xyz.size == 0:
        raise ValueError("Empty point cloud.")
    
    # Local coordinate shift
    offset = raw_xyz.min(axis=0)
    xyz_local = raw_xyz - offset
    
    # Extract features
    intensity = normalise_intensity(las.intensity)
    z_norm = get_z_norm(las, raw_xyz)
    feats = np.column_stack([xyz_local, z_norm, intensity]).astype(np.float32)
    
    # Extract labels
    raw_labels = las.classification.astype(np.int32)
    semantic = remap_classes(raw_labels, config)
    inst = get_tree_ids(las, len(raw_labels), path.name)
    
    npz_path: Optional[Path] = None
    if config.save_npz:
        if config.out_npz_dir is None:
            raise ValueError("config.save_npz=True requires config.out_npz_dir")
        # Compute voxel coordinates
        coords = np.floor(xyz_local / config.voxel_size).astype(np.int32)

        # Save NPZ (MINSU3D format)
        npz_path = Path(config.out_npz_dir) / f"{path.stem}.npz"
        np.savez_compressed(
            npz_path,
            coords=coords,
            feats=feats,
            semantic_label=semantic,
            instance_label=inst,
            offset=offset.astype(np.float64),
        )
    
    # Save PLY (ForestFormer3D format)
    # Using local coords avoids float32 quantization at large global magnitudes.
    xyz_for_ply = xyz_local if config.ply_use_local_coords else raw_xyz
    ply_path = config.out_ply_dir / f"{path.stem}.ply"
    write_forestformer_ply(
        ply_path,
        xyz=xyz_for_ply,
        semantic=semantic,
        inst=inst,
        z_norm=z_norm,
        intensity=intensity,
        config=config,
    )

    if config.ply_use_local_coords and config.save_ply_offset:
        offset_path = config.out_ply_dir / f"{path.stem}_offset.npy"
        np.save(offset_path, offset.astype(np.float64))
    
    return npz_path, ply_path


def batch_process(
    config: PreprocessingConfig,
    file_pattern: str = "*.las",
) -> Tuple[List[str], List[str], List[Tuple[str, str]]]:
    """
    Process all LAS files in the input directory.
    
    Args:
        config: Preprocessing configuration
        file_pattern: Glob pattern for LAS files
        
    Returns:
        Tuple of (npz_paths, ply_paths, errors)
    """
    las_files = sorted(config.input_dir.glob(file_pattern))
    
    if not las_files:
        print(f"❌ No files matching '{file_pattern}' found in {config.input_dir}")
        return [], [], []
    
    print(f"🚀 Found {len(las_files)} LAS files.")
    
    all_npz: List[str] = []
    all_ply = []
    errors = []
    
    for f in las_files:
        try:
            npz, ply = process_las(f, config)
            if npz is not None:
                all_npz.append(str(npz))
            all_ply.append(str(ply))
        except Exception as e:
            print(f"❌ Error in {f.name}: {e}")
            errors.append((f.name, str(e)))
    
    # Save file lists
    if config.save_npz:
        if config.out_npz_dir is None:
            raise ValueError("config.save_npz=True requires config.out_npz_dir")
        with open(Path(config.out_npz_dir) / "npz_list.txt", "w") as f:
            f.write("\n".join(all_npz))

    with open(config.out_ply_dir / "ply_list.txt", "w") as f:
        f.write("\n".join(all_ply))
    
    print(f"\n✅ Processing complete!")
    if config.save_npz:
        print(f"   Successful: {len(all_npz)} NPZ files")
    print(f"   Successful: {len(all_ply)} PLY files")
    print(f"   Errors: {len(errors)} files")
    if config.save_npz and config.out_npz_dir is not None:
        print(f"\n📄 NPZ list: {Path(config.out_npz_dir) / 'npz_list.txt'}")
    print(f"📄 PLY list: {config.out_ply_dir / 'ply_list.txt'}")
    
    return all_npz, all_ply, errors
