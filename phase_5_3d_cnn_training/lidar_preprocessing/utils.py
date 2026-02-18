"""Utility functions for validation and analysis."""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple
from pyntcloud import PyntCloud


def load_npz_summary(npz_path: Path) -> Dict:
    """
    Load and summarize a single NPZ file.
    
    Args:
        npz_path: Path to NPZ file
        
    Returns:
        Dictionary with summary statistics
    """
    data = np.load(npz_path)
    
    coords = data['coords']
    feats = data['feats']
    semantic_label = data['semantic_label']
    instance_label = data['instance_label']
    offset = data['offset']
    
    unique_classes, class_counts = np.unique(semantic_label, return_counts=True)
    unique_trees = np.unique(instance_label[instance_label > 0])
    
    return {
        'filename': npz_path.name,
        'n_points': coords.shape[0],
        'coords_shape': coords.shape,
        'feats_shape': feats.shape,
        'offset': offset,
        'unique_classes': unique_classes.tolist(),
        'class_counts': dict(zip(unique_classes.tolist(), class_counts.tolist())),
        'n_trees': len(unique_trees),
        'tree_ids': unique_trees.tolist()[:10],  # First 10 tree IDs
    }


def load_ply_summary(ply_path: Path) -> Dict:
    """
    Load and summarize a single PLY file.
    
    Args:
        ply_path: Path to PLY file
        
    Returns:
        Dictionary with summary statistics
    """
    cloud = PyntCloud.from_file(str(ply_path))
    df = cloud.points
    
    class_counts = df['semantic_seg'].value_counts().to_dict()
    unique_trees = df[df['treeID'] > 0]['treeID'].unique()
    
    return {
        'filename': ply_path.name,
        'n_points': len(df),
        'columns': df.columns.tolist(),
        'x_range': (df['x'].min(), df['x'].max()),
        'y_range': (df['y'].min(), df['y'].max()),
        'z_range': (df['z'].min(), df['z'].max()),
        'z_norm_range': (df['z_norm'].min(), df['z_norm'].max()),
        'intensity_range': (df['intensity'].min(), df['intensity'].max()),
        'class_counts': class_counts,
        'n_trees': len(unique_trees),
        'tree_ids': unique_trees[:10].tolist(),  # First 10 tree IDs
    }


def npz_to_dataframe(npz_path: Path) -> pd.DataFrame:
    """
    Convert NPZ file to pandas DataFrame for analysis.
    
    Args:
        npz_path: Path to NPZ file
        
    Returns:
        DataFrame with all point data
    """
    data = np.load(npz_path)
    
    coords = data['coords']
    feats = data['feats']
    semantic_label = data['semantic_label']
    instance_label = data['instance_label']
    
    combined_data = np.hstack([
        coords,
        feats,
        semantic_label.reshape(-1, 1),
        instance_label.reshape(-1, 1)
    ])
    
    col_names = [
        "vox_x", "vox_y", "vox_z",
        "loc_x", "loc_y", "loc_z", "z_norm", "intensity",
        "class_id",
        "tree_id"
    ]
    
    return pd.DataFrame(combined_data, columns=col_names)


def dataset_statistics(npz_list_path: Path, ignore_index: int = -100) -> Dict:
    """
    Compute statistics across entire dataset.
    
    Args:
        npz_list_path: Path to npz_list.txt file
        ignore_index: Ignore index for labels
        
    Returns:
        Dictionary with dataset-wide statistics
    """
    with open(npz_list_path, 'r') as f:
        npz_files = [Path(line.strip()) for line in f]
    
    total_points = 0
    total_trees = set()
    class_counts_global = {}
    
    for npz_path in npz_files:
        data = np.load(npz_path)
        sem = data['semantic_label']
        inst = data['instance_label']
        
        total_points += len(sem)
        total_trees.update(inst[inst > 0])
        
        unique, counts = np.unique(sem, return_counts=True)
        for cls, cnt in zip(unique, counts):
            class_counts_global[int(cls)] = class_counts_global.get(int(cls), 0) + int(cnt)
    
    # Compute class percentages
    class_percentages = {
        cls: 100 * cnt / total_points
        for cls, cnt in class_counts_global.items()
    }
    
    return {
        'n_files': len(npz_files),
        'total_points': total_points,
        'unique_trees': len(total_trees),
        'class_counts': class_counts_global,
        'class_percentages': class_percentages,
    }


def compare_npz_ply(npz_path: Path, ply_path: Path, ignore_index: int = -100) -> bool:
    """
    Verify that NPZ and PLY files contain consistent data.
    
    Args:
        npz_path: Path to NPZ file
        ply_path: Path to PLY file
        ignore_index: Ignore index used in NPZ
        
    Returns:
        True if files are consistent
    """
    # Load NPZ
    data = np.load(npz_path)
    sem_npz = data['semantic_label']
    
    # Count valid points (excluding ignore_index)
    valid_mask = sem_npz != ignore_index
    n_valid_npz = int(valid_mask.sum())
    
    # Count class distribution in NPZ (excluding ignore_index)
    unique_npz, counts_npz = np.unique(sem_npz[valid_mask], return_counts=True)
    class_dist_npz = dict(zip(unique_npz.tolist(), counts_npz.tolist()))
    
    # Load PLY
    cloud = PyntCloud.from_file(str(ply_path))
    df = cloud.points
    n_valid_ply = len(df)
    
    class_dist_ply = df['semantic_seg'].value_counts().to_dict()
    
    # Compare
    points_match = (n_valid_npz == n_valid_ply)
    classes_match = (class_dist_npz == class_dist_ply)
    
    if not points_match:
        print(f"⚠️ Point count mismatch: NPZ={n_valid_npz}, PLY={n_valid_ply}")
    
    if not classes_match:
        print(f"⚠️ Class distribution mismatch:")
        print(f"   NPZ: {class_dist_npz}")
        print(f"   PLY: {class_dist_ply}")
    
    return points_match and classes_match
