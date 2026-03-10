#!/usr/bin/env python3
"""
Example usage of the LiDAR preprocessing package.

This script demonstrates how to:
1. Configure preprocessing settings
2. Process LAS files
3. Validate outputs
4. Analyze results
"""

from pathlib import Path
from lidar_preprocessing import PreprocessingConfig, batch_process
from lidar_preprocessing.utils import (
    load_npz_summary,
    load_ply_summary,
    dataset_statistics,
    compare_npz_ply,
)


def main():
    print("=" * 80)
    print("LiDAR Preprocessing Example")
    print("=" * 80)
    
    # 1. Configure preprocessing
    config = PreprocessingConfig(
        input_dir=Path("/data/calperumResearch/site1_1_DD0001/lidar/clipped/3d_structure_obj_polar_id"),
        out_npz_dir=Path("/data/calperumResearch/site1_1_DD0001/lidar/clipped/preprocessed/minsu3d_npz/"),
        out_ply_dir=Path("/data/calperumResearch/site1_1_DD0001/lidar/clipped/preprocessed/forestformer3d_ply/"),
        voxel_size=0.10,
    )
    
    print(f"\n{config}\n")
    
    # 2. Process all LAS files
    print("\n" + "=" * 80)
    print("Processing LAS files...")
    print("=" * 80 + "\n")
    
    npz_files, ply_files, errors = batch_process(config)
    
    if errors:
        print("\n⚠️ Errors encountered during processing:")
        for fname, err in errors:
            print(f"   {fname}: {err}")
    
    if not npz_files:
        print("❌ No files were successfully processed.")
        return
    
    # 3. Validate first file
    print("\n" + "=" * 80)
    print("Validating first file...")
    print("=" * 80 + "\n")
    
    npz_path = Path(npz_files[0])
    ply_path = Path(ply_files[0])
    
    print(f"NPZ: {npz_path.name}")
    print(f"PLY: {ply_path.name}\n")
    
    # Check NPZ content
    npz_summary = load_npz_summary(npz_path)
    print("NPZ Summary:")
    print(f"  Points: {npz_summary['n_points']:,}")
    print(f"  Features shape: {npz_summary['feats_shape']}")
    print(f"  Classes: {npz_summary['unique_classes']}")
    print(f"  Trees: {npz_summary['n_trees']}\n")
    
    # Check PLY content
    ply_summary = load_ply_summary(ply_path)
    print("PLY Summary:")
    print(f"  Points: {ply_summary['n_points']:,}")
    print(f"  Columns: {ply_summary['columns']}")
    print(f"  X range: {ply_summary['x_range']}")
    print(f"  Trees: {ply_summary['n_trees']}\n")
    
    # Compare NPZ and PLY
    print("Comparing NPZ and PLY consistency...")
    is_consistent = compare_npz_ply(npz_path, ply_path, config.ignore_index)
    if is_consistent:
        print("✅ NPZ and PLY files are consistent!\n")
    else:
        print("❌ Inconsistency detected between NPZ and PLY.\n")
    
    # 4. Dataset-wide statistics
    print("\n" + "=" * 80)
    print("Dataset Statistics")
    print("=" * 80 + "\n")
    
    npz_list_path = config.out_npz_dir / "npz_list.txt"
    stats = dataset_statistics(npz_list_path, config.ignore_index)
    
    print(f"Total files: {stats['n_files']}")
    print(f"Total points: {stats['total_points']:,}")
    print(f"Unique trees: {stats['unique_trees']}\n")
    
    print("Class distribution:")
    for cls in sorted(stats['class_counts'].keys()):
        if cls == config.ignore_index:
            continue
        count = stats['class_counts'][cls]
        pct = stats['class_percentages'][cls]
        print(f"  Class {cls}: {count:>12,} ({pct:5.2f}%)")
    
    print("\n" + "=" * 80)
    print("✅ Preprocessing complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
