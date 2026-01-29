#!/usr/bin/env python3
"""
Fix classification issues in voxel data by properly handling NoData values
and ensuring correct label assignments.
"""

import numpy as np
import pandas as pd

def fix_classification_data():
    """Fix the classification issues in voxel tensors"""
    
    print("Loading and analyzing voxel data...")
    
    # Load existing data
    data = np.load("unet_voxel_tensors.npz", allow_pickle=True)
    occupancy = data['occupancy']
    dominant_labels = data['dominant_labels']
    point_counts = data['point_counts']
    voxel_size = data['voxel_size_m'].item()
    origin = data['origin'].item()
    
    print(f"Original data shape: {occupancy.shape}")
    print(f"Occupied voxels: {np.sum(occupancy):,}")
    
    # Check for invalid labels
    unique_labels, counts = np.unique(dominant_labels, return_counts=True)
    print("Current label distribution:")
    for label, count in zip(unique_labels, counts):
        print(f"  Label {label}: {count:,} voxels")
    
    # Fix the labels
    print("\nFixing invalid labels...")
    
    # Create corrected labels array
    corrected_labels = dominant_labels.copy()
    
    # Replace invalid values (-2147483648 or -1) with 255 for NoData
    invalid_mask = (dominant_labels < 0) | (dominant_labels > 3)
    corrected_labels[invalid_mask] = 255  # Use 255 as NoData value
    
    # Ensure only occupied voxels have valid labels (0-3)
    empty_voxels = occupancy == 0
    corrected_labels[empty_voxels] = 255  # NoData for empty voxels
    
    # Verify the fix
    occupied_voxels = occupancy > 0
    valid_labels = corrected_labels[occupied_voxels]
    
    # Check that all occupied voxels have valid labels
    valid_label_mask = (valid_labels >= 0) & (valid_labels <= 3)
    print(f"Occupied voxels with valid labels: {np.sum(valid_label_mask):,}/{len(valid_labels):,}")
    
    if np.sum(valid_label_mask) < len(valid_labels):
        print("WARNING: Some occupied voxels still have invalid labels!")
        invalid_occupied = valid_labels[~valid_label_mask]
        print(f"Invalid labels in occupied voxels: {np.unique(invalid_occupied)}")
    
    # Show corrected distribution
    print("\nCorrected label distribution (occupied voxels only):")
    valid_occupied_labels = corrected_labels[occupied_voxels]
    unique_corrected, counts_corrected = np.unique(valid_occupied_labels, return_counts=True)
    for label, count in zip(unique_corrected, counts_corrected):
        if label != 255:  # Don't show NoData
            print(f"  Label {label}: {count:,} voxels ({100*count/len(valid_occupied_labels):.1f}%)")
    
    # Save corrected tensors
    print("\nSaving corrected tensors...")
    
    np.savez_compressed(
        "unet_voxel_tensors_fixed.npz",
        occupancy=occupancy,
        point_counts=point_counts,
        dominant_labels=corrected_labels.astype(np.uint8),  # Use uint8 to prevent overflow
        voxel_size_m=voxel_size,
        origin=origin,
    )
    
    print("✓ Corrected tensors saved to unet_voxel_tensors_fixed.npz")
    
    return corrected_labels

def create_fixed_visualization():
    """Create visualization files with corrected classifications"""
    
    print("\nCreating fixed visualization files...")
    
    # Load corrected data
    data = np.load("unet_voxel_tensors_fixed.npz", allow_pickle=True)
    occupancy = data['occupancy']
    dominant_labels = data['dominant_labels']  # Now corrected
    voxel_size = data['voxel_size_m'].item()
    origin = data['origin'].item()
    
    # Get occupied voxel coordinates
    z_idx, y_idx, x_idx = np.where(occupancy > 0)
    
    # Convert to world coordinates
    vs_x, vs_y, vs_z = voxel_size['x'], voxel_size['y'], voxel_size['z']
    origin_x, origin_y, origin_z = origin['x'], origin['y'], origin['z']
    
    world_x = origin_x + x_idx * vs_x + vs_x/2
    world_y = origin_y + y_idx * vs_y + vs_y/2  
    world_z = origin_z + z_idx * vs_z + vs_z/2
    
    # Get labels for occupied voxels
    labels = dominant_labels[z_idx, y_idx, x_idx]
    
    # Verify no invalid labels
    print(f"Labels in visualization: {np.unique(labels)}")
    
    # Create colors based on corrected labels
    colors = np.zeros((len(labels), 3))
    color_map = {
        0: [139, 69, 19],    # SaddleBrown for bare ground
        1: [255, 215, 0],    # Gold for non-photosynthetic vegetation
        2: [34, 139, 34],    # ForestGreen for photosynthetic vegetation  
        3: [70, 70, 70],     # DimGray for shadows
        255: [128, 128, 128] # Gray for NoData (shouldn't appear in visualization)
    }
    
    for i, label in enumerate(labels):
        if label in color_map:
            colors[i] = [c/255.0 for c in color_map[label]]  # Normalize to 0-1
        else:
            colors[i] = [0.5, 0.5, 0.5]  # Gray for unexpected labels
    
    # Save as CSV
    df = pd.DataFrame({
        'X': world_x,
        'Y': world_y,
        'Z': world_z,
        'Label': labels,
        'R': (colors[:, 0] * 255).astype(np.uint8),
        'G': (colors[:, 1] * 255).astype(np.uint8),
        'B': (colors[:, 2] * 255).astype(np.uint8)
    })
    
    csv_file = "voxel_occupancy_fixed.csv"
    df.to_csv(csv_file, index=False)
    print(f"✓ Fixed CSV saved: {csv_file}")
    
    # Create LAS file with proper classification codes
    try:
        import laspy
        
        # Create LAS point cloud
        las = laspy.create(point_format=0, file_version="1.4")
        las.x = world_x
        las.y = world_y
        las.z = world_z
        
        # Set classification codes (ensure they're uint8)
        las.classification = labels.astype(np.uint8)
        
        # Set colors (LAS uses 16-bit colors: 0-65535)
        las.red = (colors[:, 0] * 65535).astype(np.uint16)
        las.green = (colors[:, 1] * 65535).astype(np.uint16)  
        las.blue = (colors[:, 2] * 65535).astype(np.uint16)
        
        las_file = "voxel_occupancy_fixed.las"
        las.write(las_file)
        print(f"✓ Fixed LAS saved: {las_file}")
        
    except ImportError:
        print("⚠ laspy not available, skipping LAS export")
    
    # Create PLY file
    try:
        import open3d as o3d
        
        # Create Open3D point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(np.column_stack([world_x, world_y, world_z]))
        pcd.colors = o3d.utility.Vector3dVector(colors)
        
        ply_file = "voxel_occupancy_fixed.ply"
        o3d.io.write_point_cloud(ply_file, pcd)
        print(f"✓ Fixed PLY saved: {ply_file}")
        
    except ImportError:
        print("⚠ open3d not available, skipping PLY export")
    
    # Print final statistics
    print(f"\nFinal statistics:")
    print(f"Total points exported: {len(labels):,}")
    print("Label distribution:")
    unique_final, counts_final = np.unique(labels, return_counts=True)
    for label, count in zip(unique_final, counts_final):
        label_name = {0: 'Bare ground', 1: 'Non-photosynthetic', 2: 'Photosynthetic', 3: 'Shadow'}.get(label, 'Unknown')
        print(f"  Label {label} ({label_name}): {count:,} points ({100*count/len(labels):.1f}%)")
    
    return df

if __name__ == "__main__":
    # Fix the classification data
    corrected_labels = fix_classification_data()
    
    # Create fixed visualization files
    df = create_fixed_visualization()
    
    print("\n✅ Classification issues fixed!")
    print("New files created:")
    print("- unet_voxel_tensors_fixed.npz")
    print("- voxel_occupancy_fixed.csv")  
    print("- voxel_occupancy_fixed.las")
    print("- voxel_occupancy_fixed.ply")