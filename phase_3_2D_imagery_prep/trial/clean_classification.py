#!/usr/bin/env python3
"""
Fix classification issues by removing voxels with invalid labels
and creating clean visualization files.
"""

import numpy as np
import pandas as pd

def create_clean_visualization():
    """Create visualization files with only valid labels (no NoData voxels)"""
    
    print("Loading and cleaning voxel data...")
    
    # Load existing data
    data = np.load("unet_voxel_tensors.npz", allow_pickle=True)
    occupancy = data['occupancy']
    dominant_labels = data['dominant_labels']
    point_counts = data['point_counts']
    voxel_size = data['voxel_size_m'].item()
    origin = data['origin'].item()
    
    # Get occupied voxel coordinates
    z_idx, y_idx, x_idx = np.where(occupancy > 0)
    labels = dominant_labels[z_idx, y_idx, x_idx]
    counts = point_counts[z_idx, y_idx, x_idx]
    
    print(f"Total occupied voxels: {len(labels):,}")
    
    # Find voxels with valid labels (0, 1, 2, 3 only)
    valid_mask = (labels >= 0) & (labels <= 3)
    
    print(f"Voxels with valid labels: {np.sum(valid_mask):,}")
    print(f"Voxels with invalid labels: {np.sum(~valid_mask):,}")
    
    # Show distribution of invalid labels
    if np.any(~valid_mask):
        invalid_labels = labels[~valid_mask]
        unique_invalid, counts_invalid = np.unique(invalid_labels, return_counts=True)
        print("Invalid labels found:")
        for label, count in zip(unique_invalid, counts_invalid):
            print(f"  {label}: {count:,} voxels")
    
    # Keep only voxels with valid labels
    z_valid = z_idx[valid_mask]
    y_valid = y_idx[valid_mask]
    x_valid = x_idx[valid_mask]
    labels_valid = labels[valid_mask]
    counts_valid = counts[valid_mask]
    
    # Convert to world coordinates
    vs_x, vs_y, vs_z = voxel_size['x'], voxel_size['y'], voxel_size['z']
    origin_x, origin_y, origin_z = origin['x'], origin['y'], origin['z']
    
    world_x = origin_x + x_valid * vs_x + vs_x/2
    world_y = origin_y + y_valid * vs_y + vs_y/2  
    world_z = origin_z + z_valid * vs_z + vs_z/2
    
    print(f"\nFinal point count: {len(labels_valid):,}")
    
    # Show final label distribution
    print("Clean label distribution:")
    unique_clean, counts_clean = np.unique(labels_valid, return_counts=True)
    for label, count in zip(unique_clean, counts_clean):
        label_name = {0: 'Bare ground', 1: 'Non-photosynthetic', 2: 'Photosynthetic', 3: 'Shadow'}.get(label, 'Unknown')
        print(f"  Label {label} ({label_name}): {count:,} points ({100*count/len(labels_valid):.1f}%)")
    
    # Create colors - use distinctive colors that work well in CloudCompare
    colors = np.zeros((len(labels_valid), 3))
    color_map = {
        0: [160, 82, 45],    # SaddleBrown for bare ground/soil
        1: [218, 165, 32],   # GoldenRod for non-photosynthetic vegetation
        2: [50, 205, 50],    # LimeGreen for photosynthetic vegetation  
        3: [105, 105, 105],  # DimGray for shadows
    }
    
    for i, label in enumerate(labels_valid):
        if label in color_map:
            colors[i] = [c/255.0 for c in color_map[label]]  # Normalize to 0-1
        else:
            colors[i] = [0.5, 0.5, 0.5]  # Gray fallback
    
    # Save as CSV
    df = pd.DataFrame({
        'X': world_x,
        'Y': world_y,
        'Z': world_z,
        'Label': labels_valid,
        'PointCount': counts_valid,
        'R': (colors[:, 0] * 255).astype(np.uint8),
        'G': (colors[:, 1] * 255).astype(np.uint8),
        'B': (colors[:, 2] * 255).astype(np.uint8)
    })
    
    csv_file = "voxel_occupancy_clean.csv"
    df.to_csv(csv_file, index=False)
    print(f"✓ Clean CSV saved: {csv_file}")
    
    # Create LAS file with proper classification codes (0-3 are valid)
    try:
        import laspy
        
        las = laspy.create(point_format=0, file_version="1.4")
        las.x = world_x
        las.y = world_y
        las.z = world_z
        
        # Classification codes 0-3 are all valid for LAS
        las.classification = labels_valid.astype(np.uint8)
        
        # Set colors (LAS uses 16-bit colors: 0-65535)
        las.red = (colors[:, 0] * 65535).astype(np.uint16)
        las.green = (colors[:, 1] * 65535).astype(np.uint16)  
        las.blue = (colors[:, 2] * 65535).astype(np.uint16)
        
        las_file = "voxel_occupancy_clean.las"
        las.write(las_file)
        print(f"✓ Clean LAS saved: {las_file}")
        
    except ImportError as e:
        print(f"⚠ laspy not available: {e}")
    except Exception as e:
        print(f"⚠ Error creating LAS: {e}")
    
    # Create PLY file
    try:
        import open3d as o3d
        
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(np.column_stack([world_x, world_y, world_z]))
        pcd.colors = o3d.utility.Vector3dVector(colors)
        
        ply_file = "voxel_occupancy_clean.ply"
        o3d.io.write_point_cloud(ply_file, pcd)
        print(f"✓ Clean PLY saved: {ply_file}")
        
    except ImportError as e:
        print(f"⚠ open3d not available: {e}")
    except Exception as e:
        print(f"⚠ Error creating PLY: {e}")
    
    # Create a height-colored version for better visualization
    print("\nCreating height-colored version...")
    
    # Normalize heights to 0-1 for color mapping
    height_norm = (world_z - world_z.min()) / (world_z.max() - world_z.min())
    
    # Create height-based colors (blue to red gradient)
    height_colors = np.zeros((len(labels_valid), 3))
    height_colors[:, 0] = height_norm  # Red increases with height
    height_colors[:, 1] = 1 - np.abs(height_norm - 0.5) * 2  # Green peaks at mid-height
    height_colors[:, 2] = 1 - height_norm  # Blue decreases with height
    
    # Save height-colored PLY
    try:
        import open3d as o3d
        
        pcd_height = o3d.geometry.PointCloud()
        pcd_height.points = o3d.utility.Vector3dVector(np.column_stack([world_x, world_y, world_z]))
        pcd_height.colors = o3d.utility.Vector3dVector(height_colors)
        
        ply_height_file = "voxel_occupancy_height_colored.ply"
        o3d.io.write_point_cloud(ply_height_file, pcd_height)
        print(f"✓ Height-colored PLY saved: {ply_height_file}")
        
    except Exception as e:
        print(f"⚠ Error creating height-colored PLY: {e}")
    
    print("\n✅ Classification cleaned successfully!")
    print(f"Removed {np.sum(~valid_mask):,} voxels with invalid labels")
    print(f"Final dataset: {len(labels_valid):,} clean points")
    
    return df

if __name__ == "__main__":
    df = create_clean_visualization()