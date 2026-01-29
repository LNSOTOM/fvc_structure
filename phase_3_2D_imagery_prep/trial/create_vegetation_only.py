#!/usr/bin/env python3
"""
Create vegetation-only visualization by removing bare ground (Label 0)
Keep only: Label 1 (Non-photosynthetic), Label 2 (Photosynthetic), Label 3 (Shadow)
"""

import numpy as np
import pandas as pd

def create_vegetation_only_files():
    """Create visualization files with only vegetation labels (1, 2, 3)"""
    
    print("Loading clean classification data...")
    
    # Load the clean data
    df = pd.read_csv("voxel_occupancy_clean.csv")
    
    print(f"Original clean data: {len(df):,} points")
    print("Original label distribution:")
    for label in sorted(df['Label'].unique()):
        count = len(df[df['Label'] == label])
        label_name = {0: 'Bare ground', 1: 'Non-photosynthetic', 2: 'Photosynthetic', 3: 'Shadow'}.get(label, 'Unknown')
        print(f"  Label {label} ({label_name}): {count:,} points ({100*count/len(df):.1f}%)")
    
    # Filter to keep only vegetation labels (1, 2, 3)
    vegetation_mask = df['Label'].isin([1, 2, 3])
    df_veg = df[vegetation_mask].copy()
    
    print(f"\nFiltered vegetation data: {len(df_veg):,} points")
    print("Vegetation-only label distribution:")
    for label in sorted(df_veg['Label'].unique()):
        count = len(df_veg[df_veg['Label'] == label])
        label_name = {1: 'Non-photosynthetic', 2: 'Photosynthetic', 3: 'Shadow'}.get(label, 'Unknown')
        print(f"  Label {label} ({label_name}): {count:,} points ({100*count/len(df_veg):.1f}%)")
    
    removed_count = len(df) - len(df_veg)
    print(f"\nRemoved {removed_count:,} bare ground points")
    
    # Update colors for vegetation-only dataset
    colors = np.zeros((len(df_veg), 3))
    color_map = {
        1: [218, 165, 32],   # GoldenRod for non-photosynthetic vegetation
        2: [50, 205, 50],    # LimeGreen for photosynthetic vegetation  
        3: [105, 105, 105],  # DimGray for shadows
    }
    
    for i, label in enumerate(df_veg['Label']):
        if label in color_map:
            colors[i] = [c/255.0 for c in color_map[label]]
        else:
            colors[i] = [0.5, 0.5, 0.5]  # Gray fallback
    
    # Update RGB columns
    df_veg['R'] = (colors[:, 0] * 255).astype(np.uint8)
    df_veg['G'] = (colors[:, 1] * 255).astype(np.uint8)
    df_veg['B'] = (colors[:, 2] * 255).astype(np.uint8)
    
    # Save vegetation-only CSV
    csv_file = "voxel_occupancy_vegetation_only.csv"
    df_veg.to_csv(csv_file, index=False)
    print(f"✓ Vegetation-only CSV saved: {csv_file}")
    
    # Create vegetation-only LAS file
    try:
        import laspy
        
        las = laspy.create(point_format=3, file_version="1.4")
        las.x = df_veg['X'].values
        las.y = df_veg['Y'].values
        las.z = df_veg['Z'].values
        
        # Classification codes
        las.classification = df_veg['Label'].values.astype(np.uint8)
        
        # RGB colors (16-bit for LAS)
        las.red = df_veg['R'].values.astype(np.uint16) * 256
        las.green = df_veg['G'].values.astype(np.uint16) * 256
        las.blue = df_veg['B'].values.astype(np.uint16) * 256
        
        las_file = "voxel_occupancy_vegetation_only.las"
        las.write(las_file)
        print(f"✓ Vegetation-only LAS saved: {las_file}")
        
    except ImportError as e:
        print(f"⚠ laspy not available: {e}")
    except Exception as e:
        print(f"⚠ Error creating LAS: {e}")
    
    # Create vegetation-only PLY file
    try:
        import open3d as o3d
        
        points = np.column_stack([df_veg['X'].values, df_veg['Y'].values, df_veg['Z'].values])
        
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(colors)
        
        ply_file = "voxel_occupancy_vegetation_only.ply"
        o3d.io.write_point_cloud(ply_file, pcd)
        print(f"✓ Vegetation-only PLY saved: {ply_file}")
        
    except ImportError as e:
        print(f"⚠ open3d not available: {e}")
    except Exception as e:
        print(f"⚠ Error creating PLY: {e}")
    
    # Create height-stratified analysis
    print("\nHeight analysis by vegetation type:")
    for label in sorted(df_veg['Label'].unique()):
        label_data = df_veg[df_veg['Label'] == label]
        label_name = {1: 'Non-photosynthetic', 2: 'Photosynthetic', 3: 'Shadow'}.get(label)
        
        heights = label_data['Z'].values
        print(f"\n{label_name} vegetation (Label {label}):")
        print(f"  Height range: {heights.min():.2f} - {heights.max():.2f} m")
        print(f"  Mean height: {heights.mean():.2f} m")
        print(f"  Std height: {heights.std():.2f} m")
        
        # Height bins
        bins = [0, 0.5, 1.0, 2.0, 5.0, 10.0]
        hist, _ = np.histogram(heights, bins=bins)
        print("  Height distribution:")
        for i in range(len(bins)-1):
            if hist[i] > 0:
                print(f"    {bins[i]:.1f}-{bins[i+1]:.1f}m: {hist[i]:,} points ({100*hist[i]/len(heights):.1f}%)")
    
    # Summary statistics
    print(f"\n✅ Vegetation-only dataset created!")
    print(f"Final point count: {len(df_veg):,}")
    print(f"Data reduction: {100*removed_count/len(df):.1f}% (removed bare ground)")
    print(f"Height range: {df_veg['Z'].min():.2f} - {df_veg['Z'].max():.2f} m")
    
    print("\nFiles created:")
    print("- voxel_occupancy_vegetation_only.csv")
    print("- voxel_occupancy_vegetation_only.las") 
    print("- voxel_occupancy_vegetation_only.ply")
    
    return df_veg

if __name__ == "__main__":
    df_vegetation = create_vegetation_only_files()