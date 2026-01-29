#!/usr/bin/env python3
"""
Compare LAS and PLY files to identify any differences
"""

import numpy as np
import pandas as pd
import laspy
import open3d as o3d

def compare_las_ply():
    """Compare LAS and PLY files point by point"""
    
    print("Loading files...")
    
    # Load LAS
    las = laspy.read('voxel_occupancy.las')
    las_points = np.column_stack([las.x, las.y, las.z])
    las_colors = np.column_stack([las.red, las.green, las.blue]) / 65535.0  # Scale to 0-1
    las_classification = las.classification
    
    # Load PLY 
    ply = o3d.io.read_point_cloud('voxel_occupancy_pointcloud.ply')
    ply_points = np.asarray(ply.points)
    ply_colors = np.asarray(ply.colors)
    
    print(f"\nFile Comparison:")
    print(f"LAS points: {len(las_points):,}")
    print(f"PLY points: {len(ply_points):,}")
    print(f"Points match: {len(las_points) == len(ply_points)}")
    
    # Compare coordinates (allowing for small floating point differences)
    coord_diff = np.abs(las_points - ply_points)
    max_coord_diff = np.max(coord_diff)
    print(f"\nCoordinate differences:")
    print(f"Max coordinate difference: {max_coord_diff:.10f}")
    print(f"Mean coordinate difference: {np.mean(coord_diff):.10f}")
    
    # Compare colors
    color_diff = np.abs(las_colors - ply_colors)
    max_color_diff = np.max(color_diff)
    print(f"\nColor differences:")
    print(f"Max color difference: {max_color_diff:.6f}")
    print(f"Mean color difference: {np.mean(color_diff):.6f}")
    
    # Check if coordinates are identical within tolerance
    coords_match = np.allclose(las_points, ply_points, atol=1e-6)
    colors_match = np.allclose(las_colors, ply_colors, atol=1e-3)
    
    print(f"\nData integrity:")
    print(f"Coordinates match (±1e-6): {coords_match}")
    print(f"Colors match (±1e-3): {colors_match}")
    
    # Check classification distribution
    print(f"\nLAS Classification distribution:")
    unique_classes, counts = np.unique(las_classification, return_counts=True)
    for class_val, count in zip(unique_classes, counts):
        print(f"  Class {class_val}: {count:,} points ({100*count/len(las_classification):.1f}%)")
    
    # Check coordinate ranges in detail
    print(f"\nDetailed coordinate ranges:")
    print(f"LAS X: {las.x.min():.6f} to {las.x.max():.6f}")
    print(f"PLY X: {ply_points[:,0].min():.6f} to {ply_points[:,0].max():.6f}")
    print(f"LAS Y: {las.y.min():.6f} to {las.y.max():.6f}")  
    print(f"PLY Y: {ply_points[:,1].min():.6f} to {ply_points[:,1].max():.6f}")
    print(f"LAS Z: {las.z.min():.6f} to {las.z.max():.6f}")
    print(f"PLY Z: {ply_points[:,2].min():.6f} to {ply_points[:,2].max():.6f}")
    
    # Sample some points for detailed comparison
    print(f"\nFirst 5 points comparison:")
    print("LAS coordinates:")
    for i in range(5):
        print(f"  [{las.x[i]:.6f}, {las.y[i]:.6f}, {las.z[i]:.6f}]")
    
    print("PLY coordinates:")
    for i in range(5):
        print(f"  [{ply_points[i,0]:.6f}, {ply_points[i,1]:.6f}, {ply_points[i,2]:.6f}]")
    
    return {
        'las_points': len(las_points),
        'ply_points': len(ply_points),
        'coords_match': coords_match,
        'colors_match': colors_match,
        'max_coord_diff': max_coord_diff,
        'max_color_diff': max_color_diff
    }

def check_viewer_issues():
    """Check for common viewer issues"""
    
    print("\n" + "="*60)
    print("COMMON VIEWER ISSUES & SOLUTIONS")
    print("="*60)
    
    print("\n🔍 Potential reasons for 'missing' points:")
    
    print("\n1. COORDINATE SYSTEM DIFFERENCES:")
    print("   - LAS uses original survey coordinates (UTM/projected)")
    print("   - Some viewers may not handle large coordinates well")
    print("   - Solution: Check if viewer supports projected coordinates")
    
    print("\n2. POINT SIZE/SCALE:")
    print("   - Points might be too small to see")
    print("   - Solution: Increase point size in viewer")
    
    print("\n3. CLASSIFICATION FILTERING:")
    print("   - LAS has classification codes (0-3)")
    print("   - Some viewers filter by classification")
    print("   - Solution: Enable all classification codes")
    
    print("\n4. COLOR SCALING:")
    print("   - LAS colors use 16-bit values (0-65535)")
    print("   - PLY colors use float values (0.0-1.0)")
    print("   - Solution: Check color scaling in viewer")
    
    print("\n5. VIEWER LIMITATIONS:")
    print("   - Some viewers have point count limits")
    print("   - Memory limitations with large datasets")
    print("   - Solution: Try different viewer or subsample")
    
    print("\n📋 RECOMMENDED VIEWERS:")
    print("   ✅ CloudCompare: Best for both LAS and PLY")
    print("   ✅ QGIS: Excellent LAS support")
    print("   ✅ FME Data Inspector: Professional grade")
    print("   ⚠️  MeshLab: Better for PLY than LAS")
    print("   ⚠️  Open3D viewer: PLY only")

def export_sample_for_testing():
    """Export a small sample for testing"""
    
    print("\n" + "="*60)
    print("CREATING SAMPLE FILES FOR TESTING")
    print("="*60)
    
    # Load original data
    data = np.load("unet_voxel_tensors.npz", allow_pickle=True)
    occupancy = data['occupancy']
    dominant_labels = data['dominant_labels']
    
    # Get a small sample of occupied voxels
    z_idx, y_idx, x_idx = np.where(occupancy > 0)
    
    # Take first 1000 points
    sample_size = min(1000, len(z_idx))
    sample_indices = np.arange(sample_size)
    
    z_sample = z_idx[sample_indices]
    y_sample = y_idx[sample_indices]
    x_sample = x_idx[sample_indices]
    
    # Convert to world coordinates
    voxel_size = data['voxel_size_m'].item()
    origin = data['origin'].item()
    
    world_x = origin['x'] + x_sample * voxel_size['x'] + voxel_size['x']/2
    world_y = origin['y'] + y_sample * voxel_size['y'] + voxel_size['y']/2
    world_z = origin['z'] + z_sample * voxel_size['z'] + voxel_size['z']/2
    
    labels = dominant_labels[z_sample, y_sample, x_sample]
    
    # Create simple CSV
    df = pd.DataFrame({
        'X': world_x,
        'Y': world_y,
        'Z': world_z,
        'Label': labels
    })
    
    df.to_csv('sample_1000_points.csv', index=False)
    print(f"✓ Created sample_1000_points.csv ({sample_size} points)")
    
    # Create simple LAS
    try:
        las = laspy.create(point_format=0, file_version="1.4")
        las.x = world_x
        las.y = world_y  
        las.z = world_z
        las.classification = labels.astype(np.uint8)
        
        las.write('sample_1000_points.las')
        print(f"✓ Created sample_1000_points.las ({sample_size} points)")
    except Exception as e:
        print(f"❌ Could not create sample LAS: {e}")
    
    print(f"\nTest these small files first to verify viewer compatibility!")

if __name__ == "__main__":
    # Run comparison
    result = compare_las_ply()
    
    # Check for viewer issues
    check_viewer_issues()
    
    # Create sample files
    export_sample_for_testing()