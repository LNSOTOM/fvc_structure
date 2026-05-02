#!/usr/bin/env python3
"""
Compare PLY files: PDAL-converted vs rayimport-converted
"""

import numpy as np
import open3d as o3d
from pathlib import Path

def compare_ply_files(pdal_ply, rayimport_ply):
    """Compare PLY files from PDAL conversion vs rayimport"""
    
    print(f"Comparing PLY files:")
    print(f"  PDAL: {pdal_ply}")
    print(f"  rayimport: {rayimport_ply}")
    
    # Load both PLY files
    try:
        pdal_cloud = o3d.io.read_point_cloud(str(pdal_ply))
        rayimport_cloud = o3d.io.read_point_cloud(str(rayimport_ply))
    except Exception as e:
        print(f"❌ Error loading PLY files: {e}")
        return
    
    # Compare point counts
    pdal_points = np.asarray(pdal_cloud.points)
    rayimport_points = np.asarray(rayimport_cloud.points)
    
    print(f"\nPoint Count Comparison:")
    print(f"  PDAL PLY: {len(pdal_points):,} points")
    print(f"  rayimport PLY: {len(rayimport_points):,} points")
    print(f"  Match: {'✅' if len(pdal_points) == len(rayimport_points) else '❌'}")
    
    # Compare coordinate ranges
    print(f"\nCoordinate Ranges:")
    print(f"  PDAL X: {pdal_points[:,0].min():.6f} to {pdal_points[:,0].max():.6f}")
    print(f"  rayimport X: {rayimport_points[:,0].min():.6f} to {rayimport_points[:,0].max():.6f}")
    
    # Compare colors if available
    pdal_colors = np.asarray(pdal_cloud.colors)
    rayimport_colors = np.asarray(rayimport_cloud.colors)
    
    if len(pdal_colors) > 0 and len(rayimport_colors) > 0:
        print(f"\nColor Comparison:")
        print(f"  PDAL colors: {len(pdal_colors):,}")
        print(f"  rayimport colors: {len(rayimport_colors):,}")
        print(f"  PDAL color range: {pdal_colors.min():.3f} to {pdal_colors.max():.3f}")
        print(f"  rayimport color range: {rayimport_colors.min():.3f} to {rayimport_colors.max():.3f}")
    
    # Check file headers by reading raw PLY
    print(f"\nHeader Analysis:")
    analyze_ply_header(pdal_ply, "PDAL")
    analyze_ply_header(rayimport_ply, "rayimport")

def analyze_ply_header(ply_path, source):
    """Analyze PLY header format"""
    try:
        with open(ply_path, 'rb') as f:
            header_text = ""
            while True:
                char = f.read(1).decode('ascii', errors='ignore')
                header_text += char
                if "end_header" in header_text:
                    break
        
        header_lines = header_text.strip().split('\n')
        
        print(f"\n{source} PLY Header:")
        for i, line in enumerate(header_lines[:10]):  # First 10 lines
            print(f"  {i+1}: {line}")
        
        # Check format
        format_line = next((line for line in header_lines if line.startswith('format')), None)
        if format_line:
            print(f"  Format: {format_line}")
        
        # Count properties
        properties = [line for line in header_lines if line.startswith('property')]
        print(f"  Properties count: {len(properties)}")
        
    except Exception as e:
        print(f"  ❌ Could not analyze {source} header: {e}")

# File paths based on your conversion script
output_dir = Path("/media/laura/laura_usb/code/fvc_structure/phase_1_image_processing/step_1_imagesAlignment/output/temp_ply")

# PDAL-converted files (from your script)
pdal_lidar_ply = output_dir / "lidar_cloud.ply"
pdal_sfm_ply = output_dir / "sfm_cloud.ply"

# rayimport-converted files (you'll need to create these)
rayimport_lidar_ply = output_dir / "20220517_SASMDD0011_AusGeoid20_raycloud.ply"
rayimport_sfm_ply = output_dir / "point_cloud_p1_raycloud.ply"

if __name__ == "__main__":
    # Compare LiDAR files
    if pdal_lidar_ply.exists() and rayimport_lidar_ply.exists():
        print("=" * 60)
        print("LIDAR FILE COMPARISON")
        print("=" * 60)
        compare_ply_files(pdal_lidar_ply, rayimport_lidar_ply)
    else:
        print("❌ LiDAR PLY files not found for comparison")
    
    # Compare SfM files
    if pdal_sfm_ply.exists() and rayimport_sfm_ply.exists():
        print("\n" + "=" * 60)
        print("SFM FILE COMPARISON")
        print("=" * 60)
        compare_ply_files(pdal_sfm_ply, rayimport_sfm_ply)
    else:
        print("❌ SfM PLY files not found for comparison")