#!/usr/bin/env python3
"""
Complete LiDAR alignment workflow using raycloud tools
"""

import subprocess
import laspy
import numpy as np
from pathlib import Path
import os

def las_to_ply(las_path, ply_path):
    """Convert LAS file to PLY format"""
    print(f"Converting {las_path} to {ply_path}")
    
    las = laspy.read(las_path)
    points = np.column_stack([las.x, las.y, las.z])
    
    with open(ply_path, 'w') as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(points)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("end_header\n")
        
        for point in points:
            f.write(f"{point[0]:.6f} {point[1]:.6f} {point[2]:.6f}\n")
    
    print(f"✅ Converted {len(points):,} points")

def ply_to_las(ply_path, output_las_path, reference_las_path):
    """Convert aligned PLY back to LAS format with proper header"""
    print(f"Converting {ply_path} back to LAS format")
    
    # Read reference LAS for header info
    ref_las = laspy.read(reference_las_path)
    
    # Read PLY file
    points = []
    with open(ply_path, 'r') as f:
        lines = f.readlines()
        
        # Find data start
        data_start = 0
        for i, line in enumerate(lines):
            if line.strip() == "end_header":
                data_start = i + 1
                break
        
        # Read points
        for line in lines[data_start:]:
            if line.strip():
                coords = line.strip().split()
                points.append([float(coords[0]), float(coords[1]), float(coords[2])])
    
    points = np.array(points)
    
    # Create new LAS file
    header = laspy.LasHeader(point_format=ref_las.header.point_format, version=ref_las.header.version)
    header.offsets = ref_las.header.offsets
    header.scales = ref_las.header.scales
    
    las_out = laspy.LasData(header)
    las_out.x = points[:, 0]
    las_out.y = points[:, 1] 
    las_out.z = points[:, 2]
    
    # Copy other attributes if they exist
    if hasattr(ref_las, 'intensity'):
        las_out.intensity = np.zeros(len(points), dtype=ref_las.intensity.dtype)
    
    las_out.write(output_las_path)
    print(f"✅ Saved aligned LAS with {len(points):,} points")

def align_las_files(source_las, target_las, output_las, rayalign_binary):
    """Complete alignment workflow"""
    
    print(f"🔄 Aligning LAS files using raycloud tools")
    print(f"  Source: {source_las}")
    print(f"  Target: {target_las}")
    print(f"  Output: {output_las}")
    
    # Create temp directory
    temp_dir = Path("/tmp/raycloud_alignment")
    temp_dir.mkdir(exist_ok=True)
    
    try:
        # Convert to PLY
        source_ply = temp_dir / "source.ply"
        target_ply = temp_dir / "target.ply"
        
        las_to_ply(source_las, source_ply)
        las_to_ply(target_las, target_ply)
        
        # Run rayalign
        print("🔄 Running rayalign...")
        cmd = [
            str(rayalign_binary),
            str(source_ply),
            str(target_ply),
            "--verbose"
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=temp_dir)
        
        if result.returncode != 0:
            print(f"❌ rayalign failed:")
            print(f"STDOUT: {result.stdout}")
            print(f"STDERR: {result.stderr}")
            return False
        
        print("✅ rayalign completed successfully")
        print(f"Output: {result.stdout}")
        
        # Convert aligned PLY back to LAS
        aligned_ply = temp_dir / "source_aligned.ply"
        if aligned_ply.exists():
            ply_to_las(aligned_ply, output_las, source_las)
        else:
            print(f"❌ Aligned PLY file not found: {aligned_ply}")
            return False
            
    finally:
        # Cleanup temp files (optional)
        # import shutil
        # shutil.rmtree(temp_dir)
        pass
    
    print("✅ Alignment workflow completed")
    return True

# Configuration
source_las = "/media/laura/laura_usb/qgis/calperumResearch/site2_1_DD0011/lidar/sfm_dense/point_cloud_micasense.laz"
target_las = "/media/laura/Extreme SSD/qgis/calperumResearch/site2_1_DD0011/lidar/cloud2453cf90763afd91.las"
output_las = "/media/laura/laura_usb/code/fvc_structure/phase_1_image_processing/step_1_imagesAlignment/sfm_aligned_to_lidar.las"

# Path to rayalign binary (adjust based on your build)
rayalign_binary = "/media/laura/laura_usb/code/fvc_structure/raycloudtools/build/rayalign"

# Check if rayalign exists
if not Path(rayalign_binary).exists():
    print(f"❌ rayalign binary not found: {rayalign_binary}")
    print("Please build raycloud tools first:")
    print("cd /media/laura/laura_usb/code/fvc_structure/raycloudtools")
    print("mkdir build && cd build")
    print("cmake .. && make -j$(nproc)")
else:
    # Run alignment
    success = align_las_files(source_las, target_las, output_las, rayalign_binary)
    
    if success:
        print(f"\n🎯 Final aligned LAS file: {output_las}")
    else:
        print(f"\n❌ Alignment failed")