#!/usr/bin/env python3
# filepath: /media/laura/laura_usb/code/fvc_structure/phase_1_image_processing/step_1_imagesAlignment/scripts_raycloudtool/1_2_alignment_raycloud.py

"""
Complete LAS/LAZ to Raycloud Pipeline
1. Convert LAS/LAZ to PLY using PDAL (with auto-dimension detection)
2. Convert PLY to rayimport-compatible format
3. Prepare files for raycloud alignment workflow
"""

import json
import subprocess
import struct
import numpy as np
from pathlib import Path
import tempfile

def check_pdal():
    """Check if PDAL is available"""
    try:
        result = subprocess.run(['pdal', '--version'], capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✅ PDAL available: {result.stdout.strip()}")
            return True
    except FileNotFoundError:
        print("❌ PDAL not found. Install with: sudo apt install pdal")
    return False

def get_las_dimensions(input_las):
    """Get available dimensions from LAS file using PDAL info"""
    
    print(f"🔍 Detecting dimensions in {Path(input_las).name}...")
    
    # Use pdal info with --schema to get dimension information
    cmd_schema = ['pdal', 'info', str(input_las), '--schema']
    result_schema = subprocess.run(cmd_schema, capture_output=True, text=True)
    
    if result_schema.returncode == 0:
        try:
            schema_info = json.loads(result_schema.stdout)
            if 'schema' in schema_info and 'dimensions' in schema_info['schema']:
                dimensions = schema_info['schema']['dimensions']
                available_dims = [dim['name'] for dim in dimensions]
                
                print(f"   Available dimensions: {', '.join(available_dims)}")
                
                # Build raycloud-compatible dimension list
                raycloud_dims = ['X', 'Y', 'Z']  # Always include coordinates
                
                # Add time if available (essential for trajectory matching)
                if 'GpsTime' in available_dims:
                    raycloud_dims.append('GpsTime')
                
                # Add RGB if available
                if all(dim in available_dims for dim in ['Red', 'Green', 'Blue']):
                    raycloud_dims.extend(['Red', 'Green', 'Blue'])
                    print("   Using RGB color format")
                
                # Add useful attributes for rayimport conversion
                optional_dims = ['Intensity', 'Classification', 'ReturnNumber', 'NumberOfReturns', 'ScanAngleRank']
                for dim in optional_dims:
                    if dim in available_dims:
                        raycloud_dims.append(dim)
                
                dims_str = ','.join(raycloud_dims)
                print(f"   Selected dimensions: {dims_str}")
                return dims_str
                    
        except (json.JSONDecodeError, KeyError) as e:
            print(f"❌ Failed to parse PDAL schema: {e}")
    
    # Fallback for known point formats
    print("   Using fallback dimensions: X,Y,Z,GpsTime,Red,Green,Blue,Intensity,Classification,ScanAngleRank")
    return "X,Y,Z,GpsTime,Red,Green,Blue,Intensity,Classification,ScanAngleRank"

def las_to_ply_pdal(input_las, output_ply):
    """Convert LAS to binary PLY using PDAL"""
    
    print(f"📊 Converting {Path(input_las).name} to PLY format...")
    
    # Auto-detect dimensions
    dims = get_las_dimensions(input_las)
    
    # Create PDAL pipeline for binary PLY
    pipeline = {
        "pipeline": [
            {
                "type": "readers.las",
                "filename": str(input_las)
            },
            {
                "type": "writers.ply",
                "filename": str(output_ply),
                "storage_mode": "binary_little_endian",
                "dims": dims
            }
        ]
    }
    
    # Write pipeline to temporary file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(pipeline, f, indent=2)
        pipeline_file = f.name
    
    try:
        # Run PDAL pipeline
        cmd = ['pdal', 'pipeline', pipeline_file]
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"❌ PDAL conversion failed: {result.stderr}")
            return False
        
        # Validate output
        if Path(output_ply).exists():
            size_mb = Path(output_ply).stat().st_size / (1024 * 1024)
            print(f"✅ PLY created: {size_mb:.1f} MB")
            return True
        
        return False
        
    finally:
        # Cleanup pipeline file
        try:
            Path(pipeline_file).unlink()
        except:
            pass

def analyze_ply_header(ply_file):
    """Analyze PLY header to determine binary structure"""
    
    print(f"🔍 Analyzing PLY header: {Path(ply_file).name}")
    
    properties = []
    vertex_count = 0
    
    with open(ply_file, 'rb') as f:
        while True:
            line = f.readline().decode('ascii').strip()
            
            if line.startswith('element vertex'):
                vertex_count = int(line.split()[-1])
            elif line.startswith('property'):
                parts = line.split()
                if len(parts) >= 3:
                    prop_type = parts[1]
                    prop_name = parts[2]
                    properties.append((prop_name, prop_type))
            elif line == 'end_header':
                break
    
    print(f"   Vertex count: {vertex_count:,}")
    print(f"   Properties: {len(properties)}")
    for name, ptype in properties:
        print(f"     {name}: {ptype}")
    
    return properties, vertex_count

def convert_pdal_ply_to_rayimport(input_ply, output_ply, max_points=50000):
    """Convert PDAL PLY to rayimport-compatible format with smart header analysis"""
    
    print(f"\n🔄 Converting {Path(input_ply).name} to rayimport format...")
    print(f"   Sample size: {max_points:,} points")
    
    # Analyze PLY structure
    properties, vertex_count = analyze_ply_header(input_ply)
    
    if not properties:
        print("❌ Could not read PLY header")
        return False
    
    # Build struct format based on properties
    format_chars = '<'  # little endian
    prop_map = {}
    
    for i, (name, ptype) in enumerate(properties):
        prop_map[name.lower()] = i
        
        if ptype == 'float64' or ptype == 'double':
            format_chars += 'd'
        elif ptype == 'float32' or ptype == 'float':
            format_chars += 'f'
        elif ptype == 'uint16':
            format_chars += 'H'
        elif ptype == 'uint8' or ptype == 'uchar':
            format_chars += 'B'
        elif ptype == 'int32':
            format_chars += 'i'
        else:
            print(f"⚠️  Unknown property type: {ptype} for {name}")
            format_chars += 'f'  # default to float
    
    record_size = struct.calcsize(format_chars)
    print(f"   Binary format: {format_chars}")
    print(f"   Record size: {record_size} bytes")
    
    # Limit sample size
    sample_size = min(max_points, vertex_count)
    print(f"   Processing: {sample_size:,} / {vertex_count:,} points")
    
    # Read and convert data
    converted_data = []
    
    with open(input_ply, 'rb') as f:
        # Skip header
        while True:
            line = f.readline().decode('ascii').strip()
            if line == 'end_header':
                break
        
        # Read binary data
        for i in range(sample_size):
            record = f.read(record_size)
            if len(record) < record_size:
                print(f"   End of file at point {i}")
                break
            
            # Unpack data
            values = struct.unpack(format_chars, record)
            
            # Extract required fields (with fallbacks)
            try:
                x = float(values[prop_map.get('x', 0)])
                y = float(values[prop_map.get('y', 1)])
                z = float(values[prop_map.get('z', 2)])
                
                # Time (prefer GpsTime, fallback to index-based)
                time_idx = prop_map.get('gpstime', prop_map.get('time', 3))
                time = float(values[time_idx]) if time_idx < len(values) else float(i)
                
                # Colors (convert uint16 to uint8)
                red_idx = prop_map.get('red', -1)
                green_idx = prop_map.get('green', -1)
                blue_idx = prop_map.get('blue', -1)
                
                if red_idx >= 0 and red_idx < len(values):
                    red = min(255, int(values[red_idx]) // 256) if values[red_idx] > 255 else int(values[red_idx])
                    green = min(255, int(values[green_idx]) // 256) if green_idx >= 0 and values[green_idx] > 255 else int(values[green_idx]) if green_idx >= 0 else red
                    blue = min(255, int(values[blue_idx]) // 256) if blue_idx >= 0 and values[blue_idx] > 255 else int(values[blue_idx]) if blue_idx >= 0 else red
                else:
                    red = green = blue = 128  # default gray
                
                # Intensity (fallback to ScanAngleRank or default)
                intensity_idx = prop_map.get('intensity', prop_map.get('scananglerank', -1))
                intensity = float(values[intensity_idx]) if intensity_idx >= 0 and intensity_idx < len(values) else 50.0
                
                # Pack in rayimport format
                converted_record = struct.pack('<ffffBBBf', 
                                             np.float32(x), np.float32(y), np.float32(z), np.float32(time),
                                             np.uint8(red), np.uint8(green), np.uint8(blue), 
                                             np.float32(intensity))
                converted_data.append(converted_record)
                
            except (IndexError, ValueError) as e:
                print(f"⚠️  Error processing point {i}: {e}")
                continue
            
            if (i + 1) % 10000 == 0:
                print(f"   Processed {i + 1:,} points")
    
    if not converted_data:
        print("❌ No data converted")
        return False
    
    # Write rayimport-compatible PLY
    rayimport_header = [
        'ply',
        'format binary_little_endian 1.0',
        'comment converted for rayimport compatibility',
        f'element vertex {len(converted_data)}',
        'property float x',
        'property float y',
        'property float z',
        'property float time',
        'property uchar red',
        'property uchar green',
        'property uchar blue',
        'property float intensity',
        'end_header'
    ]
    
    with open(output_ply, 'wb') as f:
        # Write header
        for line in rayimport_header:
            f.write((line + '\n').encode('ascii'))
        
        # Write binary data
        for record in converted_data:
            f.write(record)
    
    size_mb = Path(output_ply).stat().st_size / (1024 * 1024)
    print("✅ Rayimport PLY created")
    print(f"   Output: {output_ply}")
    print(f"   Points: {len(converted_data):,}")
    print(f"   Size: {size_mb:.1f} MB")
    
    return True

def check_input_files(file_paths):
    """Check if input files exist and show info"""
    for file_path in file_paths:
        if Path(file_path).exists():
            size_mb = Path(file_path).stat().st_size / (1024 * 1024)
            print(f"✅ {Path(file_path).name}: {size_mb:.1f} MB")
        else:
            print(f"❌ {Path(file_path).name}: NOT FOUND")

def main():
    """Main pipeline execution"""
    
    print("🚁 Complete LAS/LAZ to Raycloud Pipeline")
    print("=" * 50)
    
    # Input files
    # L1 (2022-05-19) to match the L1 trajectory date
    lidar_path = "/media/laura/laura_usb/qgis/calperumResearch/site1_1_DD0001/lidar/20220519_SASMDD0001_triple_AusGeoid.las"
    # L2
    sfm_path = "/media/laura/laura_usb/qgis/calperumResearch/site1_1_DD0001/lidar_l2/cloud2d765c57b8471e2c.las"
    
    # Output directories
    temp_dir = Path("/media/laura/laura_usb/code/fvc_structure/phase_1_image_processing/step_1_imagesAlignment/output/temp_ply")
    rayimport_dir = Path("/media/laura/laura_usb/code/fvc_structure/phase_1_image_processing/step_1_imagesAlignment/output/rayimport_samples")
    
    temp_dir.mkdir(parents=True, exist_ok=True)
    rayimport_dir.mkdir(parents=True, exist_ok=True)
    
    # Check prerequisites
    if not check_pdal():
        return False
    
    print("\n🔍 Checking input files...")
    check_input_files([lidar_path, sfm_path])
    
    # Step 1: Convert LAS/LAZ to PLY using PDAL
    print("\n📊 Step 1: LAS/LAZ → PLY Conversion")
    print("-" * 30)
    
    lidar_ply = temp_dir / "lidar_l1_cloud.ply"
    sfm_ply = temp_dir / "lidar_l2__cloud.ply"
    
    # Convert LiDAR
    if Path(lidar_path).exists():
        if not las_to_ply_pdal(lidar_path, lidar_ply):
            print("❌ LiDAR conversion failed")
            return False
    
    # Convert SfM
    if Path(sfm_path).exists():
        if not las_to_ply_pdal(sfm_path, sfm_ply):
            print("❌ SfM conversion failed")
            return False
    
    # Step 2: Convert PLY to rayimport format
    print("\n🔄 Step 2: PLY → Rayimport Format")
    print("-" * 30)
    
    lidar_rayimport = rayimport_dir / "lidar_l1_rayimport.ply"
    sfm_rayimport = rayimport_dir / "lidar_l2_rayimport.ply"
    
    # Convert LiDAR PLY
    if lidar_ply.exists():
        convert_pdal_ply_to_rayimport(lidar_ply, lidar_rayimport, max_points=50000)
    
    # Convert SfM PLY
    if sfm_ply.exists():
        convert_pdal_ply_to_rayimport(sfm_ply, sfm_rayimport, max_points=50000)
    
    # Step 3: Show next steps
    print("\n🎯 Next Steps: Raycloud Workflow")
    print("-" * 30)
    
    trajectory_file = "/media/laura/laura_usb/code/fvc_structure/phase_1_image_processing/step_1_imagesAlignment/output/rayimport_samples/trajectory_rayimport.txt"
    
    print("1. Convert trajectory (if not done):")
    print("   python3 2_convert_trajectory.py")
    
    print("\n2. Test rayimport:")
    print("   cd /media/laura/laura_usb/code/fvc_structure/raycloudtools/build")
    
    if lidar_rayimport.exists():
        print(f"   ./bin/rayimport-0.0.1 '{lidar_rayimport}' '{trajectory_file}'")
    
    if sfm_rayimport.exists():
        print(f"   ./bin/rayimport-0.0.1 '{sfm_rayimport}' 473156,6235679,150")
    
    print("\n3. Run alignment:")
    print("   ./bin/rayalign-0.0.1 [sfm_raycloud.ply] [lidar_raycloud.ply] --verbose")
    
    print("\n✅ Pipeline complete!")
    return True

if __name__ == "__main__":
    main()