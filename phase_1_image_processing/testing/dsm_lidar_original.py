##TEST 2: GOOD
import subprocess
import json
import sys
import os

def create_dsm(las_input, dsm_output, resolution=0.03):
    print(f"--- Generating DSM for {las_input} ---")
    print(f"Resolution: {resolution} meters ({resolution*100} cm)")

    pipeline = {
        "pipeline": [
            {
                "type": "readers.las",
                "filename": las_input
            },
            {
                "type": "writers.gdal",
                "filename": dsm_output,
                "resolution": resolution,
                # // This parameter tells PDAL how to calculate a value for a cell
                # // that *has* points. "max" takes the highest point's Z value.
                "output_type": "max", #For each grid cell that contains one or more LiDAR points, it assigns the maximum Z value from those points to the cell
                # // This is the value assigned to any grid cell that has NO points
                # // falling within its boundary.
                "nodata": -9999.0,
                "gdaldriver": "GTiff",
                "gdalopts": "COMPRESS=LZW"
            }
        ]
    }

    pipeline_json = json.dumps(pipeline)

    print("Executing PDAL pipeline...")
    try:
        subprocess.run(
            ['pdal', 'pipeline', '--stdin'],
            input=pipeline_json,  # Fixed here
            check=True,
            capture_output=True,
            text=True
        )
        print(f"✅ DSM successfully created at: {dsm_output}")

    except FileNotFoundError:
        print("\n--- ERROR ---")
        print("PDAL executable not found. Ensure PDAL is installed and available in your PATH.")
        sys.exit(1)
    except subprocess.CalledProcessError as e:
        print("\n--- PDAL Pipeline Failed ---")
        print("PDAL STDERR:\n", e.stderr)
        sys.exit(1)


if __name__ == "__main__":
    # --- Hard-coded file paths ---
    # The command-line argument parser has been replaced with these variables.
    
    ##lidar L1
    # SITE 1
    # input_path = "/media/laura/laura_usb/uas_data/DD0001/lidar/20220519_SASMDD0001_triple_AusGeoid.las"
    # output_path = "/media/laura/laura_usb/uas_data/DD0001/lidar/dsm/SASMDD0001_l1_dsm_03.tif"
    
    #SITE 11
    # input_path = "/data/calperumResearch/site2_1_DD0011/lidar/20220517_SASMDD0011_AusGeoid20.las"
    # output_path = "/data/calperumResearch/site2_1_DD0011/lidar/dsm/SASMDD0011_l1_dsm_03.tif"
    
    ##lidar L2
    #SITE 1
    # input_path = "/media/laura/laura_usb/uas_data/DD0001/lidar_l2/cloud2d765c57b8471e2c.las"
    # output_path = "/media/laura/laura_usb/uas_data/DD0001/lidar_l2/dsm/SASMDD0001_l2_dsm_03.tif"
    
    #SITE 11
    input_path = "/data/calperumResearch/site2_1_DD0011/lidar_l2/cloud3358f88f2d9dc868.las"
    output_path = "/data/calperumResearch/site2_1_DD0011/lidar_l2/dsm/SASMDD0011_l2_dsm_03.tif"
 
    
    #lidar Riegl VUX-120
    # input_path = "/media/laura/laura_usb/uas_data/DD0001/riegl_vux120/SASMDD0001_riegl_80_crosshatch.laz"
    # output_path = "/media/laura/laura_usb/uas_data/DD0001/riegl_vux120/dsm/SASMDD0001_riegl_dsm_03.tif"


    dsm_resolution = 0.03  # 5cm resolution in meters

    # --- Run the conversion ---
    # First, check if the input file actually exists to provide a clear error.
    if not os.path.exists(input_path):
        print(f"Error: Input file not found at the specified path: {input_path}")
    else:
        # If the file exists, call the function with the hard-coded paths.
        create_dsm(input_path, output_path, dsm_resolution)
        

# --- Generating DSM for /data/calperumResearch/site2_1_DD0011/lidar_l2/cloud3358f88f2d9dc868.las ---
# Resolution: 0.03 meters (3.0 cm)
# Executing PDAL pipeline...
# ✅ DSM successfully created at: /data/calperumResearch/site2_1_DD0011/lidar_l2/dsm/SASMDD0011_l2_dsm_03.tif