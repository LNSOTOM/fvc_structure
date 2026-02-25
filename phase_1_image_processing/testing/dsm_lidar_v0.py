#%%
import pdal
import json
import os
import subprocess

pipeline_dict = {
    "pipeline": [
        "/media/laura/laura_usb/uas_data/DD0001/lidar/cloud2d765c57b8471e2c.las",
        # "/media/laura/Extreme SSD/qgis/calperumResearch/site2_1_DD0011/lidar/cloud2453cf90763afd91.las",
        {
            "filename": "/media/laura/laura_usb/uas_data/DD0001/lidar/dsm/SASMDD0001_l1_dsm_03.tif",
            # "filename": "/media/laura/laura_usb/uas_data/DD0001/lidar/dsm/SASMDD0001_l1_dsm_05.tif",
            # "filename": "/media/laura/Extreme SSD/qgis/calperumResearch/site2_1_DD0011/lidar/cloud2453cf90763afd91_dsm_3cm.tif",
            "gdaldriver": "GTiff",
            # "output_type": "max",      # Use "max" for DSM, "min" for DTM
            # "dimension": "Z",          # Ensure this is specified
            "output_type":"all",
            "resolution": "0.03",      # 3 cm resolution 0.03/ 0.05
            "data_type": "float32",
            "nodata": -9999,
            "window_size": 1,  # Disable interpolation by using a small window size
            "radius": 0,  # Ensure no interpolation radius
            "type": "writers.gdal"
        }
    ]
}

pipeline = pdal.Pipeline(json.dumps(pipeline_dict))
pipeline.execute()

print("✅ DSM written using simplified PDAL JSON-style pipeline.")


# %%
import pdal
import json
import os
import subprocess

# 1_Convert Point Clouds to DSMs
# Input/Output paths
lidar_path = "/media/laura/laura_usb/uas_data/DD0001/lidar/cloud2d765c57b8471e2c.las"
output_dsm = "/media/laura/laura_usb/uas_data/DD0001/lidar/dsm/SASMDD0001_l1_dsm_05.tif"

# lidar_path = "/media/laura/Extreme SSD/qgis/calperumResearch/site2_1_DD0011/lidar/cloud2453cf90763afd91.las"
# output_dsm = "/media/laura/laura_usb/qgis/calperumResearch/site2_1_DD0011/lidar/cloud2453cf90763afd91_dsm_5cm.tif"

# Ensure output directory exists
output_dir = os.path.dirname(output_dsm)
os.makedirs(output_dir, exist_ok=True)

# Updated pipeline for DSM generation with 5cm resolution and interpolation disabled
pdal_pipeline = {
    "pipeline": [
        {
            "type": "readers.las",
            "filename": lidar_path
        },
        {
            "type": "writers.gdal",
            "filename": output_dsm,
            "resolution": 0.05,  # 5cm resolution
            "output_type": "max",  # DSM
            "data_type": "float32",
            "gdaldriver": "GTiff",
            "nodata": -9999,
            "window_size": 1,  # Disable interpolation by using a small window size
            "radius": 0  # Ensure no interpolation radius
        }
    ]
}

# Save and run pipeline
pipeline_file = "lidar_dsm_pipeline.json"
with open(pipeline_file, "w") as f:
    json.dump(pdal_pipeline, f, indent=4)

print("🔄 Running PDAL pipeline to generate DSM from LiDAR...")
try:
    subprocess.run(["pdal", "pipeline", pipeline_file], check=True)
    print(f"✅ DSM saved to: {output_dsm}")
except subprocess.CalledProcessError as e:
    print(f"❌ PDAL pipeline failed with error: {e}")

# %%
##TEST 2: GOOD
import subprocess
import json
import sys
import os

def create_dsm(las_input, dsm_output, resolution=0.05):
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
    
    input_path = "/media/laura/laura_usb/uas_data/DD0001/lidar/20220519_SASMDD0001_triple_AusGeoid.las"
    output_path = "/media/laura/laura_usb/uas_data/DD0001/lidar/dsm/SASMDD0001_l1_dsm_03.tif"
    dsm_resolution = 0.03  # 5cm resolution in meters

    # --- Run the conversion ---
    # First, check if the input file actually exists to provide a clear error.
    if not os.path.exists(input_path):
        print(f"Error: Input file not found at the specified path: {input_path}")
    else:
        # If the file exists, call the function with the hard-coded paths.
        create_dsm(input_path, output_path, dsm_resolution)
# %%
##TEST 2
import subprocess
import json
import sys
import os

def create_dsm(las_input, dsm_output, resolution=0.05):
    """
    Converts a LAS/LAZ file to a DSM GeoTIFF using a PDAL pipeline.

    For a DSM, this function grids the point cloud and uses the maximum
    Z value within each grid cell.

    Args:
        las_input (str): Path to the input LAS or LAZ file.
        dsm_output (str): Path for the output GeoTIFF DSM file.
        resolution (float): The desired resolution of the DSM in the units
                            of the LAS file's coordinate system (e.g., meters).
    """
    print(f"--- Generating DSM for {las_input} ---")
    print(f"Resolution: {resolution} meters ({resolution*100} cm)")

    # PDAL pipeline defined as a Python dictionary, then converted to JSON.
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
                "output_type": "max",      # Use 'max' for DSM
                "nodata": -9999.0,         # Value for empty pixels
                "gdaldriver": "GTiff",
                "gdalopts": "COMPRESS=LZW"
            }
        ]
    }

    pipeline_json = json.dumps(pipeline)

    print("Executing PDAL pipeline...")
    try:
        # Execute the PDAL pipeline from the command line via stdin
        subprocess.run(
            ['pdal', 'pipeline', '--stdin'],
            input=pipeline_json.encode('utf-8'),
            check=True,
            capture_output=True,
            text=True
        )
        print(f"✅ DSM successfully created at: {dsm_output}")

    except FileNotFoundError:
        print("\n--- ERROR ---")
        print("PDAL executable not found. This is a system configuration issue.")
        print("Please ensure PDAL is installed and that the 'pdal' command is available in your system's PATH.")
        sys.exit(1)
    except subprocess.CalledProcessError as e:
        print("\n--- PDAL Pipeline Failed ---")
        print("PDAL STDERR:\n", e.stderr)
        sys.exit(1)


if __name__ == "__main__":
    # --- Hard-coded file paths ---
    # The command-line argument parser has been replaced with these variables.
    
    input_path = "/media/laura/laura_usb/uas_data/DD0001/lidar/20220519_SASMDD0001_triple_AusGeoid.las"
    output_path = "/media/laura/laura_usb/uas_data/DD0001/lidar/dsm/SASMDD0001_l1_dsm_05.tif"
    dsm_resolution = 0.05  # 5cm resolution in meters

    # --- Run the conversion ---
    # First, check if the input file actually exists to provide a clear error.
    if not os.path.exists(input_path):
        print(f"Error: Input file not found at the specified path: {input_path}")
    else:
        # If the file exists, call the function with the hard-coded paths.
        create_dsm(input_path, output_path, dsm_resolution)
# %%
