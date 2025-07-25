

#%%%
# 1_Convert Point Clouds to DEMs
import os
import json
import subprocess

# Input/Output paths
lidar_path = "/media/laura/Extreme SSD/qgis/calperumResearch/site2_1_DD0011/lidar/cloud2453cf90763afd91.las"
output_dem = "/media/laura/laura_usb/qgis/calperumResearch/site2_1_DD0011/lidar/cloud2453cf90763afd91_dem_1cm.tif"

# Ensure output directory exists
output_dir = os.path.dirname(output_dem)
os.makedirs(output_dir, exist_ok=True)

# Updated pipeline without unsupported CSF arguments
pdal_pipeline = {
    "pipeline": [
        {
            "type": "readers.las",
            "filename": lidar_path
        },
        {
            "type": "filters.csf"  # No parameters — use defaults
        },
        {
            "type": "filters.range",
            "limits": "Classification[2:2]"  # Only ground points
        },
        {
            "type": "writers.gdal",
            "filename": output_dem,
            "resolution": 0.01,
            "output_type": "min",
            "dimension": "Z",
            "data_type": "float32",
            "gdaldriver": "GTiff",
            "nodata": -9999
        }
    ]
}

# Save and run pipeline
pipeline_file = "lidar_dem_pipeline.json"
with open(pipeline_file, "w") as f:
    json.dump(pdal_pipeline, f, indent=4)

print("🔄 Running PDAL pipeline to generate DEM from LiDAR...")
try:
    subprocess.run(["pdal", "pipeline", pipeline_file], check=True)
    print(f"✅ DEM saved to: {output_dem}")
except subprocess.CalledProcessError as e:
    print(f"❌ PDAL pipeline failed with error: {e}")
  
    
# %%
#  DEM Normalization Script: [0, 1] Range
import rasterio
import numpy as np
from rasterio.enums import Resampling
from rasterio import Affine

# === Input/Output Paths ===
input_dem = "/media/laura/laura_usb/qgis/calperumResearch/site2_1_DD0011/lidar/cloud2453cf90763afd91_dem_5cm.tif"
normalized_output = input_dem.replace(".tif", "_normalized.tif")

# === Load DEM ===
with rasterio.open(input_dem) as src:
    profile = src.profile
    dem = src.read(1).astype('float32')
    nodata = profile.get("nodata", -9999)
    mask = (dem != nodata)

    # === Normalize to [0,1] ignoring nodata ===
    valid = dem[mask]
    min_val = valid.min()
    max_val = valid.max()

    norm_dem = np.full_like(dem, nodata, dtype='float32')
    norm_dem[mask] = (valid - min_val) / (max_val - min_val)

    # === Update profile for output ===
    profile.update(dtype='float32', nodata=nodata)

# === Save normalized DEM ===
with rasterio.open(normalized_output, "w", **profile) as dst:
    dst.write(norm_dem, 1)

print(f"✅ Normalized DEM saved to: {normalized_output}")

# %%
import os
import json
import subprocess

# === Input and Output Paths ===
sfm_input = "/media/laura/laura_usb/qgis/calperumResearch/site2_1_DD0011/lidar/sfm_dense/point_cloud_micasense.laz"
output_dsm = "/media/laura/laura_usb/qgis/calperumResearch/site2_1_DD0011/lidar/sfm_dense/point_cloud_micasense_dsm_5cm.tif"

# === Ensure output directory exists ===
os.makedirs(os.path.dirname(output_dsm), exist_ok=True)

# === Define PDAL pipeline ===
pdal_pipeline = {
    "pipeline": [
        {
            "type": "readers.las",
            "filename": sfm_input
        },
        {
            "type": "writers.gdal",
            "filename": output_dsm,
            "resolution": 0.05,           # 5 cm per pixel
            "output_type": "max",        # DSM = max Z per pixel
            "dimension": "Z",
            "data_type": "float32",
            "gdaldriver": "GTiff",
            "nodata": -9999
        }
    ]
}

# === Save pipeline to JSON and run ===
pipeline_file = "sfm_dsm_pipeline.json"
with open(pipeline_file, "w") as f:
    json.dump(pdal_pipeline, f, indent=4)

print("🔄 Running PDAL pipeline to create DSM from dense SfM point cloud...")
try:
    subprocess.run(["pdal", "pipeline", pipeline_file], check=True)
    print(f"✅ DSM saved to: {output_dsm}")
except subprocess.CalledProcessError as e:
    print(f"❌ PDAL pipeline failed with error: {e}")

# %%
