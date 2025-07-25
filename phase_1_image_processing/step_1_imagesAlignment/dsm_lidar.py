

#%%
import pdal
import json

pipeline_dict = {
    "pipeline": [
        "/media/laura/Extreme SSD/qgis/calperumResearch/site2_1_DD0011/lidar/cloud2453cf90763afd91.las",
        {
            "filename": "/media/laura/Extreme SSD/qgis/calperumResearch/site2_1_DD0011/lidar/cloud2453cf90763afd91_dsm_3cm.tif",
            "gdaldriver": "GTiff",
            "output_type": "max",      # Use "max" for DSM, "min" for DTM
            "dimension": "Z",          # Ensure this is specified
            "resolution": "0.03",      # 3 cm resolution
            "data_type": "float32",
            "nodata": -9999,
            "type": "writers.gdal"
        }
    ]
}

pipeline = pdal.Pipeline(json.dumps(pipeline_dict))
pipeline.execute()

print("✅ DSM written using simplified PDAL JSON-style pipeline.")


# %%
# 1_Convert Point Clouds to DSMs
import os
import json
import subprocess

# Input/Output paths
lidar_path = "/media/laura/Extreme SSD/qgis/calperumResearch/site2_1_DD0011/lidar/cloud2453cf90763afd91.las"
output_dsm = "/media/laura/laura_usb/qgis/calperumResearch/site2_1_DD0011/lidar/cloud2453cf90763afd91_dsm_3cm.tif"

# Ensure output directory exists
output_dir = os.path.dirname(output_dsm)
os.makedirs(output_dir, exist_ok=True)

# Updated pipeline without unsupported CSF arguments
pdal_pipeline = {
     "pipeline": [
        {
            "type": "readers.las",
            "filename": lidar_path
        },
        {
            "type": "writers.gdal",
            "filename": output_dsm,
            "resolution": 0.03,
            "output_type": "max",  # DSM
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

print("🔄 Running PDAL pipeline to generate DSM from LiDAR...")
try:
    subprocess.run(["pdal", "pipeline", pipeline_file], check=True)
    print(f"✅ DEM saved to: {output_dsm}")
except subprocess.CalledProcessError as e:
    print(f"❌ PDAL pipeline failed with error: {e}")
  
    
# %%
