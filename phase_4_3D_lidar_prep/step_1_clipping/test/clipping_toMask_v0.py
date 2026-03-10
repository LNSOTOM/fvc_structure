#%%
import os
import json
from pathlib import Path
import pdal
import rasterio

# === Step 1: Inputs ===

# 1. The original, unclipped LAS/LAZ file for the tile.
input_las = Path("/media/laura/laura_usb/uas_data/DD0001/lidar/clipped/raster/tiles_multispectral.26.las")

# 2. The reference multispectral raster (5cm resolution) to define the processing area.
aoi_raster_5cm = Path("/media/laura/laura_usb/qgis/calperumResearch/site1_1_DD0001/inputs/predictors/tiles_3072/stacked/tiles_multispectral.26.tif")

# 3. The TIF MASK (5cm resolution) providing the structure class values (1-4).
aoi_mask_tif_5cm = Path("/media/laura/laura_usb/uas_data/DD0001/imagery/inputs/masks/tiles_3072/raw/structure_class/annotation_raster/downsample_5cm_nearest/mask_structure_3072.26.tif")

# === Step 1: Outputs ===
raster_name = aoi_raster_5cm.stem
out_dir = Path("./lidar_processing_steps")
out_dir.mkdir(parents=True, exist_ok=True)
# The output is a classified .las file with raw Z values, ready for Step 2.
output_classified_las = out_dir / f"{raster_name}_step1_classified.las"

# === Get raster CRS and bounds for processing ===
with rasterio.open(aoi_raster_5cm) as src:
    target_epsg = src.crs.to_epsg()
    minx, miny, maxx, maxy = src.bounds
bounds_str = f"([{minx},{maxx}],[{miny},{maxy}])"
print(f"Target EPSG: {target_epsg}")

# === Build the PDAL Pipeline for Step 1 ===
pipeline_step1 = {
    "pipeline": [
        { "type": "readers.las", "filename": str(input_las) },
        { "type": "filters.reprojection", "out_srs": f"EPSG:{target_epsg}" },
        { "type": "filters.crop", "bounds": bounds_str },
        # Assign the class value from the 5cm TIF mask to the 'Classification' dimension.
        {
            "type": "filters.colorization",
            "raster": str(aoi_mask_tif_5cm),
            "dimensions": "Classification:1"
        },
        # Keep only the points that were successfully classified (values 1-4).
        { "type": "filters.range", "limits": "Classification[1:4]" },
        # Write the final classified point cloud with raw Z-values.
        {
            "type": "writers.las",
            "filename": str(output_classified_las),
            "minor_version": 4,
            "extra_dims": "all"
        }
    ]
}

# Execute the pipeline for Step 1
print("\n--- EXECUTING STEP 1: Clipping and Classifying ---")
pipe1 = pdal.Pipeline(json.dumps(pipeline_step1))
point_count1 = pipe1.execute()
print(f"✅ Step 1 complete.")
print(f"   Points written: {point_count1}")
print(f"   Classified LAS file with raw Z-values saved to: {output_classified_las}")

# %%
##test2

import os
import json
from pathlib import Path
import pdal
import rasterio

# === Step 1: Inputs ===

# 1. The original, unclipped LAS/LAZ file for the tile.
input_las = Path("/media/laura/laura_usb/uas_data/DD0001/lidar/clipped/raster/tiles_multispectral.26.las")

# 2. The reference multispectral raster (5cm resolution) to define the processing area.
aoi_raster_5cm = Path("/media/laura/laura_usb/qgis/calperumResearch/site1_1_DD0001/inputs/predictors/tiles_3072/stacked/tiles_multispectral.26.tif")

# 3. The TIF MASK (5cm resolution) providing the structure class values (1-4).
aoi_mask_tif_5cm = Path("/media/laura/laura_usb/uas_data/DD0001/imagery/inputs/masks/tiles_3072/raw/structure_class/annotation_raster/downsample_5cm_nearest/mask_structure_3072.26.tif")

# === Step 1: Outputs ===
raster_name = aoi_raster_5cm.stem
out_dir = Path("./lidar_processing_steps")
out_dir.mkdir(parents=True, exist_ok=True)
# The output is a classified .las file with raw Z values, ready for Step 2.
output_classified_las = out_dir / f"{raster_name}_step1_classified.las"

# === Get raster CRS and bounds for processing ===
with rasterio.open(aoi_raster_5cm) as src:
    target_epsg = src.crs.to_epsg()
    minx, miny, maxx, maxy = src.bounds
bounds_str = f"([{minx},{maxx}],[{miny},{maxy}])"
print(f"Target EPSG: {target_epsg}")

# === Build the PDAL Pipeline for Step 1 ===
pipeline_step1 = {
    "pipeline": [
        { "type": "readers.las", "filename": str(input_las) },
        { "type": "filters.reprojection", "out_srs": f"EPSG:{target_epsg}" },
        { "type": "filters.crop", "bounds": bounds_str },
        # Assign the class value from the 5cm TIF mask to the 'Classification' dimension.
        {
            "type": "filters.colorization",
            "raster": str(aoi_mask_tif_5cm),
            "dimensions": "Classification:1"
        },
        # Add a new dimension to store the structure class.
        # This must be done before we can assign values to it.
        {
            "type": "filters.add_dimension",
            "name": "StructureClass"
          
        },
        # Copy the structure class from 'Classification' to a new 'StructureClass' dimension.
        # This preserves the label before ground classification overwrites 'Classification'.
        {
            "type": "filters.assign",
            "value": "StructureClass = Classification"
        },
        # Keep only the points that were successfully classified (values 1-4).
        { "type": "filters.range", "limits": "Classification[1:4]" },
        # Write the final classified point cloud with raw Z-values.
        {
            "type": "writers.las",
            "a_srs": f"EPSG:{target_epsg}",
            "filename": str(output_classified_las),
            "minor_version": 4,
            "extra_dims": "all"
        }
    ]
}

# Execute the pipeline for Step 1
print("\n--- EXECUTING STEP 1: Clipping and Classifying ---")
pipe1 = pdal.Pipeline(json.dumps(pipeline_step1))
point_count1 = pipe1.execute()
print(f"✅ Step 1 complete.")
print(f"   Points written: {point_count1}")
print(f"   Classified LAS file with raw Z-values saved to: {output_classified_las}")

# %%
