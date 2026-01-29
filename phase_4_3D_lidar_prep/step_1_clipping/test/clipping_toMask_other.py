#%%
# clip with raster mask
import os
import json
from pathlib import Path

import pdal
import rasterio

# === Inputs ===

# 1. The original, unclipped LAS/LAZ file for the tile.
# input_las = Path("/media/laura/laura_usb/uas_data/DD0001/imagery/clipped/raster/tiles_multispectral.109.las")
input_las = Path("/data/calperumResearch/site1_1_DD0001/imagery/clipped/raster/tiles_multispectral.109.las")

#l1
# input_las = Path("/media/laura/laura_usb/uas_data/DD0001/lidar/clipped/coreg/icp/icp_tiles_multispectral.109.las")
# input_las = Path("/data/calperumResearch/site1_1_DD0001/lidar/clipped/raster/coreg/tiles_multispectral.109.las")
#l2
# input_las = Path("/media/laura/laura_usb/uas_data/DD0001/lidar_l2/clipped/raster/coreg/gauss/icp/icp_tiles_multispectral.109.las")


# 2. The reference multispectral raster (5cm resolution).
#    This is used to get the correct CRS and bounding box for processing.
# aoi_raster_5cm = Path("/media/laura/laura_usb/qgis/calperumResearch/site1_1_DD0001/inputs/predictors/tiles_3072/stacked/tiles_multispectral.109.tif")
aoi_raster_5cm = Path("/data/calperumResearch/site1_1_DD0001/imagery/inputs/predictors/tiles_3072/stacked/tiles_multispectral.109.tif")


# 3. The TIF MASK (5cm resolution) providing the structure class values (1-4).
# aoi_mask_tif_5cm = Path("/media/laura/laura_usb/uas_data/DD0001/imagery/inputs/masks/tiles_3072/raw/structure_class/annotation_raster/downsample_5cm_nearest/mask_structure_3072.109.tif")
aoi_mask_tif_5cm = Path("/data/calperumResearch/site1_1_DD0001/imagery/inputs/masks/tiles_3072/structure_class/annotation_raster/downsample_5cm_nearest/mask_structure_3072.109.tif")


# === Outputs ===
raster_name = aoi_raster_5cm.stem
# out_dir = Path("/media/laura/laura_usb/uas_data/DD0001/imagery/clipped/mask")
out_dir = Path("/data/calperumResearch/site1_1_DD0001/imagery/clipped/mask")

#L1
# out_dir = Path("/media/laura/laura_usb/uas_data/DD0001/lidar/clipped/mask/coreg/icp")
#L2
# out_dir = Path("/media/laura/laura_usb/uas_data/DD0001/lidar_l2/clipped/mask/coreg/icp")


out_dir.mkdir(parents=True, exist_ok=True)
# The final output is a classified .las file with raw Z values.
output_classified_las = out_dir / f"mask_{raster_name}.las"

# === Get raster CRS and bounds from the 5cm reference raster ===
with rasterio.open(aoi_raster_5cm) as src:
    target_epsg = src.crs.to_epsg()
    minx, miny, maxx, maxy = src.bounds

bounds_str = f"([{minx},{maxx}],[{miny},{maxy}])"
print(f"Target EPSG: {target_epsg}")
print(f"Processing bounds defined by 5cm raster.")

# === Build the PDAL Pipeline: Classify, Filter, and Save as LAS ===
# This pipeline does NOT modify the Z (elevation) values.
pipeline = {
    "pipeline": [
        # STAGE 1: Read the source LAS file.
        {
            "type": "readers.las",
            "filename": str(input_las)
        },
        # STAGE 2: Reproject points to match the target raster CRS.
        # {
        #     "type": "filters.reprojection",
        #     "out_srs": f"EPSG:{target_epsg}"
        # },
        # STAGE 3: Fast crop to the bounding box of the 5cm raster.
        # {
        #     "type": "filters.crop",
        #     "bounds": bounds_str
        # },
        # STAGE 4: Assign the class value from the 5cm TIF mask to each point.
        # This assigns the mask's pixel value to the standard 'Classification' dimension.
        {
            "type": "filters.colorization",
            "raster": str(aoi_mask_tif_5cm),
            "dimensions": "Classification:64"
        },
        # STAGE 5: Keep only the points that were successfully classified (values 1-4).
        # {
        #     "type": "filters.range",
        #     "limits": "Classification[64:68]" # This should match your classes
        # },
        # STAGE 6: Write the final classified point cloud.
        # The raw Z dimension is automatically passed through and saved.
        {
            "type": "writers.las",
            "filename": str(output_classified_las),
            "minor_version": 4,
            "extra_dims": "all" # Ensures the 'Classification' dimension is saved
        }
    ]
}

# Execute the PDAL pipeline
print("\nExecuting PDAL pipeline to create classified LAS file...")
pipe = pdal.Pipeline(json.dumps(pipeline))
point_count = pipe.execute()

print(f"\n✅ PDAL processing complete.")
print(f"   Points written to new LAS file: {point_count}")
print(f"   Final classified LAS file saved to: {output_classified_las}")
print(f"   This file retains the original, raw Z-values for each point.")

# %%
