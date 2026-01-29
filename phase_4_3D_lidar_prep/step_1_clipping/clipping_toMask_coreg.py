# %%
# clip with raster mask - Multiple files
import os
import json
from pathlib import Path

import pdal
import rasterio
import laspy            # <-- NEW
from tqdm import tqdm

# === Inputs ===

# site 1
LAS_DIR = Path("/data/calperumResearch/site1_1_DD0001/lidar/clipped/raster/coreg/icp")
RASTER_DIR = Path("/data/calperumResearch/site1_1_DD0001/imagery/inputs/predictors/tiles_3072/stacked")
MASK_DIR = Path("/data/calperumResearch/site1_1_DD0001/imagery/inputs/masks/tiles_3072/structure_class/annotation_raster/downsample_5cm_nearest")
OUT_DIR = Path("/data/calperumResearch/site1_1_DD0001/lidar/clipped/mask/coreg/icp_top")

# site 11 
# LAS_DIR = Path("/data/calperumResearch/site2_1_DD0011/lidar/clipped/raster/coreg/icp")
# RASTER_DIR = Path("/data/calperumResearch/site2_1_DD0011/imagery/inputs/predictors/tiles_3072/stacked")
# MASK_DIR = Path("/data/calperumResearch/site2_1_DD0011/imagery/inputs/masks/tiles_3072/structure_class/annotation_raster/downsample_5cm_nearest")
# OUT_DIR = Path("/data/calperumResearch/site2_1_DD0011/lidar/clipped/mask/coreg/icp_top")

OUT_DIR.mkdir(parents=True, exist_ok=True)

# Find all LAS files
las_files = sorted(LAS_DIR.glob("tiles_multispectral*.las"))

if not las_files:
    print(f"❌ No LAS files found in {LAS_DIR}")
    exit(1)

print(f"Found {len(las_files)} LAS files to process")


# Process each file
for input_las in tqdm(las_files, desc="Processing tiles"):
    # Extract tile ID (e.g., "tiles_multispectral.33.las" -> "33")
    tile_id = input_las.stem.split(".")[-1]

    # Construct corresponding raster and mask paths
    aoi_raster_5cm = RASTER_DIR / f"tiles_multispectral.{tile_id}.tif"
    aoi_mask_tif_5cm = MASK_DIR / f"mask_structure_3072.{tile_id}.tif"

    # Validate files exist
    if not aoi_raster_5cm.exists():
        print(f"⚠️  Skipping tile {tile_id}: raster not found at {aoi_raster_5cm}")
        continue
    if not aoi_mask_tif_5cm.exists():
        print(f"⚠️  Skipping tile {tile_id}: mask not found at {aoi_mask_tif_5cm}")
        continue

    # Output file
    raster_name = aoi_raster_5cm.stem
    output_classified_las = OUT_DIR / f"mask_{raster_name}.las"

    # Skip if already processed
    if output_classified_las.exists():
        print(f"⏩ Skipping tile {tile_id}: output already exists")
        continue

    # --- Preserve original scale/offset ---
    in_las = laspy.read(input_las)
    sx, sy, sz = in_las.header.scales
    ox, oy, oz = in_las.header.offsets
    minor_ver = int(in_las.header.version.minor)

    # === Read raster bounds ===
    with rasterio.open(aoi_raster_5cm) as src:
        target_epsg = src.crs.to_epsg()
        minx, miny, maxx, maxy = src.bounds

    bounds_str = f"([{minx},{maxx}],[{miny},{maxy}])"


    # === Build the PDAL pipeline ===
    pipeline = {
        "pipeline": [
            # Stage 1: read LAS
            {
                "type": "readers.las",
                "filename": str(input_las)
            },

            # Stage 2: crop to raster extent
            {
                "type": "filters.crop",
                "bounds": bounds_str
            },

            # Stage 3: Paint mask pixel onto Classification
            {
                "type": "filters.colorization",
                "raster": str(aoi_mask_tif_5cm),
                "dimensions": "Classification"
            },

            # Stage 4: keep only 64–68
            {
                "type": "filters.range",
                "limits": "Classification[64:68]"
            },

            # Stage 5: write classified LAS
            {
                "type": "writers.las",
                "filename": str(output_classified_las),
                "minor_version": minor_ver,
                "extra_dims": "all",
                "scale_x": sx,
                "scale_y": sy,
                "scale_z": sz,
                "offset_x": ox,
                "offset_y": oy,
                "offset_z": oz,
                "forward": "all"
            }
        ]
    }

    # Execute pipeline
    try:
        pipe = pdal.Pipeline(json.dumps(pipeline))
        n = pipe.execute()
        print(f"✅ Tile {tile_id}: {n:,} points → {output_classified_las.name}")
    except Exception as e:
        print(f"❌ Error processing tile {tile_id}: {e}")

print(f"\n🎉 All processing complete.")
print(f"   Output files retain the original location and raw Z-values for each point.")
