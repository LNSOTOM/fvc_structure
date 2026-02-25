#%%
# for one file
## Works for raster
import os
import json
from pathlib import Path

import pdal
import rasterio
import fiona
from shapely.geometry import shape
from shapely.ops import unary_union, transform as shp_transform
from shapely.validation import make_valid
from shapely.wkt import dumps as wkt_dumps
from pyproj import CRS, Transformer

# === Inputs ===
#medium - using aligned LiDAR file that accounts for displacement correction
# input_las = Path("/media/laura/laura_usb/uas_data/DD0001/lidar/20220519_SASMDD0001_triple_AusGeoid_coreg.las")
# input_las = Path("/media/laura/laura_usb/uas_data/DD0001/imagery/SASMDD0001_point_cloud_p1.las")
# input_las = Path("/media/laura/laura_usb/uas_data/DD0001/lidar/clipped/coreg/tiles_multispectral_reg.las")
# input_las = Path("/media/laura/laura_usb/uas_data/DD0001/lidar/clipped/raster/tiles_multispectral.26.las")
input_las = Path("/data/calperumResearch/site1_1_DD0001/imagery/clipped/raster/tiles_multispectral.109.las")

# aoi_raster = Path("/media/laura/laura_usb/qgis/calperumResearch/site1_1_DD0001/inputs/predictors/tiles_3072/stacked/tiles_multispectral.26.tif")
# aoi_vector = Path("/media/laura/laura_usb/uas_data/DD0001/imagery/inputs/masks/tiles_3072/raw/structure_class/annotation_shp/mask_fvc_3072.26.shp")

aoi_raster_5cm = Path("/data/calperumResearch/site1_1_DD0001/imagery/inputs/predictors/tiles_3072/stacked/tiles_multispectral.109.tif")
aoi_vector = Path("/data/calperumResearch/site1_1_DD0001/imagery/inputs/masks/tiles_3072/structure_class/annotations_shp/mask_fvc_3072.109.shp")



#low
# input_las = Path("/media/laura/laura_usb/qgis/calperumResearch/site2_1_DD0011/lidar/sfm_dense/point_cloud_p1.laz")
# aoi_raster = Path("/media/laura/laura_usb/qgis/calperumResearch/site1_1_DD0001/inputs/predictors/tiles_3072/stacked/tiles_multispectral.93.tif")
# aoi_vector = Path("/media/laura/laura_usb/uas_data/DD0001/imagery/inputs/masks/tiles_3072/raw/structure_class/annotation_shp/mask_fvc_3072.93.shp")


# === Outputs ===
raster_name = aoi_raster.stem  # Extract the raster name (e.g., "tiles_multispectral.26")
out_dir = Path("/media/laura/laura_usb/uas_data/DD0001/lidar/clipped/coreg")
out_dir.mkdir(parents=True, exist_ok=True)
shp_clip_out = out_dir / f"{raster_name}.las"  # Use raster name directly for output


# === Get raster CRS and bounds ===
with rasterio.open(aoi_raster) as src:
    raster_crs = CRS.from_wkt(src.crs.to_wkt())
    target_epsg = raster_crs.to_epsg()
    minx, miny, maxx, maxy = src.bounds

# Slightly pad bounds to avoid edge numeric issues
pad = max((maxx - minx), (maxy - miny)) * 1e-9
bounds_str = f"([{minx - pad},{maxx + pad}],[{miny - pad},{maxy + pad}])"

print(f"Raster EPSG: {target_epsg}")
print(f"Raster bounds: {minx:.3f}, {miny:.3f}, {maxx:.3f}, {maxy:.3f}")

# === Read shapefile geometry, unify, reproject to raster CRS ===
with fiona.open(aoi_vector, "r") as src:
    shp_crs = CRS.from_wkt(src.crs_wkt) if src.crs_wkt else CRS.from_user_input(src.crs)
    geoms = [shape(feat["geometry"]) for feat in src]

poly = unary_union(geoms)

# Fix geometry, drop Z/M
poly = make_valid(poly).buffer(0)
if not shp_crs == raster_crs:
    transformer = Transformer.from_crs(shp_crs, raster_crs, always_xy=True)
    poly = shp_transform(lambda x, y, z=None: transformer.transform(x, y), poly)
poly = shp_transform(lambda x, y, z=None: (x, y), poly)

# Add a small positive buffer to catch edge points (5 cm)
poly = poly.buffer(0.05)

if poly.is_empty:
    raise ValueError("AOI polygon is empty after reprojection/validation.")

# Export WKT (PDAL accepts 'polygon' as WKT)
polygon_wkt = wkt_dumps(poly, rounding_precision=6)

# === Build PDAL pipeline ===
pipeline = {
    "pipeline": [
        # 1) Read LAS
        {"type": "readers.las", "filename": str(input_las), "tag": "src"},

        # 2) Reproject LAS to raster CRS
        {"type": "filters.reprojection", "out_srs": f"EPSG:{target_epsg}", "inputs": ["src"], "tag": "rpj"},

        # 3) Crop to raster bounds
        {"type": "filters.crop", "bounds": bounds_str, "inputs": ["rpj"], "tag": "crop_bounds"},

        # 4) Crop to shapefile polygon
        {"type": "filters.crop", "polygon": polygon_wkt, "inputs": ["crop_bounds"], "tag": "crop_poly"},

        # 5) Write clipped LAS
        {"type": "writers.las", "filename": str(shp_clip_out), "minor_version": 4, "inputs": ["crop_poly"]}
    ]
}

# Execute pipeline
pipe = pdal.Pipeline(json.dumps(pipeline))
cnt = pipe.execute()

print(f"✅ PDAL points processed: {cnt}")
print(f"✅ Shapefile-polygon clip: {shp_clip_out}")


#%%
## Works for raster multiple files
# --- Clip a single large LAS into per-tile LAS files using raster tiles + matching mask shapefiles ---
# Requirements: pdal, rasterio, fiona, shapely, pyproj, tqdm (optional but nice)

import os
import re
import json
from pathlib import Path

import pdal
import rasterio
import fiona
from shapely.geometry import shape
from shapely.ops import unary_union, transform as shp_transform
from shapely.validation import make_valid
from shapely.wkt import dumps as wkt_dumps
from pyproj import CRS, Transformer
from tqdm import tqdm

# =========================
# CONFIG — EDIT THESE PATHS
# =========================
# Big LAS for the site (source to clip) - using aligned file with displacement correction
#MEDIUM
INPUT_LAS = Path("/media/laura/laura_usb/uas_data/DD0001/imagery/clipped/raster/tiles_multispectral.26.las")
# INPUT_LAS = Path("/media/laura/laura_usb/uas_data/DD0001/lidar/clipped/raster/tiles_multispectral.26.las")
# INPUT_LAS = Path("/media/laura/laura_usb/uas_data/DD0001/lidar/clipped/raster/coreg/tiles_multispectral.26.las")

# Folder with raster tiles (e.g., tiles_multispectral.106.tif, tiles_multispectral.107.tif, …)
RASTER_DIR = Path("/media/laura/laura_usb/qgis/calperumResearch/site1_1_DD0001/inputs/predictors/tiles_3072/stacked")
# Folder with matching mask shapefiles (e.g., mask_fvc_3072.106.shp)
MASK_DIR = Path("/media/laura/laura_usb/uas_data/DD0001/imagery/inputs/masks/tiles_3072/raw/structure_class/annotation_shp")

# # Output folder for per-tile clipped LAS files
OUT_DIR = Path("/media/laura/laura_usb/uas_data/DD0001/imagery/clipped/mask_shp")
# OUT_DIR = Path("/media/laura/laura_usb/uas_data/DD0001/lidar/clipped/mask_shp")
# OUT_DIR = Path("/media/laura/laura_usb/uas_data/DD0001/lidar/clipped/mask_shp/coreg")

OUT_DIR.mkdir(parents=True, exist_ok=True)

#LOW
# INPUT_LAS = Path("/media/laura/laura_usb/qgis/calperumResearch/site2_1_DD0011/lidar/20220517_SASMDD0011_AusGeoid20.las")
# # Folder with raster tiles (e.g., tiles_multispectral.106.tif, tiles_multispectral.107.tif, …)
# RASTER_DIR = Path("/media/laura/laura_usb/qgis/calperumResearch/site2_1_DD0011/inputs/predictors/tiles_3072/stacked")
# # Folder with matching mask shapefiles (e.g., mask_fvc_3072.106.shp)
# MASK_DIR = Path("/media/laura/laura_usb/uas_data/DD0011/imagery/inputs/masks/tiles_3072/raw/structure_class/annotation_shp")

# Output folder for per-tile clipped LAS files
# OUT_DIR = Path("/media/laura/laura_usb/uas_data/DD0011/lidar/clipped")
# OUT_DIR.mkdir(parents=True, exist_ok=True)




# File name patterns (edit if your names differ)
# We extract tile index from the raster name using the last ".<index>" pattern
RASTER_GLOB = "tiles_multispectral.*.tif"  # all your rasters
MASK_TEMPLATE = "mask_fvc_3072.{index}.shp"  # how to find the matching mask by index

# Reprojection / geometry safety knobs
EDGE_PAD_FRAC = 1e-9      # pad bounds to avoid floating-point edge misses
POLY_BUFFER_M = 0.05      # small positive buffer to catch boundary points (meters in projected CRS)
SKIP_IF_EXISTS = True     # skip making output if the target LAS already exists
VERBOSE_ERRORS = True


def extract_tile_index_from_name(name: str) -> str | None:
    """
    Extracts the tile index from a name like 'tiles_multispectral.106.tif' -> '106'
    Uses the last '.<digits>' group before the extension.
    """
    m = re.search(r"\.(\d+)(?:\.[^.]+)?$", name)
    return m.group(1) if m else None


def read_raster_bounds_and_crs(raster_path: Path):
    with rasterio.open(raster_path) as src:
        raster_crs = CRS.from_wkt(src.crs.to_wkt())
        minx, miny, maxx, maxy = src.bounds
    # slight padding to bounds to avoid numeric edge misses
    pad = max((maxx - minx), (maxy - miny)) * EDGE_PAD_FRAC
    bounds_str = f"([{minx - pad},{maxx + pad}],[{miny - pad},{maxy + pad}])"
    return raster_crs, bounds_str, (minx, miny, maxx, maxy)


def read_merged_mask_polygon(mask_path: Path, target_crs: CRS, buffer_m: float) -> str:
    """
    Read all features in the shapefile, union them, make valid, drop Z/M, reproject to target CRS,
    add a small buffer, and return WKT for PDAL.
    """
    with fiona.open(mask_path, "r") as src:
        shp_crs = CRS.from_wkt(src.crs_wkt) if getattr(src, "crs_wkt", None) else CRS.from_user_input(src.crs)
        geoms = [shape(feat["geometry"]) for feat in src]

    if not geoms:
        raise ValueError(f"No geometries in {mask_path}")

    poly = unary_union(geoms)
    poly = make_valid(poly).buffer(0)  # fix, dissolve slivers

    # reproject if needed
    if shp_crs != target_crs:
        transformer = Transformer.from_crs(shp_crs, target_crs, always_xy=True)
        poly = shp_transform(lambda x, y, z=None: transformer.transform(x, y), poly)

    # drop Z/M if any
    poly = shp_transform(lambda x, y, z=None: (x, y), poly)

    # gentle outward buffer for safety at edges
    if buffer_m and buffer_m > 0:
        poly = poly.buffer(buffer_m)

    if poly.is_empty:
        raise ValueError("AOI polygon empty after validation/reprojection.")

    return wkt_dumps(poly, rounding_precision=6)


def build_pdal_pipeline(input_las: Path, target_epsg: int, bounds_str: str, polygon_wkt: str, out_path: Path) -> dict:
    """
    Build PDAL pipeline JSON for: read -> reproject -> crop bounds -> crop polygon -> write
    """
    return {
        "pipeline": [
            {"type": "readers.las", "filename": str(input_las), "tag": "src"},
            {"type": "filters.reprojection", "out_srs": f"EPSG:{target_epsg}", "inputs": ["src"], "tag": "rpj"},
            {"type": "filters.crop", "bounds": bounds_str, "inputs": ["rpj"], "tag": "crop_bounds"},
            {"type": "filters.crop", "polygon": polygon_wkt, "inputs": ["crop_bounds"], "tag": "crop_poly"},
            {"type": "writers.las", "filename": str(out_path), "minor_version": 4, "inputs": ["crop_poly"]},
        ]
    }


def main():
    rasters = sorted(RASTER_DIR.glob(RASTER_GLOB))
    if not rasters:
        print(f"⚠️  No rasters found in {RASTER_DIR} with pattern {RASTER_GLOB}")
        return

    print(f"Found {len(rasters)} raster tiles in {RASTER_DIR}")

    successes, skipped, failures = 0, 0, 0
    for rpath in tqdm(rasters, desc="Clipping LAS per tile", unit="tile"):
        try:
            tile_index = extract_tile_index_from_name(rpath.name)
            if tile_index is None:
                failures += 1
                if VERBOSE_ERRORS:
                    print(f"\n❌ Could not extract tile index from raster name: {rpath.name}")
                continue

            mask_path = MASK_DIR / MASK_TEMPLATE.format(index=tile_index)
            if not mask_path.exists():
                skipped += 1
                print(f"\n⏭️  Mask missing → skip: {mask_path.name}")
                continue

            out_path = OUT_DIR / f"{rpath.stem}.las"
            if SKIP_IF_EXISTS and out_path.exists():
                skipped += 1
                # Uncomment if you want a quieter run:
                # print(f"⏭️  Exists → {out_path.name}")
                continue

            # Read raster CRS + bounds
            raster_crs, bounds_str, (minx, miny, maxx, maxy) = read_raster_bounds_and_crs(rpath)
            target_epsg = raster_crs.to_epsg()

            # Collect polygon WKT in raster CRS
            polygon_wkt = read_merged_mask_polygon(mask_path, raster_crs, POLY_BUFFER_M)

            # Build and run PDAL
            pipeline_dict = build_pdal_pipeline(INPUT_LAS, target_epsg, bounds_str, polygon_wkt, out_path)
            pipe = pdal.Pipeline(json.dumps(pipeline_dict))
            cnt = pipe.execute()

            successes += 1
            print(f"\n✅ {rpath.name} → {out_path.name} | points processed: {cnt}")

        except Exception as e:
            failures += 1
            if VERBOSE_ERRORS:
                print(f"\n💥 Error on {rpath.name}: {e}")

    print("\n=== Summary ===")
    print(f"  ✅ Success: {successes}")
    print(f"  ⏭️  Skipped: {skipped}")
    print(f"  ❌ Failed : {failures}")
    print(f"  Output dir: {OUT_DIR}")


if __name__ == "__main__":
    main()
    
# %%
# clip with raster: GOOD
import os
import json
from pathlib import Path

import pdal
import rasterio

# === Inputs ===

# 1. The original, unclipped LAS/LAZ file for the tile.
# input_las = Path("/media/laura/laura_usb/uas_data/DD0001/imagery/clipped/raster/tiles_multispectral.109.las")
# input_las = Path("/media/laura/laura_usb/uas_data/DD0001/lidar/clipped/raster/tiles_multispectral.26.las")
input_las = Path("/media/laura/laura_usb/uas_data/DD0001/lidar/clipped/coreg/icp/iicp_tiles_multispectral.26.las")

# 2. The reference multispectral raster (5cm resolution).
#    This is used to get the correct CRS and bounding box for processing.
aoi_raster_5cm = Path("/media/laura/laura_usb/qgis/calperumResearch/site1_1_DD0001/inputs/predictors/tiles_3072/stacked/tiles_multispectral.26.tif")

# 3. The TIF MASK (5cm resolution) providing the structure class values (1-4).
aoi_mask_tif_5cm = Path("/media/laura/laura_usb/uas_data/DD0001/imagery/inputs/masks/tiles_3072/raw/structure_class/annotation_raster/downsample_5cm_nearest/mask_structure_3072.26.tif")

# === Outputs ===
raster_name = aoi_raster_5cm.stem
# out_dir = Path("/media/laura/laura_usb/uas_data/DD0001/imagery/clipped/mask")
# out_dir = Path("/media/laura/laura_usb/uas_data/DD0001/lidar/clipped/mask")
out_dir = Path("/media/laura/laura_usb/uas_data/DD0001/lidar/clipped/mask/coreg/icp")

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
        {
            "type": "filters.reprojection",
            "out_srs": f"EPSG:{target_epsg}"
        },
        # STAGE 3: Fast crop to the bounding box of the 5cm raster.
        {
            "type": "filters.crop",
            "bounds": bounds_str
        },
        # STAGE 4: Assign the class value from the 5cm TIF mask to each point.
        # This assigns the mask's pixel value to the standard 'Classification' dimension.
        {
            "type": "filters.colorization",
            "raster": str(aoi_mask_tif_5cm),
            "dimensions": "Classification:65"
        },
        # STAGE 5: Keep only the points that were successfully classified (values 1-4).
        {
            "type": "filters.range",
            "limits": "Classification[65:68]"
        },
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
