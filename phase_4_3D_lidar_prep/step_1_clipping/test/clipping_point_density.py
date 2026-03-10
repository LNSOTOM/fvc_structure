
#%%
### with point density estimation and adaptive buffer
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
input_las = Path("/media/laura/laura_usb/qgis/calperumResearch/site1_1_DD0001/lidar/20220519_SASMDD0001_triple_AusGeoid.las")
aoi_raster = Path("/media/laura/laura_usb/qgis/calperumResearch/site1_1_DD0001/inputs/predictors/tiles_3072/stacked/tiles_multispectral.26.tif")
aoi_vector = Path("/media/laura/laura_usb/uas_data/DD0001/imagery/inputs/masks/tiles_3072/raw/structure_class/annotation_shp/mask_fvc_3072.26.shp")


# === Outputs ===
raster_name = aoi_raster.stem  # Extract the raster name (e.g., "tiles_multispectral.26")
out_dir = Path("/media/laura/laura_usb/uas_data/DD0001/lidar/clipped")
out_dir.mkdir(parents=True, exist_ok=True)
shp_clip_out = out_dir / f"{raster_name}.las"  # Use raster name directly for output


# === Get raster CRS and bounds ===
with rasterio.open(aoi_raster) as src:
    raster_crs = CRS.from_wkt(src.crs.to_wkt())
    target_epsg = raster_crs.to_epsg()
    minx, miny, maxx, maxy = src.bounds
    resx, resy = src.res  # Extract raster resolution (pixel size in x and y directions)

# Slightly pad bounds to avoid edge numeric issues
pad = max((maxx - minx), (maxy - miny)) * 1e-9
bounds_str = f"([{minx - pad},{maxx + pad}],[{miny - pad},{maxy + pad}])"

print(f"Raster EPSG: {target_epsg}")
print(f"Raster bounds: {minx:.3f}, {miny:.3f}, {maxx:.3f}, {maxy:.3f}")
print(f"Raster resolution: {resx:.3f} x {resy:.3f}")

# === Estimate Point Density ===
# Build a pre-pipeline to count points within raster bounds
pre_pipeline = {
    "pipeline": [
        {"type": "readers.las", "filename": str(input_las), "tag": "src"},
        {"type": "filters.reprojection", "out_srs": f"EPSG:{target_epsg}", "inputs": ["src"], "tag": "rpj"},
        {"type": "filters.crop", "bounds": bounds_str, "inputs": ["rpj"], "tag": "crop_bounds"},
        {"type": "writers.null"}  # Null writer to count points without writing output
    ]
}

# Execute the pre-pipeline
pre_pipe = pdal.Pipeline(json.dumps(pre_pipeline))
n_points = pre_pipe.execute()  # Number of points within bounds

if n_points <= 0:
    raise ValueError("No points found within raster bounds; cannot estimate point density.")

# Calculate nominal point spacing (m) assuming quasi-uniform sampling
tile_area = (maxx - minx) * (maxy - miny)  # Area of raster bounds in square meters
point_spacing = (tile_area / n_points) ** 0.5  # Nominal spacing = sqrt(area / points)

# Constrain spacing to a reasonable range (e.g., 5 mm to 20 cm)
point_spacing = max(0.005, min(point_spacing, 0.20))

# Adjust buffer size based on point spacing
buffer_size = max(point_spacing, 0.5 * (max(abs(resx), abs(resy))))  # At least half a pixel size

print(f"Estimated point density: {n_points / tile_area:.2f} points/m²")
print(f"Nominal point spacing: {point_spacing:.4f} m")
print(f"Adjusted buffer size: {buffer_size:.4f} m")

# === Read shapefile geometry, unify, reproject to raster CRS ===
with fiona.open(aoi_vector, "r") as src:
    shp_crs = CRS.from_wkt(src.crs_wkt) if src.crs_wkt else CRS.from_user_input(src.crs)
    geoms = [shape(feat["geometry"]) for feat in src]

poly = unary_union(geoms)  # Combine all geometries into a single polygon

# Fix geometry, drop Z/M
poly = make_valid(poly).buffer(0)
if not shp_crs == raster_crs:
    transformer = Transformer.from_crs(shp_crs, raster_crs, always_xy=True)
    poly = shp_transform(lambda x, y, z=None: transformer.transform(x, y), poly)
poly = shp_transform(lambda x, y, z=None: (x, y), poly)

if poly.is_empty:
    raise ValueError("AOI polygon is empty after reprojection/validation.")

# Apply buffer to the AOI polygon
poly = poly.buffer(buffer_size)
if poly.is_empty:
    raise ValueError("AOI polygon became empty after applying buffer.")

polygon_wkt = wkt_dumps(poly, rounding_precision=6)

# === Build Final PDAL Pipeline ===
pipeline = {
    "pipeline": [
        {"type": "readers.las", "filename": str(input_las), "tag": "src"},
        {"type": "filters.reprojection", "out_srs": f"EPSG:{target_epsg}", "inputs": ["src"], "tag": "rpj"},
        {"type": "filters.crop", "bounds": bounds_str, "inputs": ["rpj"], "tag": "crop_bounds"},
        {"type": "filters.crop", "polygon": polygon_wkt, "inputs": ["crop_bounds"], "tag": "crop_poly"},
        {
            "type": "writers.las",
            "filename": str(shp_clip_out),
            "major_version": 1,
            "minor_version": 4,
            "a_srs": f"EPSG:{target_epsg}",
            "scale_x": 0.01,
            "scale_y": 0.01,
            "scale_z": 0.01,
            "inputs": ["crop_poly"]
        }
    ]
}

# Execute the final pipeline
pipe = pdal.Pipeline(json.dumps(pipeline))
writer_count = pipe.execute()

print(f"✅ Points processed: {writer_count}")
print(f"✅ Output written to: {shp_clip_out}")
# %%
