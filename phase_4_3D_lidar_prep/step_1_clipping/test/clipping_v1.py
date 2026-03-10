
#%%
## Works for raster
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

# input_las = Path("/media/laura/laura_usb/uas_data/DD0001/lidar/cloud2d765c57b8471e2c.las")
# aoi_raster = Path("/media/laura/Extreme SSD/qgis/calperumResearch/site1_1_DD0001/inputs/predictors/tiles_3072/stacked/tiles_multispectral.26.tif")
# aoi_vector = Path("/media/laura/Extreme SSD/qgis/calperumResearch/site1_1_DD0001/inputs/masks/tiles_3072/raw/fvc_class/annotation_shp/mask_fvc_3072.26.shp")

# input_las = Path("/media/laura/Extreme SSD/qgis/calperumResearch/site2_1_DD0011/lidar/cloud2453cf90763afd91.las")
# input_las = Path("/media/laura/laura_usb/qgis/calperumResearch/site2_1_DD0011/lidar/thinned_voxel_auto_3cm/cloud2453cf90763afd91_cropped_only.las")
# input_las = Path("/media/laura/laura_usb/qgis/calperumResearch/site2_1_DD0011/lidar/thinned_voxel_auto_3cm/SASMDD0011_rgb_pc_thin20mm_from_ref.las")

##low
# input_las = Path("/media/laura/Extreme SSD/qgis/calperumResearch/site2_1_DD0011/lidar/cloud2453cf90763afd91.las")
# input_las = Path("/media/laura/laura_usb/qgis/calperumResearch/site2_1_DD0011/lidar/20220517_SASMDD0011_AusGeoid20.las") #with AHD
# # input_las = Path("/media/laura/laura_usb/qgis/calperumResearch/site2_1_DD0011/lidar_l2/cloud3358f88f2d9dc868.las")
# aoi_raster = Path("/media/laura/Extreme SSD/qgis/calperumResearch/site2_1_DD0011/inputs/predictors/tiles_3072/stacked/tiles_multispectral.33.tif")
# aoi_vector = Path("/media/laura/Extreme SSD/qgis/calperumResearch/site2_1_DD0011/inputs/masks/tiles_3072/raw/fvc_class/annotation_shp/mask_fvc_3072.33.shp")

##medium
# input_las = Path("/media/laura/Extreme SSD/qgis/calperumResearch/site1_1_DD0001/lidar/cloud1457eb47b07af181.las") #L1
input_las = Path("/media/laura/laura_usb/qgis/calperumResearch/site1_1_DD0001/lidar/20220519_SASMDD0001_triple_AusGeoid.las") #L1 with AHD
# input_las = Path("/media/laura/laura_usb/qgis/calperumResearch/site1_1_DD0001/lidar_l2/cloud2d765c57b8471e2c.las") #L2
aoi_raster = Path("/media/laura/Extreme SSD/qgis/calperumResearch/site1_1_DD0001/inputs/predictors/tiles_3072/stacked/tiles_multispectral.26.tif") #26, 109 tile
aoi_vector = Path("/media/laura/Extreme SSD/qgis/calperumResearch/site1_1_DD0001/inputs/masks/tiles_3072/raw/fvc_class/annotation_shp/mask_fvc_3072.26.shp")


# === Outputs ===
##low
# out_dir = Path("/media/laura/laura_usb/qgis/calperumResearch/site2_1_DD0011/lidar/trim")
# out_dir = Path("/media/laura/laura_usb/qgis/calperumResearch/site2_1_DD0011/lidar_l2/trim")\
    
#medium
out_dir = Path("/media/laura/laura_usb/qgis/calperumResearch/site1_1_DD0001/lidar/clipped")
# out_dir = Path("/media/laura/laura_usb/qgis/calperumResearch/site1_1_DD0001/lidar_l2/clipped")

out_dir.mkdir(parents=True, exist_ok=True)
raster_clip_out = out_dir / (input_las.stem + "_tiles_multispectral.26.las")
shp_clip_out    = out_dir / (input_las.stem + "_clip_shp_polygon.las")

# === Get raster CRS + bounds ===
with rasterio.open(aoi_raster) as src:
    raster_crs = CRS.from_wkt(src.crs.to_wkt())
    target_epsg = raster_crs.to_epsg()
    minx, miny, maxx, maxy = src.bounds

# Slightly pad bounds to avoid edge numeric issues
pad = max((maxx - minx), (maxy - miny)) * 1e-9
bounds_str = f"([{minx - pad},{maxx + pad}],[{miny - pad},{maxy + pad}])"

print(f"Raster EPSG: {target_epsg}")
print(f"Raster bounds: {minx:.3f}, {miny:.3f}, {maxx:.3f}, {maxy:.3f}")

# === Read shapefile geometry, unify, reproject → raster CRS, clean ===
with fiona.open(aoi_vector, "r") as src:
    if src.crs_wkt:
        shp_crs = CRS.from_wkt(src.crs_wkt)
    elif src.crs:
        shp_crs = CRS.from_user_input(src.crs)
    else:
        shp_crs = raster_crs
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

minx_p, miny_p, maxx_p, maxy_p = poly.bounds
print(f"AOI polygon area (raster CRS units^2): {poly.area:.3f}")
print(f"AOI bbox: {minx_p:.3f}, {miny_p:.3f}, {maxx_p:.3f}, {maxy_p:.3f}")

# Export WKT (PDAL accepts 'polygon' as WKT)
polygon_wkt = wkt_dumps(poly, rounding_precision=6)

# === Build PDAL pipeline (branch to 2 outputs) ===
stages = []

# 1) Read LAS
stages.append({"type": "readers.las", "filename": str(input_las), "tag": "src"})

# 2) Reproject LAS → raster CRS
stages.append({
    "type": "filters.reprojection",
    "out_srs": f"EPSG:{target_epsg}",
    "inputs": ["src"],
    "tag": "rpj"
})

# 3) Crop to raster bounds (fast pre-clip)
stages.append({
    "type": "filters.crop",
    "bounds": bounds_str,
    "inputs": ["rpj"],
    "tag": "crop_bounds"
})

# 4) Write: clipped to raster extent
stages.append({
    "type": "writers.las",
    "filename": str(raster_clip_out),
    "minor_version": 4,
    "inputs": ["crop_bounds"]
})

# 5) Crop that branch by polygon (precise)
stages.append({
    "type": "filters.crop",
    "polygon": polygon_wkt,   # WKT in same CRS as LAS (raster CRS)
    "inputs": ["crop_bounds"],
    "tag": "crop_poly"
})

# 6) Write: clipped to shapefile polygon
stages.append({
    "type": "writers.las",
    "filename": str(shp_clip_out),
    "minor_version": 4,
    "inputs": ["crop_poly"]
})

pipeline = {"pipeline": stages}
pipe = pdal.Pipeline(json.dumps(pipeline))
cnt = pipe.execute()

print(f"✅ PDAL points processed: {cnt}")
print(f"✅ Raster-extent clip: {raster_clip_out}")
print(f"✅ Shapefile-polygon clip: {shp_clip_out}")

# %%
