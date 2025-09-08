#%%
## Works for raster extent clipping only
import json
from pathlib import Path
import pdal
import rasterio
from pyproj import CRS

# === Inputs ===

# Example LAS (replace with your file)
input_las = Path("/media/laura/laura_usb/qgis/calperumResearch/site1_1_DD0001/lidar/20220519_SASMDD0001_triple_AusGeoid.las")

# Example raster AOI
aoi_raster = Path("/media/laura/Extreme SSD/qgis/calperumResearch/site1_1_DD0001/inputs/predictors/tiles_3072/stacked/tiles_multispectral.26.tif")

# === Outputs ===
out_dir = Path("/media/laura/laura_usb/qgis/calperumResearch/site1_1_DD0001/lidar/clipped")
out_dir.mkdir(parents=True, exist_ok=True)

raster_clip_out = out_dir / (input_las.stem + "_tiles_multispectral.26.las")

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

# === Build PDAL pipeline ===
stages = [
    # 1) Read LAS
    {"type": "readers.las", "filename": str(input_las), "tag": "src"},

    # 2) Reproject LAS → raster CRS
    {
        "type": "filters.reprojection",
        "out_srs": f"EPSG:{target_epsg}",
        "inputs": ["src"],
        "tag": "rpj"
    },

    # 3) Crop to raster bounds (fast pre-clip)
    {
        "type": "filters.crop",
        "bounds": bounds_str,
        "inputs": ["rpj"],
        "tag": "crop_bounds"
    },

    # 4) Write: clipped to raster extent
    {
        "type": "writers.las",
        "filename": str(raster_clip_out),
        "minor_version": 4,
        "inputs": ["crop_bounds"]
    }
]

pipeline = {"pipeline": stages}
pipe = pdal.Pipeline(json.dumps(pipeline))
cnt = pipe.execute()

print(f"✅ PDAL points processed: {cnt}")
print(f"✅ Raster-extent clip: {raster_clip_out}")
<<<<<<< HEAD
#%%
=======
print(f"✅ Shapefile-polygon clip: {shp_clip_out}")

# %%
### keep normal
### keep normal (robust + verified)
import json
from pathlib import Path
import subprocess

import pdal
import rasterio
import fiona
from shapely.geometry import shape
from shapely.ops import unary_union, transform as shp_transform
from shapely.validation import make_valid
from shapely.wkt import dumps as wkt_dumps
from pyproj import CRS, Transformer

# === Inputs ===
# input_las = Path("/media/laura/Extreme SSD/qgis/calperumResearch/site1_1_DD0001/lidar/cloud1457eb47b07af181.las")
input_las = Path("/media/laura/laura_usb/uas_data/DD0001/lidar/cloud2d765c57b8471e2c.las")
aoi_raster = Path("/media/laura/Extreme SSD/qgis/calperumResearch/site1_1_DD0001/inputs/predictors/tiles_3072/stacked/tiles_multispectral.26.tif")
aoi_vector = Path("/media/laura/Extreme SSD/qgis/calperumResearch/site1_1_DD0001/inputs/masks/tiles_3072/raw/fvc_class/annotation_shp/mask_fvc_3072.26.shp")

# === Outputs ===
# out_dir = Path("/media/laura/Laura 102/uas_data/site1_1_DD0001/lidar/trim/normals")
out_dir = Path("/media/laura/laura_usb/uas_data/DD0001/lidar/trim")
out_dir.mkdir(parents=True, exist_ok=True)
raster_clip_out = out_dir / (input_las.stem + "_clip_raster_extent.las")
shp_clip_out    = out_dir / (input_las.stem + "_clip_shp_polygon.las")

# === Get raster CRS + bounds ===
with rasterio.open(aoi_raster) as src:
    raster_crs = CRS.from_wkt(src.crs.to_wkt())
    target_epsg = raster_crs.to_epsg()
    minx, miny, maxx, maxy = src.bounds

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
poly = make_valid(poly).buffer(0)  # fix geometry
if shp_crs != raster_crs:
    transformer = Transformer.from_crs(shp_crs, raster_crs, always_xy=True)
    poly = shp_transform(lambda x, y, z=None: transformer.transform(x, y), poly)
poly = shp_transform(lambda x, y, z=None: (x, y), poly)
poly = poly.buffer(0.05)  # small inclusion buffer

if poly.is_empty:
    raise ValueError("AOI polygon is empty after reprojection/validation.")

minx_p, miny_p, maxx_p, maxy_p = poly.bounds
print(f"AOI area: {poly.area:.3f}  bbox: {minx_p:.3f},{miny_p:.3f},{maxx_p:.3f},{maxy_p:.3f}")

polygon_wkt = wkt_dumps(poly, rounding_precision=6)

# === Build PDAL pipeline (bounds clip + normals) -> (polygon clip + normals) ===
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

# 3) Crop to raster bounds
stages.append({
    "type": "filters.crop",
    "bounds": bounds_str,
    "inputs": ["rpj"],
    "tag": "crop_bounds"
})

# 4) Normals for raster-extent branch
stages.append({
    "type": "filters.normal",
    "knn": 16,
    "inputs": ["crop_bounds"],
    "tag": "norm_bounds"
})

# 5) Write: raster extent (preserve structure + extras)
stages.append({
    "type": "writers.las",
    "filename": str(raster_clip_out),
    "forward": "all",
    "extra_dims": "all",
    "inputs": ["norm_bounds"]
})

# 6) Polygon crop from bounds branch
stages.append({
    "type": "filters.crop",
    "polygon": polygon_wkt,
    "inputs": ["crop_bounds"],
    "tag": "crop_poly"
})

# 7) Normals for polygon branch
stages.append({
    "type": "filters.normal",
    "knn": 16,
    "inputs": ["crop_poly"],
    "tag": "norm_poly"
})

# 8) Write: polygon clip (preserve structure + extras)
stages.append({
    "type": "writers.las",
    "filename": str(shp_clip_out),
    "forward": "all",
    "extra_dims": "all",
    "inputs": ["norm_poly"]
})

pipeline = {"pipeline": stages}

# === Execute with verbose logging so we can see PDAL errors if any ===
try:
    pipe = pdal.Pipeline(json.dumps(pipeline))
    cnt = pipe.execute()
    print(f"✅ PDAL points processed: {cnt}")
except RuntimeError as e:
    # Some PDAL builds put useful detail only in CLI verbose; dump JSON & run CLI
    pj = out_dir / "debug_normals_pipeline.json"
    pj.write_text(json.dumps(pipeline, indent=2))
    print(f"⚠️ Pipeline failed in bindings. Saved JSON: {pj}")
    print("Re-running via CLI for detailed error...")
    res = subprocess.run(
        ["pdal", "pipeline", str(pj), "--verbose=6"],
        text=True, capture_output=True
    )
    print("----- PDAL STDOUT -----\n", res.stdout)
    print("----- PDAL STDERR -----\n", res.stderr)
    res.check_returncode()  # will raise with full stderr if non-zero

print(f"✅ Raster-extent clip (with normals): {raster_clip_out}")
print(f"✅ Shapefile-polygon clip (with normals): {shp_clip_out}")

# === Verify normals actually exist in outputs ===
try:
    import laspy
    for p in (raster_clip_out, shp_clip_out):
        las = laspy.read(str(p))
        dims = set(las.point_format.dimension_names)
        has_normals = {"NormalX","NormalY","NormalZ"}.issubset(dims)
        print(f"{p.name}: normals present? {has_normals}  dims={sorted(dims)[:12]} ...")
except Exception as _:
    print("laspy not available for verification; skipping dim check.")

# %%
##improved
# keep normal (robust + verified) — GDAL-less PDAL + Python drape
import json
from pathlib import Path
import subprocess

import pdal
import rasterio
import fiona
from shapely.geometry import shape
from shapely.ops import unary_union, transform as shp_transform
from shapely.validation import make_valid
from shapely.wkt import dumps as wkt_dumps
from pyproj import CRS, Transformer

# === Inputs ===
input_las = Path("/media/laura/Extreme SSD/qgis/calperumResearch/site1_1_DD0001/lidar/cloud1457eb47b07af181.las")
aoi_raster = Path("/media/laura/Extreme SSD/qgis/calperumResearch/site1_1_DD0001/inputs/predictors/tiles_3072/stacked/tiles_multispectral.26.tif")
aoi_vector = Path("/media/laura/Extreme SSD/qgis/calperumResearch/site1_1_DD0001/inputs/masks/tiles_3072/raw/fvc_class/annotation_shp/mask_fvc_3072.26.shp")

# === Outputs ===
out_dir = Path("/media/laura/Laura 102/uas_data/site1_1_DD0001/lidar/trim/normals")
out_dir.mkdir(parents=True, exist_ok=True)
raster_clip_out = out_dir / (input_las.stem + "_clip_raster_extent.las")
shp_clip_out    = out_dir / (input_las.stem + "_clip_shp_polygon.las")

# Draped outputs + normals-after-drape
raster_clip_draped = raster_clip_out.with_name(raster_clip_out.stem + "_draped.las")
shp_clip_draped    = shp_clip_out.with_name(shp_clip_out.stem + "_draped.las")
raster_clip_draped_norm = raster_clip_draped.with_name(raster_clip_draped.stem + "_normals.las")
shp_clip_draped_norm    = shp_clip_draped.with_name(shp_clip_draped.stem + "_normals.las")

# === Get raster CRS + bounds (+ nodata) ===
with rasterio.open(aoi_raster) as src:
    raster_crs = CRS.from_wkt(src.crs.to_wkt())
    target_epsg = raster_crs.to_epsg()
    minx, miny, maxx, maxy = src.bounds
    raster_nodata = src.nodata

pad = max((maxx - minx), (maxy - miny)) * 1e-9
bounds_str = f"([{minx - pad},{maxx + pad}],[{miny - pad},{maxy + pad}])"

print(f"Raster EPSG: {target_epsg}")
print(f"Raster bounds: {minx:.3f}, {miny:.3f}, {maxx:.3f}, {maxy:.3f}")
print(f"Raster NoData: {raster_nodata}")

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
poly = make_valid(poly).buffer(0)
if shp_crs != raster_crs:
    transformer = Transformer.from_crs(shp_crs, raster_crs, always_xy=True)
    poly = shp_transform(lambda x, y, z=None: transformer.transform(x, y), poly)
poly = shp_transform(lambda x, y, z=None: (x, y), poly)
poly = poly.buffer(0.05)
if poly.is_empty:
    raise ValueError("AOI polygon is empty after reprojection/validation.")
polygon_wkt = wkt_dumps(poly, rounding_precision=6)

# === PDAL pipeline (no filters.gdal) ===
stages = []
stages.append({"type": "readers.las", "filename": str(input_las), "tag": "src"})
stages.append({"type": "filters.reprojection", "out_srs": f"EPSG:{target_epsg}", "inputs": ["src"], "tag": "rpj"})
stages.append({"type": "filters.crop", "bounds": bounds_str, "inputs": ["rpj"], "tag": "crop_bounds"})

# normals on raster-extent branch (pre-drape)
stages.append({"type": "filters.normal", "knn": 16, "inputs": ["crop_bounds"], "tag": "norm_bounds"})
stages.append({"type": "writers.las", "filename": str(raster_clip_out), "forward": "all", "extra_dims": "all", "inputs": ["norm_bounds"]})

# polygon branch
stages.append({"type": "filters.crop", "polygon": polygon_wkt, "inputs": ["crop_bounds"], "tag": "crop_poly"})
stages.append({"type": "filters.normal", "knn": 16, "inputs": ["crop_poly"], "tag": "norm_poly"})
stages.append({"type": "writers.las", "filename": str(shp_clip_out), "forward": "all", "extra_dims": "all", "inputs": ["norm_poly"]})

pipeline = {"pipeline": stages}

try:
    pipe = pdal.Pipeline(json.dumps(pipeline))
    cnt = pipe.execute()
    print(f"✅ PDAL points processed: {cnt}")
except RuntimeError as e:
    pj = out_dir / "debug_normals_pipeline.json"
    pj.write_text(json.dumps(pipeline, indent=2))
    print(f"⚠️ Pipeline failed in bindings. Saved JSON: {pj}")
    res = subprocess.run(["pdal", "pipeline", str(pj), "--verbose=6"], text=True, capture_output=True)
    print("----- PDAL STDOUT -----\n", res.stdout)
    print("----- PDAL STDERR -----\n", res.stderr)
    res.check_returncode()

print(f"✅ Raster-extent clip (pre-drape normals): {raster_clip_out}")
print(f"✅ Shapefile-polygon clip (pre-drape normals): {shp_clip_out}")

# === Python drape (overwrite Z with raster elevation; keep Z_orig; handle NoData) ===
import laspy
import numpy as np
from rasterio.vrt import WarpedVRT

def drape_las_to_raster(in_las: Path, raster_tif: Path, out_las: Path,
                        keep_original_z: bool = True, drop_nodata: bool = False):
    las = laspy.read(in_las)
    xs, ys, zs = las.x, las.y, las.z
    n = len(xs)

    with rasterio.open(raster_tif) as src:
        # VRT ensures sampling in raster's CRS (we already reprojected LAS to match)
        vrt = WarpedVRT(src, dst_crs=src.crs)
        nodata = src.nodata
        RasterZ = np.empty(n, dtype=np.float32)
        chunk = 1_000_000
        for i0 in range(0, n, chunk):
            i1 = min(i0 + chunk, n)
            coords = np.column_stack([xs[i0:i1], ys[i0:i1]])
            vals = np.array([v for v in vrt.sample(coords)], dtype=np.float32).squeeze()
            RasterZ[i0:i1] = vals

    nodata_mask = np.zeros(n, dtype=bool)
    if nodata is not None:
        nodata_mask = np.isclose(RasterZ, nodata)

    if drop_nodata and nodata is not None and nodata_mask.any():
        keep = ~nodata_mask
        print(f"   Dropping {int(nodata_mask.sum())}/{n} points on raster NoData")
        src_las = laspy.read(in_las)
        las = laspy.LasData(las.header)
        las.points = src_las.points[keep]
        xs, ys, zs = las.x, las.y, las.z
        RasterZ = RasterZ[keep]
        nodata_mask = np.zeros(len(xs), dtype=bool)

    if keep_original_z:
        if "Z_orig" not in las.point_format.dimension_names:
            las.add_extra_dim(laspy.ExtraBytesParams(name="Z_orig", type=np.float64))
        las.Z_orig = zs.astype(np.float64)

    if "RasterZ" not in las.point_format.dimension_names:
        las.add_extra_dim(laspy.ExtraBytesParams(name="RasterZ", type=np.float32))
    las.RasterZ = RasterZ

    newZ = np.where(nodata_mask, zs, RasterZ).astype(las.z.dtype)
    las.z = newZ
    las.write(out_las)
    print(f"✅ Draped LAS written: {out_las} | NoData kept: {int(nodata_mask.sum())}")

# Drape both outputs
drape_las_to_raster(raster_clip_out, aoi_raster, raster_clip_draped, keep_original_z=True, drop_nodata=False)
drape_las_to_raster(shp_clip_out, aoi_raster, shp_clip_draped,    keep_original_z=True, drop_nodata=False)

# === Recompute normals on the *draped* clouds (so normals match new Z) ===
def recompute_normals(in_las: Path, out_las: Path, k=16):
    pipe = {
        "pipeline": [
            {"type":"readers.las", "filename": str(in_las)},
            {"type":"filters.normal", "knn": int(k)},
            {"type":"writers.las", "filename": str(out_las), "forward":"all", "extra_dims":"all"}
        ]
    }
    p = pdal.Pipeline(json.dumps(pipe)); p.execute()

recompute_normals(raster_clip_draped, raster_clip_draped_norm, k=16)
recompute_normals(shp_clip_draped,    shp_clip_draped_norm,    k=16)

print(f"✅ Raster-extent clip (draped + normals): {raster_clip_draped_norm}")
print(f"✅ Shapefile-polygon clip (draped + normals): {shp_clip_draped_norm}")

# === Quick verification ===
try:
    import laspy, numpy as np
    for p in (raster_clip_draped_norm, shp_clip_draped_norm):
        las = laspy.read(str(p))
        dims = set(las.point_format.dimension_names)
        print(f"{p.name}: has normals? { {'NormalX','NormalY','NormalZ'}.issubset(dims) } | has RasterZ? { 'RasterZ' in dims } | has Z_orig? { 'Z_orig' in dims }")
        if "RasterZ" in dims:
            dz = las.z - las.RasterZ
            print(f"   Z - RasterZ (min/mean/max): {float(np.nanmin(dz)):.3e} / {float(np.nanmean(dz)):.3e} / {float(np.nanmax(dz)):.3e}")
except Exception:
    pass


>>>>>>> 25d0e1f (phase1_data processing)
