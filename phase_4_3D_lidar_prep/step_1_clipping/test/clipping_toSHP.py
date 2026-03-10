
#%%
from pathlib import Path
import subprocess, json
import rasterio
import fiona
from pyproj import CRS

input_las = Path("/media/laura/laura_usb/qgis/calperumResearch/site1_1_DD0001/lidar/20220519_SASMDD0001_triple_AusGeoid.las")
aoi_raster = Path("/media/laura/laura_usb/qgis/calperumResearch/site1_1_DD0001/inputs/predictors/tiles_3072/stacked/tiles_multispectral.26.tif")
aoi_vector = Path("/media/laura/laura_usb/uas_data/DD0001/imagery/inputs/masks/tiles_3072/raw/structure_class/annotation_shp/mask_fvc_3072.26.shp")

# LAS CRS
meta = subprocess.check_output(["pdal", "info", "--metadata", str(input_las)])
las_srs_wkt = json.loads(meta.decode())["metadata"]["srs"].get("wkt")
las_epsg = CRS.from_wkt(las_srs_wkt).to_epsg() if las_srs_wkt else None
print("LAS EPSG:", las_epsg)

# Raster CRS
with rasterio.open(aoi_raster) as src:
    ras_epsg = CRS.from_wkt(src.crs.to_wkt()).to_epsg()
print("Raster EPSG:", ras_epsg)

# Vector CRS
with fiona.open(aoi_vector) as vsrc:
    v_wkt = getattr(vsrc, "crs_wkt", None)
    v_crs = CRS.from_wkt(v_wkt) if v_wkt else CRS.from_user_input(vsrc.crs)
print("Vector EPSG:", v_crs.to_epsg())

# %%
# Clip to tile bounds + apply per-tile XY shift from GeFolki
from pathlib import Path
import json
import subprocess, json as pyjson
import numpy as np
import pdal
import rasterio
from rasterio.windows import from_bounds
from shapely.geometry import Polygon
from shapely.ops import transform as shp_transform
from pyproj import CRS, Transformer

# ==== INPUTS ====
input_las   = Path("/media/laura/laura_usb/qgis/calperumResearch/site1_1_DD0001/lidar/20220519_SASMDD0001_triple_AusGeoid.las")
aoi_raster  = Path("/media/laura/laura_usb/qgis/calperumResearch/site1_1_DD0001/inputs/predictors/tiles_3072/stacked/tiles_multispectral.26.tif")

# GeFolki outputs: displacement fields that warp LiDAR DSM -> RGB DSM
FLOW_U_PATH = Path("/media/laura/laura_usb/qgis/calperumResearch/site1_1_DD0001/flows/flow_u.npy")  # +x pixels (columns)
FLOW_V_PATH = Path("/media/laura/laura_usb/qgis/calperumResearch/site1_1_DD0001/flows/flow_v.npy")  # +y pixels (rows)

# A raster whose grid matches the flow fields (the RGB DSM you registered to)
FLOW_REF_TIF = Path("/media/laura/laura_usb/qgis/calperumResearch/site1_1_DD0001/rasters/rgb_dsm_aligned.tif")

out_dir = Path("/media/laura/laura_usb/uas_data/DD0001/lidar/clipped")
out_dir.mkdir(parents=True, exist_ok=True)
out_path = out_dir / f"{aoi_raster.stem}.las"

# ==== Read raster CRS + bounds (target EPSG) ====
with rasterio.open(aoi_raster) as src:
    raster_crs = CRS.from_wkt(src.crs.to_wkt())
    target_epsg = raster_crs.to_epsg()
    minx, miny, maxx, maxy = src.bounds

# Tiny pad to avoid numeric edge misses
pad = max((maxx - minx), (maxy - miny)) * 1e-9
minx -= pad; miny -= pad; maxx += pad; maxy += pad

# ==== LAS CRS (should be 7854 but we read it robustly) ====
info = subprocess.check_output(["pdal", "info", "--metadata", str(input_las)])
las_wkt = pyjson.loads(info.decode())["metadata"]["srs"].get("wkt")
if not las_wkt:
    raise RuntimeError("LAS has no SRS; set EXPECTED_LAS_EPSG and use override_srs.")
las_crs = CRS.from_wkt(las_wkt)

# ==== Transform tile bbox into LAS CRS & build LAS-space bounds for fast pre-crop ====
if raster_crs != las_crs:
    to_las = Transformer.from_crs(raster_crs, las_crs, always_xy=True)
    x0, y0 = to_las.transform(minx, miny)
    x1, y1 = to_las.transform(maxx, maxy)
    minx_las, miny_las, maxx_las, maxy_las = min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1)
else:
    minx_las, miny_las, maxx_las, maxy_las = minx, miny, maxx, maxy

bounds_las_str = f"([{minx_las},{maxx_las}],[{miny_las},{maxy_las}])"

# ==== Compute per-tile XY shift from GeFolki flow fields ====
# We sample the flow inside the tile footprint in the FLOW_REF_TIF grid, then convert pixels -> meters
with rasterio.open(FLOW_REF_TIF) as ref:
    ref_crs = CRS.from_wkt(ref.crs.to_wkt())
    if ref_crs != raster_crs:
        raise RuntimeError("FLOW_REF_TIF CRS differs from tile raster CRS. Reproject flow or use matching grid.")
    # Window covering the tile bounds in the flow grid
    win = from_bounds(minx, miny, maxx, maxy, transform=ref.transform)
    # Clamp window to array bounds
    win = win.round_offsets().round_lengths()
    rows = range(int(win.row_off), int(win.row_off + win.height))
    cols = range(int(win.col_off), int(win.col_off + win.width))
    # Pixel size in meters (assume square pixels)
    px_w = ref.transform.a
    px_h = -ref.transform.e  # positive meters

# Load the flow arrays (memory-map ok)
u = np.load(FLOW_U_PATH)  # cols (+x)
v = np.load(FLOW_V_PATH)  # rows (+y)
u_tile = u[np.ix_(rows, cols)]
v_tile = v[np.ix_(rows, cols)]

# Robust central tendency (ignore NaNs)
u_med = np.nanmedian(u_tile)
v_med = np.nanmedian(v_tile)

# Convert pixels to meters in the target CRS
dx_m = u_med * px_w
dy_m = v_med * px_h

print(f"Per-tile flow (median): u={u_med:.3f} px, v={v_med:.3f} px  ->  dx={dx_m:.3f} m, dy={dy_m:.3f} m")

# NOTE on sign: These flows are typically defined to warp LiDAR -> RGB.
# If you discover the shift is in the opposite direction, flip signs: dx_m, dy_m = -dx_m, -dy_m

# ==== Build PDAL pipeline ====
pipe_json = {
    "pipeline": [
        # 1) Read LAS (override_srs if needed)
        {"type": "readers.las", "filename": str(input_las), "tag": "src"},

        # 2) Fast pre-crop in LAS CRS using tile bbox transformed to LAS SRS
        {"type": "filters.crop", "bounds": bounds_las_str, "inputs": ["src"], "tag": "crop_las"},

        # 3) Reproject to the raster CRS for output consistency
        {"type": "filters.reprojection", "out_srs": f"EPSG:{target_epsg}", "inputs": ["crop_las"], "tag": "rpj"},

        # 4) Apply per-tile rigid XY shift (meters, in target CRS)
        {
            "type": "filters.transformation",
            "matrix": f"1 0 0 {dx_m}  0 1 0 {dy_m}  0 0 1 0  0 0 0 1",
            "inputs": ["rpj"],
            "tag": "xy_shift"
        },

        # 5) Write corrected tile LAS
        {"type": "writers.las", "filename": str(out_path), "minor_version": 4, "inputs": ["xy_shift"]}
    ]
}

pipe = pdal.Pipeline(json.dumps(pipe_json))
cnt = pipe.execute()
print(f"✅ Points processed: {cnt}")
print(f"✅ Wrote: {out_path}")

# %%
