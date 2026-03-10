#%%
from pathlib import Path
import json as pyjson
import numpy as np
import rasterio
from rasterio.windows import from_bounds, bounds as win_bounds
from pyproj import CRS
import subprocess, json
import pdal
from scipy.ndimage import zoom

# -------------------
# Inputs (edit me)
# -------------------
LAS_IN   = Path("/media/laura/laura_usb/qgis/calperumResearch/site1_1_DD0001/lidar/20220519_SASMDD0001_triple_AusGeoid.las")
TILE_TIF = Path("/media/laura/laura_usb/qgis/calperumResearch/site1_1_DD0001/inputs/predictors/tiles_3072/stacked/tiles_multispectral.26.tif")
# Use a DSM or any single-band raster that co-registers with your labels (RGB-DSM recommended)
REF_DSM  = TILE_TIF  # <-- replace with RGB DSM tile if you have it

OUT_DIR  = Path("/media/laura/laura_usb/uas_data/DD0001/lidar/clipped/aligned")
OUT_DIR.mkdir(parents=True, exist_ok=True)
LAS_OUT  = OUT_DIR / f"{TILE_TIF.stem}.las"
TMP_DSM  = OUT_DIR / f"{TILE_TIF.stem}_lidar_dsm.tif"

# -------------------
# 1) Read tile bounds & grid
# -------------------
with rasterio.open(TILE_TIF) as src:
    raster_crs = CRS.from_wkt(src.crs.to_wkt())
    transform  = src.transform
    target_epsg = raster_crs.to_epsg()
    # raw tile bounds
    minx, miny, maxx, maxy = src.bounds
    # pixel sizes (positive meters)
    res_x = transform.a
    res_y = -transform.e

# tiny pad to avoid numeric edge misses
pad = max(maxx - minx, maxy - miny) * 1e-9
minx_p, miny_p, maxx_p, maxy_p = minx - pad, miny - pad, maxx + pad, maxy + pad

# snap the (padded) bounds to the tile grid so geotransforms match exactly
win = from_bounds(minx_p, miny_p, maxx_p, maxy_p, transform=transform)
win = win.round_offsets().round_lengths()

# snapped bounds back in map units
minx_g, miny_g, maxx_g, maxy_g = win_bounds(win, transform)
bounds_str = f"([{minx_g},{maxx_g}],[{miny_g},{maxy_g}])"

# handy indices for slicing flow arrays later
row0, col0 = int(win.row_off), int(win.col_off)
nrows, ncols = int(win.height), int(win.width)

# -------------------
# 2) Rasterize LiDAR to DSM on EXACT tile grid (PDAL writers.gdal)
#    We pre-crop in LAS CRS (fast) then reproject
# -------------------
# Discover LAS CRS
meta = subprocess.check_output(["pdal", "info", "--metadata", str(LAS_IN)])
las_wkt = json.loads(meta.decode())["metadata"]["srs"].get("wkt")
if not las_wkt:
    raise RuntimeError("LAS SRS missing; set override_srs if you know it.")
# Build explicit bounds string in TILE CRS (writers.gdal understands 'bounds' when combined with out_srs)
bounds_str = f"([{minx},{maxx}],[{miny},{maxy}])"

# ---- PDAL: rasterize LiDAR -> DSM on the tile grid (no out_srs here) ----
pipe_dsm = {
    "pipeline": [
        {"type": "readers.las", "filename": str(LAS_IN), "tag": "in"},
        {"type": "filters.reprojection", "out_srs": f"EPSG:{target_epsg}", "inputs": ["in"], "tag": "rpj"},
        {
            "type": "writers.gdal",
            "filename": str(TMP_DSM),
            "gdaldriver": "GTiff",
            "dimension": "Z",
            "output_type": "max",
            "resolution": float(res_x),
            "bounds": bounds_str,
            "window_size": 3,
            "radius": 0.0,
            "nodata": -9999.0,
            "data_type": "float32",
        },
    ]
}

cnt = pdal.Pipeline(pyjson.dumps(pipe_dsm)).execute()
print(f"Rasterized LiDAR to DSM: {TMP_DSM} (points: {cnt})")
# -------------------
# 3) Phase correlation: estimate sub-pixel (dx,dy) in pixels
# -------------------
def read_band_as_f32(path):
    with rasterio.open(path) as s:
        arr = s.read(1, masked=True).filled(np.nan).astype(np.float32)
    return arr

# Choose a band from REF_DSM; if RGB use grayscale or NDVI, but here we just read band 1
ref = read_band_as_f32(REF_DSM)
lid = read_band_as_f32(TMP_DSM)

# Normalize & handle NaNs
def zscore(a):
    m = np.nanmean(a); sd = np.nanstd(a)
    return (a - m) / (sd + 1e-6)
ref_n = zscore(ref)
lid_n = zscore(lid)

# Replace NaNs with 0 for FFT
ref_n = np.nan_to_num(ref_n, nan=0.0)
lid_n = np.nan_to_num(lid_n, nan=0.0)

# Resample lid_n to match ref_n dimensions if needed
if ref_n.shape != lid_n.shape:
    scale_factors = (ref_n.shape[0] / lid_n.shape[0], ref_n.shape[1] / lid_n.shape[1])
    lid_n = zoom(lid_n, scale_factors, order=1)  # Bilinear interpolation
    print(f"Resampled LiDAR DSM to match reference DSM dimensions: {lid_n.shape}")

# Phase correlation (FFT cross-power spectrum)
F1 = np.fft.rfft2(ref_n)
F2 = np.fft.rfft2(lid_n)
R = F1 * np.conj(F2)
R /= (np.abs(R) + 1e-12)
r = np.fft.irfft2(R, s=ref_n.shape)

# Peak location
peak = np.unravel_index(np.argmax(r), r.shape)
shp = np.array(ref_n.shape, dtype=float)
# Convert wrap-around peak to signed shift (rows, cols)
shift_rc = np.array(peak, dtype=float)
shift_rc[shift_rc > shp/2] -= shp[shift_rc > shp/2]
# Optional: sub-pixel refinement with quadratic fit around peak (omitted for brevity)

dy_px, dx_px = shift_rc  # rows, cols
print(f"Estimated shift: dx={dx_px:.3f} px, dy={dy_px:.3f} px")

# Convert to meters (target CRS)
dx_m = dx_px * res_x
dy_m = dy_px * res_y
print(f"Estimated shift: dx={dx_m:.3f} m, dy={dy_m:.3f} m")

# -------------------
# 4) Apply rigid shift to LiDAR & write LAS
# -------------------
pipe_fix = {
    "pipeline":[
        {"type":"readers.las","filename":str(LAS_IN),"tag":"in"},
        {"type":"filters.reprojection","out_srs":f"EPSG:{target_epsg}","inputs":["in"],"tag":"rpj"},
        {"type":"filters.crop","bounds":bounds_str,"inputs":["rpj"],"tag":"crop"},
        {
          "type":"filters.transformation",
          "matrix": f"1 0 0 {dx_m}  0 1 0 {dy_m}  0 0 1 0  0 0 0 1",
          "inputs":["crop"], "tag":"shift"
        },
        {"type":"writers.las","filename":str(LAS_OUT),"minor_version":4,"inputs":["shift"]}
    ]
}
cnt2 = pdal.Pipeline(pyjson.dumps(pipe_fix)).execute()
print(f"✅ Wrote corrected LAS: {LAS_OUT}  (points processed: {cnt2})")

# %%
