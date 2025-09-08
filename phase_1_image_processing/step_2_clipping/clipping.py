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
#%%
