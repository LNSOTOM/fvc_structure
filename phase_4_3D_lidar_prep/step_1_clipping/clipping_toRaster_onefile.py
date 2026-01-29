#%%
# --- Clip LAS to the raster's bounds (crop in LAS CRS first, then reproject) ---
# Requirements: pdal, rasterio, shapely, pyproj

from pathlib import Path
import json

import pdal
import rasterio
from shapely.geometry import Polygon
from shapely.ops import transform as shp_transform
from shapely.wkt import dumps as wkt_dumps
from pyproj import CRS, Transformer

# === Inputs ===
# input_las = Path("/media/laura/laura_usb/uas_data/DD0001/imagery/SASMDD0001_rgb_pc.las")
# input_las = Path("/media/laura/laura_usb/uas_data/DD0001/lidar/20220519_SASMDD0001_triple_AusGeoid.las")
# input_las = Path("/media/laura/laura_usb/uas_data/DD0001/lidar/20220519_SASMDD0001_triple_AusGeoid_coreg.las")
# input_las = Path("/media/laura/laura_usb/uas_data/DD0001/lidar/20220519_SASMDD0001_triple_AusGeoid_coreg_gauss_3.0.las")
# input_las = Path("/media/laura/laura_usb/uas_data/DD0001/lidar_l2/cloud2d765c57b8471e2c.las")
# input_las = Path("/media/laura/laura_usb/uas_data/DD0001/lidar_l2/cloud2d765c57b8471e2c_coreg.las")
# input_las = Path("/media/laura/laura_usb/uas_data/DD0001/lidar_l2/cloud2d765c57b8471e2c_coreg_bilin2.las")
# input_las = Path("/media/laura/laura_usb/uas_data/DD0001/lidar_l2/cloud2d765c57b8471e2c_coreg_gauss.las")
# input_las = Path("/media/laura/laura_usb/uas_data/DD0001/lidar_l2/cloud2d765c57b8471e2c_coreg_gauss_near33.0.las")
# input_las = Path("/media/laura/laura_usb/uas_data/DD0001/riegl_vux120/SASMDD0001_riegl_80_crosshatch.laz")

# === Inputs === NVIDIA
##site 1
# input_las = Path("/data/calperumResearch/site1_1_DD0001/imagery/SASMDD0001_rgb_pc.las")
# input_las = Path("/data/calperumResearch/site1_1_DD0001/lidar/20220519_SASMDD0001_triple_AusGeoid.las")
# input_las = Path("/data/calperumResearch/site1_1_DD0001/lidar/20220519_SASMDD0001_triple_AusGeoid_coreg.las")

## L1 + site 11
# input_las = Path("/data/calperumResearch/site2_1_DD0011/imagery/SASMDD0011_rgb_pc.las")
# input_las = Path("/data/calperumResearch/site2_1_DD0011/lidar/20220517_SASMDD0011_AusGeoid20.las")
# input_las = Path("/data/calperumResearch/site2_1_DD0011/lidar/20220517_SASMDD0011_AusGeoid20_coreg.las")

## L2 + site 11
input_las = Path("/data/calperumResearch/site2_1_DD0011/lidar_l2/cloud3358f88f2d9dc868.las")
# input_las = Path("/data/calperumResearch/site2_1_DD0011/lidar_l2/cloud3358f88f2d9dc868_coreg.las")


# === AIO ===
# aoi_raster = Path("/media/laura/laura_usb/qgis/calperumResearch/site1_1_DD0001/inputs/predictors/tiles_3072/stacked/tiles_multispectral.109.tif")
# aoi_vector = Path("/media/.../mask_fvc_3072.93.shp")  # OPTIONAL, only if you want polygon crop after reprojection

# === AIO === NVIDIA
#site 1
# aoi_raster = Path("/data/calperumResearch/site1_1_DD0001/imagery/inputs/predictors/tiles_3072/stacked/tiles_multispectral.26.tif")

#site 11
aoi_raster = Path("/data/calperumResearch/site2_1_DD0011/imagery/inputs/predictors/tiles_3072/stacked/tiles_multispectral.33.tif")


# === Output ===
# out_dir = Path("/media/laura/laura_usb/uas_data/DD0001/imagery/clipped/raster")
# out_dir = Path("/media/laura/laura_usb/uas_data/DD0001/lidar/clipped/raster")
# out_dir = Path("/media/laura/laura_usb/uas_data/DD0001/lidar/clipped/raster/coreg")
# out_dir = Path("/media/laura/laura_usb/uas_data/DD0001/lidar/clipped/raster/coreg/gauss")
# out_dir = Path("/media/laura/laura_usb/uas_data/DD0001/lidar_l2/clipped/raster")
# out_dir = Path("/media/laura/laura_usb/uas_data/DD0001/lidar_l2/clipped/raster/coreg")
# out_dir = Path("/media/laura/laura_usb/uas_data/DD0001/lidar_l2/clipped/raster/coreg/bilin")
# out_dir = Path("/media/laura/laura_usb/uas_data/DD0001/lidar_l2/clipped/raster/coreg/gauss33_near")
# out_dir = Path("/media/laura/laura_usb/uas_data/DD0001/riegl_vux120/clipped/raster")

# === Output === NVIDIA
#site1
# out_dir = Path("/data/calperumResearch/site1_1_DD0001/imagery/clipped/raster")
# out_dir = Path("/data/calperumResearch/site1_1_DD0001/lidar/clipped/raster/original")
# out_dir = Path("/data/calperumResearch/site1_1_DD0001/lidar/clipped/raster/coreg")

# L1 + site11
# out_dir = Path("/data/calperumResearch/site2_1_DD0011/imagery/clipped/raster")
# # out_dir = Path("/data/calperumResearch/site2_1_DD0011/lidar/clipped/raster/original")
# out_dir = Path("/data/calperumResearch/site2_1_DD0011/lidar/clipped/raster/coreg")

# L2 + site1
out_dir = Path("/data/calperumResearch/site2_1_DD0011/lidar_l2/clipped/raster/original")
# out_dir = Path("/data/calperumResearch/site2_1_DD0011/lidar_l2/clipped/raster/coreg")

out_dir.mkdir(parents=True, exist_ok=True)
out_path = out_dir / f"{aoi_raster.stem}.las"

# === Read raster CRS + bounds ===
with rasterio.open(aoi_raster) as src:
    raster_crs = CRS.from_wkt(src.crs.to_wkt())
    target_epsg = raster_crs.to_epsg()
    minx, miny, maxx, maxy = src.bounds

# Tiny pad to avoid numeric edge misses
pad = max((maxx - minx), (maxy - miny)) * 1e-9
minx -= pad; miny -= pad; maxx += pad; maxy += pad

# === Determine LAS/native CRS of the input point cloud ===
# PDAL can report SRS, but for a simple script we assume EPSG in the LAS header or known ahead of time.
# If unknown, you can use pdal info to fetch it. Here we do it inline with pdal info:
import subprocess, json as pyjson
info = subprocess.check_output(["pdal", "info", "--metadata", str(input_las)])
las_meta = pyjson.loads(info.decode("utf-8"))
las_srs_wkt = las_meta["metadata"]["srs"].get("wkt", None)
if not las_srs_wkt:
    raise RuntimeError("Could not read LAS SRS from file metadata.")
las_crs = CRS.from_wkt(las_srs_wkt)

# === Transform the raster bounds (envelope) into LAS CRS and build a LAS-CRS bounds string ===
# Build a polygon of the raster bbox in raster CRS
bbox_poly = Polygon([(minx, miny), (maxx, miny), (maxx, maxy), (minx, maxy)])
if raster_crs != las_crs:
    to_las = Transformer.from_crs(raster_crs, las_crs, always_xy=True)
    bbox_poly_las = shp_transform(lambda x, y, z=None: to_las.transform(x, y), bbox_poly)
else:
    bbox_poly_las = bbox_poly

# Get min/max in LAS CRS
minx_las, miny_las, maxx_las, maxy_las = bbox_poly_las.bounds
bounds_las_str = f"([{minx_las},{maxx_las}],[{miny_las},{maxy_las}])"

print(f"Raster CRS EPSG: {target_epsg}")
print(f"Raster bounds: {minx:.3f}, {miny:.3f}, {maxx:.3f}, {maxy:.3f}")
print(f"LAS CRS: {las_crs.to_string()}")
print(f"LAS-space crop bounds: {bounds_las_str}")

# === OPTIONAL: polygon crop after reprojection (uncomment if needed) ===
# from shapely.validation import make_valid
# import fiona
# from shapely.geometry import shape
# from shapely.ops import unary_union
# from pyproj import Transformer
#
# with fiona.open(aoi_vector, "r") as vsrc:
#     shp_crs = CRS.from_wkt(getattr(vsrc, "crs_wkt", None)) if getattr(vsrc, "crs_wkt", None) else CRS.from_user_input(vsrc.crs)
#     geoms = [shape(f["geometry"]) for f in vsrc if f.get("geometry")]
#
# poly = unary_union(geoms)
# poly = make_valid(poly).buffer(0)
# if shp_crs != raster_crs:
#     tx = Transformer.from_crs(shp_crs, raster_crs, always_xy=True)
#     poly = shp_transform(lambda x, y, z=None: tx.transform(x, y), poly)
# poly = shp_transform(lambda x, y, z=None: (x, y), poly).buffer(0.05)  # 5 cm safety buffer
# polygon_wkt = wkt_dumps(poly, rounding_precision=6)

# === Build PDAL pipeline ===
# Order: read -> crop (LAS CRS bounds) -> reproject to raster CRS -> [optional polygon crop] -> write
pipeline = {
    "pipeline": [
        {"type": "readers.las", "filename": str(input_las), "tag": "src"},

        # Pre-crop in LAS/native CRS using transformed raster bounds
        {"type": "filters.crop", "bounds": bounds_las_str, "inputs": ["src"], "tag": "crop_las"},

        # Reproject to the raster CRS for output consistency
        {"type": "filters.reprojection", "out_srs": f"EPSG:{target_epsg}", "inputs": ["crop_las"], "tag": "rpj"},

        # # OPTIONAL polygon crop (uncomment these two lines if using mask polygon after reprojection)
        # {"type": "filters.crop", "polygon": polygon_wkt, "inputs": ["rpj"], "tag": "crop_poly"},
        # {"type": "writers.las", "filename": str(out_path), "minor_version": 4, "inputs": ["crop_poly"]},

        # Bounds-only output:
        {"type": "writers.las", "filename": str(out_path), "minor_version": 4, "inputs": ["rpj"]},
    ]
}

# Execute
pipe = pdal.Pipeline(json.dumps(pipeline))
cnt = pipe.execute()

print(f"✅ PDAL points processed: {cnt}")
print(f"✅ Bounds-only clip written: {out_path}")


# %%
##### for multiple file slower than Dask version
# --- Clip LAS to the raster's bounds (crop in LAS CRS first, then reproject) ---
# Requirements: pdal, rasterio, shapely, pyproj

from pathlib import Path
import json
import subprocess
import json as pyjson

import pdal
import rasterio
from shapely.geometry import Polygon
from shapely.ops import transform as shp_transform
from pyproj import CRS, Transformer


# === Inputs === NVIDIA
input_las = Path("/data/calperumResearch/site1_1_DD0001/imagery/SASMDD0001_rgb_pc.las")
# input_las = Path("/data/calperumResearch/site1_1_DD0001/lidar/20220519_SASMDD0001_triple_AusGeoid.las")
# input_las = Path("/data/calperumResearch/site1_1_DD0001/lidar/20220519_SASMDD0001_triple_AusGeoid_coreg.las")

# Folder containing all stacked tiles (AOIs)
aoi_raster_dir = Path("/data/calperumResearch/site1_1_DD0001/imagery/inputs/predictors/tiles_3072/stacked/")

# === Output === NVIDIA
out_dir = Path("/data/calperumResearch/site1_1_DD0001/imagery/clipped/raster")
# out_dir = Path("/data/calperumResearch/site1_1_DD0001/lidar/clipped/raster/original")
# out_dir = Path("/data/calperumResearch/site1_1_DD0001/lidar/clipped/raster/coreg")
out_dir.mkdir(parents=True, exist_ok=True)

# Optional: only process certain pattern, e.g. tiles_multispectral.*.tif
raster_pattern = "tiles_multispectral*.tif"

# Optional: skip if output already exists
SKIP_EXISTING = True

# === Determine LAS/native CRS of the input point cloud ONCE ===
info = subprocess.check_output(["pdal", "info", "--metadata", str(input_las)])
las_meta = pyjson.loads(info.decode("utf-8"))
las_srs_wkt = las_meta["metadata"]["srs"].get("wkt", None)
if not las_srs_wkt:
    raise RuntimeError("Could not read LAS SRS from file metadata.")
las_crs = CRS.from_wkt(las_srs_wkt)

print(f"LAS CRS: {las_crs.to_string()}")

# === Loop over all AOI rasters in the folder ===
raster_paths = sorted(aoi_raster_dir.glob(raster_pattern))

print(f"Found {len(raster_paths)} AOI rasters in {aoi_raster_dir}")

for aoi_raster in raster_paths:
    # Build output path per tile
    out_path = out_dir / f"{aoi_raster.stem}.las"

    if SKIP_EXISTING and out_path.exists():
        print(f"⏭️  Skipping existing output: {out_path.name}")
        continue

    print("\n------------------------------------------")
    print(f"🔹 Processing raster: {aoi_raster.name}")
    print(f"🔹 Output LAS:        {out_path.name}")

    # === Read raster CRS + bounds ===
    with rasterio.open(aoi_raster) as src:
        raster_crs = CRS.from_wkt(src.crs.to_wkt())
        target_epsg = raster_crs.to_epsg()
        minx, miny, maxx, maxy = src.bounds

    # Tiny pad to avoid numeric edge misses
    pad = max((maxx - minx), (maxy - miny)) * 1e-9
    minx_p = minx - pad
    miny_p = miny - pad
    maxx_p = maxx + pad
    maxy_p = maxy + pad

    # === Transform raster bounds into LAS CRS and build a LAS-CRS bounds string ===
    bbox_poly = Polygon([(minx_p, miny_p), (maxx_p, miny_p),
                         (maxx_p, maxy_p), (minx_p, maxy_p)])

    if raster_crs != las_crs:
        to_las = Transformer.from_crs(raster_crs, las_crs, always_xy=True)
        bbox_poly_las = shp_transform(lambda x, y, z=None: to_las.transform(x, y),
                                      bbox_poly)
    else:
        bbox_poly_las = bbox_poly

    minx_las, miny_las, maxx_las, maxy_las = bbox_poly_las.bounds
    bounds_las_str = f"([{minx_las},{maxx_las}],[{miny_las},{maxy_las}])"

    print(f"  Raster CRS EPSG: {target_epsg}")
    print(f"  Raster bounds:   {minx_p:.3f}, {miny_p:.3f}, {maxx_p:.3f}, {maxy_p:.3f}")
    print(f"  LAS-space crop:  {bounds_las_str}")

    # === Build PDAL pipeline for this AOI ===
    pipeline = {
        "pipeline": [
            {"type": "readers.las",
             "filename": str(input_las),
             "tag": "src"},

            # Pre-crop in LAS/native CRS using transformed raster bounds
            {"type": "filters.crop",
             "bounds": bounds_las_str,
             "inputs": ["src"],
             "tag": "crop_las"},

            # Reproject to the raster CRS for output consistency
            {"type": "filters.reprojection",
             "out_srs": f"EPSG:{target_epsg}",
             "inputs": ["crop_las"],
             "tag": "rpj"},

            # Bounds-only output:
            {"type": "writers.las",
             "filename": str(out_path),
             "minor_version": 4,
             "inputs": ["rpj"]},
        ]
    }

    pipe = pdal.Pipeline(json.dumps(pipeline))
    cnt = pipe.execute()

    print(f"✅ PDAL points processed: {cnt}")
    print(f"✅ Bounds-only clip written: {out_path}")


#%%
### for multiple file with Dask
import os
import time
import json
import subprocess
from pathlib import Path

import pdal
import rasterio
from shapely.geometry import Polygon
from shapely.ops import transform as shp_transform
from pyproj import CRS, Transformer

from dask.distributed import Client, as_completed
from tqdm import tqdm

# ---------------- CONFIG ----------------

# === Inputs === NVIDIA
# L1 + site1
# # INPUT_LAS = Path("/data/calperumResearch/site1_1_DD0001/imagery/SASMDD0001_rgb_pc.las")
# INPUT_LAS = Path("/data/calperumResearch/site1_1_DD0001/lidar/20220519_SASMDD0001_triple_AusGeoid.las")
# # INPUT_LAS = Path("/data/calperumResearch/site1_1_DD0001/lidar/20220519_SASMDD0001_triple_AusGeoid_coreg.las")

# L2 + site1
# INPUT_LAS = Path("/data/calperumResearch/site1_1_DD0001/lidar_l2/cloud2d765c57b8471e2c.las")
INPUT_LAS = Path("/data/calperumResearch/site1_1_DD0001/lidar_l2/cloud2d765c57b8471e2c_coreg.las")

AOI_RASTER_DIR = Path("/data/calperumResearch/site1_1_DD0001/imagery/inputs/predictors/tiles_3072/stacked/")
RASTER_PATTERN = "tiles_multispectral*.tif"

# # === Output === NVIDIA
# L1 + site1
# # OUT_DIR = Path("/data/calperumResearch/site1_1_DD0001/imagery/clipped/raster")
# OUT_DIR = Path("/data/calperumResearch/site1_1_DD0001/lidar/clipped/raster/original")
# # OUT_DIR = Path("/data/calperumResearch/site1_1_DD0001/lidar/clipped/raster/coreg")

# L2 + site1
# OUT_DIR = Path("/data/calperumResearch/site1_1_DD0001/lidar_l2/clipped/raster/original")
OUT_DIR = Path("/data/calperumResearch/site1_1_DD0001/lidar_l2/clipped/raster/coreg")


# L1 + site11
# INPUT_LAS = Path("/data/calperumResearch/site2_1_DD0011/imagery/SASMDD0011_rgb_pc.las")
# INPUT_LAS = Path("/data/calperumResearch/site2_1_DD0011/lidar/20220517_SASMDD0011_AusGeoid20.las")
# INPUT_LAS = Path("/data/calperumResearch/site2_1_DD0011/lidar/20220517_SASMDD0011_AusGeoid20_coreg.las")

## L2 + site 11
# INPUT_LAS = Path("/data/calperumResearch/site2_1_DD0011/lidar_l2/cloud3358f88f2d9dc868.las")
# INPUT_LAS = Path("/data/calperumResearch/site2_1_DD0011/lidar_l2/cloud3358f88f2d9dc868_coreg.las")


# AOI_RASTER_DIR = Path("/data/calperumResearch/site2_1_DD0011/imagery/inputs/predictors/tiles_3072/stacked/")
# RASTER_PATTERN = "tiles_multispectral*.tif"

# === Output === NVIDIA
# L1 + Site 11
# OUT_DIR = Path("/data/calperumResearch/site2_1_DD0011/imagery/clipped/raster")
# OUT_DIR = Path("/data/calperumResearch/site2_1_DD0011/lidar/clipped/raster/original")
# OUT_DIR = Path("/data/calperumResearch/site2_1_DD0011/lidar/clipped/raster/coreg")

# L2 + site 11
# OUT_DIR = Path("/data/calperumResearch/site2_1_DD0011/lidar_l2/clipped/raster/original")
# OUT_DIR = Path("/data/calperumResearch/site2_1_DD0011/lidar_l2/clipped/raster/coreg")


OUT_DIR.mkdir(parents=True, exist_ok=True)

SKIP_EXISTING = True
SIGNATURE_FILE = OUT_DIR / "clip_signature.txt"


def get_las_srs_wkt(las_path: Path) -> str:
    info = subprocess.check_output(["pdal", "info", "--metadata", str(las_path)])
    las_meta = json.loads(info.decode("utf-8"))
    las_srs_wkt = las_meta["metadata"]["srs"].get("wkt", None)
    if not las_srs_wkt:
        raise RuntimeError(f"Could not read LAS SRS from file metadata: {las_path}")
    return las_srs_wkt


def process_clip_task(task_data):
    """
    Single clipping task executed on a Dask worker.
    task_data = (raster_path_str, input_las_str, las_srs_wkt, out_dir_str, skip_existing)
    """
    raster_path_str, input_las_str, las_srs_wkt, out_dir_str, skip_existing = task_data

    raster_path = Path(raster_path_str)
    input_las = Path(input_las_str)
    out_dir = Path(out_dir_str)

    try:
        out_path = out_dir / f"{raster_path.stem}.las"

        if skip_existing and out_path.exists():
            return {"status": "skipped", "file": raster_path.name, "output": out_path.name}

        # ⭐ Debug: show which tile this worker is handling
        print(f"[Worker] Clipping {raster_path.name} -> {out_path.name}")

        las_crs = CRS.from_wkt(las_srs_wkt)

        with rasterio.open(raster_path) as src:
            raster_crs = CRS.from_wkt(src.crs.to_wkt())
            target_epsg = raster_crs.to_epsg()
            minx, miny, maxx, maxy = src.bounds

        pad = max((maxx - minx), (maxy - miny)) * 1e-9
        minx_p = minx - pad
        miny_p = miny - pad
        maxx_p = maxx + pad
        maxy_p = maxy + pad

        bbox_poly = Polygon([(minx_p, miny_p),
                             (maxx_p, miny_p),
                             (maxx_p, maxy_p),
                             (minx_p, maxy_p)])

        if raster_crs != las_crs:
            to_las = Transformer.from_crs(raster_crs, las_crs, always_xy=True)
            bbox_poly_las = shp_transform(lambda x, y, z=None: to_las.transform(x, y),
                                          bbox_poly)
        else:
            bbox_poly_las = bbox_poly

        minx_las, miny_las, maxx_las, maxy_las = bbox_poly_las.bounds
        bounds_las_str = f"([{minx_las},{maxx_las}],[{miny_las},{maxy_las}])"

        pipeline = {
            "pipeline": [
                {"type": "readers.las", "filename": str(input_las), "tag": "src"},
                {"type": "filters.crop", "bounds": bounds_las_str, "inputs": ["src"], "tag": "crop_las"},
                {"type": "filters.reprojection", "out_srs": f"EPSG:{target_epsg}", "inputs": ["crop_las"], "tag": "rpj"},
                {"type": "writers.las", "filename": str(out_path), "minor_version": 4, "inputs": ["rpj"]},
            ]
        }

        pipe = pdal.Pipeline(json.dumps(pipeline))
        cnt = pipe.execute()
        del pipe

        return {
            "status": "success",
            "file": raster_path.name,
            "output": out_path.name,
            "points": int(cnt),
            "epsg": target_epsg,
        }

    except Exception as e:
        return {"status": "error", "file": raster_path.name, "error": str(e)}


def clip_las_to_tiles_with_dask():
    start_time = time.time()

    raster_paths = sorted(AOI_RASTER_DIR.glob(RASTER_PATTERN))
    print(f"Found {len(raster_paths)} AOI rasters in {AOI_RASTER_DIR}")
    print("Tiles:", [rp.name for rp in raster_paths])
    if not raster_paths:
        raise SystemExit("No rasters found, check RASTER_PATTERN and folder.")

    las_srs_wkt = get_las_srs_wkt(INPUT_LAS)
    las_crs = CRS.from_wkt(las_srs_wkt)
    print(f"LAS CRS: {las_crs.to_string()}")

    tasks = [
        (str(rp), str(INPUT_LAS), las_srs_wkt, str(OUT_DIR), SKIP_EXISTING)
        for rp in raster_paths
    ]
    print(f"🚀 Using Dask to process {len(tasks)} clipping tasks in parallel")

    # threads instead of processes: fewer nanny warnings
    with Client(processes=False) as client:
        print(f"Dask dashboard: {client.dashboard_link}")

        futures = [client.submit(process_clip_task, task) for task in tasks]

        successful = skipped = failed = total_points = 0

        with tqdm(total=len(futures), desc="Clipping LAS to tiles") as pbar:
            for future in as_completed(futures):
                result = future.result()
                status = result.get("status", "unknown")

                if status == "success":
                    successful += 1
                    total_points += result.get("points", 0)
                    print(f"✅ {result['file']} → {result['output']} "
                          f"({result['points']} pts, EPSG:{result['epsg']})")
                elif status == "skipped":
                    skipped += 1
                else:
                    failed += 1
                    print(f"❌ Error processing {result.get('file', 'unknown')}: "
                          f"{result.get('error', 'Unknown error')}")

                pbar.update(1)

    elapsed = time.time() - start_time

    with open(SIGNATURE_FILE, "w") as f:
        f.write(f"Date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Input LAS: {INPUT_LAS}\n")
        f.write(f"AOI dir: {AOI_RASTER_DIR}\n")
        f.write(f"Pattern: {RASTER_PATTERN}\n")
        f.write(f"Output dir: {OUT_DIR}\n")
        f.write(f"Total rasters: {len(raster_paths)}\n")
        f.write(f"Successful tiles: {successful}\n")
        f.write(f"Skipped tiles: {skipped}\n")
        f.write(f"Failed tiles: {failed}\n")
        f.write(f"Total points processed: {total_points}\n")
        f.write(f"Elapsed time (s): {elapsed:.2f}\n")

    print(f"\n✅ Done clipping LAS in {elapsed:.2f} seconds")
    print(f"   - Successful: {successful}")
    print(f"   - Skipped:    {skipped}")
    print(f"   - Failed:     {failed}")
    print(f"   - Total points processed: {total_points}")
    print(f"   - Signature:  {SIGNATURE_FILE}")


if __name__ == "__main__":
    clip_las_to_tiles_with_dask()
    

# %%
