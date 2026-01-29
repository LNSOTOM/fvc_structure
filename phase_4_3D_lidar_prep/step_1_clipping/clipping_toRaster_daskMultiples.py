#%%
# --- Clip LAS to the raster's bounds (crop in LAS CRS first, then reproject) ---
# Requirements: pdal, rasterio, shapely, pyproj

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
