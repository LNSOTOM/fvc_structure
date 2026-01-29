import json
from pathlib import Path
from typing import List

import pdal
import rasterio
from shapely.geometry import Polygon
from shapely.ops import transform as shp_transform
from pyproj import CRS, Transformer
from dask.distributed import Client, as_completed
from tqdm import tqdm
import time

from .core import _read_las_crs, _raster_bounds_and_epsg, _raster_bbox_in_las_crs


def _clip_for_raster(input_las: Path, raster: Path, out_path: Path, las_crs: CRS) -> int:
    bounds, target_epsg, raster_crs = _raster_bounds_and_epsg(raster)
    minx_las, miny_las, maxx_las, maxy_las = _raster_bbox_in_las_crs(bounds, raster_crs, las_crs)
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
    return int(cnt)


def clip_many_dask(
    input_las: Path,
    aoi_raster_dir: Path,
    out_dir: Path,
    pattern: str = "tiles_multispectral*.tif",
    skip_existing: bool = True,
    write_signature: bool = True,
) -> List[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    rasters = sorted(aoi_raster_dir.glob(pattern))
    if not rasters:
        return []

    las_crs = _read_las_crs(input_las)

    outputs: List[Path] = []
    start = time.time()
    with Client(processes=False) as client:
        print(f"Dask dashboard: {client.dashboard_link}")
        futures = []
        for raster in rasters:
            out_path = out_dir / f"{raster.stem}.las"
            if skip_existing and out_path.exists():
                # keep outputs list consistent with produced files
                outputs.append(out_path)
                continue
            futures.append(client.submit(_clip_for_raster, input_las, raster, out_path, las_crs))
            outputs.append(out_path)

        successful = skipped = failed = total_points = 0
        with tqdm(total=len(futures), desc="Clipping LAS to tiles") as pbar:
            for fut in as_completed(futures):
                try:
                    pts = fut.result()
                    successful += 1
                    total_points += int(pts)
                except Exception as e:
                    failed += 1
                    print(f"❌ Worker error: {e}")
                pbar.update(1)

    elapsed = time.time() - start
    if write_signature:
        sig = out_dir / "clip_signature.txt"
        try:
            with open(sig, "w") as f:
                f.write(f"Date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Input LAS: {input_las}\n")
                f.write(f"AOI dir: {aoi_raster_dir}\n")
                f.write(f"Pattern: {pattern}\n")
                f.write(f"Output dir: {out_dir}\n")
                f.write(f"Total rasters: {len(rasters)}\n")
                f.write(f"Successful tiles: {successful}\n")
                f.write(f"Skipped tiles: {skipped}\n")
                f.write(f"Failed tiles: {failed}\n")
                f.write(f"Total points processed: {total_points}\n")
                f.write(f"Elapsed time (s): {elapsed:.2f}\n")
            print(f"Signature written: {sig}")
        except Exception as e:
            print(f"Could not write signature: {e}")

    print(f"✅ Done clipping. Successful: {successful}, Failed: {failed}, Elapsed: {elapsed:.2f}s")
    return outputs
