"""
DSM Generation Module

Create Digital Surface Models (DSM) from LAS/LAZ point clouds using PDAL,
with optional Dask parallel execution, Xarray/RioXarray analytics, and
(optional) STAC Item metadata sidecars for each DSM.

Notebook:
    from dsm_lidar import create_dsm, batch_create_dsm_parallel

    create_dsm(
        las_input="/path/to/input.las",
        dsm_output="/path/to/output.tif",
        resolution=0.03
    )
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

# ---------------------------------------------------------------------
# Optional deps
# ---------------------------------------------------------------------
try:
    import dask
    from dask import delayed, compute
    from dask.diagnostics import ProgressBar

    HAS_DASK = True
except Exception:
    HAS_DASK = False

try:
    import xarray as xr
    import rioxarray

    HAS_XARRAY = True
except Exception:
    HAS_XARRAY = False

try:
    import numpy as np

    HAS_NUMPY = True
except Exception:
    HAS_NUMPY = False


PathLike = Union[str, Path]


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
def _run(cmd: List[str], *, stdin: Optional[str] = None, timeout: Optional[int] = None) -> subprocess.CompletedProcess:
    """Run a subprocess command and raise a clean error on failure."""
    try:
        return subprocess.run(
            cmd,
            input=stdin,
            text=True,
            capture_output=True,
            check=True,
            timeout=timeout,
        )
    except FileNotFoundError as e:
        raise RuntimeError(f"Executable not found: {cmd[0]}. Is it installed and in PATH?") from e
    except subprocess.CalledProcessError as e:
        msg = e.stderr.strip() if e.stderr else str(e)
        raise RuntimeError(f"Command failed: {' '.join(cmd)}\n{msg}") from e


def estimate_processing_time(las_input: PathLike, resolution: float) -> Dict[str, float]:
    """
    Estimate DSM processing time and output size using `pdal info --summary`.
    Falls back to a file-size heuristic if summary parsing fails.
    """
    las_input = Path(las_input)
    file_size_mb = las_input.stat().st_size / (1024 * 1024)

    try:
        cp = _run(["pdal", "info", "--summary", str(las_input)], timeout=15)
        info = json.loads(cp.stdout)
        summary = info["summary"]

        x_range = summary["bounds"]["maxx"] - summary["bounds"]["minx"]
        y_range = summary["bounds"]["maxy"] - summary["bounds"]["miny"]
        grid_cols = int(x_range / resolution)
        grid_rows = int(y_range / resolution)
        grid_cells = grid_cols * grid_rows

        # Very rough: adjust for your hardware if you want
        estimated_seconds = (grid_cells / 1_000_000) * 3.0

        return {
            "file_size_mb": float(file_size_mb),
            "grid_cols": float(grid_cols),
            "grid_rows": float(grid_rows),
            "grid_cells": float(grid_cells),
            "estimated_seconds": float(estimated_seconds),
            "point_count": float(summary.get("num_points", 0)),
        }
    except Exception:
        return {
            "file_size_mb": float(file_size_mb),
            "estimated_seconds": float(file_size_mb * 2.0),
        }


# ---------------------------------------------------------------------
# DSM creation (PDAL)
# ---------------------------------------------------------------------
def create_dsm(
    las_input: PathLike,
    dsm_output: PathLike,
    resolution: float = 0.03,
    output_type: str = "max",
    nodata: float = -9999.0,
    compression: str = "LZW",
    verbose: bool = True,
    show_progress: bool = True,
    gdalopts_extra: Optional[List[str]] = None,
    stac: bool = False,
    stac_site_id: Optional[str] = None,
    stac_product_level: Optional[str] = None,
    stac_datetime: Optional[str] = None,
) -> bool:
    """
    Convert a LAS/LAZ file to a DSM GeoTIFF using PDAL.

    If `stac=True`, a STAC Item JSON sidecar will be written next to the DSM.

    Returns True if successful.
    """
    las_input = Path(las_input)
    dsm_output = Path(dsm_output)

    if not las_input.exists():
        raise FileNotFoundError(f"Input LAS file not found: {las_input}")

    dsm_output.parent.mkdir(parents=True, exist_ok=True)

    if verbose:
        print("--- Generating DSM ---")
        print(f"Input:      {las_input}")
        print(f"Output:     {dsm_output}")
        print(f"Resolution: {resolution} m ({resolution * 100:.1f} cm)")
        print(f"Output type:{output_type}")

        try:
            est = estimate_processing_time(las_input, resolution)
            if "grid_cells" in est:
                print("\nEstimates:")
                print(f"  Point count:  {int(est['point_count']):,}")
                print(f"  Grid size:    {int(est['grid_cols']):,} × {int(est['grid_rows']):,} cells")
                print(f"  Total cells:  {int(est['grid_cells']):,}")
                print(f"  Est. time:    ~{est['estimated_seconds']:.0f} s")
        except Exception:
            pass

    # GDAL options
    gdalopts = [f"COMPRESS={compression}"]
    # Good defaults for large rasters
    gdalopts += ["TILED=YES", "BIGTIFF=IF_SAFER"]
    if gdalopts_extra:
        gdalopts += gdalopts_extra

    pipeline = {
        "pipeline": [
            {"type": "readers.las", "filename": str(las_input)},
            {
                "type": "writers.gdal",
                "filename": str(dsm_output),
                "resolution": resolution,
                "output_type": output_type,
                "nodata": nodata,
                "gdaldriver": "GTiff",
                "gdalopts": ",".join(gdalopts),
            },
        ]
    }
    pipeline_json = json.dumps(pipeline)

    if verbose:
        print("\n⏳ Executing PDAL pipeline...")
        if show_progress:
            print("   (Large files / high resolution may take a while)\n")

    processing = {"done": False}

    def _spinner() -> None:
        spinner = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
        i = 0
        t0 = time.time()
        while not processing["done"]:
            elapsed = time.time() - t0
            sys.stdout.write(f"\r   Processing... {spinner[i % len(spinner)]} ({elapsed:.0f}s)")
            sys.stdout.flush()
            i += 1
            time.sleep(0.1)
        sys.stdout.write("\r" + " " * 60 + "\r")
        sys.stdout.flush()

    spinner_thread: Optional[threading.Thread] = None
    if show_progress and verbose:
        spinner_thread = threading.Thread(target=_spinner, daemon=True)
        spinner_thread.start()

    try:
        t0 = time.time()
        _run(["pdal", "pipeline", "--stdin"], stdin=pipeline_json)
        processing["done"] = True
        if spinner_thread:
            spinner_thread.join(timeout=0.5)

        elapsed = time.time() - t0
        if verbose:
            print(f"✅ DSM created: {dsm_output}")
            print(f"   Time: {elapsed:.1f} s")

        if stac:
            if stac_site_id is None or stac_product_level is None:
                raise ValueError("stac_site_id and stac_product_level are required when stac=True")
            stac_path = write_dsm_stac_item(
                dsm_path=dsm_output,
                site_id=stac_site_id,
                product_level=stac_product_level,
                resolution=resolution,
                datetime=stac_datetime,
                overwrite=True,
            )
            if verbose:
                print(f"🧾 STAC Item written: {stac_path}")

        return True

    finally:
        processing["done"] = True


# ---------------------------------------------------------------------
# Batch processing
# ---------------------------------------------------------------------
def batch_create_dsm(
    input_output_pairs: List[Dict[str, PathLike]],
    resolution: float = 0.03,
    **kwargs: Any,
) -> Dict[str, bool]:
    """Sequential DSM creation for a list of {'input': ..., 'output': ...} dicts."""
    results: Dict[str, bool] = {}

    for i, pair in enumerate(input_output_pairs, 1):
        print(f"\n{'=' * 70}")
        print(f"Processing {i}/{len(input_output_pairs)}")
        print(f"{'=' * 70}")

        try:
            ok = create_dsm(pair["input"], pair["output"], resolution=resolution, **kwargs)
            results[str(pair["output"])] = bool(ok)
        except Exception as e:
            print(f"❌ Failed: {e}")
            results[str(pair["output"])] = False

    print(f"\n{'=' * 70}")
    print("BATCH PROCESSING SUMMARY")
    print(f"{'=' * 70}")
    good = sum(1 for v in results.values() if v)
    print(f"✅ Successful: {good}/{len(results)}")
    print(f"❌ Failed:     {len(results) - good}/{len(results)}")
    return results


def batch_create_dsm_parallel(
    input_output_pairs: List[Dict[str, PathLike]],
    resolution: float = 0.03,
    n_workers: Optional[int] = None,
    scheduler: str = "processes",
    **kwargs: Any,
) -> Dict[str, bool]:
    """
    Parallel DSM creation using Dask (file-level parallelism).

    NOTE: PDAL is a separate process; using Dask 'processes' is usually best.
    """
    if not HAS_DASK:
        print("⚠️ Dask not available. Install with: pip install dask")
        print("   Falling back to sequential processing...")
        return batch_create_dsm(input_output_pairs, resolution=resolution, **kwargs)

    print(f"{'=' * 70}")
    print(f"PARALLEL BATCH PROCESSING ({len(input_output_pairs)} files)")
    print(f"{'=' * 70}")
    print(f"Workers:    {n_workers if n_workers else 'auto'}")
    print(f"Scheduler:  {scheduler}")
    print(f"Resolution: {resolution} m")

    tasks = []
    for pair in input_output_pairs:
        task = delayed(create_dsm)(
            pair["input"],
            pair["output"],
            resolution=resolution,
            show_progress=False,
            **kwargs,
        )
        tasks.append((str(pair["output"]), task))

    print(f"\nProcessing {len(tasks)} files in parallel...")
    with ProgressBar():
        results_list = compute(
            *[t for _, t in tasks],
            num_workers=n_workers,
            scheduler=scheduler,
        )

    results = {out: bool(ok) for (out, _), ok in zip(tasks, results_list)}

    print(f"\n{'=' * 70}")
    print("PARALLEL PROCESSING SUMMARY")
    print(f"{'=' * 70}")
    good = sum(1 for v in results.values() if v)
    print(f"✅ Successful: {good}/{len(results)}")
    print(f"❌ Failed:     {len(results) - good}/{len(results)}")
    return results


# ---------------------------------------------------------------------
# Xarray / Dask raster analytics
# ---------------------------------------------------------------------
def load_dsm_xarray(dsm_path: PathLike, chunks: Optional[Dict[str, int]] = None) -> "xr.DataArray":
    """
    Load DSM as a Dask-backed Xarray DataArray.

    Default chunking is set for large rasters.
    """
    if not HAS_XARRAY:
        raise ImportError("Xarray + rioxarray required. Install with: pip install xarray rioxarray")

    dsm = rioxarray.open_rasterio(
        dsm_path,
        chunks=chunks or {"x": 1024, "y": 1024},
        masked=True,
    )

    # Remove band dimension if single band
    if "band" in dsm.dims and dsm.sizes.get("band", 1) == 1:
        dsm = dsm.squeeze("band", drop=True)

    dsm.name = Path(dsm_path).stem
    return dsm


def compare_dsms(dsm_paths: List[PathLike], labels: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    Compare DSMs with Dask-friendly statistics (compute once at the end).
    """
    if not HAS_XARRAY or not HAS_NUMPY:
        raise ImportError("Xarray + numpy required for DSM comparison")

    if labels is None:
        labels = [f"DSM_{i+1}" for i in range(len(dsm_paths))]

    print(f"{'=' * 70}")
    print("DSM COMPARISON")
    print(f"{'=' * 70}\n")

    dsms: Dict[str, "xr.DataArray"] = {}
    lazy_stats: Dict[str, Dict[str, Any]] = {}

    for path, label in zip(dsm_paths, labels):
        path = Path(path)
        if not path.exists():
            print(f"⚠️ Skipping {label}: not found")
            continue

        print(f"Loading {label}: {path.name}")
        dsm = load_dsm_xarray(path)
        dsms[label] = dsm

        lazy_stats[label] = {
            "path": str(path),
            "shape": tuple(dsm.shape),
            "resolution": float(dsm.rio.resolution()[0]),
            "crs": str(dsm.rio.crs),
            "bounds": tuple(dsm.rio.bounds()),
            "nodata": float(dsm.rio.nodata) if dsm.rio.nodata is not None else None,
            "file_size_mb": float(path.stat().st_size / (1024 * 1024)),
            # lazy reductions (compute later)
            "min": dsm.min(),
            "max": dsm.max(),
            "mean": dsm.mean(),
            "std": dsm.std(),
        }

    # compute all lazy stats in one go (fewer graphs, faster)
    stats: Dict[str, Dict[str, Any]] = {}
    for label, st in lazy_stats.items():
        stats[label] = dict(st)
        for k in ["min", "max", "mean", "std"]:
            stats[label][k] = float(st[k].compute())

    print(f"\n{'=' * 70}")
    print("STATISTICS COMPARISON")
    print(f"{'=' * 70}")
    print(f"{'Metric':<20} " + " ".join([f"{lbl:<15}" for lbl in stats.keys()]))
    print("-" * 70)

    metrics = ["resolution", "shape", "min", "max", "mean", "std", "file_size_mb"]
    for metric in metrics:
        row = []
        for lbl in stats.keys():
            val = stats[lbl].get(metric, "N/A")
            if isinstance(val, float):
                row.append(f"{val:<15.4f}")
            else:
                row.append(f"{str(val):<15}")
        print(f"{metric:<20} " + " ".join(row))

    return {"stats": stats, "datasets": dsms}


def dsm_difference(
    dsm1_path: PathLike,
    dsm2_path: PathLike,
    output_path: Optional[PathLike] = None,
    chunks: Optional[Dict[str, int]] = None,
) -> "xr.DataArray":
    """
    Compute DSM difference (dsm2 - dsm1) with grid alignment safeguards.
    """
    if not HAS_XARRAY:
        raise ImportError("Xarray required. Install with: pip install xarray rioxarray")

    print("Computing DSM difference...")
    dsm1 = load_dsm_xarray(dsm1_path, chunks=chunks)
    dsm2 = load_dsm_xarray(dsm2_path, chunks=chunks)

    # safest for comparisons: require identical grids
    dsm1, dsm2 = xr.align(dsm1, dsm2, join="exact")

    diff = (dsm2 - dsm1).rename("elevation_change")
    diff.rio.write_crs(dsm1.rio.crs, inplace=True)
    diff.rio.write_nodata(dsm1.rio.nodata, inplace=True)

    print("\nDifference Statistics:")
    print(f"  Mean change: {float(diff.mean().compute()):.3f} m")
    print(f"  Std dev:     {float(diff.std().compute()):.3f} m")
    print(f"  Min change:  {float(diff.min().compute()):.3f} m")
    print(f"  Max change:  {float(diff.max().compute()):.3f} m")

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"\nSaving difference to: {output_path}")
        diff.rio.to_raster(str(output_path), compress="LZW", tiled=True, windowed=True)

    return diff


# ---------------------------------------------------------------------
# STAC Item sidecar for DSM outputs (extra function you requested)
# ---------------------------------------------------------------------
def write_dsm_stac_item(
    dsm_path: PathLike,
    site_id: str,
    product_level: str,
    resolution: float,
    datetime: Optional[str] = None,
    collection: str = "dsm-products",
    overwrite: bool = False,
) -> Path:
    """
    Write a minimal STAC Item JSON for a DSM GeoTIFF next to the file.

    Output: <dsm_path>.stac-item.json
    """
    if not HAS_XARRAY:
        raise ImportError("xarray + rioxarray required for STAC generation (pip install xarray rioxarray)")

    import uuid
    from datetime import datetime as dt

    dsm_path = Path(dsm_path)
    stac_path = dsm_path.with_suffix(".stac-item.json")

    if stac_path.exists() and not overwrite:
        return stac_path

    dsm = load_dsm_xarray(dsm_path)

    left, bottom, right, top = dsm.rio.bounds()
    bbox = [left, bottom, right, top]

    # polygon footprint
    geometry = {
        "type": "Polygon",
        "coordinates": [[
            [left, bottom],
            [left, top],
            [right, top],
            [right, bottom],
            [left, bottom],
        ]],
    }

    crs = dsm.rio.crs
    epsg = int(crs.to_epsg()) if crs is not None and crs.to_epsg() is not None else None

    if datetime is None:
        datetime = dt.utcnow().isoformat(timespec="seconds") + "Z"

    item = {
        "type": "Feature",
        "stac_version": "1.0.0",
        "id": dsm_path.stem,
        "collection": collection,
        "geometry": geometry,
        "bbox": bbox,
        "properties": {
            "datetime": datetime,
            "proj:epsg": epsg,
            "gsd": float(resolution),
            "site_id": site_id,
            "product": "DSM",
            "processing:level": product_level,
            "processing:software": "PDAL",
            "nodata": float(dsm.rio.nodata) if dsm.rio.nodata is not None else None,
        },
        "assets": {
            "dsm": {
                "href": str(dsm_path),
                "type": "image/tiff; application=geotiff",
                "roles": ["data"],
                "title": "Digital Surface Model",
            }
        },
        "links": [],
    }

    with open(stac_path, "w") as f:
        json.dump(item, f, indent=2)

    return stac_path


# ---------------------------------------------------------------------
# Example configurations
# ---------------------------------------------------------------------
SITE_CONFIGS = {
    "DD0001_L1": {
        "input": "/media/laura/laura_usb/uas_data/DD0001/lidar/20220519_SASMDD0001_triple_AusGeoid.las",
        "output": "/media/laura/laura_usb/uas_data/DD0001/lidar/dsm/SASMDD0001_l1_dsm_03.tif",
        "site_id": "DD0001",
        "level": "L1",
    },
    "DD0011_L1": {
        "input": "/data/calperumResearch/site2_1_DD0011/lidar/20220517_SASMDD0011_AusGeoid20.las",
        "output": "/data/calperumResearch/site2_1_DD0011/lidar/dsm/SASMDD0011_l1_dsm_03.tif",
        "site_id": "DD0011",
        "level": "L1",
    },
    "DD0001_L2": {
        "input": "/media/laura/laura_usb/uas_data/DD0001/lidar_l2/cloud2d765c57b8471e2c.las",
        "output": "/media/laura/laura_usb/uas_data/DD0001/lidar_l2/dsm/SASMDD0001_l2_dsm_03.tif",
        "site_id": "DD0001",
        "level": "L2",
    },
    "DD0011_L2": {
        "input": "/data/calperumResearch/site2_1_DD0011/lidar_l2/cloud3358f88f2d9dc868.las",
        "output": "/data/calperumResearch/site2_1_DD0011/lidar_l2/dsm/SASMDD0011_l2_dsm_03.tif",
        "site_id": "DD0011",
        "level": "L2",
    },
    "DD0001_Riegl": {
        "input": "/media/laura/laura_usb/uas_data/DD0001/riegl_vux120/SASMDD0001_riegl_80_crosshatch.laz",
        "output": "/media/laura/laura_usb/uas_data/DD0001/riegl_vux120/dsm/SASMDD0001_riegl_dsm_03.tif",
        "site_id": "DD0001",
        "level": "RIEGL",
    },
}


if __name__ == "__main__":
    ACTIVE_SITE = "DD0011_L2"
    cfg = SITE_CONFIGS[ACTIVE_SITE]
    dsm_resolution = 0.03

    print(f"{'=' * 70}")
    print(f"Processing: {ACTIVE_SITE}")
    print(f"{'=' * 70}\n")

    try:
        create_dsm(
            las_input=cfg["input"],
            dsm_output=cfg["output"],
            resolution=dsm_resolution,
            verbose=True,
            stac=True,  # ✅ write STAC sidecar
            stac_site_id=cfg["site_id"],
            stac_product_level=cfg["level"],
        )
    except Exception as e:
        print(f"\n❌ Error: {e}")
        sys.exit(1)
