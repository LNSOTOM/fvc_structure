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

import argparse
import importlib
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

try:
    yaml = importlib.import_module("yaml")

    HAS_YAML = True
except Exception:
    yaml = None
    HAS_YAML = False


PathLike = Union[str, Path]


@dataclass(frozen=True)
class DsmJob:
    """Reusable LAS/LAZ-to-DSM job definition."""

    las_input: PathLike
    dsm_output: PathLike
    site_id: Optional[str] = None
    product_level: Optional[str] = None
    sensor: Optional[str] = None
    # Optional upstream provenance metadata.
    source_workflow: Optional[str] = None


def _first_present(data: Dict[str, Any], keys: List[str], default: Any = None) -> Any:
    for key in keys:
        if key in data and data[key] is not None:
            return data[key]
    return default


def load_dsm_config(config_path: PathLike) -> Dict[str, Any]:
    """Load a DSM job config from YAML or JSON."""
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"DSM config not found: {config_path}")

    suffix = config_path.suffix.lower()
    with open(config_path, "r") as f:
        if suffix in {".yml", ".yaml"}:
            if not HAS_YAML:
                raise ImportError("PyYAML is required for .yml/.yaml config files. Install with: pip install pyyaml")
            data = yaml.safe_load(f) or {}
        elif suffix == ".json":
            data = json.load(f)
        else:
            if HAS_YAML:
                data = yaml.safe_load(f) or {}
            else:
                data = json.load(f)

    if not isinstance(data, dict):
        raise ValueError(f"DSM config must contain a mapping/object: {config_path}")
    return data


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


def recommend_dsm_resolution(
    las_input: PathLike,
    candidate_resolutions: Optional[List[float]] = None,
    max_estimated_minutes: float = 15.0,
    max_grid_cells: Optional[int] = 250_000_000,
) -> Dict[str, Any]:
    """
    Estimate candidate DSM resolutions and choose the finest one within limits.

    Returns a dict with the chosen resolution and all per-resolution estimates.
    """
    if candidate_resolutions is None:
        candidate_resolutions = [0.10, 0.05, 0.03]

    if not candidate_resolutions:
        raise ValueError("candidate_resolutions must contain at least one value")

    candidates = sorted({float(res) for res in candidate_resolutions}, reverse=True)
    if any(res <= 0 for res in candidates):
        raise ValueError("candidate_resolutions must be positive")

    estimates = []
    for resolution in candidates:
        estimate = estimate_processing_time(las_input, resolution)
        estimate["resolution"] = resolution
        estimate["estimated_minutes"] = float(estimate.get("estimated_seconds", 0.0)) / 60.0
        grid_cells = estimate.get("grid_cells")
        estimate["within_limits"] = (
            estimate["estimated_minutes"] <= max_estimated_minutes
            and (max_grid_cells is None or grid_cells is None or grid_cells <= max_grid_cells)
        )
        estimates.append(estimate)

    eligible = [estimate for estimate in estimates if estimate["within_limits"]]
    if eligible:
        selected = min(eligible, key=lambda estimate: estimate["resolution"])
    else:
        selected = max(estimates, key=lambda estimate: estimate["resolution"])

    return {
        "selected_resolution": float(selected["resolution"]),
        "selected_estimate": selected,
        "estimates": estimates,
        "max_estimated_minutes": float(max_estimated_minutes),
        "max_grid_cells": max_grid_cells,
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


def create_dsm_from_job(
    job: DsmJob,
    resolution: float = 0.03,
    stac: Optional[bool] = None,
    **kwargs: Any,
) -> bool:
    """Create a DSM from a reusable job definition."""
    write_stac = bool(job.site_id and job.product_level) if stac is None else stac
    return create_dsm(
        las_input=job.las_input,
        dsm_output=job.dsm_output,
        resolution=resolution,
        stac=write_stac,
        stac_site_id=job.site_id,
        stac_product_level=job.product_level,
        **kwargs,
    )


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


def job_from_site_config(site_name: str) -> DsmJob:
    """Build a reusable job from one of the predefined site configs."""
    if site_name not in SITE_CONFIGS:
        available = ", ".join(sorted(SITE_CONFIGS))
        raise KeyError(f"Unknown site config: {site_name}. Available: {available}")

    cfg = SITE_CONFIGS[site_name]
    return DsmJob(
        las_input=cfg["input"],
        dsm_output=cfg["output"],
        site_id=cfg.get("site_id"),
        product_level=cfg.get("level"),
        sensor=site_name,
    )


def job_from_config(config_path: PathLike) -> DsmJob:
    """Build a DSM job from a YAML/JSON config file."""
    cfg = load_dsm_config(config_path)
    job_cfg = cfg.get("job", cfg)
    if not isinstance(job_cfg, dict):
        raise ValueError("DSM config 'job' section must be a mapping/object")

    site_name = _first_present(job_cfg, ["site"])
    if site_name:
        job = job_from_site_config(str(site_name))
        return DsmJob(
            las_input=_first_present(job_cfg, ["las_input", "input"], job.las_input),
            dsm_output=_first_present(job_cfg, ["dsm_output", "output"], job.dsm_output),
            site_id=_first_present(job_cfg, ["site_id"], job.site_id),
            product_level=_first_present(job_cfg, ["product_level", "level"], job.product_level),
            sensor=_first_present(job_cfg, ["sensor"], job.sensor),
            source_workflow=_first_present(job_cfg, ["source_workflow"], job.source_workflow),
        )

    las_input = _first_present(job_cfg, ["las_input", "input"])
    dsm_output = _first_present(job_cfg, ["dsm_output", "output"])
    if not las_input or not dsm_output:
        raise ValueError("DSM config must define las_input and dsm_output")

    return DsmJob(
        las_input=las_input,
        dsm_output=dsm_output,
        site_id=_first_present(job_cfg, ["site_id"]),
        product_level=_first_present(job_cfg, ["product_level", "level"]),
        sensor=_first_present(job_cfg, ["sensor"]),
        source_workflow=_first_present(job_cfg, ["source_workflow"], DsmJob.source_workflow),
    )


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Create a DSM GeoTIFF from a georeferenced LAS/LAZ point cloud."
        )
    )
    parser.add_argument("--config", help="YAML/JSON config file with DSM input/output paths and defaults")
    parser.add_argument("--site", choices=sorted(SITE_CONFIGS), help="Run one predefined site configuration")
    parser.add_argument("--las-input", help="Input georeferenced LAS/LAZ file")
    parser.add_argument("--dsm-output", help="Output DSM GeoTIFF path")
    parser.add_argument("--site-id", help="Site identifier written to the optional STAC sidecar")
    parser.add_argument("--product-level", help="Product level written to the optional STAC sidecar")
    parser.add_argument("--datetime", dest="stac_datetime", help="STAC datetime, for example 2024-01-31T00:00:00Z")
    parser.add_argument("--resolution", type=float, help="DSM grid resolution in metres (default: 0.03)")
    parser.add_argument("--output-type", help="PDAL writers.gdal output_type, usually max for DSM")
    parser.add_argument("--nodata", type=float, help="NoData value for the DSM (default: -9999.0)")
    parser.add_argument("--compression", help="GeoTIFF compression option (default: LZW)")
    parser.add_argument("--no-stac", action="store_true", help="Do not write a STAC sidecar")
    parser.add_argument("--estimate-only", action="store_true", help="Print processing estimate without creating a DSM")
    parser.add_argument("--list-sites", action="store_true", help="List predefined site configurations and exit")
    parser.add_argument("--quiet", action="store_true", help="Reduce console output")
    return parser


def _job_from_args(args: argparse.Namespace, cfg: Optional[Dict[str, Any]] = None) -> DsmJob:
    cfg = cfg or {}
    job_cfg = cfg.get("job", cfg)
    if not isinstance(job_cfg, dict):
        raise ValueError("DSM config 'job' section must be a mapping/object")

    if args.site:
        job = job_from_site_config(args.site)
        return DsmJob(
            las_input=args.las_input or _first_present(job_cfg, ["las_input", "input"], job.las_input),
            dsm_output=args.dsm_output or _first_present(job_cfg, ["dsm_output", "output"], job.dsm_output),
            site_id=args.site_id or job.site_id,
            product_level=args.product_level or job.product_level,
            sensor=_first_present(job_cfg, ["sensor"], job.sensor),
            source_workflow=_first_present(job_cfg, ["source_workflow"], job.source_workflow),
        )

    config_site = _first_present(job_cfg, ["site"])
    if config_site:
        job = job_from_site_config(str(config_site))
        return DsmJob(
            las_input=args.las_input or _first_present(job_cfg, ["las_input", "input"], job.las_input),
            dsm_output=args.dsm_output or _first_present(job_cfg, ["dsm_output", "output"], job.dsm_output),
            site_id=args.site_id or _first_present(job_cfg, ["site_id"], job.site_id),
            product_level=args.product_level or _first_present(job_cfg, ["product_level", "level"], job.product_level),
            sensor=_first_present(job_cfg, ["sensor"], job.sensor),
            source_workflow=_first_present(job_cfg, ["source_workflow"], job.source_workflow),
        )

    las_input = args.las_input or _first_present(job_cfg, ["las_input", "input"])
    dsm_output = args.dsm_output or _first_present(job_cfg, ["dsm_output", "output"])
    if not las_input or not dsm_output:
        raise ValueError("Provide --config, --site, or both --las-input and --dsm-output")

    return DsmJob(
        las_input=las_input,
        dsm_output=dsm_output,
        site_id=args.site_id or _first_present(job_cfg, ["site_id"]),
        product_level=args.product_level or _first_present(job_cfg, ["product_level", "level"]),
        sensor=_first_present(job_cfg, ["sensor"]),
        source_workflow=_first_present(job_cfg, ["source_workflow"], DsmJob.source_workflow),
    )


def _processing_option(
    args: argparse.Namespace,
    cfg: Dict[str, Any],
    name: str,
    default: Any,
) -> Any:
    if getattr(args, name) is not None:
        return getattr(args, name)
    return cfg.get(name, default)


def _datetime_option(value: Any) -> Optional[str]:
    if value is None:
        return None
    if hasattr(value, "isoformat"):
        return value.isoformat().replace("+00:00", "Z")
    return str(value)


def main(argv: Optional[List[str]] = None) -> int:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)

    if args.list_sites:
        print("Available site configurations:")
        for name in sorted(SITE_CONFIGS):
            cfg = SITE_CONFIGS[name]
            print(f"  {name}: {cfg['input']} -> {cfg['output']}")
        return 0

    try:
        cfg = load_dsm_config(args.config) if args.config else {}
        job = _job_from_args(args, cfg)
        resolution = float(_processing_option(args, cfg, "resolution", 0.03))
        output_type = str(_processing_option(args, cfg, "output_type", "max"))
        nodata = float(_processing_option(args, cfg, "nodata", -9999.0))
        compression = str(_processing_option(args, cfg, "compression", "LZW"))
        stac_datetime = _datetime_option(args.stac_datetime or cfg.get("stac_datetime") or cfg.get("datetime"))
        stac = False if args.no_stac else cfg.get("stac")

        if args.estimate_only:
            estimate = estimate_processing_time(job.las_input, resolution)
            print(json.dumps(estimate, indent=2))
            return 0

        create_dsm_from_job(
            job,
            resolution=resolution,
            output_type=output_type,
            nodata=nodata,
            compression=compression,
            verbose=not args.quiet,
            stac=stac,
            stac_datetime=stac_datetime,
        )
        return 0
    except Exception as e:
        print(f"\nError: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
