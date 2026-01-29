"""
Core functions for ground classification and Z normalization.
"""

import os
import json
import subprocess
from pathlib import Path
import numpy as np
import laspy
from scipy.spatial import cKDTree


def run_smrf_ground_filter(
    input_las: str | Path,
    output_las: str | Path,
    pipeline_json: str | Path = None,
    *,
    slope: float = 0.15,
    window: float = 3.5,
    threshold: float = 0.25,
    scalar: float = 1.35,
    compress: bool = False,
    save_pipeline: bool = True,
) -> Path:
    """
    Extract ground points using PDAL's SMRF (Simple Morphological Filter).

    Parameters
    ----------
    input_las : str | Path
        Input LAS file path
    output_las : str | Path
        Output LAS file path (ground points only)
    pipeline_json : str | Path, optional
        Path to save PDAL pipeline JSON
    slope : float, default=0.15
        SMRF slope parameter
    window : float, default=3.5
        SMRF window size
    threshold : float, default=0.25
        SMRF elevation threshold
    scalar : float, default=1.35
        SMRF scalar parameter
    compress : bool, default=False
        Apply LAZ compression
    save_pipeline : bool, default=True
        Save pipeline JSON to disk

    Returns
    -------
    Path
        Path to output ground LAS file
    """
    input_las = Path(input_las)
    output_las = Path(output_las)

    # Read header to preserve scale/offset
    in_las = laspy.read(input_las)
    sx, sy, sz = in_las.header.scales
    ox, oy, oz = in_las.header.offsets

    if pipeline_json is None:
        pipeline_json = output_las.with_suffix(".json")
    else:
        pipeline_json = Path(pipeline_json)

    ground_filter = {
        "type": "filters.smrf",
        "slope": slope,
        "window": window,
        "threshold": threshold,
        "scalar": scalar,
    }

    writer = {
        "type": "writers.las",
        "filename": str(output_las),
        "scale_x": sx,
        "scale_y": sy,
        "scale_z": sz,
        "offset_x": ox,
        "offset_y": oy,
        "offset_z": oz,
        "forward": "all",
    }
    if compress:
        writer["compression"] = "laszip"

    pipeline = {
        "pipeline": [
            {"type": "readers.las", "filename": str(input_las)},
            ground_filter,
            {"type": "filters.range", "limits": "Classification[2:2]"},
            writer,
        ]
    }

    os.makedirs(output_las.parent, exist_ok=True)
    os.makedirs(pipeline_json.parent, exist_ok=True)

    with open(pipeline_json, "w") as f:
        json.dump(pipeline, f, indent=2)
    if save_pipeline:
        print(f"💾 Saved PDAL SMRF pipeline: {pipeline_json}")

    print(f"▶️  Running PDAL SMRF on {input_las.name}")
    try:
        res = subprocess.run(
            ["pdal", "pipeline", str(pipeline_json), "--verbose=6"],
            check=True, text=True, capture_output=True
        )
        if res.stdout:
            print(res.stdout)
    except subprocess.CalledProcessError as e:
        print("----- PDAL SMRF STDOUT -----")
        print(e.stdout or "")
        print("----- PDAL SMRF STDERR -----")
        print(e.stderr or "")
        raise

    if not output_las.exists():
        raise FileNotFoundError("❌ SMRF ground LAS not written. Check PDAL output.")

    print(f"✅ SMRF ground LAS: {output_las}")
    return output_las


def run_csf_ground_filter(
    input_las: str | Path,
    output_las: str | Path,
    *,
    resolution: float = 0.5,
    iterations: int = 500,
    threshold: float = 0.01,
    smooth: bool = False,
    rigidness: int = 2,
    time_step: float | None = None,
    compress: bool = False,
    save_pipeline: bool = True,
) -> Path:
    """
    Extract ground points using PDAL's CSF (Cloth Simulation Filter).

    Parameters
    ----------
    input_las : str | Path
        Input LAS file path
    output_las : str | Path
        Output LAS file path (ground points only)
    resolution : float, default=0.5
        Cloth grid resolution
    iterations : int, default=500
        Maximum iterations
    threshold : float, default=0.01
        Classification threshold
    smooth : bool, default=False
        Enable cloth smoothing
    rigidness : int, default=2
        Cloth rigidness (1=steep, 2=relief, 3=flat)
    time_step : float, optional
        Time step for simulation
    compress : bool, default=False
        Apply LAZ compression
    save_pipeline : bool, default=True
        Save pipeline JSON to disk

    Returns
    -------
    Path
        Path to output ground LAS file
    """
    input_las = Path(input_las)
    output_las = Path(output_las)
    output_las.parent.mkdir(parents=True, exist_ok=True)

    pipeline_json = output_las.with_suffix(".json")

    # Read header from input LAS
    in_las = laspy.read(input_las)
    sx, sy, sz = in_las.header.scales
    ox, oy, oz = in_las.header.offsets

    ground_filter = {
        "type": "filters.csf",
        "resolution": resolution,
        "iterations": iterations,
        "threshold": threshold,
        "smooth": smooth,
        "rigidness": rigidness,
    }
    if time_step is not None:
        ground_filter["time_step"] = time_step

    writer = {
        "type": "writers.las",
        "filename": str(output_las),
        "scale_x": sx,
        "scale_y": sy,
        "scale_z": sz,
        "offset_x": ox,
        "offset_y": oy,
        "offset_z": oz,
        "forward": "all",
    }
    if compress:
        writer["compression"] = "laszip"

    pipeline = {
        "pipeline": [
            {"type": "readers.las", "filename": str(input_las)},
            ground_filter,
            {"type": "filters.range", "limits": "Classification[2:2]"},
            writer,
        ]
    }

    with open(pipeline_json, "w") as f:
        json.dump(pipeline, f, indent=2)
    if save_pipeline:
        print(f"💾 Saved PDAL CSF pipeline: {pipeline_json}")

    print(f"▶️  Running PDAL CSF on {input_las.name}")
    try:
        subprocess.run(["pdal", "pipeline", str(pipeline_json), "--verbose=6"], check=True)
    except subprocess.CalledProcessError as e:
        print(f"❌ PDAL CSF failed: {e}")
        raise

    if not output_las.exists():
        raise FileNotFoundError("❌ Ground LAS not written. Check PDAL output.")

    print(f"✅ CSF ground LAS: {output_las}")
    return output_las


def normalize_las_z(
    input_las: str | Path,
    ground_las: str | Path,
    output_las: str | Path,
    *,
    k: int = 3,
    r_max: float = 1.5,
    extra_dim_name: str = "Z_norm",
) -> None:
    """
    Compute height above ground (HAG) using inverse-distance weighted kNN.

    Adds a new ExtraBytes dimension (default: Z_norm) to the LAS file.

    Parameters
    ----------
    input_las : str | Path
        Input LAS file path (all points)
    ground_las : str | Path
        Ground LAS file path (ground points only)
    output_las : str | Path
        Output LAS file path with Z_norm dimension
    k : int, default=3
        Number of nearest neighbors for interpolation
    r_max : float, default=1.5
        Maximum search radius (currently unused)
    extra_dim_name : str, default="Z_norm"
        Name of the extra dimension to store HAG values
    """
    input_las = Path(input_las)
    ground_las = Path(ground_las)
    output_las = Path(output_las)
    output_las.parent.mkdir(parents=True, exist_ok=True)

    # Load point clouds
    las_all = laspy.read(input_las)
    all_xyz = np.column_stack((las_all.x, las_all.y, las_all.z))

    las_ground = laspy.read(ground_las)
    ground_xyz = np.column_stack((las_ground.x, las_ground.y, las_ground.z))
    if len(ground_xyz) == 0:
        raise ValueError("❌ No ground points found in ground LAS.")

    # Check headers for consistency
    if not np.allclose(las_all.header.scales, las_ground.header.scales):
        print("⚠️ Warning: scale mismatch between input and ground LAS.")
    if not np.allclose(las_all.header.offsets, las_ground.header.offsets):
        print("⚠️ Warning: offset mismatch between input and ground LAS.")

    # Build KDTree for 2D nearest neighbor search
    tree = cKDTree(ground_xyz[:, :2])
    dist, idx = tree.query(all_xyz[:, :2], k=k, workers=-1)
    dist = np.maximum(dist, 1e-6)

    if dist.ndim == 1:  # ensure 2D shape
        dist = dist[:, None]
        idx = idx[:, None]

    # Inverse distance weights
    w = 1.0 / dist
    w /= w.sum(axis=1, keepdims=True)

    # Interpolated ground elevation
    gz_est = np.sum(ground_xyz[idx, 2] * w, axis=1)

    # Height Above Ground (HAG)
    hag = np.maximum(all_xyz[:, 2] - gz_est, 0.0).astype(np.float32)

    # Add or overwrite extra dimension
    if extra_dim_name not in las_all.point_format.dimension_names:
        las_all.add_extra_dim(laspy.ExtraBytesParams(name=extra_dim_name, type=np.float32))
    las_all[extra_dim_name] = hag

    las_all.header.software_id = "groundnorm_HAG"
    las_all.write(output_las)

    print(f"✅ Wrote normalized LAS: {output_las}")
    print(f"   HAG stats (m): min={hag.min():.3f}, median={np.median(hag):.3f}, max={hag.max():.3f}")
    print(f"   Ground pts: {len(ground_xyz):,}, All pts: {len(all_xyz):,}, k={k}")
