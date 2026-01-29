# run_csf_ground_filter.py (SMRF part)

import os
import json
import subprocess
from pathlib import Path
import laspy  # NEW

def run_smrf_ground_filter(
    input_las: str | Path,
    output_las: str | Path,
    pipeline_json: str | Path = None,
    *,
    slope: float = 0.15,
    window: float = 2.0,
    threshold: float = 0.15,
    scalar: float = 1.25,
    compress: bool = False,
    save_pipeline: bool = True,
) -> Path:

    input_las = Path(input_las)
    output_las = Path(output_las)

    # --- Read header to copy scales/offsets ---
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
        # keep same quantisation as input
        "scale_x": sx,
        "scale_y": sy,
        "scale_z": sz,
        "offset_x": ox,
        "offset_y": oy,
        "offset_z": oz,
        # forward VLR/CRS info if your PDAL version supports it
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
        print(f"💾 Saved PDAL SMRF pipeline JSON: {pipeline_json}")

    print(f"▶️  Running PDAL SMRF ground filter on {input_las.name}")
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
        raise FileNotFoundError("❌ SMRF Ground LAS was not written. Check PDAL output.")

    print(f"✅ SMRF Ground LAS written: {output_las}")
    return output_las


def run_csf_ground_filter(
    input_las: str | Path,
    output_las: str | Path,
    *,
    resolution: float = 0.2,
    iterations: int = 500,
    threshold: float = 0.1,
    smooth: bool = False,
    rigidness: int = 2,
    time_step: float | None = None,
    compress: bool = False,
    save_pipeline: bool = True,
) -> Path:
    input_las = Path(input_las)
    output_las = Path(output_las)
    output_las.parent.mkdir(parents=True, exist_ok=True)

    pipeline_json = output_las.with_suffix(".json")

    # --- Read header from input LAS ---
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
        print(f"💾 Saved PDAL pipeline JSON: {pipeline_json}")

    print(f"▶️  Running PDAL CSF ground filter on {input_las.name}")
    try:
        subprocess.run(["pdal", "pipeline", str(pipeline_json), "--verbose=6"], check=True)
    except subprocess.CalledProcessError as e:
        print(f"❌ PDAL CSF failed: {e}")
        raise

    if not output_las.exists():
        raise FileNotFoundError("❌ Ground LAS not written — check PDAL output logs.")

    print(f"✅ CSF Ground LAS written: {output_las}")
    return output_las
