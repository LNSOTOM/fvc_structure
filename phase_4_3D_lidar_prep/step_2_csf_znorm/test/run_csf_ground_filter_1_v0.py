
# run_csf_ground_filter.py
import os
import json
import subprocess
from pathlib import Path


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
    """
    Run PDAL Cloth Simulation Filter (CSF) to classify ground points and
    output ground-only LAS (Classification = 2).

    Parameters
    ----------
    input_las : str or Path
        Path to input LAS/LAZ file.
    output_las : str or Path
        Output LAS/LAZ with only ground points.
    resolution : float
        Cloth grid resolution (meters). Default = 0.2 for UAS point clouds.
    iterations : int
        Number of iterations for cloth simulation (default = 500).
    threshold : float
        Height threshold for ground detection.
    smooth : bool
        Apply smoothing to cloth (default False).
    rigidness : int
        Cloth rigidness (1–3). 1 = flexible (better for rugged terrain).
    time_step : float or None
        Optional time-step parameter for simulation.
    compress : bool
        Save output with LASzip compression if True.
    """
    input_las = Path(input_las)
    output_las = Path(output_las)
    output_las.parent.mkdir(parents=True, exist_ok=True)

    pipeline_json = output_las.with_suffix(".json")

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

    writer = {"type": "writers.las", "filename": str(output_las)}
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

    # Save JSON
    with open(pipeline_json, "w") as f:
        json.dump(pipeline, f, indent=2)
    if save_pipeline:
        print(f"💾 Saved PDAL pipeline JSON: {pipeline_json}")

    # Run PDAL
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
