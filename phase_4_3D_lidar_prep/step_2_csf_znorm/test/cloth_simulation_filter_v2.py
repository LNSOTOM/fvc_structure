# %%
###test4
import os
import json
import subprocess
from pathlib import Path


def run_csf(  # keeps your original name & call pattern
    input_las: str | Path,
    output_las: str | Path,
    pipeline_json: str | Path = None,
    *,
    method: str = "smrf",          # choose between "smrf" and "csf"
    # CSF parameters (used when method="csf")
    resolution: float = 0.2,
    iterations: int = 500,
    threshold: float = 0.1,
    smooth: bool = False,
    csf_rigidness: int | None = 2,
    csf_time_step: float | None = None,
    # SMRF parameters (used when method="smrf")
    smrf_slope: float = 0.15,
    smrf_window: float = 2.0,
    smrf_threshold: float = 0.15,
    smrf_scalar: float = 1.25,
    compress: bool = False,
    save_pipeline: bool = True,
) -> Path:
    """Run PDAL ground filtering via SMRF (default) or CSF and write ground-only LAS."""
    input_las  = Path(input_las)
    output_las = Path(output_las)

    # Default JSON path if not given
    if pipeline_json is None:
        pipeline_json = output_las.with_suffix("").with_name(output_las.stem + "_pipeline.json")
    else:
        pipeline_json = Path(pipeline_json)

    writer = {"type": "writers.las", "filename": str(output_las)}
    if compress:
        writer["compression"] = "laszip"  # requires PDAL with LASzip support

    method_normalized = method.lower()
    if method_normalized not in {"smrf", "csf"}:
        raise ValueError("method must be either 'smrf' or 'csf'")

    if method_normalized == "smrf":
        ground_filter = {
            "type": "filters.smrf",
            "slope": smrf_slope,
            "window": smrf_window,
            "threshold": smrf_threshold,
            "scalar": smrf_scalar,
        }
    else:
        ground_filter = {
            "type": "filters.csf",
            "resolution": resolution,
            "iterations": iterations,
            "threshold": threshold,
            "smooth": smooth,
        }
        if csf_rigidness is not None:
            ground_filter["rigidness"] = csf_rigidness
        if csf_time_step is not None:
            ground_filter["time_step"] = csf_time_step

    pipeline = {
        "pipeline": [
            {"type": "readers.las", "filename": str(input_las)},
            ground_filter,
            {"type": "filters.range", "limits": "Classification[2:2]"},
            writer
        ]
    }

    os.makedirs(output_las.parent, exist_ok=True)
    os.makedirs(pipeline_json.parent, exist_ok=True)

    # Save JSON so PDAL can run it
    with open(pipeline_json, "w") as f:
        json.dump(pipeline, f, indent=2)
    if save_pipeline:
        print(f"💾 Saved PDAL pipeline JSON: {pipeline_json}")

    chosen_filter = "SMRF" if method_normalized == "smrf" else "CSF"
    print(f"▶️  Running PDAL {chosen_filter} pipeline...")

    # Run with verbose; show real PDAL error if it fails
    try:
        res = subprocess.run(
            ["pdal", "pipeline", str(pipeline_json), "--verbose=6"],
            check=True, text=True, capture_output=True
        )
        if res.stdout:
            print(res.stdout)
    except subprocess.CalledProcessError as e:
        print("----- PDAL STDOUT -----")
        print(e.stdout or "")
        print("----- PDAL STDERR -----")
        print(e.stderr or "")
        raise

    if not output_las.exists():
        raise FileNotFoundError(
            f"❌ Ground LAS was not written. Check PDAL output for {chosen_filter} classification results."
        )

    print(f"✅ Ground LAS ({chosen_filter}): {output_las}")
    return output_las


# === Example usage (tuned for UAS + low shrubs) ===
raster_clip = "/media/laura/laura_usb/uas_data/DD0001/lidar/clipped/raster/coreg/tiles_multispectral.26.las"

# Run SMRF (default)
ground_las_smrf = run_csf(
    input_las=raster_clip,
    output_las="/media/laura/laura_usb/uas_data/DD0001/lidar/clipped/csf/smrf_tiles_multispectral.26.las",
    pipeline_json="/media/laura/laura_usb/uas_data/DD0001/lidar/clipped/csf/smrf_tiles_multispectral.26.json",
    method="smrf",
    smrf_threshold=0.20,
    smrf_window=2.8,
    smrf_slope=0.10,
    smrf_scalar=1.35,
)

# Run CSF for comparison
ground_las_csf = run_csf(
    input_las=raster_clip,
    output_las="/media/laura/laura_usb/uas_data/DD0001/lidar/clipped/csf/csf_tiles_multispectral.26.las",
    pipeline_json="/media/laura/laura_usb/uas_data/DD0001/lidar/clipped/csf/csf_tiles_multispectral.26.json",
    method="csf",
    threshold=0.09,
    iterations=500,
    csf_rigidness=1,
)


# %%
# process_csf_hag.py
from run_csf_ground_filter import run_csf_ground_filter
from normalize_las_z import normalize_las_z

# === Step 1: Extract Ground with CSF ===
input_tile = "/media/laura/laura_usb/uas_data/DD0001/lidar/clipped/raster/coreg/tiles_multispectral.26.las"
ground_out = "/media/laura/laura_usb/uas_data/DD0001/lidar/clipped/csf/csf_tiles_multispectral.26.las"

run_csf_ground_filter(
    input_las=input_tile,
    output_las=ground_out,
    resolution=0.2,  #0.2
    iterations=500,
    threshold=0.01,  #0.09  #0.03
    rigidness=1,   # flexible cloth for shrubs
    smooth=True,  #False
)

# === Step 2: Normalize Z (Height Above Ground) ===
norm_out = "/media/laura/laura_usb/uas_data/DD0001/lidar/clipped/z_norm/z_norm_tiles_multispectral.26.las"

normalize_las_z(
    input_las=input_tile,
    ground_las=ground_out,
    output_las=norm_out,
    k=3
)


# %%
