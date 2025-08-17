

# %% CSF + HAG on the raster-extent clipped LAS
import os
import json
from pathlib import Path
import pdal

# === INPUT: use your raster-extent clip as source ===
raster_clip_las = Path("/media/laura/Laura 102/uas_data/site1_1_DD0001/lidar/trim/cloud1457eb47b07af181_clip_raster_extent.las")

# === OUTPUTS ===
out_dir = raster_clip_las.parent  # same folder as the clip, or change if you prefer
csf_hag_all = out_dir / (raster_clip_las.stem.replace("_clip_raster_extent", "") + "_csf_hag_all.las")
ground_only = out_dir / (raster_clip_las.stem.replace("_clip_raster_extent", "") + "_ground_only.las")

os.makedirs(out_dir, exist_ok=True)

# === PDAL pipeline: CSF → HAG → write both all-points and ground-only ===
stages = [
    {"type": "readers.las", "filename": str(raster_clip_las), "tag": "src"},

    # Cloth Simulation Filtering (classifies ground as class = 2)
    {
        "type": "filters.csf",
        "resolution": 0.2,   # tune to your density/terrain
        "iterations": 500,
        "threshold": 0.1,
        "smooth": False,
        "inputs": ["src"],
        "tag": "csf"
    },

    # Height Above Ground using a Delaunay TIN built from ground points
    {
        "type": "filters.hag_delaunay",
        "inputs": ["csf"],
        "tag": "hag"
    },

    # Write all points with Classification + HeightAboveGround
    {
        "type": "writers.las",
        "filename": str(csf_hag_all),
        "minor_version": 4,
        "inputs": ["hag"]
    },

    # Ground-only branch (class == 2)
    {
        "type": "filters.range",
        "limits": "Classification[2:2]",
        "inputs": ["hag"],
        "tag": "ground"
    },
    {
        "type": "writers.las",
        "filename": str(ground_only),
        "minor_version": 4,
        "inputs": ["ground"]
    }
]

pipeline = {"pipeline": stages}
pipe = pdal.Pipeline(json.dumps(pipeline))
count = pipe.execute()

print(f"✅ PDAL points processed: {count}")
print(f"✅ CSF+HAG written: {csf_hag_all}")
print(f"✅ Ground-only written: {ground_only}")

# Optional: quick sanity check of the new dimension names
meta = pipe.metadata  # dict
print("Dimensions present (first writer):")
dims = meta.get("metadata", {}).get("writers.las", {})
print(dims if isinstance(dims, dict) else "(see LAS header in a viewer)")

# %%
####test 2
import os
import json
import subprocess
from pathlib import Path

def run_csf(
    input_las: str | Path,
    output_las: str | Path,
    pipeline_json: str | Path = None,
    *,
    resolution: float = 0.2,
    iterations: int = 500,
    threshold: float = 0.1,
    smooth: bool = False,
    compress: bool = False,
    save_pipeline: bool = True
) -> Path:
    """
    Run PDAL Cloth Simulation Filtering (CSF) and write ground-only LAS.
    Ground = Classification 2 (filtered via filters.range).

    Args:
        input_las: path to input LAS/LAZ (e.g., your raster-extent clip).
        output_las: path to ground-only LAS/LAZ to write.
        pipeline_json: optional path to save the PDAL pipeline JSON.
                       If None, saves next to output LAS.
        resolution, iterations, threshold, smooth: CSF parameters.
        compress: if True, attempt LASzip compression.
        save_pipeline: if True, writes pipeline JSON.

    Returns:
        Path to the ground-only LAS/LAZ.
    """
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

    pipeline = {
        "pipeline": [
            {"type": "readers.las", "filename": str(input_las)},
            {
                "type": "filters.csf",
                "resolution": resolution,
                "iterations": iterations,
                "threshold": threshold,
                "smooth": smooth
            },
            {"type": "filters.range", "limits": "Classification[2:2]"},
            writer
        ]
    }

    os.makedirs(output_las.parent, exist_ok=True)
    os.makedirs(pipeline_json.parent, exist_ok=True)

    if save_pipeline:
        with open(pipeline_json, "w") as f:
            json.dump(pipeline, f, indent=2)
        print(f"💾 Saved PDAL pipeline JSON: {pipeline_json}")

    subprocess.run(["pdal", "pipeline", str(pipeline_json)], check=True)

    if not output_las.exists():
        raise FileNotFoundError("❌ Ground LAS was not written. Likely 0 ground points found.")

    print(f"✅ Ground LAS: {output_las}")
    return output_las


# === Example usage ===
raster_clip = "/media/laura/Laura 102/uas_data/site1_1_DD0001/lidar/trim/cloud1457eb47b07af181_clip_raster_extent.las"

# ground_las = run_csf(
#     input_las=raster_clip,
#     output_las="/media/laura/Laura 102/uas_data/site1_1_DD0001/lidar/trim/csf/cloud1457eb47b07af181_clip_raster_extent_csf_ground.las",
#     resolution=0.2,
#     iterations=500,
#     threshold=0.1,
#     smooth=False
# )

ground_las = run_csf(
    input_las=raster_clip,
    output_las="/media/laura/Laura 102/uas_data/site1_1_DD0001/lidar/trim/csf/smooth/cloud1457eb47b07af181_clip_raster_extent_csf_ground.las",
    pipeline_json="/home/laura/Documents/uas_data/Calperum/site1_supersite_DD0001/lidars/trim/csf/smooth/csf_clip_pipeline.json",
    resolution=0.08,     # 5–10 cm is a good start for UAS
    iterations=1000,
    threshold=0.04,      # tighten to avoid low veg
    smooth=True,         # slope smoothing helps terrain
)


# %%
#### test 3
import os
import json
import subprocess
from pathlib import Path

def run_csf(
    input_las: str | Path,
    output_las: str | Path,
    pipeline_json: str | Path = None,
    *,
    resolution: float = 0.2,
    iterations: int = 500,
    threshold: float = 0.1,
    smooth: bool = False,
    compress: bool = False,
    save_pipeline: bool = True
) -> Path:
    """
    Run PDAL Cloth Simulation Filtering (CSF) and write ground-only LAS.
    Ground = Classification 2 (filtered via filters.range).

    Args:
        input_las: path to input LAS/LAZ (e.g., your raster-extent clip).
        output_las: path to ground-only LAS/LAZ to write.
        pipeline_json: optional path to save the PDAL pipeline JSON.
                       If None, saves next to output LAS.
        resolution, iterations, threshold, smooth: CSF parameters.
        compress: if True, attempt LASzip compression.
        save_pipeline: if True, writes pipeline JSON.

    Returns:
        Path to the ground-only LAS/LAZ.
    """
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

    pipeline = {
        "pipeline": [
            {"type": "readers.las", "filename": str(input_las)},
            {
                "type": "filters.csf",
                "resolution": resolution,
                "iterations": iterations,
                "threshold": threshold,
                "smooth": smooth,
                "rigidness": 1           # softer cloth follows terrain; helps avoid classifying shrubs as ground
            },
            {"type": "filters.range", "limits": "Classification[2:2]"},
            writer
        ]
    }

    os.makedirs(output_las.parent, exist_ok=True)
    os.makedirs(pipeline_json.parent, exist_ok=True)

    # Always write the JSON so pdal can read it
    with open(pipeline_json, "w") as f:
        json.dump(pipeline, f, indent=2)
    if save_pipeline:
        print(f"💾 Saved PDAL pipeline JSON: {pipeline_json}")

    # Run with verbose and show real PDAL error if it fails
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
        raise FileNotFoundError("❌ Ground LAS was not written. Likely 0 ground points found.")

    print(f"✅ Ground LAS: {output_las}")
    return output_las


# === Example usage ===
raster_clip = "/media/laura/Laura 102/uas_data/site1_1_DD0001/lidar/trim/cloud1457eb47b07af181_clip_raster_extent.las"

ground_las = run_csf(
    input_las=raster_clip,
    output_las="/media/laura/Laura 102/uas_data/site1_1_DD0001/lidar/trim/csf/smoothest/cloud1457eb47b07af181_clip_raster_extent_csf_ground.las",
    pipeline_json="/home/laura/Documents/uas_data/Calperum/site1_supersite_DD0001/lidars/trim/csf/smoothest/csf_clip_pipeline.json",
    resolution=0.08,     # 5–10 cm is a good start for UAS
    iterations=1000,
    threshold=0.04,      # tighten to avoid low veg
    smooth=True,         # slope smoothing helps terrain
)

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
    # (CSF args kept for compatibility; not used by SMRF)
    resolution: float = 0.2,
    iterations: int = 500,
    threshold: float = 0.1,
    smooth: bool = False,
    # SMRF parameters (used)
    smrf_slope: float = 0.15,       # terrain slope factor (0–1)
    smrf_window: float = 2.0,       # base window size (meters)
    smrf_threshold: float = 0.15,   # elevation diff (meters)
    smrf_scalar: float = 1.25,      # window growth factor
    compress: bool = False,
    save_pipeline: bool = True
) -> Path:
    """
    Run PDAL SMRF (Simple Morphological Filter) and write ground-only LAS.
    Ground = Classification 2 (kept via filters.range).

    Args:
        input_las: source LAS/LAZ (e.g., your raster-extent clip).
        output_las: ground-only LAS/LAZ to write.
        pipeline_json: optional path to save the PDAL pipeline JSON (if None, saved next to output).
        smrf_slope, smrf_window, smrf_threshold, smrf_scalar: SMRF parameters.
        compress: if True and PDAL supports LASzip, compress output.
        save_pipeline: if True, write the pipeline JSON to disk.
    """
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

    pipeline = {
        "pipeline": [
            {"type": "readers.las", "filename": str(input_las)},
            {
                "type": "filters.smrf",
                "slope": smrf_slope,
                "window": smrf_window,
                "threshold": smrf_threshold,
                "scalar": smrf_scalar
            },
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
        raise FileNotFoundError("❌ Ground LAS was not written. Possibly 0 ground points after SMRF.")

    print(f"✅ Ground LAS: {output_las}")
    return output_las


# === Example usage (tuned for UAS + low shrubs) ===
raster_clip = "/media/laura/Laura 102/uas_data/site1_1_DD0001/lidar/trim/cloud1457eb47b07af181_clip_raster_extent.las"

ground_las = run_csf(
    input_las=raster_clip,
    output_las="/media/laura/Laura 102/uas_data/site1_1_DD0001/lidar/trim/csf/smrf/cloud1457eb47b07af181_clip_raster_extent_smrf_ground.las",
    pipeline_json="/home/laura/Documents/uas_data/Calperum/site1_supersite_DD0001/lidars/trim/csf/smrf/smrf_clip_pipeline.json",
    # SMRF params to reduce shrub-as-ground:
    smrf_slope=0.15,        # decrease if too aggressive on slopes
    smrf_window=2.0,        # try 1.0–3.0 for UAS
    smrf_threshold=0.12,    # tighten if shrubs still sneak in (e.g., 0.10)
    smrf_scalar=1.2,        # 1.1–1.4 commonly useful
    # old CSF args are ignored
)

# %%
