
# %%
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


# === tuned for UAS + low shrubs) ===
#low
# raster_clip = "/media/laura/Laura 102/uas_data/site1_1_DD0001/lidar/trim/cloud1457eb47b07af181_clip_raster_extent.las"

#medium
raster_clip = "/media/laura/laura_usb/qgis/calperumResearch/site1_1_DD0001/lidar/clipped/20220519_SASMDD0001_triple_AusGeoid_tiles_multispectral.26.las"
# raster_clip = "/media/laura/laura_usb/qgis/calperumResearch/site1_1_DD0001/lidar/clipped/20220519_SASMDD0001_triple_AusGeoid_tiles_multispectral.109.las"


ground_las = run_csf(
    input_las=raster_clip,
    output_las="/media/laura/laura_usb/qgis/calperumResearch/site1_1_DD0001/lidar/clipped/csf/triple_AusGeoid_tiles_multispectral.26-smrf_ground.las",
    pipeline_json="/media/laura/laura_usb/qgis/calperumResearch/site1_1_DD0001/lidar/clipped/csf/smrf_clip_pipeline.json",
    # SMRF params to reduce shrub-as-ground:
    smrf_slope=0.15,        # decrease if too aggressive on slopes
    smrf_window=2.0,        # try 1.0–3.0 for UAS
    smrf_threshold=0.12,    # tighten if shrubs still sneak in (e.g., 0.10)
    smrf_scalar=1.2,        # 1.1–1.4 commonly useful
)
