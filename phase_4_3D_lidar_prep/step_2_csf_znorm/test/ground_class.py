#%%
import os
import json
from pathlib import Path
import pdal

# === Inputs ===
# This script uses the output from Step 1.
raster_name = "tiles_multispectral.26"
step1_output_dir = Path("/media/laura/laura_usb/code/fvc_structure/phase_3_3D_lidar_prep/step_1_clipping/lidar_processing_steps")
input_classified_las = step1_output_dir / f"{raster_name}_step1_classified.las"

# === Outputs ===
output_ground_classified_las = step1_output_dir / f"{raster_name}_step2_ground_classified.las"

if not input_classified_las.exists():
    raise FileNotFoundError(f"Input file from Step 1 not found: {input_classified_las}")

# === PDAL Pipeline for Step 2 ===
pipeline_step2 = {
    "pipeline": [
        # 1. Read the classified point cloud from Step 1.
        { "type": "readers.las", "filename": str(input_classified_las) },

        # 2. Classify points into Ground (2) and Non-Ground (1).
        #    This overwrites the previous classification values.
        { "type": "filters.smrf",
          "slope": 0.15,
          "window": 2.0,
          "threshold": 0.12,
          "scalar": 1.2
        },

        # --- CRITICAL CHANGE ---
        # We REMOVED the "filters.range" that deleted non-ground points.
        # Now, all points are kept.

        # 3. Write the ground-classified file.
        {
            "type": "writers.las",
            "filename": str(output_ground_classified_las),
            "minor_version": 4
        }
    ]
}

print("\n--- EXECUTING STEP 2: Ground Classification ---")
pipe2 = pdal.Pipeline(json.dumps(pipeline_step2))
point_count2 = pipe2.execute()
print(f"✅ Step 2 complete.")
print(f"   Points written: {point_count2}")
print(f"   Final ground-classified LAS file saved to: {output_ground_classified_las}")

# %%
##test2
import os
import json
from pathlib import Path
import pdal
import rasterio

def classify_ground_conservative(
    input_las: Path,
    output_las: Path,
    reference_raster: Path,
    *,
    smrf_slope: float = 0.15,
    smrf_window: float = 1.8,  # Slightly reduced for finer detail
    smrf_threshold: float = 0.08, # Tightened to exclude low vegetation
    smrf_scalar: float = 1.2
) -> Path:
    """
    Runs a more conservative PDAL SMRF classification to better distinguish
    low vegetation from the true ground surface.

    Args:
        input_las: The source LAS/LAZ file.
        output_las: Path to save the ground-classified LAS/LAZ file.
        reference_raster: A raster used to define the processing extent and CRS.
        smrf_*: Tunable parameters for the Simple Morphological Filter.
    """
    print("--- Step 1: Running Conservative Ground Classification ---")
    
    with rasterio.open(reference_raster) as src:
        target_epsg = src.crs.to_epsg()
        minx, miny, maxx, maxy = src.bounds
    bounds_str = f"([{minx},{maxx}],[{miny},{maxy}])"
    print(f"Target EPSG: {target_epsg}")
    print(f"SMRF Parameters: window={smrf_window}m, threshold={smrf_threshold}m")

    pipeline = {
        "pipeline": [
            {"type": "readers.las", "filename": str(input_las)},
            {"type": "filters.reprojection", "out_srs": f"EPSG:{target_epsg}"},
            {"type": "filters.crop", "bounds": bounds_str},
            # This stage applies the tuned SMRF parameters
            {
                "type": "filters.smrf",
                "slope": smrf_slope,
                "window": smrf_window,
                "threshold": smrf_threshold,
                "scalar": smrf_scalar
            },
            # The output contains ALL points, now classified as Ground(2) or Non-Ground(1)
            {"type": "writers.las", "filename": str(output_las)}
        ]
    }

    try:
        output_las.parent.mkdir(parents=True, exist_ok=True)
        pipe = pdal.Pipeline(json.dumps(pipeline))
        count = pipe.execute()
        print(f"✅ Ground classification successful.")
        print(f"   Wrote {count} re-classified points to: {output_las}")
    except Exception as e:
        print(f"❌ PDAL pipeline failed: {e}")
        raise

    if not output_las.exists():
        raise FileNotFoundError("Output LAS file was not written.")

    return output_las


# === This block runs when you execute the script directly ===
if __name__ == "__main__":
    
    # === Inputs ===
    input_las = Path("/media/laura/laura_usb/uas_data/DD0001/lidar/clipped/raster/tiles_multispectral.26.las")
    aoi_raster_5cm = Path("/media/laura/laura_usb/qgis/calperumResearch/site1_1_DD0001/inputs/predictors/tiles_3072/stacked/tiles_multispectral.26.tif")

    # === Outputs ===
    raster_name = aoi_raster_5cm.stem
    out_dir = Path("./lidar_processing_steps_correct_order")
    output_ground_classified_las = out_dir / f"{raster_name}_step1_ground_classified_conservative.las"

    # --- Run the Classification with the new, tighter parameters ---
    classify_ground_conservative(
        input_las=input_las,
        output_las=output_ground_classified_las,
        reference_raster=aoi_raster_5cm,
        # You can continue to tune these if needed. If shrubs are still
        # being included, try reducing smrf_threshold even more (e.g., to 0.08).
        smrf_threshold=0.08,
        smrf_window=1.8
    )

    print("\n--- NEXT STEP ---")
    print("Visually inspect the new output file in CloudCompare.")
    print("Then, run the 'apply_mask_override.py' script to apply your final classes.")

# %%
