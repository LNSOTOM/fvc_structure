
#%%
import os
import json
from pathlib import Path
import pdal

# === Inputs ===
# This script uses the output from Step 2.
raster_name = "tiles_multispectral.26"
step2_output_dir = Path("/media/laura/laura_usb/code/fvc_structure/phase_3_3D_lidar_prep/step_1_clipping/lidar_processing_steps")
input_ground_classified_las = step2_output_dir / f"{raster_name}_step2_ground_classified.las"

# === Outputs ===
output_normalized_las = step2_output_dir / f"{raster_name}_step3_final_normalized.las"

if not input_ground_classified_las.exists():
    raise FileNotFoundError(f"Input file from Step 2 not found: {input_ground_classified_las}")

# === PDAL Pipeline for Step 3 ===
pipeline_step3 = {
    "pipeline": [
        # 1. Read the ground-classified point cloud from Step 2.
        { "type": "readers.las", "filename": str(input_ground_classified_las) },

        # 2. Calculate Height Above Ground.
        #    This filter automatically uses points with Classification=2 to build
        #    a terrain model and calculates the height for ALL points.
        { "type": "filters.hag_nn" },

        # 3. Write the final file with the new 'HeightAboveGround' dimension.
        {
            "type": "writers.las",
            "filename": str(output_normalized_las),
            "minor_version": 4,
            "extra_dims": "all"
        }
    ]
}

print("\n--- EXECUTING STEP 3: Normalizing Height ---")
pipe3 = pdal.Pipeline(json.dumps(pipeline_step3))
point_count3 = pipe3.execute()
print(f"✅ Step 3 complete.")
print(f"   Points written: {point_count3}")
print(f"   Final normalized LAS file saved to: {output_normalized_las}")


# %%
##test 2
import json
from pathlib import Path
import pdal

# --- Configuration ---
# File paths are set based on your previous scripts and logs.

# Input: Your accurately classified point cloud.
GROUND_CLASSIFIED_FILE = Path("/media/laura/laura_usb/code/fvc_structure/phase_3_3D_lidar_prep/step_2_csf/lidar_processing_steps_correct_order/tiles_multispectral.26_step1_ground_classified_conservative.las")

# Output: The final, correctly normalized LAS file.
NORMALIZED_OUTPUT_FILE = Path("/media/laura/laura_usb/code/fvc_structure/phase_3_3D_lidar_prep/step_1_clipping/lidar_processing_steps/tiles_multispectral.26_normalized_height.las")

# --- Pre-run Check ---
if not GROUND_CLASSIFIED_FILE.exists():
    raise FileNotFoundError(f"Input file not found: {GROUND_CLASSIFIED_FILE}")

# === Build the Corrected PDAL Normalization Pipeline ===
pipeline_definition = {
    "pipeline": [
        {
            "type": "readers.las",
            "filename": str(GROUND_CLASSIFIED_FILE)
        },
        {
            "type": "filters.hag_nn"
        },
        # --- KEY CORRECTION ---
        # Instead of 'assign', we use 'ferry' to explicitly move the new
        # dimension into Z. This is a more robust method.
        {
            "type": "filters.ferry",
            "dimensions": "HeightAboveGround => Z"
        },
        {
            "type": "writers.las",
            "filename": str(NORMALIZED_OUTPUT_FILE),
            # --- KEY CORRECTION ---
            # We explicitly set the scale and offset for the new Z values
            # to create a clean header for our normalized data. This prevents
            # the "under the plane" issue.
            "scale_z": 0.01,
            "offset_z": "auto"
        }
    ]
}

# --- Execute the Pipeline ---
try:
    print("\n--- EXECUTING CORRECTED NORMALIZATION PIPELINE ---")
    pipeline = pdal.Pipeline(json.dumps(pipeline_definition))
    point_count = pipeline.execute()
    print(f"✅ Normalization complete.")
    print(f"   Points written: {point_count}")
    print(f"   Final normalized LAS file saved to: {NORMALIZED_OUTPUT_FILE}")

except Exception as e:
    print(f"An error occurred during PDAL execution: {e}")

# %%
