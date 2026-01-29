
# %%
# process_csf_hag.py
import importlib
import sys
from pathlib import Path
from tqdm import tqdm

# Force reload of modules to avoid caching issues
if 'run_csf_ground_filter_1' in sys.modules:
    importlib.reload(sys.modules['run_csf_ground_filter_1'])
if 'normalise_las_z_2' in sys.modules:
    importlib.reload(sys.modules['normalise_las_z_2'])

from run_csf_ground_filter_1 import run_csf_ground_filter, run_smrf_ground_filter
from normalise_las_z_2 import normalize_las_z

#SITE 1
# Input folder containing LAS tiles (L1)
# input_folder = Path("/data/calperumResearch/site1_1_DD0001/lidar/clipped/raster/coreg/icp")
# output_base = Path("/data/calperumResearch/site1_1_DD0001/lidar/clipped/raster")

# Input folder (L2)
# input_folder = Path("/media/laura/laura_usb/uas_data/DD0001/lidar_l2/clipped/raster/coreg/gauss/icp")
# output_base = Path("/media/laura/laura_usb/uas_data/DD0001/lidar_l2/clipped/raster")


#SITE 11
# Input folder containing LAS tiles (L1)
input_folder = Path("/data/calperumResearch/site2_1_DD0011/lidar/clipped/raster/coreg/icp")
output_base = Path("/data/calperumResearch/site2_1_DD0011/lidar/clipped/raster")



# Output directories
ground_classification_smrf_dir = output_base / "ground_classification" / "smrf"
ground_classification_csf_dir = output_base / "ground_classification" / "csf"
z_norm_smrf_dir = output_base / "z_norm" / "smrf"
z_norm_csf_dir = output_base / "z_norm" / "csf"

# Create output directories if they don't exist
ground_classification_smrf_dir.mkdir(parents=True, exist_ok=True)
ground_classification_csf_dir.mkdir(parents=True, exist_ok=True)
z_norm_smrf_dir.mkdir(parents=True, exist_ok=True)
z_norm_csf_dir.mkdir(parents=True, exist_ok=True)

# Get all LAS files in the input folder
las_files = sorted(input_folder.glob("*.las"))

if not las_files:
    print(f"No LAS files found in {input_folder}")
    sys.exit(1)

print(f"Found {len(las_files)} LAS files to process")

# Process each tile
for input_tile in tqdm(las_files, desc="Processing tiles"):
    tile_name = input_tile.stem
    print(f"\n=== Processing {tile_name} ===")
    
    # Define output paths for this tile
    smrf_ground = ground_classification_smrf_dir / f"smrf_{tile_name}.las"
    smrf_json = ground_classification_smrf_dir / f"smrf_{tile_name}.json"
    csf_ground = ground_classification_csf_dir / f"csf_{tile_name}.las"
    norm_smrf = z_norm_smrf_dir / f"z_norm_smrf_{tile_name}.las"
    norm_csf = z_norm_csf_dir / f"z_norm_csf_{tile_name}.las"
    
    # === Step 1: Extract Ground with SMRF ===
    try:
        ground_out_smrf = run_smrf_ground_filter(
            input_las=str(input_tile),
            output_las=str(smrf_ground),
            pipeline_json=str(smrf_json),
            slope=0.15,
            window=3.5,
            threshold=0.25,
            scalar=1.35,
        )
    except Exception as e:
        print(f"SMRF failed for {tile_name}: {e}")
        continue

    # === Step 2: Extract Ground with CSF ===
    try:
        ground_out_csf = run_csf_ground_filter(
            input_las=str(input_tile),
            output_las=str(csf_ground),
            resolution=0.5,
            iterations=500,
            threshold=0.01,
            rigidness=2,
            smooth=False,
        )
    except Exception as e:
        print(f"CSF failed for {tile_name}: {e}")
        continue

    # === Step 3a: Normalize Z using SMRF ground ===
    try:
        normalize_las_z(
            input_las=str(input_tile),
            ground_las=ground_out_smrf,
            output_las=str(norm_smrf),
            k=3
        )
    except Exception as e:
        print(f"SMRF normalization failed for {tile_name}: {e}")

    # === Step 3b: Normalize Z using CSF ground ===
    try:
        normalize_las_z(
            input_las=str(input_tile),
            ground_las=ground_out_csf,
            output_las=str(norm_csf),
            k=3
        )
    except Exception as e:
        print(f"CSF normalization failed for {tile_name}: {e}")

print("\n✅ All tiles processed!")


# %%
