
# %%
# process_csf_hag.py
import importlib
import sys

# Force reload of modules to avoid caching issues
if 'run_csf_ground_filter_1_onefile' in sys.modules:
    importlib.reload(sys.modules['run_csf_ground_filter_1_onefile'])
if 'normalise_las_z_2' in sys.modules:
    importlib.reload(sys.modules['normalise_las_z_2'])

from run_csf_ground_filter_1_onefile import run_csf_ground_filter, run_smrf_ground_filter
from normalise_las_z_2 import normalize_las_z


# Input tile (L1)
# input_tile = "/media/laura/laura_usb/uas_data/DD0001/lidar/clipped/raster/coreg/icp/icp_tiles_multispectral.109.las"
input_tile = "/data/calperumResearch/site1_1_DD0001/lidar/clipped/raster/coreg/icp/tiles_multispectral.113.las"

# Input tile (L2)
# input_tile = "/media/laura/laura_usb/uas_data/DD0001/lidar_l2/clipped/raster/coreg/gauss/icp/icp_tiles_multispectral.109.las"

# === Step 1: Extract Ground with SMRF ===
ground_out_smrf = run_smrf_ground_filter(
    input_las=input_tile,
    #l1
    output_las="/data/calperumResearch/site1_1_DD0001/lidar/clipped/raster/ground_classification/smrf/smrf_tiles_multispectral.113.las",
    pipeline_json="/data/calperumResearch/site1_1_DD0001/lidar/clipped/raster/ground_classification/smrf/smrf_tiles_multispectral.113.json",
    #l2
    # output_las="/media/laura/laura_usb/uas_data/DD0001/lidar_l2/clipped/raster/ground_classification/smrf_tiles_multispectral.109.las",
    # pipeline_json="/media/laura/laura_usb/uas_data/DD0001/lidar_l2/clipped/raster/ground_classification/smrf_tiles_multispectral.109.json",
    slope=0.15, #0.10
    window=3.5,  #2.8
    threshold=0.25,  #0.20
    scalar=1.35,
)

# === Step 2: Extract Ground with CSF ===
ground_out_csf = run_csf_ground_filter(
    input_las=input_tile,
    #l1
    output_las="/data/calperumResearch/site1_1_DD0001/lidar/clipped/raster/ground_classification/csf/csf_tiles_multispectral.113.las",
    #l2
    # output_las="/media/laura/laura_usb/uas_data/DD0001/lidar_l2/clipped/raster/ground_classification/csf_tiles_multispectral.109.las",
    resolution=0.5,  #0.2 #0.3
    iterations=500,
    threshold=0.01,  #0.01, #0.09  #0.03
    rigidness=2,   # flexible cloth for shrubs= 1
    smooth=False,  #False=# Disable post-smoothing
)

# === Step 3a: Normalize Z (Height Above Ground) using SMRF ground ===
#l1
norm_out_smrf = "/data/calperumResearch/site1_1_DD0001/lidar/clipped/raster/z_norm/smrf/z_norm_smrf_tiles_multispectral.113.las"
#l2
# norm_out_smrf = "/media/laura/laura_usb/uas_data/DD0001/lidar_l2/clipped/z_norm/z_norm_smrf_tiles_multispectral.109.las"

normalize_las_z(
    input_las=input_tile,
    ground_las=ground_out_smrf,  # Using SMRF ground for normalization
    output_las=norm_out_smrf,
    k=3
)

# === Step 3b: Normalize Z (Height Above Ground) using CSF ground ===
#l1
norm_out_csf = "/data/calperumResearch/site1_1_DD0001/lidar/clipped/raster/z_norm/csf/z_norm_csf_tiles_multispectral.113.las"
#l2
# norm_out_csf = "/media/laura/laura_usb/uas_data/DD0001/lidar_l2/clipped/z_norm/z_norm_csf_tiles_multispectral.109.las"

normalize_las_z(
    input_las=input_tile,
    ground_las=ground_out_csf,  # Using CSF ground for normalization
    output_las=norm_out_csf,
    k=3
)


# %%
