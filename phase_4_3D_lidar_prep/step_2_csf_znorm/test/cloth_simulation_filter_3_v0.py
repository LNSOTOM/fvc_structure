
# %%
# process_csf_hag.py
from run_csf_ground_filter_1 import run_csf_ground_filter
from normalise_las_z_2 import normalize_las_z

# === Step 1: Extract Ground with CSF ===
# L1
# input_tile = "/media/laura/laura_usb/uas_data/DD0001/lidar/clipped/raster/coreg/icp/icp_tiles_multispectral.109.las"
# ground_out = "/media/laura/laura_usb/uas_data/DD0001/lidar/clipped/csf/csf_tiles_multispectral.109.las"
# L2
input_tile = "/media/laura/laura_usb/uas_data/DD0001/lidar_l2/clipped/raster/coreg/gauss/icp/icp_tiles_multispectral.109.las"
ground_out = "/media/laura/laura_usb/uas_data/DD0001/lidar_l2/clipped/raster/csf/csf_tiles_multispectral.109.las"

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
norm_out = "/media/laura/laura_usb/uas_data/DD0001/lidar_l2/clipped/z_norm/z_norm_tiles_multispectral.109.las"

normalize_las_z(
    input_las=input_tile,
    ground_las=ground_out,
    output_las=norm_out,
    k=3
)


# %%
