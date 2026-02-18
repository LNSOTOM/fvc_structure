#%%
import sys
import numpy as np
import rasterio
import time
import pylab as pl
from pathlib import Path

# === GeFolki dependencies ===
# Workstation
# sys.path.append('/media/laura/laura_usb/code/fvc_structure/phase_1_image_processing/gefolki/python')
# NVIDIA Instance
sys.path.append('/home/nvidia/projects/fvc_structure/phase_1_image_processing/gefolki/python')

from python.algorithm import EFolki
from python.tools import wrapData

# === Configuration ===
Base_band = 0

# === Input paths ===
# Workstation
# base_image_path   = Path("/media/laura/laura_usb/uas_data/DD0001/imagery/dsm/SASMDD0001_rgb_dsm_03.tif")
# moving_image_path = Path("/media/laura/laura_usb/uas_data/DD0001/lidar_l2/dsm/crop_to_rgb/SASMDD0001_l2_dsm_03.tif")

# NVIDIA
base_image_path   = Path("/data/calperumResearch/site1_1_DD0001/imagery/dsm/SASMDD0001_rgb_dsm_03.tif")
moving_image_path = Path("/data/calperumResearch/site1_1_DD0001/lidar_l2/dsm/crop_to_rgb/SASMDD0001_l2_dsm_03.tif")

#%%
# === Function Definitions ===

def load_data_and_mask(path, band_idx):
    """Load raster band data with its valid data mask (True=valid, False=nodata)."""
    with rasterio.open(path) as src:
        if band_idx >= src.count:
            raise ValueError(f"Band {band_idx} out of range for {path}")
        valid_mask = src.read_masks(band_idx + 1) > 0
        arr = src.read(band_idx + 1).astype(np.float32)
        arr[~valid_mask] = 0
        return arr, valid_mask

def normalize(arr):
    """Normalize array to zero mean, unit variance for stable EFolki convergence."""
    arr = arr.copy()
    arr = (arr - np.nanmean(arr)) / (np.nanstd(arr) + 1e-6)
    arr[np.isnan(arr)] = 0
    return arr

#%%
# === Load Input Rasters ===
base_data, base_mask = load_data_and_mask(base_image_path, Base_band)
moving_data, moving_mask = load_data_and_mask(moving_image_path, 0)

# === Verify Metadata Consistency ===
with rasterio.open(base_image_path) as base_src, rasterio.open(moving_image_path) as mov_src:
    if base_src.crs != mov_src.crs:
        print(f"⚠️ CRS mismatch: {base_src.crs} vs {mov_src.crs}")
    if base_src.res != mov_src.res:
        print(f"⚠️ Resolution mismatch: {base_src.res} vs {mov_src.res}")

#%%
# === Initial Visualization ===
pl.figure(figsize=(8,8)); pl.imshow(base_data, cmap='gray'); pl.title("Base (RGB DSM)")
pl.figure(figsize=(8,8)); pl.imshow(moving_data, cmap='gray'); pl.title("Moving (LiDAR DSM)")
pl.show()

#%%
# === Define Processing Mask ===
processing_mask = base_mask & moving_mask
base_data[~processing_mask] = 0
moving_data[~processing_mask] = 0

pl.figure(figsize=(8,8)); pl.imshow(processing_mask, cmap='gray'); pl.title("Valid Overlap Region")
pl.show()

#%%
# === Normalize Inputs ===
base_norm   = normalize(base_data)
moving_norm = normalize(moving_data)

#%%
# === Run GeFolki Optical Flow ===
print("🔹 Calculating optical flow (EFolki)...")
t0 = time.perf_counter()

u, v = EFolki(base_norm, moving_norm,
              iteration=8,
              radius=[32, 16, 8, 4],
              rank=4,
              levels=7)

t1 = time.perf_counter()
print(f"✅ Flow calculated in {(t1 - t0)/60:.3f} minutes")

# Mask flow outside valid region
u[~processing_mask] = 0
v[~processing_mask] = 0

# Visualize flow norm
N = np.sqrt(u**2 + v**2)
pl.figure(figsize=(8,8)); pl.imshow(N, cmap='viridis'); pl.title("Norm of Optical Flow"); pl.colorbar(); pl.show()

#%%
# === Warp LiDAR DSM and Its Mask ===
print("🔹 Warping LiDAR DSM to RGB DSM space...")
imag_reg = wrapData(moving_data, u, v)
warped_mask = wrapData(moving_mask.astype('float32'), u, v) > 0.5

pl.figure(figsize=(8,8)); pl.imshow(imag_reg, cmap='gray'); pl.title("Registered LiDAR DSM"); pl.show()

#%%
# === Save Results ===
print("\n--- Saving Outputs ---")

flow_dir = base_image_path.parent
aligned_path = moving_image_path.with_name(moving_image_path.stem + "_coreg.tif")

with rasterio.open(base_image_path) as src:
    profile = src.profile
    transform = src.transform
    crs = src.crs
    nodata_value = src.nodata if src.nodata is not None else -9999.0

# Apply nodata mask
imag_reg[~warped_mask] = nodata_value

# Update profile and write
profile.update(dtype='float32', nodata=nodata_value, transform=transform, crs=crs)
with rasterio.open(aligned_path, "w", **profile) as dst:
    dst.write(imag_reg.astype('float32'), 1)

print(f"✅ Registered DSM saved to: {aligned_path}")

# Save optical flow fields
np.save(flow_dir / "flow_u_03_l2.npy", u)
np.save(flow_dir / "flow_v_03_l2.npy", v)
print(f"💾 Flow fields saved to: {flow_dir}")


#%%
# === Optional: Residual Quality Check ===
residual = base_data - imag_reg
pl.figure(figsize=(8,8))
pl.imshow(residual, cmap='RdBu', vmin=-0.5, vmax=0.5)
pl.title("Residual (Base - Registered LiDAR)")
pl.colorbar(label="m")
pl.show()

# %%
