#%%
import sys
import numpy as np
import rasterio
import time
import pylab as pl
from pathlib import Path
from rasterio.warp import reproject, Resampling

# === GeFolki dependencies (vendored library, resolved relative to this file) ===
from phase2_config import add_gefolki_to_path, load_config

add_gefolki_to_path()

from python.algorithm import EFolki
from python.tools import wrapData

# === Configuration (flow section of the job config) ===
flow_cfg = load_config()["flow"]

Base_band = int(flow_cfg.get("base_band", 0))

# === Input paths ===
base_image_path = Path(flow_cfg["base_image"])
moving_image_path = Path(flow_cfg["moving_image"])


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

def preprocess_dsm(arr, mask):
    """Remove absolute offset and normalize for EFolki."""
    arr = arr.astype(np.float32)
    arr[~mask] = np.nan
    arr = arr - np.nanmean(arr)          # remove mean (offset)
    arr = arr / (np.nanstd(arr) + 1e-6)  # standardize contrast
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
    if base_src.transform != mov_src.transform:
        print("⚠️ Pixel grid differs → Reprojecting LiDAR DSM to RGB grid.")
        dst = np.empty_like(base_data)
        reproject(
            source=moving_data,
            destination=dst,
            src_transform=mov_src.transform,
            src_crs=mov_src.crs,
            dst_transform=base_src.transform,
            dst_crs=base_src.crs,
            resampling=Resampling.bilinear
        )
        moving_data = dst

#%%
# === Initial Visualization ===
pl.figure(figsize=(8,8))
pl.imshow(base_data, cmap='gray')
pl.title("Base (RGB DSM)")

pl.figure(figsize=(8,8))
pl.imshow(moving_data, cmap='gray')
pl.title("Moving (LiDAR DSM)")

pl.show()

#%%
# === Define Processing Mask ===
processing_mask = base_mask & moving_mask
base_data[~processing_mask] = 0
moving_data[~processing_mask] = 0

pl.figure(figsize=(8,8))
pl.imshow(processing_mask, cmap='gray')
pl.title("Valid Overlap Region")
pl.show()

#%%
# === Normalize Inputs ===
base_norm   = preprocess_dsm(base_data, processing_mask)
moving_norm = preprocess_dsm(moving_data, processing_mask)

#%%
# === Run GeFolki Optical Flow ===
print("🔹 Calculating optical flow (EFolki)...")
t0 = time.perf_counter()

u, v = EFolki(base_norm, moving_norm,
              iteration=int(flow_cfg.get("iteration", 8)),
              radius=list(flow_cfg.get("radius", [8, 4, 2])),
              rank=int(flow_cfg.get("rank", 4)),
              levels=int(flow_cfg.get("levels", 6)))

t1 = time.perf_counter()
print(f"✅ Flow calculated in {(t1 - t0)/60:.3f} minutes")

# Mask flow outside valid region
u[~processing_mask] = 0
v[~processing_mask] = 0

# Flow diagnostics
print(f"Mean flow: u={np.nanmean(u):.3f}, v={np.nanmean(v):.3f}, std(u)={np.nanstd(u):.3f}, std(v)={np.nanstd(v):.3f}")

# Visualize flow norm
N = np.sqrt(u**2 + v**2)
pl.figure(figsize=(8,8))
pl.imshow(N, cmap='viridis')
pl.title("Norm of Optical Flow")
pl.colorbar()
pl.show()

#%%
# === Warp LiDAR DSM and Its Mask ===
# print("🔹 Warping LiDAR DSM to RGB DSM space...")
# imag_reg = wrapData(moving_data.astype('float64'), u, v)
# warped_mask = wrapData(moving_mask.astype('float32'), u, v) > 0.5

# pl.figure(figsize=(8,8))
# pl.imshow(imag_reg, cmap='gray')
# pl.title("Registered LiDAR DSM")
# pl.show()

#%%
#same as before but palette changed
# === Register the LiDAR data AND its footprint ===
print("Registering image...\n")
# 1. Warp the LiDAR elevation data
imag_reg = wrapData(moving_data, u, v)

# 2. Warp the LiDAR's original footprint (its mask) using the SAME flow field.
#    This is the key step to correctly track where the nodata areas should be.
warped_mask = wrapData(moving_mask.astype('float32'), u, v) > 0.5

pl.figure(figsize=(8,8)); pl.imshow(imag_reg); pl.title("Registered LiDAR DSM")
pl.show()

#%%
# === Save Results ===
print("\n--- Saving Outputs ---")

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
flow_u_path = Path(flow_cfg["flow_u_output"])
flow_v_path = Path(flow_cfg["flow_v_output"])
flow_u_path.parent.mkdir(parents=True, exist_ok=True)
flow_v_path.parent.mkdir(parents=True, exist_ok=True)
np.save(flow_u_path, u)
np.save(flow_v_path, v)
print(f"💾 Flow fields saved to: {flow_u_path} and {flow_v_path}")

#%%
# === Optional: Residual Quality Check ===
residual = base_data - imag_reg
vmin, vmax = np.percentile(residual[processing_mask], [1, 99])
pl.figure(figsize=(8,8))
pl.imshow(residual, cmap='RdBu', vmin=vmin, vmax=vmax)
pl.title("Residual (Base - Registered LiDAR)")
pl.colorbar(label="m")
pl.show()

# Save difference GeoTIFF (use a distinct name so the registered DSM is not overwritten)
diff_path = moving_image_path.with_name(moving_image_path.stem + "_coreg_residual.tif")
profile.update(dtype='float32')
with rasterio.open(diff_path, "w", **profile) as dst:
    dst.write(residual.astype('float32'), 1)
print(f"💾 Difference map saved to: {diff_path}")
# %%
