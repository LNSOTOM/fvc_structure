#%%

import sys
import numpy as np
import rasterio
import time
import pylab as pl
from pathlib import Path
from PIL import Image
from matplotlib import cm

# === GeFolki dependencies ===
sys.path.append('/media/laura/laura_usb/code/fvc_structure/phase_1_image_processing/gefolki/python')
from python.algorithm import EFolki
from python.tools import wrapData

# === Toggle: Select band from base image ===
Base_band = 0  # 0=Red, 1=Green, 2=Blue (if using RGB orthomosaic)

# === Input paths ===
# base_image_path = Path("/media/laura/laura_usb/uas_data/DD0001/imagery/dsm/SASMDD0001_rgb_dsm_05.tif")
# moving_image_path = Path("/media/laura/laura_usb/uas_data/DD0001/lidar/dsm/crop_to_rgb/SASMDD0001_l1_dsm_05.tif")
base_image_path = Path("/media/laura/laura_usb/uas_data/DD0001/imagery/dsm/SASMDD0001_rgb_dsm_05.tif")
moving_image_path = Path("/media/laura/laura_usb/uas_data/DD0001/lidar/dsm/crop_to_rgb/SASMDD0001_l1_dsm_03.tif")  # normalized to [0,1]



# === Load selected band from base RGB image ===
def load_band(path, band_idx):
    with rasterio.open(path) as src:
        if band_idx >= src.count:
            raise ValueError(f"Requested band index {band_idx} exceeds total bands ({src.count})")
        arr = src.read(band_idx + 1).astype(np.float32)
        nodata = src.nodata if src.nodata is not None else -9999
        arr[arr == nodata] = 0
        profile = src.profile
        return arr, profile


# === Load base and moving ===
def resample_image(image, reference_shape):
    """Resample image to the shape of a reference image."""
    return np.array(Image.fromarray(image).resize((reference_shape[1], reference_shape[0]), Image.BILINEAR))

base_band_data, profile = load_band(base_image_path, Base_band)
moving_image_orig, _ = load_band(moving_image_path, 0)  # DEM is single-band

# Resample moving image to match base image dimensions
moving_image = resample_image(moving_image_orig, base_band_data.shape)

# === Plot base band and moving ===
pl.figure()
pl.imshow(base_band_data, cmap='gray')
pl.title("Base image DSM from RGB dense cloud")

pl.figure()
pl.imshow(moving_image, cmap='gray')
pl.title("Moving image DSM from LiDAR")

# %%
# === Apply mask from base ===
mask = base_band_data > 0.0
base_band_data *= mask
moving_image *= mask

pl.figure()
pl.imshow(moving_image, cmap='gray')
pl.title("Moving image masked")
pl.colorbar()

# %%
# === Calculate optical flow ===
print("Calculating flow...\n")
t0 = time.perf_counter()
u, v = EFolki(base_band_data, moving_image, iteration=8, radius=[16, 8, 4], rank=4, levels=6)
t1 = time.perf_counter()
print(f"✅ Flow calculated in {(t1 - t0)/60:.3f} minutes")

# === Norm of flow ===
N = np.sqrt(u**2 + v**2)
pl.figure()
pl.imshow(N, cmap='viridis')
pl.title("Norm of Optical Flow")
pl.colorbar()

# %%
# === Register image ===
print("Registering image...\n")
imag_reg = wrapData(moving_image, u, v)

pl.figure()
pl.imshow(imag_reg)
pl.title("Registered image")

#%%
pl.figure()
pl.title("Before/After Overlay")
pl.imshow(moving_image, cmap='gray')
pl.imshow(imag_reg, cmap=cm.inferno, alpha=0.5)

#%%
pl.figure()
pl.imshow(np.abs(moving_image - imag_reg), cmap='magma')
pl.title("Absolute Difference (Before vs Registered)")
pl.colorbar()


#%%
# === Save flow and registered image ===
flow_dir = base_image_path.parent
np.save(flow_dir / "flow_u_03.npy", u)
np.save(flow_dir / "flow_v_03.npy", v)
print(f"💾 Flow saved to: {flow_dir/'flow_u_03.npy'} and {flow_dir/'flow_v_03.npy'}")

aligned_path = moving_image_path.with_name(moving_image_path.stem + "_coreg.tif")
profile.update(dtype='float32')

with rasterio.open(aligned_path, "w", **profile) as dst:
    dst.write(imag_reg.astype('float32'), 1)

print(f"✅ Registered image saved to: {aligned_path}")


# %%
# === OPTIONAL: Apply warp to all bands in a multiband image ===
print("\n🌀 Applying warp to all bands in a 3D array...")


# === Load full multiband stack (example: multispectral tif) ===
# For demo: simulate a stack with 3 identical bands
moving_stack = np.stack([moving_image] * 3, axis=2)

# Apply mask to every band
expanded_mask = np.expand_dims(mask, axis=2)
moving_stack *= expanded_mask

# Allocate output
processed_array = np.empty_like(moving_stack)

# Warp each band
t0 = time.perf_counter()
for band in range(moving_stack.shape[2]):
    print(f"Processing band {band + 1}/{moving_stack.shape[2]}...")
    im = moving_stack[:, :, band]
    processed_array[:, :, band] = wrapData(im, u, v)
t1 = time.perf_counter()
print("⏱️ It took %.3f minutes to apply warp to all bands\n" % ((t1 - t0) / 60))
# %%
