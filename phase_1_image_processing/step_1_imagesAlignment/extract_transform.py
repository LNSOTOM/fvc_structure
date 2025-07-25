#%%
import rasterio

# Paths to original and aligned DEMs
original_dem_path = "/media/laura/laura_usb/qgis/calperumResearch/site2_1_DD0011/lidar/cloud2453cf90763afd91_dem_5cm_normalized_resampled.tif"
aligned_dem_path = original_dem_path.replace(".tif", "_aligned_to_micasense.tif")

# Load transforms
with rasterio.open(original_dem_path) as src_orig:
    transform_orig = src_orig.transform

with rasterio.open(aligned_dem_path) as src_aligned:
    transform_aligned = src_aligned.transform

# Extract origins (top-left coordinates)
x_orig, y_orig = transform_orig.c, transform_orig.f
x_aligned, y_aligned = transform_aligned.c, transform_aligned.f

# Compute translation
dx = x_aligned - x_orig
dy = y_aligned - y_orig

print("📐 Original origin: ", (x_orig, y_orig))
print("📐 Aligned origin : ", (x_aligned, y_aligned))
print("📦 Estimated translation (meters):")
print("   ΔX:", dx)
print("   ΔY:", dy)

# %%
