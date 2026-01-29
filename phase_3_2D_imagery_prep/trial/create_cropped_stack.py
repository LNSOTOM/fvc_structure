#!/usr/bin/env python3
"""
Create a cropped and properly formatted version of the dominant labels stack
"""

import numpy as np
import rasterio
import pandas as pd

# Load the existing tensors
print("Loading voxel tensors...")
data = np.load("unet_voxel_tensors.npz", allow_pickle=True)

occupancy = data['occupancy']
dominant_labels = data['dominant_labels']
voxel_size = data['voxel_size_m'].item()
origin = data['origin'].item()

print(f"Original tensor shape: {occupancy.shape}")
print(f"Occupied voxels: {np.sum(occupancy):,}")

# Load CSV to find the actual data bounds
df = pd.read_csv("xyz_norm_with_labels.csv")

# Load mask to get transform
mask_path = "/media/laura/laura_usb/qgis/calperumResearch/site1_1_DD0001/inputs/masks/tiles_3072/raw/fvc_class/annotation_raster/dowsample_5cm_bilinear/mask_fvc_3072.26.tif"
with rasterio.open(mask_path) as src:
    transform = src.transform
    crs = src.crs

# Convert point coordinates to pixel coordinates
cols, rows = rasterio.transform.rowcol(
    transform,
    df["X"].to_numpy(),
    df["Y"].to_numpy(),
    op=float,
)
cols = np.array(cols, dtype=int)
rows = np.array(rows, dtype=int)

# Find the bounding box of actual data
min_row, max_row = rows.min(), rows.max()
min_col, max_col = cols.min(), cols.max()

# Add some padding
padding = 50  # pixels
min_row = max(0, min_row - padding)
max_row = min(occupancy.shape[1] - 1, max_row + padding)
min_col = max(0, min_col - padding)
max_col = min(occupancy.shape[2] - 1, max_col + padding)

print(f"Data bounds: rows {min_row}-{max_row}, cols {min_col}-{max_col}")
print(f"Cropped size: {max_row - min_row + 1} x {max_col - min_col + 1}")

# Crop the data
cropped_dominant = dominant_labels[:, min_row:max_row+1, min_col:max_col+1]
cropped_occupancy = occupancy[:, min_row:max_row+1, min_col:max_col+1]

print(f"Cropped tensor shape: {cropped_dominant.shape}")

# Create a cropped transform
cropped_transform = rasterio.transform.from_bounds(
    transform.c + min_col * transform.a,  # west
    transform.f + (max_row + 1) * transform.e,  # south
    transform.c + (max_col + 1) * transform.a,  # east
    transform.f + min_row * transform.e,  # north
    max_col - min_col + 1,  # width
    max_row - min_row + 1   # height
)

# Prepare data for export with proper nodata handling
nz, ny, nx = cropped_dominant.shape

# Set nodata value
nodata_value = 255  # Use 255 as nodata instead of -1

# Create the output with proper nodata
output_data = cropped_dominant.copy().astype(np.uint8)
output_data[cropped_occupancy == 0] = nodata_value

# Only export layers that have some data
layers_with_data = []
for z in range(nz):
    if np.any(cropped_occupancy[z] > 0):
        layers_with_data.append(z)

print(f"Layers with data: {len(layers_with_data)} out of {nz}")

# Save cropped version
profile = {
    "driver": "GTiff",
    "dtype": "uint8",
    "nodata": nodata_value,
    "width": nx,
    "height": ny,
    "count": len(layers_with_data),
    "transform": cropped_transform,
    "crs": crs,
    "compress": "lzw",
}

output_file = "dominant_labels_cropped.tif"
print(f"Saving cropped stack to {output_file}...")

with rasterio.open(output_file, "w", **profile) as dst:
    for band_idx, z in enumerate(layers_with_data, 1):
        dst.write(output_data[z, :, :], band_idx)
        # Set band description
        dst.set_band_description(band_idx, f"Height_slice_{z*voxel_size['z']:.2f}m")

print("✓ Cropped stack saved!")

# Create summary statistics
print("\nSummary:")
print(f"- Original size: {occupancy.shape[1]} x {occupancy.shape[2]} pixels")
print(f"- Cropped size: {ny} x {nx} pixels")
print(f"- Size reduction: {(1 - (ny*nx)/(occupancy.shape[1]*occupancy.shape[2]))*100:.1f}%")
print(f"- Layers with data: {len(layers_with_data)}")
print(f"- Height range: {layers_with_data[0]*voxel_size['z']:.2f}m to {layers_with_data[-1]*voxel_size['z']:.2f}m")
print(f"- Nodata value: {nodata_value}")
print("- Label range: 0-3 (vegetation classes)")

# Show label statistics for the cropped data
print("\nLabel statistics in cropped data:")
for z in layers_with_data[:5]:  # Show first 5 layers
    layer_data = output_data[z, :, :]
    valid_data = layer_data[layer_data != nodata_value]
    if len(valid_data) > 0:
        unique_labels = np.unique(valid_data)
        print(f"  Layer {z} ({z*voxel_size['z']:.2f}m): labels {unique_labels}, {len(valid_data)} valid pixels")