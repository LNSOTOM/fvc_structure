#!/usr/bin/env python3
"""
Convert labelled point CSV to 3D U-Net tensors as described in README_3d_unet.md
"""

from pathlib import Path
import json
import numpy as np
import pandas as pd
import rasterio

# --- configuration ---------------------------------------------------------
workdir = Path(".")  # Current directory
csv_path = "xyz_norm_with_labels.csv"
meta_path = "voxel_points_meta.json"  # This might not exist, we'll handle it
mask_path = Path("/media/laura/laura_usb/qgis/calperumResearch/site1_1_DD0001/inputs/masks/tiles_3072/raw/fvc_class/annotation_raster/dowsample_5cm_bilinear/mask_fvc_3072.26.tif")
voxel_size_override = 0.05  # Force 5cm cubic voxels to match downsampled mask

print("Starting 3D U-Net tensor creation...")
print(f"CSV path: {csv_path}")
print(f"Mask path: {mask_path}")

# --- load inputs -----------------------------------------------------------
try:
    df = pd.read_csv(csv_path)
    print(f"Loaded CSV with {len(df)} points")
    print(f"Columns: {list(df.columns)}")
    print(f"Unique labels: {df['Label'].unique()}")
except FileNotFoundError:
    print(f"Error: CSV file not found: {csv_path}")
    print("Please run voxel_point.py first to generate the CSV file.")
    exit(1)

# Try to load metadata, create fallback if not available
try:
    meta = json.loads(Path(meta_path).read_text())
    print("Loaded metadata from file")
except FileNotFoundError:
    print("Metadata file not found, creating from CSV data...")
    meta = {
        "z_range": {
            "min": float(df["Z_norm"].min()),
            "max": float(df["Z_norm"].max())
        }
    }
    print(f"Z range: {meta['z_range']['min']:.3f} to {meta['z_range']['max']:.3f}")

try:
    with rasterio.open(mask_path) as src:
        transform = src.transform
        width, height = src.width, src.height
        pixel_size = abs(transform.a)
        crs = src.crs
        assert np.isclose(pixel_size, abs(transform.e)), "Mask pixels must be square"
        print(f"Mask dimensions: {width} x {height}, pixel size: {pixel_size:.3f}")
except FileNotFoundError:
    print(f"Error: Mask file not found: {mask_path}")
    print("Please check the mask file path.")
    exit(1)

vs_xy = pixel_size
vs_z = voxel_size_override or pixel_size
z_min = meta["z_range"]["min"]
z_max = meta["z_range"]["max"]

print(f"Voxel size: XY={vs_xy:.3f}m, Z={vs_z:.3f}m")

# --- index computation -----------------------------------------------------
cols, rows = rasterio.transform.rowcol(
    transform,
    df["X"].to_numpy(),
    df["Y"].to_numpy(),
    op=float,
)
cols = cols.astype(np.int64)
rows = rows.astype(np.int64)

iz = np.floor((df["Z_norm"].to_numpy() - z_min) / vs_z).astype(np.int64)
iz = np.clip(iz, 0, None)
nz = int(iz.max()) + 1

nx, ny = width, height
shape = (nz, ny, nx)

print(f"Tensor shape (Z, Y, X): {shape}")
print(f"Total voxels: {np.prod(shape):,}")

# --- allocate volumes ------------------------------------------------------
print("Creating voxel tensors...")
occupancy = np.zeros(shape, dtype=np.uint8)
point_counts = np.zeros(shape, dtype=np.uint32)
dominant = np.full(shape, -1, dtype=np.int32)

# flatten indices for aggregation
flat = iz * (ny * nx) + rows * nx + cols
uniq, inv = np.unique(flat, return_inverse=True)

np.add.at(point_counts.ravel(), flat, 1)
occupancy.ravel()[flat] = 1

labels = df["Label"].to_numpy(dtype=np.int32)
uniq_labels = np.unique(labels)
label_idx = np.searchsorted(uniq_labels, labels)

votes = np.zeros((uniq.size, uniq_labels.size), dtype=np.uint32)
np.add.at(votes, (inv, label_idx), 1)
majority = uniq_labels[votes.argmax(axis=1)]
dominant.ravel()[uniq] = majority
dominant[occupancy == 0] = -1

print(f"Occupied voxels: {np.sum(occupancy):,} ({100*np.sum(occupancy)/np.prod(shape):.2f}%)")
print(f"Max points per voxel: {np.max(point_counts)}")

# --- save tensors ---------------------------------------------------------
output_file = "unet_voxel_tensors.npz"
print(f"Saving tensors to {output_file}...")

np.savez_compressed(
    output_file,
    occupancy=occupancy,
    point_counts=point_counts,
    dominant_labels=dominant,
    voxel_size_m=dict(x=vs_xy, y=vs_xy, z=vs_z),
    origin=dict(x=transform.c, y=transform.f, z=z_min),
)

print("✓ Tensors saved successfully!")

# --- Optional: Create raster stack for QA -----------------------------------
print("Creating raster stack for QA...")
stack = dominant  # (Z, Y, X)

profile = {
    "driver": "GTiff",  # Use regular GeoTIFF instead of COG for simplicity
    "dtype": "int16",
    "width": nx,
    "height": ny,
    "count": nz,
    "transform": transform,
    "crs": crs,
    "compress": "lzw",
}

output_tiff = "dominant_labels_stack.tif"
with rasterio.open(output_tiff, "w", **profile) as dst:
    for band in range(nz):
        dst.write(stack[band, :, :], band + 1)

print(f"✓ Raster stack saved to {output_tiff}")
print(f"✓ Multi-band raster with {nz} bands (Z slices)")

print("\nSummary:")
print(f"- Input points: {len(df):,}")
print(f"- Voxel grid: {shape} = {np.prod(shape):,} voxels")
print(f"- Occupancy: {np.sum(occupancy):,} voxels ({100*np.sum(occupancy)/np.prod(shape):.2f}%)")
print(f"- Labels: {len(uniq_labels)} unique classes")
print(f"- Output files: {output_file}, {output_tiff}")