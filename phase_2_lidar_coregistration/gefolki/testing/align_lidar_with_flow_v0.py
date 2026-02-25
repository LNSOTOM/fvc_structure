#%%
import numpy as np
import laspy
import rasterio
from pathlib import Path
from scipy.ndimage import map_coordinates

# Input paths
las_in = Path("/media/laura/laura_usb/uas_data/DD0001/lidar/20220519_SASMDD0001_triple_AusGeoid.las")
# Use the RGB DSM as reference grid (the target we're aligning to)
grid = Path("/media/laura/laura_usb/uas_data/DD0001/imagery/dsm/SASMDD0001_rgb_dsm_03.tif")
# Flow computed from LiDAR DSM -> RGB DSM alignment
flow_u = np.load("/media/laura/laura_usb/uas_data/DD0001/imagery/dsm/flow_u_03.npy")
flow_v = np.load("/media/laura/laura_usb/uas_data/DD0001/imagery/dsm/flow_v_03.npy")

# --- Raster grid info (for world<->pixel transforms) ---
with rasterio.open(grid) as src:
    transform = src.transform
    H, W = src.height, src.width

# Sanity: flow arrays should match raster dimensions
if flow_u.shape != (H, W) or flow_v.shape != (H, W):
    raise ValueError(f"Flow shape {flow_u.shape} / {flow_v.shape} does not match raster size {(H, W)}")

inv = ~transform       # affine inverse for (x,y) -> (row,col)
eps = 1e-3             # small epsilon to keep coords inside bounds

out = las_in.with_name(las_in.stem + "_aligned_to_rgb.las")
total_in, total_written = 0, 0

print("Applying optical flow displacement to LiDAR points...")

# Stream through the LAS in chunks to avoid huge RAM usage
with laspy.open(las_in) as reader:
    header = reader.header
    with laspy.open(out, mode="w", header=header) as writer:
        for pts in reader.chunk_iterator(5_000_000):   # adjust chunk size for your RAM
            total_in += len(pts)

            # World -> raster coords
            x = pts.x
            y = pts.y
            row, col = inv * (x, y)

            # Valid where point lies inside DSM (we'll drop the rest)
            valid = (col >= 0) & (col < W - 1) & (row >= 0) & (row < H - 1)
            if not np.any(valid):
                continue  # nothing from this chunk overlaps DSM

            # Interpolate flow at valid locations (bilinear)
            rowc = np.clip(row[valid], 0, H - 1 - eps)
            colc = np.clip(col[valid], 0, W - 1 - eps)
            coords = np.vstack((rowc, colc))
            dx_pix = map_coordinates(flow_u, coords, order=1, mode="nearest")
            dy_pix = map_coordinates(flow_v, coords, order=1, mode="nearest")

            # Pixel -> world (meters); transform.a > 0, transform.e typically < 0 (north-up)
            dx = dx_pix * transform.a
            dy = dy_pix * transform.e

            # Build a filtered point record with only valid points
            sub = pts[valid]        # keeps all dimensions (intensity, RGB, etc.)
            xv = sub.x.copy()
            yv = sub.y.copy()

            # Apply displacement to align LiDAR to RGB DSM
            xv += dx
            yv += dy

            # Overwrite XY (Z remains unchanged)
            sub.x = xv
            sub.y = yv

            # Write only the valid (inside-DSM) points
            writer.write_points(sub)
            total_written += len(sub)

print(f"✅ Wrote {out}")
print(f"   kept {total_written} / {total_in} points ({total_written/total_in:.1%}) that overlapped the DSM")

#%%
# Read and display flow data in column format
import numpy as np, rasterio
from pathlib import Path
import pandas as pd

print("Loading flow arrays...")
flow_u = np.load("/media/laura/laura_usb/uas_data/DD0001/imagery/dsm/flow_u_03.npy")
flow_v = np.load("/media/laura/laura_usb/uas_data/DD0001/imagery/dsm/flow_v_03.npy")

with rasterio.open("/media/laura/laura_usb/uas_data/DD0001/imagery/dsm/SASMDD0001_rgb_dsm_03.tif") as src:
    T = src.transform
    ax = T.a  # pixel width  (>0)
    ay = T.e  # pixel height (<0 for north-up)

# Convert to world coordinates
dx_m = flow_u * ax  # X displacement in meters
dy_m = flow_v * ay  # Y displacement in meters
mag_m = np.hypot(dx_m, dy_m)

# Flatten arrays and create sample of flow data
valid_mask = np.isfinite(dx_m) & np.isfinite(dy_m) & (mag_m > 0.001)  # Filter out near-zero displacements
indices = np.where(valid_mask)

# Sample every 1000th point to show more meaningful displacements
sample_step = 1000
sample_indices = slice(0, len(indices[0]), sample_step)

# Create DataFrame with flow data
flow_data = pd.DataFrame({
    'row': indices[0][sample_indices],
    'col': indices[1][sample_indices], 
    'flow_u_pix': flow_u[indices][sample_indices],
    'flow_v_pix': flow_v[indices][sample_indices],
    'dx_m': dx_m[indices][sample_indices],
    'dy_m': dy_m[indices][sample_indices],
    'magnitude_m': mag_m[indices][sample_indices]
})

print(f"\n=== FLOW DATA SAMPLE (every {sample_step}th non-zero pixel) ===")
print(f"Total pixels with significant displacement (>0.1mm): {np.sum(valid_mask):,}")
print(f"Showing {len(flow_data)} sample points:\n")
print(flow_data.round(4))

# Calculate displacement magnitudes and components
# Convert pixel displacements to world coordinates (meters)
dx_m = flow_u * ax  # X displacement in meters (ax > 0)
dy_m = flow_v * ay  # Y displacement in meters (ay < 0 for north-up, so this handles the sign)

# Combined magnitude
mag_pix = np.hypot(flow_u, flow_v)
mag_m = np.hypot(dx_m, dy_m)

# Filter out invalid values
valid_mask = np.isfinite(mag_m) & np.isfinite(dx_m) & np.isfinite(dy_m)
mag_m_valid = mag_m[valid_mask]
dx_m_valid = dx_m[valid_mask]
dy_m_valid = dy_m[valid_mask]

if len(mag_m_valid) > 0:
    # Calculate median values
    median_mag = np.median(mag_m_valid)
    median_x = np.median(dx_m_valid)
    median_y = np.median(dy_m_valid)
    
    print("=== MEDIAN DISPLACEMENT VALUES ===")
    print(f"Median magnitude: {median_mag:.4f} m")
    print(f"Median X displacement: {median_x:.4f} m")
    print(f"Median Y displacement: {median_y:.4f} m")
    print()
    
    # Calculate percentiles first
    q_mag = np.percentile(mag_m_valid, [50, 90, 99, 99.9])
    
    print("=== ROBUST ALIGNMENT APPROACH ===")
    print("To get closer to median values:")
    print("1. Apply median translation first in CloudCompare:")
    print(f"   X: {median_x:.7f}")
    print(f"   Y: {median_y:.7f}")
    print("   Z: 0.0000000")
    print()
    print("2. Then run optical flow alignment on residual displacement")
    print("3. Or filter out extreme displacements (>99th percentile)")
    print(f"   Current 99th percentile: {q_mag[2]:.3f} m = {q_mag[2]*100:.1f} cm")
    print()
    
    # Show what happens if we filter extreme displacements
    moderate_mask = mag_m_valid <= q_mag[2]  # Keep only displacements <= 99th percentile
    mag_filtered = mag_m_valid[moderate_mask]
    dx_filtered = dx_m_valid[moderate_mask]
    dy_filtered = dy_m_valid[moderate_mask]
    
    print("=== IF FILTERING EXTREME DISPLACEMENTS (>99th percentile) ===")
    print(f"Filtered median magnitude: {np.median(mag_filtered):.4f} m = {np.median(mag_filtered)*100:.1f} cm")
    print(f"Filtered median X: {np.median(dx_filtered):.4f} m = {np.median(dx_filtered)*100:.1f} cm")
    print(f"Filtered median Y: {np.median(dy_filtered):.4f} m = {np.median(dy_filtered)*100:.1f} cm")
    print(f"Points kept: {len(mag_filtered):,} / {len(mag_m_valid):,} ({len(mag_filtered)/len(mag_m_valid)*100:.1f}%)")
    print()
    
    # Combined magnitude statistics
    q_mag = np.percentile(mag_m_valid, [50, 90, 99, 99.9])
    print(f"Flow magnitude (m) - p50: {q_mag[0]:.3f}, p90: {q_mag[1]:.3f}, p99: {q_mag[2]:.3f}, p99.9: {q_mag[3]:.3f}")
    print(f"Mean magnitude: {np.mean(mag_m_valid):.3f} m")
    print(f"Max magnitude: {np.max(mag_m_valid):.3f} m")
    print()
    
    # X displacement statistics
    q_x = np.percentile(dx_m_valid, [50, 90, 99, 99.9])
    print(f"X displacement (m) - p50: {q_x[0]:.3f}, p90: {q_x[1]:.3f}, p99: {q_x[2]:.3f}, p99.9: {q_x[3]:.3f}")
    print(f"Mean X displacement: {np.mean(dx_m_valid):.3f} m")
    print(f"X range: [{np.min(dx_m_valid):.3f}, {np.max(dx_m_valid):.3f}] m")
    print()
    
    # Y displacement statistics
    q_y = np.percentile(dy_m_valid, [50, 90, 99, 99.9])
    print(f"Y displacement (m) - p50: {q_y[0]:.3f}, p90: {q_y[1]:.3f}, p99: {q_y[2]:.3f}, p99.9: {q_y[3]:.3f}")
    print(f"Mean Y displacement: {np.mean(dy_m_valid):.3f} m")
    print(f"Y range: [{np.min(dy_m_valid):.3f}, {np.max(dy_m_valid):.3f}] m")
else:
    print("No valid flow values found")

# %%