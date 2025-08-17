#%%
import numpy as np
import laspy
from scipy.interpolate import RegularGridInterpolator
import rasterio
from pathlib import Path

# === Input paths ===
las_path = Path("/media/laura/Extreme SSD/qgis/calperumResearch/site2_1_DD0011/lidar/cloud2453cf90763afd91.las")
original_dem_path = Path("/media/laura/laura_usb/qgis/calperumResearch/site2_1_DD0011/lidar/thinned_voxel/dem_3cm_lidar_resampled_to_sfm.tif")
aligned_dem_path = Path("/media/laura/laura_usb/qgis/calperumResearch/site2_1_DD0011/lidar/thinned_voxel/dem_3cm_lidar_resampled_to_sfm_aligned_to_sfm.tif")
flow_u_path = Path("/media/laura/laura_usb/qgis/calperumResearch/site2_1_DD0011/lidar/thinned_voxel/flow_u.npy")
flow_v_path = Path("/media/laura/laura_usb/qgis/calperumResearch/site2_1_DD0011/lidar/thinned_voxel/flow_v.npy")

# === Output LAS ===
output_las = las_path.with_name(aligned_dem_path.stem + "_aligned_gefolki_3d.las")

# === Load flow and elevation displacement ===
flow_u = np.load(flow_u_path)
flow_v = np.load(flow_v_path)

with rasterio.open(original_dem_path) as src_orig, rasterio.open(aligned_dem_path) as src_aligned:
    dem_orig = src_orig.read(1).astype(np.float32)
    dem_aligned = src_aligned.read(1).astype(np.float32)
    transform = src_aligned.transform
    profile = src_aligned.profile

    dz = dem_aligned - dem_orig  # vertical displacement

# === Create grid for interpolation
xres = transform[0]
yres = -transform[4]
x0 = transform[2]
y0 = transform[5]

height, width = dem_aligned.shape
x_coords = x0 + np.arange(width) * xres
y_coords = y0 - np.arange(height) * yres  # row-based indexing

# === Interpolators for dx, dy, dz
interp_dx = RegularGridInterpolator((y_coords, x_coords), flow_u, bounds_error=False, fill_value=0)
interp_dy = RegularGridInterpolator((y_coords, x_coords), flow_v, bounds_error=False, fill_value=0)
interp_dz = RegularGridInterpolator((y_coords, x_coords), dz,     bounds_error=False, fill_value=0)

# === Load original LAS
las = laspy.read(str(las_path))
points_xy = np.vstack((las.y, las.x)).T

# === Interpolate displacement
dx = interp_dx(points_xy)
dy = interp_dy(points_xy)
dz = interp_dz(points_xy)

# === Apply displacement
las.x = las.x + dx
las.y = las.y + dy
las.z = las.z + dz

# === Save aligned LAS
las.write(str(output_las))
print(f"✅ 3D GeFolki-aligned LAS saved to:\n{output_las}")

#%%
import open3d as o3d
import laspy
import numpy as np
from pathlib import Path

# === Paths ===
source_path = Path("/media/laura/Extreme SSD/qgis/calperumResearch/site2_1_DD0011/lidar/dem_3cm_lidar_resampled_to_sfm_aligned_to_sfm_aligned_gefolki_3d.las")
target_path = Path("/media/laura/laura_usb/qgis/calperumResearch/site2_1_DD0011/lidar/thinned_voxel/point_cloud_p1_thinned_3cm.las")
output_path = source_path.with_name(target_path.stem + "_icp_refined.las")

# === Helper: Load LAS to Open3D point cloud
def las_to_o3d(las_file):
    las = laspy.read(str(las_file))
    pts = np.vstack((las.x, las.y, las.z)).T
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    return pcd, las

# Load GeFolki-aligned LiDAR (source)
source_pcd, source_las = las_to_o3d(source_path)

# Load SfM point cloud (target)
target_pcd, _ = las_to_o3d(target_path)

# ICP parameters
threshold = 0.15  # meters
init_transform = np.eye(4)

# Run ICP refinement
print("🔄 Running ICP refinement...")
icp_result = o3d.pipelines.registration.registration_icp(
    source=source_pcd,
    target=target_pcd,
    max_correspondence_distance=threshold,
    init=init_transform,
    estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint()
)

print("✅ ICP done.")
print("📐 Transformation matrix:\n", icp_result.transformation)

# Apply ICP transformation
source_pcd.transform(icp_result.transformation)

# Save refined LAS
refined_pts = np.asarray(source_pcd.points)
refined_las = laspy.create(point_format=source_las.header.point_format, file_version=source_las.header.version)
refined_las.x, refined_las.y, refined_las.z = refined_pts[:, 0], refined_pts[:, 1], refined_pts[:, 2]
refined_las.write(str(output_path))

# Optional: Save transformation matrix
np.savetxt(output_path.with_suffix(".txt"), icp_result.transformation, fmt="%.8f")

print(f"✅ ICP-refined LAS saved to:\n  {output_path}")

# %%
#tetss
import numpy as np
import laspy
from pathlib import Path
import rasterio
from scipy.interpolate import RegularGridInterpolator

# === Input paths ===
las_path = Path("/media/laura/Extreme SSD/qgis/calperumResearch/site2_1_DD0011/lidar/cloud2453cf90763afd91.las")
original_dem_path = Path("/media/laura/laura_usb/qgis/calperumResearch/site2_1_DD0011/lidar/sfm_dense/point_cloud_p1.laz")
aligned_dem_path = Path("/media/laura/laura_usb/qgis/calperumResearch/site2_1_DD0011/lidar/cloud2453cf90763afd91_dsm_3cm_resampled_to_sfm_aligned_to_rgb_aligned_gefolki_3d.las")
flow_u_path = Path("/media/laura/laura_usb/qgis/calperumResearch/site2_1_DD0011/lidar/sfm_dense/flow_u.npy")
flow_v_path = Path("/media/laura/laura_usb/qgis/calperumResearch/site2_1_DD0011/lidar/sfm_dense/flow_v.npy")

# === Output LAS path ===
output_las = las_path.with_name(aligned_dem_path.stem + "_aligned_gefolki_3d.las")

# === Load DEMs and compute dz
with rasterio.open(original_dem_path) as src_orig, rasterio.open(aligned_dem_path) as src_aligned:
    dem_orig = src_orig.read(1).astype(np.float32)
    dem_aligned = src_aligned.read(1).astype(np.float32)
    transform = src_aligned.transform

    # Handle nodata
    nodata_orig = src_orig.nodata if src_orig.nodata is not None else 9999
    nodata_aligned = src_aligned.nodata if src_aligned.nodata is not None else 9999

    dem_orig[dem_orig == nodata_orig] = np.nan
    dem_aligned[dem_aligned == nodata_aligned] = np.nan

    dz = dem_aligned - dem_orig
    dz[np.isnan(dz)] = 0  # Safe for interpolation

# === Print stats
print("📊 dz displacement summary:")
print(f"Shape: {dz.shape}")
print(f"Min: {np.nanmin(dz):.3f} m")
print(f"Max: {np.nanmax(dz):.3f} m")
print(f"Mean: {np.nanmean(dz):.3f} m")
print(f"Std: {np.nanstd(dz):.3f} m")

# === Load optical flow fields
flow_u = np.load(flow_u_path)
flow_v = np.load(flow_v_path)
assert flow_u.shape == dz.shape, "Mismatch between flow field and DEM shape."

# === Set up grid for interpolation
xres, yres = transform.a, -transform.e
x0, y0 = transform.c, transform.f
height, width = dz.shape
x_coords = x0 + np.arange(width) * xres
y_coords = y0 - np.arange(height) * yres  # raster top-down

# === Interpolators
interp_dx = RegularGridInterpolator((y_coords, x_coords), flow_u, bounds_error=False, fill_value=0)
interp_dy = RegularGridInterpolator((y_coords, x_coords), flow_v, bounds_error=False, fill_value=0)
interp_dz = RegularGridInterpolator((y_coords, x_coords), dz,     bounds_error=False, fill_value=0)

# === Load LAS and extract original coordinates
las = laspy.read(str(las_path))
x, y, z = las.x.copy(), las.y.copy(), las.z.copy()
points_xy = np.column_stack((y, x))  # order = (Y, X)

# === Interpolate and apply displacements
dx = interp_dx(points_xy) * xres
dy = interp_dy(points_xy) * yres
dz = interp_dz(points_xy)

las.x = x + dx
las.y = y + dy
las.z = z + dz

# === Save output LAS
las.write(str(output_las))
print(f"✅ 3D GeFolki-aligned LAS saved to:\n{output_las}")

# 📊 dz displacement summary:
# Shape: (9477, 8672)
# Min: -45.902 m
# Max: 7.460 m
# Mean: -5.354 m
# Std: 12.418 m
# %%
##test 3
import numpy as np
import laspy
from pathlib import Path
import rasterio
from scipy.interpolate import RegularGridInterpolator

def apply_gefolki_transform(
    las_input_path: Path,
    output_path: Path,
    dsm_sfm_path: Path,
    dsm_lidar_aligned_path: Path,
    flow_u_path: Path,
    flow_v_path: Path
):
    # === Load DSMs
    with rasterio.open(dsm_sfm_path) as src_sfm, rasterio.open(dsm_lidar_aligned_path) as src_lidar:
        dsm_sfm = src_sfm.read(1).astype(np.float32)
        dsm_lidar = src_lidar.read(1).astype(np.float32)
        transform = src_lidar.transform

        # Handle NoData
        nodata_sfm = src_sfm.nodata if src_sfm.nodata is not None else 9999
        nodata_lidar = src_lidar.nodata if src_lidar.nodata is not None else 9999
        dsm_sfm[dsm_sfm == nodata_sfm] = np.nan
        dsm_lidar[dsm_lidar == nodata_lidar] = np.nan

        dz = dsm_lidar - dsm_sfm
        dz[np.isnan(dz)] = 0

    print("📊 dz displacement stats:")
    print(f"Min: {np.nanmin(dz):.3f}, Max: {np.nanmax(dz):.3f}, Mean: {np.nanmean(dz):.3f}, Std: {np.nanstd(dz):.3f}")

    # === Load optical flow
    flow_u = np.load(flow_u_path)
    flow_v = np.load(flow_v_path)
    assert flow_u.shape == dz.shape, "Flow shape mismatch"

    # === Interpolation grid
    xres, yres = transform.a, -transform.e
    x0, y0 = transform.c, transform.f
    height, width = dz.shape
    x_coords = x0 + np.arange(width) * xres
    y_coords = y0 - np.arange(height) * yres

    interp_dx = RegularGridInterpolator((y_coords, x_coords), flow_u, bounds_error=False, fill_value=0)
    interp_dy = RegularGridInterpolator((y_coords, x_coords), flow_v, bounds_error=False, fill_value=0)
    interp_dz = RegularGridInterpolator((y_coords, x_coords), dz,     bounds_error=False, fill_value=0)

    # === Load point cloud
    las = laspy.read(str(las_input_path))
    x, y, z = las.x.copy(), las.y.copy(), las.z.copy()
    points_xy = np.column_stack((y, x))  # (Y, X)

    # === Interpolate displacements
    dx = interp_dx(points_xy) * xres
    dy = interp_dy(points_xy) * yres
    dz = interp_dz(points_xy)

    # === Apply transformation
    las.x = x + dx
    las.y = y + dy
    las.z = z + dz

    # === Save
    las.write(str(output_path))
    print(f"✅ Saved aligned LAS to:\n{output_path}")

# %%
# Paths
sfm_dense = Path("/media/laura/laura_usb/qgis/calperumResearch/site2_1_DD0011/lidar/sfm_dense/point_cloud_p1.laz")
lidar_raw = Path("/media/laura/Extreme SSD/qgis/calperumResearch/site2_1_DD0011/lidar/cloud2453cf90763afd91.las")

dsm_sfm = Path("//media/laura/laura_usb/qgis/calperumResearch/site2_1_DD0011/lidar/sfm_dense/SASMDD0011_dsm_3cm.tif")
dsm_lidar_aligned = Path("/media/laura/laura_usb/qgis/calperumResearch/site2_1_DD0011/lidar/cloud2453cf90763afd91_dsm_3cm_resampled_to_sfm_aligned_to_rgb.tif")

flow_u = Path("/media/laura/laura_usb/qgis/calperumResearch/site2_1_DD0011/lidar/sfm_dense/flow_u.npy")
flow_v = Path("/media/laura/laura_usb/qgis/calperumResearch/site2_1_DD0011/lidar/sfm_dense/flow_v.npy")

# Apply to SfM dense cloud
apply_gefolki_transform(
    las_input_path=sfm_dense,
    output_path=sfm_dense.with_name("SASMDD0011_dense_cloud_aligned.las"),
    dsm_sfm_path=dsm_sfm,
    dsm_lidar_aligned_path=dsm_lidar_aligned,
    flow_u_path=flow_u,
    flow_v_path=flow_v
)

# Apply to LiDAR cloud
apply_gefolki_transform(
    las_input_path=lidar_raw,
    output_path=lidar_raw.with_name("cloud2453cf90763afd91_aligned.las"),
    dsm_sfm_path=dsm_sfm,
    dsm_lidar_aligned_path=dsm_lidar_aligned,
    flow_u_path=flow_u,
    flow_v_path=flow_v
)

# %%
###test3 
import numpy as np
import laspy
from pathlib import Path
import rasterio
from scipy.interpolate import RegularGridInterpolator

# === Input paths ===
las_path = Path("/media/laura/Extreme SSD/qgis/calperumResearch/site2_1_DD0011/lidar/cloud2453cf90763afd91.las")
sfm_dsm_path = Path("/media/laura/laura_usb/qgis/calperumResearch/site2_1_DD0011/lidar/sfm_dense/SASMDD0011_dsm_3cm.tif")
lidar_dsm_aligned_path = Path("/media/laura/laura_usb/qgis/calperumResearch/site2_1_DD0011/lidar/cloud2453cf90763afd91_dsm_3cm_resampled_to_sfm_aligned_to_rgb.tif")
flow_u_path = Path("/media/laura/laura_usb/qgis/calperumResearch/site2_1_DD0011/lidar/sfm_dense/flow_u.npy")
flow_v_path = Path("/media/laura/laura_usb/qgis/calperumResearch/site2_1_DD0011/lidar/sfm_dense/flow_v.npy")

# === Output path ===
output_las = las_path.with_name(lidar_dsm_aligned_path.stem + "_aligned_gefolki_3d.las")

# === Load DSMs and calculate vertical displacement
with rasterio.open(sfm_dsm_path) as src_sfm, rasterio.open(lidar_dsm_aligned_path) as src_lidar:
    dem_sfm = src_sfm.read(1).astype(np.float32)
    dem_lidar = src_lidar.read(1).astype(np.float32)
    transform = src_lidar.transform

    nodata_sfm = src_sfm.nodata if src_sfm.nodata is not None else 9999
    nodata_lidar = src_lidar.nodata if src_lidar.nodata is not None else 9999

    dem_sfm[dem_sfm == nodata_sfm] = np.nan
    dem_lidar[dem_lidar == nodata_lidar] = np.nan

    dz_grid = dem_lidar - dem_sfm
    dz_grid[np.isnan(dz_grid)] = 0

# === Print dz stats
print("📊 dz displacement stats:")
print(f"Min: {np.min(dz_grid):.3f} m | Max: {np.max(dz_grid):.3f} m | Mean: {np.mean(dz_grid):.3f} m | Std: {np.std(dz_grid):.3f} m")

# === Load GeFolki optical flow
flow_u = np.load(flow_u_path)
flow_v = np.load(flow_v_path)

assert flow_u.shape == dz_grid.shape, "Mismatch in flow and DSM dimensions."

# === Build interpolation grid
xres, yres = transform.a, -transform.e
x0, y0 = transform.c, transform.f
height, width = dz_grid.shape
x_coords = x0 + np.arange(width) * xres
y_coords = y0 - np.arange(height) * yres

interp_dx = RegularGridInterpolator((y_coords, x_coords), flow_u, bounds_error=False, fill_value=0)
interp_dy = RegularGridInterpolator((y_coords, x_coords), flow_v, bounds_error=False, fill_value=0)
interp_dz = RegularGridInterpolator((y_coords, x_coords), dz_grid, bounds_error=False, fill_value=0)

# === Load LiDAR LAS
las = laspy.read(str(las_path))
x, y, z = las.x.copy(), las.y.copy(), las.z.copy()

# === Interpolate displacements
points_xy = np.column_stack((y, x))  # (Y, X) order for raster space
dx = interp_dx(points_xy) * xres
dy = interp_dy(points_xy) * yres
dz = interp_dz(points_xy)

# === Apply displacements
las.x = x + dx
las.y = y + dy
las.z = z + dz

# === Save aligned LAS
las.write(str(output_las))
print(f"✅ LiDAR LAS aligned and saved to:\n{output_las}")

# %%
#otro good
import numpy as np
import matplotlib.pyplot as plt
import rasterio
from pathlib import Path

# === Input paths ===
sfm_dsm_path = Path("/media/laura/laura_usb/qgis/calperumResearch/site2_1_DD0011/lidar/sfm_dense/SASMDD0011_dsm_3cm.tif")
lidar_dsm_aligned_path = Path("/media/laura/laura_usb/qgis/calperumResearch/site2_1_DD0011/lidar/cloud2453cf90763afd91_dsm_3cm_resampled_to_sfm_aligned_to_rgb.tif")

# === Load both DSMs
with rasterio.open(sfm_dsm_path) as src_sfm, rasterio.open(lidar_dsm_aligned_path) as src_lidar:
    dsm_sfm = src_sfm.read(1).astype(np.float32)
    dsm_lidar = src_lidar.read(1).astype(np.float32)

    # Handle NoData
    nodata_sfm = src_sfm.nodata if src_sfm.nodata is not None else 9999
    nodata_lidar = src_lidar.nodata if src_lidar.nodata is not None else 9999

    dsm_sfm[dsm_sfm == nodata_sfm] = np.nan
    dsm_lidar[dsm_lidar == nodata_lidar] = np.nan

    # Calculate dz: LiDAR DSM - SfM DSM
    dz = dsm_lidar - dsm_sfm
    dz[np.isnan(dz)] = 0  # Optional: set NaNs to 0 for visualization

# === Print stats
print("📊 dz (LiDAR - SfM) summary:")
print(f"Min:  {np.nanmin(dz):.3f} m")
print(f"Max:  {np.nanmax(dz):.3f} m")
print(f"Mean: {np.nanmean(dz):.3f} m")
print(f"Std:  {np.nanstd(dz):.3f} m")

# === Plot
plt.figure(figsize=(10, 8))
plt.imshow(dz, cmap='RdBu', vmin=-5, vmax=5)
plt.colorbar(label='dz (LiDAR - SfM) [m]')
plt.title("Vertical Difference Map: LiDAR DSM - SfM DSM")
plt.tight_layout()
plt.show()

# %%
###improve
import numpy as np
import rasterio
import matplotlib.pyplot as plt
from pathlib import Path

# === Input paths ===
sfm_dsm_path = Path("/media/laura/laura_usb/qgis/calperumResearch/site2_1_DD0011/lidar/sfm_dense/SASMDD0011_dsm_3cm.tif")
lidar_dsm_aligned_path = Path("/media/laura/laura_usb/qgis/calperumResearch/site2_1_DD0011/lidar/cloud2453cf90763afd91_dsm_3cm_resampled_to_sfm_aligned_to_rgb.tif")

# === Load DSMs ===
with rasterio.open(sfm_dsm_path) as src_sfm, rasterio.open(lidar_dsm_aligned_path) as src_lidar:
    dsm_sfm = src_sfm.read(1).astype(np.float32)
    dsm_lidar = src_lidar.read(1).astype(np.float32)

    nodata_sfm = src_sfm.nodata if src_sfm.nodata is not None else 9999
    nodata_lidar = src_lidar.nodata if src_lidar.nodata is not None else 9999

# === Replace nodata with NaN
dsm_sfm[dsm_sfm == nodata_sfm] = np.nan
dsm_lidar[dsm_lidar == nodata_lidar] = np.nan

# === Compute vertical difference
dz = dsm_lidar - dsm_sfm

# === Apply mask for valid data and dz ∈ [-5, 5]
valid_mask = (~np.isnan(dz)) & (dz >= -5) & (dz <= 5)
dz_clipped = dz[valid_mask]

# === Summary statistics
print("📊 Filtered dz stats (LiDAR - SfM within ±5m):")
print(f"Min:  {np.nanmin(dz_clipped):.3f} m")
print(f"Max:  {np.nanmax(dz_clipped):.3f} m")
print(f"Mean: {np.nanmean(dz_clipped):.3f} m")
print(f"Std:  {np.nanstd(dz_clipped):.3f} m")
print(f"Percentiles [5, 25, 50, 75, 95]: {np.percentile(dz_clipped, [5, 25, 50, 75, 95])}")

# === Histogram
plt.figure(figsize=(10, 4))
plt.hist(dz_clipped, bins=100, color='steelblue', edgecolor='k')
plt.title("Histogram of Vertical Differences (LiDAR DSM - SfM DSM)")
plt.xlabel("dz (m)")
plt.ylabel("Pixel Count")
plt.grid(True)
plt.tight_layout()
plt.show()

# %%
##clip
import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.transform import from_origin
import matplotlib.pyplot as plt
from pathlib import Path

# === Input paths ===
sfm_dsm_path = Path("/media/laura/laura_usb/qgis/calperumResearch/site2_1_DD0011/lidar/sfm_dense/SASMDD0011_dsm_3cm.tif")
lidar_dsm_aligned_path = Path("/media/laura/laura_usb/qgis/calperumResearch/site2_1_DD0011/lidar/cloud2453cf90763afd91_dsm_3cm_resampled_to_sfm_aligned_to_rgb.tif")

# === Output paths ===
dz_out_path = lidar_dsm_aligned_path.with_name("dz_lidar_minus_sfm_clipped.tif")
mask_out_path = lidar_dsm_aligned_path.with_name("valid_mask_dz_5m.tif")

# === Load data
with rasterio.open(sfm_dsm_path) as src_sfm, rasterio.open(lidar_dsm_aligned_path) as src_lidar:
    dsm_sfm = src_sfm.read(1).astype(np.float32)
    dsm_lidar = src_lidar.read(1).astype(np.float32)
    transform = src_lidar.transform
    crs = src_lidar.crs
    profile = src_lidar.profile.copy()

nodata_sfm = profile.get("nodata", 9999)
dsm_sfm[dsm_sfm == nodata_sfm] = np.nan
dsm_lidar[dsm_lidar == nodata_sfm] = np.nan

# === Calculate dz
dz = dsm_lidar - dsm_sfm
dz[np.isnan(dz)] = np.nan

# === Apply mask and set outside range to NaN
valid_mask = (dz >= -5) & (dz <= 5)
dz_clipped = np.where(valid_mask, dz, np.nan)

# === Export dz_clipped
profile.update(dtype='float32', count=1, nodata=np.nan)

with rasterio.open(dz_out_path, 'w', **profile) as dst:
    dst.write(dz_clipped, 1)

print(f"✅ Exported clipped dz map to:\n{dz_out_path}")

# === Export valid mask as binary (1 = valid, 0 = invalid)
mask_array = valid_mask.astype(np.uint8)
profile.update(dtype='uint8', nodata=0)

with rasterio.open(mask_out_path, 'w', **profile) as dst:
    dst.write(mask_array, 1)

print(f"✅ Exported valid dz mask to:\n{mask_out_path}")

# %%
#plot
# === Plot outputs ===
fig, axs = plt.subplots(1, 3, figsize=(18, 6))

# 1. Original dz map
im0 = axs[0].imshow(dz, cmap='coolwarm', vmin=-5, vmax=5)
axs[0].set_title("Raw dz (LiDAR - SfM)")
fig.colorbar(im0, ax=axs[0], fraction=0.046)

# 2. Clipped dz
im1 = axs[1].imshow(dz_clipped, cmap='RdBu_r', vmin=-5, vmax=5)
axs[1].set_title("Clipped dz (±5 m)")
fig.colorbar(im1, ax=axs[1], fraction=0.046)

# 3. Valid Mask
im2 = axs[2].imshow(valid_mask, cmap='Greens')
axs[2].set_title("Valid Mask (|dz| ≤ 5 m)")
fig.colorbar(im2, ax=axs[2], fraction=0.046)

plt.suptitle("Vertical Displacement Analysis: LiDAR vs SfM", fontsize=14)
plt.tight_layout()
plt.show()

# %%
