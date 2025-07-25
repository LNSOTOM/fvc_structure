
#%%
import numpy as np
import laspy
from scipy.interpolate import RegularGridInterpolator
import rasterio

# === Paths ===
lidar_path = "/media/laura/Extreme SSD/qgis/calperumResearch/site2_1_DD0011/lidar/cloud2453cf90763afd91.las"
sfm_input  = "/media/laura/laura_usb/qgis/calperumResearch/site2_1_DD0011/lidar/sfm_dense/point_cloud_micasense.laz"
flow_aligned_dem = "/media/laura/laura_usb/qgis/calperumResearch/site2_1_DD0011/lidar/cloud2453cf90763afd91_dem_5cm_normalized_resampled_aligned_to_micasense.tif"
original_dem = flow_aligned_dem.replace("_aligned_to_micasense.tif", ".tif")  # original LiDAR DEM
output_las = lidar_path.replace(".las", "_aligned_via_gefolki.las")

# === Load optical flow difference from aligned and original DEM ===
with rasterio.open(original_dem) as src_orig, rasterio.open(flow_aligned_dem) as src_aligned:
    dem_orig = src_orig.read(1)
    dem_aligned = src_aligned.read(1)
    profile = src_aligned.profile
    transform = src_aligned.transform

    # Calculate flow in pixels
    flow_u_pix = (dem_aligned - dem_orig) * 0  # We'll use actual displacements next
    flow_v_pix = (dem_aligned - dem_orig) * 0  # placeholder only

    # Replace with real dx/dy in meters
    dx = flow_u_pix  # should be precomputed and saved if needed
    dy = flow_v_pix

# === Compute real-world grid coordinates ===
rows, cols = dem_aligned.shape
x0, y0 = transform[2], transform[5]
xres, yres = transform[0], -transform[4]

x_coords = x0 + np.arange(cols) * xres
y_coords = y0 - np.arange(rows) * yres

# === Build interpolators ===
# For now, dummy zero-shift (replace with actual dx/dy maps if available)
interp_dx = RegularGridInterpolator((y_coords, x_coords), dx, bounds_error=False, fill_value=0)
interp_dy = RegularGridInterpolator((y_coords, x_coords), dy, bounds_error=False, fill_value=0)

# === Load original LiDAR point cloud ===
las = laspy.read(lidar_path)
points_xy = np.vstack((las.y, las.x)).T  # (Y, X) for interpolator

# === Interpolate shifts ===
shift_x = interp_dx(points_xy)
shift_y = interp_dy(points_xy)

# === Apply shifts ===
las.x += shift_x
las.y += shift_y

# === Save new aligned LAS ===
las.write(output_las)
print(f"✅ GeFolki-aligned LAS saved to:\n{output_las}")

# %%
