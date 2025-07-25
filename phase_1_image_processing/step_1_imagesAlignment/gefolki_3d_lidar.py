
#%%
import numpy as np
import laspy
from rasterio.transform import rowcol
import rasterio

# === 1. Load flow fields from GeFolki ===
flow_u = np.load("/media/laura/laura_usb/qgis/calperumResearch/site2_1_DD0011/lidar/sfm_dense/flow_u.npy") # X displacement in pixels
flow_v = np.load("/media/laura/laura_usb/qgis/calperumResearch/site2_1_DD0011/lidar/sfm_dense/flow_v.npy") # Y displacement in pixels


# === 2. Load DSM used in GeFolki to get geo-transform ===
with rasterio.open("/media/laura/laura_usb/qgis/calperumResearch/site2_1_DD0011/lidar/sfm_dense/SASMDD0011_dsm_3cm.tif") as dsm:
    transform = dsm.transform
    height, width = dsm.height, dsm.width
    crs = dsm.crs

# === 3. Load original LiDAR point cloud ===
las = laspy.read("/media/laura/Extreme SSD/qgis/calperumResearch/site2_1_DD0011/lidar/cloud2453cf90763afd91.las")
x, y, z = las.x.copy(), las.y.copy(), las.z.copy()

# === 4. Convert world XY to pixel (row, col) ===
cols, rows = rowcol(transform, x, y)

# Keep only points within bounds of flow arrays
mask = (rows >= 0) & (rows < height) & (cols >= 0) & (cols < width)
cols_valid = cols[mask]
rows_valid = rows[mask]

# === 5. Compute average shift from GeFolki displacements ===
u_disp = flow_u[rows_valid, cols_valid]
v_disp = flow_v[rows_valid, cols_valid]

# Convert pixel displacements to world units (meters)
dx_m = np.mean(u_disp) * transform.a
dy_m = np.mean(v_disp) * abs(transform.e)

print(f"📐 Applying average XY shift: dx = {dx_m:.4f} m, dy = {dy_m:.4f} m")

# === 6. Apply rigid XY transform to full LiDAR cloud ===
x_new = x + dx_m
y_new = y + dy_m
z_new = z  # No change

# === 7. Save new shifted LAS (rigidly aligned to RGB DSM) ===
header = las.header
shifted_las = laspy.LasData(header)
shifted_las.x = x_new
shifted_las.y = y_new
shifted_las.z = z_new

# Preserve all other fields
for dim in las.point_format.dimensions:
    if dim.name not in ['X', 'Y', 'Z']:
        shifted_las[dim.name] = las[dim.name]

out_path = "/media/laura/Extreme SSD/qgis/calperumResearch/site2_1_DD0011/lidar/lidar_gefolki_rigid.las"
shifted_las.write(out_path)

print(f"✅ Rigid XY shift from GeFolki applied and saved to: {out_path}")

# 📐 Applying average XY shift: dx = 2.5616 m, dy = -0.1155 m
# ✅ Rigid XY shift from GeFolki applied and saved to: /media/laura/Extreme SSD/qgis/calperumResearch/site2_1_DD0011/lidar/lidar_gefolki_rigid.las# %%


#%%
###test 2 - better
import numpy as np
import laspy
import rasterio
from rasterio.transform import rowcol

# === INPUTS ===
lidar_path = "/media/laura/Extreme SSD/qgis/calperumResearch/site2_1_DD0011/lidar/cloud2453cf90763afd91.las"
ref_dsm_path = "/media/laura/laura_usb/qgis/calperumResearch/site2_1_DD0011/lidar/sfm_dense/SASMDD0011_dsm_3cm.tif"
flow_u_path = "/media/laura/laura_usb/qgis/calperumResearch/site2_1_DD0011/lidar/sfm_dense/flow_u.npy"
flow_v_path = "/media/laura/laura_usb/qgis/calperumResearch/site2_1_DD0011/lidar/sfm_dense/flow_v.npy"
output_las_path = "/media/laura/laura_usb/qgis/calperumResearch/site2_1_DD0011/lidar/lidar_gefolki_shifted_xyz.las"

# === Load flow fields ===
flow_u = np.load(flow_u_path)
flow_v = np.load(flow_v_path)

# === Load DSM for georeferencing and Z correction ===
with rasterio.open(ref_dsm_path) as ref_dsm:
    ref_array = ref_dsm.read(1)
    transform = ref_dsm.transform
    crs = ref_dsm.crs
    height, width = ref_dsm.height, ref_dsm.width

# === Load original LiDAR point cloud ===
las = laspy.read(lidar_path)
x, y, z = las.x.copy(), las.y.copy(), las.z.copy()

# === Convert world XY to raster row/col ===
cols, rows = rowcol(transform, x, y)
valid = (rows >= 0) & (rows < height) & (cols >= 0) & (cols < width)

# Keep only valid matches for flow/Z diff
cols = cols[valid]
rows = rows[valid]
x_valid = x[valid]
y_valid = y[valid]
z_valid = z[valid]

# === Average XY shift from flow ===
dx = np.mean(flow_u[rows, cols]) * transform.a
dy = np.mean(flow_v[rows, cols]) * abs(transform.e)

# === Z offset using DSM minus LiDAR (reverse sign to shift LiDAR up) ===
# === Z offset using DSM minus LiDAR (gentle correction) ===
ref_z = ref_array[rows, cols]
z_offset = np.nanmean(ref_z - z_valid)
dz = -0.5 * z_offset  # Apply half of the inverse shift


print(f"📐 Applying shift: dx={dx:.3f}, dy={dy:.3f}, dz={dz:.3f}")

# === Apply transformation ===
x_shifted = x + dx
y_shifted = y + dy
z_shifted = z + dz

# === Save new shifted LAS ===
header = las.header
new_las = laspy.LasData(header)
new_las.x = x_shifted
new_las.y = y_shifted
new_las.z = z_shifted

# Copy extra fields like classification
for dim in las.point_format.dimensions:
    if dim.name not in ['X', 'Y', 'Z']:
        new_las[dim.name] = las[dim.name]

new_las.write(output_las_path)
print(f"✅ Saved corrected LiDAR LAS to: {output_las_path}")

# %%
