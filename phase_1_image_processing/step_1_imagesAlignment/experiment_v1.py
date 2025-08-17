
#%%
import rasterio
from pathlib import Path
import numpy as np

lidar_dsm_aligned_path = Path("/media/laura/laura_usb/qgis/calperumResearch/site2_1_DD0011/lidar/cloud2453cf90763afd91_dsm_3cm_resampled_to_sfm_aligned_to_rgb.tif")

with rasterio.open(lidar_dsm_aligned_path) as src:
    dsm_data = src.read(1)
    nodata = src.nodata
    transform = src.transform
    width, height = src.width, src.height

# 2. Get Valid Pixel Coordinates
# === Generate full meshgrid of row/col indices
rows, cols = np.meshgrid(np.arange(height), np.arange(width), indexing="ij")

# === Mask valid pixels
valid_mask = dsm_data != nodata

# === Get row/col of valid pixels
valid_rows = rows[valid_mask]
valid_cols = cols[valid_mask]

#  3. Convert Valid Pixels to Real-World Coordinates
# === Convert row/col to (x, y) using transform
valid_x, valid_y = rasterio.transform.xy(transform, valid_rows, valid_cols)

# Convert to NumPy arrays and stack into coordinates
valid_coords = np.column_stack([np.array(valid_x), np.array(valid_y)])

# %%
##test 2
import laspy
from scipy.spatial import cKDTree

las_path = Path("/media/laura/Extreme SSD/qgis/calperumResearch/site2_1_DD0011/lidar/cloud2453cf90763afd91.las")
las = laspy.read(str(las_path))
x, y, z = las.x, las.y, las.z

# Build KD-tree for efficient XY matching
tree = cKDTree(valid_coords)
query_coords = np.column_stack((x, y))
distances, _ = tree.query(query_coords, distance_upper_bound=0.05)  # max 5 cm tolerance

valid_lidar_mask = distances != np.inf

# %%
clipped_las = laspy.LasData(las.header)
clipped_las.points = las.points[valid_lidar_mask]

output_path = las_path.with_name("cloud2453cf90763afd91_clipped_to_aligned_lidar_dsm.las")
clipped_las.write(str(output_path))
print(f"✅ Clipped LiDAR saved to:\n{output_path}")

# %%
##test 3
import numpy as np
import laspy
import rasterio
from affine import Affine
from scipy.ndimage import map_coordinates
from pathlib import Path

# === INPUT FILES ===
lidar_path = Path("/media/laura/Extreme SSD/qgis/calperumResearch/site2_1_DD0011/lidar/cloud2453cf90763afd91.las")  # or .laz
flow_u_path = Path("/media/laura/laura_usb/qgis/calperumResearch/site2_1_DD0011/lidar/sfm_dense/flow_u.npy")        # horizontal shift (cols)
flow_v_path = Path("/media/laura/laura_usb/qgis/calperumResearch/site2_1_DD0011/lidar/sfm_dense/flow_v.npy")        # vertical shift (rows)
rgb_dsm_path = Path("/media/laura/Extreme SSD/qgis/calperumResearch/site2_1_DD0011/lidar/cloud2453cf90763afd91_DSM_3cm.tif")  # DSM for geotransform
output_las_path = Path("/media/laura/laura_usb/uas_data/DD0011/DD0011-lidar_aligned_to_rgb.las")

# === LOAD SHIFT FIELDS ===
flow_u = np.load(flow_u_path)
flow_v = np.load(flow_v_path)

# === READ RGB DSM GEOREFERENCE ===
with rasterio.open(rgb_dsm_path) as dsm:
    transform: Affine = dsm.transform
    width = dsm.width
    height = dsm.height
    crs = dsm.crs
    res_x, res_y = transform.a, -transform.e  # should be (0.03, 0.03)

print(f"Affine transform: {transform}")
print(f"Resolution X: {res_x}, Resolution Y: {res_y}")

# === OPEN LIDAR POINT CLOUD ===
las = laspy.read(lidar_path)
x_coords = las.x
y_coords = las.y
z_coords = las.z

# === CONVERT (X, Y) TO (ROW, COL) PIXEL INDICES IN DSM ===
inv_transform = ~transform
cols, rows = inv_transform * (x_coords, y_coords)

print("Row range:", rows.min(), rows.max())
print("Col range:", cols.min(), cols.max())
print("Flow shape:", flow_u.shape)

# === CLAMP ROWS AND COLS TO DSM EXTENT ===
cols_clipped = np.clip(cols, 0, flow_u.shape[1] - 1)
rows_clipped = np.clip(rows, 0, flow_u.shape[0] - 1)

# === INTERPOLATE SHIFTS FROM GEFOLKI ===
dx_pixels = map_coordinates(flow_u, [rows_clipped, cols_clipped], order=1, mode='nearest')
dy_pixels = map_coordinates(flow_v, [rows_clipped, cols_clipped], order=1, mode='nearest')

print(f"Sample dx (pixels): {dx_pixels[:5]}")
print(f"Sample dy (pixels): {dy_pixels[:5]}")

# === INVERT THE SHIFT TO ALIGN LIDAR TO RGB ===
dx_meters = -dx_pixels * res_x
dy_meters = -dy_pixels * res_y

# === APPLY SHIFT TO X, Y; Z REMAINS ===
x_aligned = x_coords + dx_meters
y_aligned = y_coords + dy_meters

# === SAVE THE ALIGNED POINT CLOUD ===
aligned_las = laspy.LasData(las.header)
aligned_las.x = x_aligned
aligned_las.y = y_aligned
aligned_las.z = z_coords
aligned_las.write(output_las_path)

print(f"✅ Aligned LiDAR point cloud saved to: {output_las_path}")


# %%
# 1. Print Your DSM Transform
print(f"Affine transform: {transform}")
print(f"Resolution X: {transform.a}, Resolution Y: {-transform.e}")

# Affine transform: | 0.03, 0.00, 473145.68|
# | 0.00,-0.03, 6235905.53|
# | 0.00, 0.00, 1.00|
# Resolution X: 0.03, Resolution Y: 0.03
# %%
# 2. Print a Sample Displacement
print(f"Sample dx in pixels: {dx_pixels[:10]}")
print(f"Converted to meters: {dx_meters[:10]}")

# Sample dx (pixels): [0. 0. 0. 0. 0.]
# Sample dy (pixels): [0. 0. 0. 0. 0.]
# %%
# 3. Print the Row/Col Ranges
print("Row range:", rows.min(), rows.max())
print("Col range:", cols.min(), cols.max())
print("Flow shape:", flow_u.shape)

# Row range: 0.8999999761581421 9623.0
# Col range: 0.0 6533.980000000447
# Flow shape: (9477, 8672)
# %%
# Check RGB DSM bounds
with rasterio.open(rgb_dsm_path) as src:
    bounds = src.bounds
    print(f"RGB DSM bounds:\nLeft: {bounds.left}\nRight: {bounds.right}\nBottom: {bounds.bottom}\nTop: {bounds.top}")

# RGB DSM bounds:
# Left: 473145.68468093954
# Right: 473341.70468093955
# Bottom: 6235616.8406857895
# Top: 6235905.53068579
# %%
# Check LiDAR point bounds
print(f"LiDAR X range: {x_coords.min()} to {x_coords.max()}")
print(f"LiDAR Y range: {y_coords.min()} to {y_coords.max()}")

# LiDAR X range: 473145.68468093954 to 473341.7040809395
# LiDAR Y range: 6235616.8406857895 to 6235905.50368579

# %%
##test4
import numpy as np
import laspy
import rasterio
from affine import Affine
from scipy.ndimage import map_coordinates
from scipy.spatial import cKDTree
from pathlib import Path

# === INPUT FILES ===
lidar_path = Path("/media/laura/Extreme SSD/qgis/calperumResearch/site2_1_DD0011/lidar/cloud2453cf90763afd91.las")  # or .laz
rgb_path = Path("/media/laura/laura_usb/uas_data/DD0011/point_cloud_p1.laz")  # DSM for geotransform
flow_u_path = Path("/media/laura/laura_usb/qgis/calperumResearch/site2_1_DD0011/lidar/sfm_dense/flow_u.npy")        # horizontal shift (cols)
flow_v_path = Path("/media/laura/laura_usb/qgis/calperumResearch/site2_1_DD0011/lidar/sfm_dense/flow_v.npy")        # vertical shift (rows)
output_path = Path("/media/laura/laura_usb/uas_data/DD0011/DD0011-lidar_aligned_to_rgb.las")


# === LOAD FLOW FIELDS ===
flow_u = np.load(flow_u_path)
flow_v = np.load(flow_v_path)

# === LOAD DSM GEOREFERENCE ===
with rasterio.open(rgb_dsm_path) as dsm:
    transform: Affine = dsm.transform
    width = dsm.width
    height = dsm.height
    res_x, res_y = transform.a, -transform.e

# === LOAD LIDAR CLOUD ===
las = laspy.read(lidar_path)
x_coords = las.x
y_coords = las.y
z_coords = las.z

# === APPLY XY ALIGNMENT USING FLOW ===
inv_transform = ~transform
cols, rows = inv_transform * (x_coords, y_coords)

cols_clipped = np.clip(cols, 0, flow_u.shape[1] - 1)
rows_clipped = np.clip(rows, 0, flow_u.shape[0] - 1)

dx_pixels = map_coordinates(flow_u, [rows_clipped, cols_clipped], order=1, mode='nearest')
dy_pixels = map_coordinates(flow_v, [rows_clipped, cols_clipped], order=1, mode='nearest')

dx_meters = -dx_pixels * res_x
dy_meters = -dy_pixels * res_y

x_aligned = x_coords + dx_meters
y_aligned = y_coords + dy_meters

# === LOAD RGB REFERENCE CLOUD FOR Z ALIGNMENT ===
rgb_las = laspy.read(rgb_path)
rgb_xyz = np.vstack((rgb_las.x, rgb_las.y, rgb_las.z)).T

# === MATCH XY LOCATIONS TO COMPUTE Z OFFSET ===
lidar_aligned_xy = np.vstack((x_aligned, y_aligned)).T
tree = cKDTree(rgb_xyz[:, :2])
dist, idx = tree.query(lidar_aligned_xy, k=1, distance_upper_bound=0.05)

valid = idx < rgb_xyz.shape[0]
if np.sum(valid) == 0:
    raise ValueError("❌ No valid overlaps found for Z alignment.")

matched_lidar_z = z_coords[valid]
matched_rgb_z = rgb_xyz[idx[valid], 2]

z_offset = np.median(matched_rgb_z - matched_lidar_z)
print(f"✅ Computed Z offset to apply: {z_offset:.4f} m")

# === APPLY FULL XYZ ALIGNMENT ===
z_aligned = z_coords + z_offset

# === SAVE FINAL LAS ===
# Copy header and create new LAS object
final_las = laspy.LasData(las.header)

# Apply aligned XYZ
final_las.x = x_aligned
final_las.y = y_aligned
final_las.z = z_aligned

# === Preserve all extra dimensions ===
for dim in las.point_format.dimensions:
    name = dim.name
    if name not in ("X", "Y", "Z"):  # already replaced above
        setattr(final_las, name, getattr(las, name))

# === Write output ===
final_las.write(output_path)


print(f"✅ Fully aligned LAS saved to: {output_path}")

# %%
##test 5
import numpy as np
import laspy
import rasterio
from affine import Affine
from scipy.ndimage import map_coordinates
from scipy.spatial import cKDTree
from pathlib import Path

# === INPUT FILES ===
lidar_path = Path("/media/laura/Extreme SSD/qgis/calperumResearch/site2_1_DD0011/lidar/cloud2453cf90763afd91.las")  # LiDAR input
rgb_path = Path("/media/laura/laura_usb/uas_data/DD0011/point_cloud_p1.laz")   # RGB reference
flow_u_path = Path("/media/laura/laura_usb/qgis/calperumResearch/site2_1_DD0011/lidar/sfm_dense/flow_u.npy")        # horizontal shift (cols)
flow_v_path = Path("/media/laura/laura_usb/qgis/calperumResearch/site2_1_DD0011/lidar/sfm_dense/flow_v.npy")        # vertical shift (rows)
rgb_dsm_path = Path("/media/laura/laura_usb/qgis/calperumResearch/site2_1_DD0011/lidar/sfm_dense/SASMDD0011_dsm_3cm.tif")
output_path = Path("/media/laura/laura_usb/uas_data/DD0011/lidar_xyz_gefolki_aligned_to_rgb.las")

# === LOAD DSM TRANSFORM (affine) ===
with rasterio.open(rgb_dsm_path) as dsm:
    transform: Affine = dsm.transform
    res_x, res_y = transform.a, -transform.e
    print(f"✅ DSM pixel size: {res_x} m")

# === LOAD FLOW FIELDS ===
flow_u = np.load(flow_u_path)
flow_v = np.load(flow_v_path)

# === LOAD LiDAR ===
las = laspy.read(lidar_path)
x_coords = las.x
y_coords = las.y
z_coords = las.z

# === APPLY GeFolki XY SHIFT ===
inv_transform = ~transform
cols, rows = inv_transform * (x_coords, y_coords)

cols_clipped = np.clip(cols, 0, flow_u.shape[1] - 1)
rows_clipped = np.clip(rows, 0, flow_u.shape[0] - 1)

dx_pixels = map_coordinates(flow_u, [rows_clipped, cols_clipped], order=1, mode='nearest')
dy_pixels = map_coordinates(flow_v, [rows_clipped, cols_clipped], order=1, mode='nearest')

dx_meters = -dx_pixels * res_x
dy_meters = -dy_pixels * res_y

x_aligned = x_coords + dx_meters
y_aligned = y_coords + dy_meters

# === LOAD RGB dense cloud for Z alignment ===
rgb_las = laspy.read(rgb_path)
rgb_xyz = np.vstack((rgb_las.x, rgb_las.y, rgb_las.z)).T

# === Z ALIGNMENT USING KDTree ===
lidar_xy = np.vstack((x_aligned, y_aligned)).T
tree = cKDTree(rgb_xyz[:, :2])
_, idx = tree.query(lidar_xy, k=1, distance_upper_bound=0.05)

valid = idx < rgb_xyz.shape[0]
if np.sum(valid) == 0:
    raise ValueError("❌ No valid Z matches found in overlap.")

z_diff = rgb_xyz[idx[valid], 2] - z_coords[valid]
z_offset = np.median(z_diff)
print(f"✅ Median Z offset to apply: {z_offset:.4f} m")

# === APPLY FULL XYZ TRANSFORMATION ===
z_aligned = z_coords + z_offset

# === SAVE FINAL LAS (preserve attributes) ===
aligned_las = laspy.LasData(las.header)
aligned_las.x = x_aligned
aligned_las.y = y_aligned
aligned_las.z = z_aligned

for dim in las.point_format.dimensions:
    if dim.name not in ("X", "Y", "Z"):
        setattr(aligned_las, dim.name, getattr(las, dim.name))

aligned_las.write(output_path)
print(f"✅ Fully aligned LAS saved to: {output_path}")

# %%
#test 6 - tes
import laspy
import numpy as np
from pathlib import Path

# Paths
rgb_path = Path("/media/laura/laura_usb/uas_data/DD0011/point_cloud_p1.laz")
aligned_lidar_path = Path("/media/laura/laura_usb/uas_data/DD0011/lidar_xyz_gefolki_aligned_to_rgb.las")

# === Load RGB dense cloud ===
rgb_las = laspy.read(rgb_path)
rgb_x, rgb_y, rgb_z = rgb_las.x, rgb_las.y, rgb_las.z

# === Load LiDAR-aligned cloud ===
lidar_las = laspy.read(aligned_lidar_path)
lidar_x, lidar_y, lidar_z = lidar_las.x, lidar_las.y, lidar_las.z

# === Compute bounding box centers ===
def center(coords):
    return (np.min(coords) + np.max(coords)) / 2

rgb_center = np.array([center(rgb_x), center(rgb_y), center(rgb_z)])
lidar_center = np.array([center(lidar_x), center(lidar_y), center(lidar_z)])

# === Compute shift (RGB - LiDAR) ===
offset = np.round(rgb_center - lidar_center, 4)

print("📦 RGB Center:", tuple(np.round(rgb_center, 4)))
print("📦 LiDAR Center:", tuple(np.round(lidar_center, 4)))
print("📐 Offset to apply (X, Y, Z):", tuple(offset))

# === Apply shift to LiDAR
lidar_las.x += offset[0]
lidar_las.y += offset[1]
lidar_las.z += offset[2]

# === Save
output_path = Path("/media/laura/laura_usb/uas_data/DD0011/lidar_xyz_gefolki_finalaligned_to_rgb.las")
lidar_las.write(output_path)

print(f"✅ Final corrected LiDAR saved to: {output_path}")


# %%
##test 7 - next
import laspy
import numpy as np
from pathlib import Path

# Input and output
input_las = Path("/media/laura/laura_usb/uas_data/DD0011/lidar_xyz_gefolki_aligned_to_rgb.las")
output_las = Path("/media/laura/laura_usb/uas_data/DD0011/lidar_xyz_final_aligned_to_rgb.las")

# Load LiDAR cloud
las = laspy.read(input_las)

# === FINAL OFFSET — update based on CloudCompare measurement ===
# Example values (you must replace with actual box center difference)
x_shift = 473208.87 - 473240.6092
y_shift = 6235753.76 - 6235762.6057
z_shift = 49.79 - 35.6283

print(f"Applying offset: X={x_shift:.4f}, Y={y_shift:.4f}, Z={z_shift:.4f}")

# Apply shift
las.x += x_shift
las.y += y_shift
las.z += z_shift

# Save final LAS
las.write(output_las)
print(f"✅ Final aligned LAS saved to: {output_las}")


# %%
##test 8
import laspy
import numpy as np
import rasterio
from affine import Affine
from scipy.ndimage import map_coordinates
from pathlib import Path

# === Paths
input_las = Path("/media/laura/laura_usb/uas_data/DD0011/lidar_xyz_gefolki_aligned_to_rgb.las")
output_las = Path("/media/laura/laura_usb/uas_data/DD0011/lidar_xyz_finalAffineAligned_to_rgb.las")
flow_u_path = Path("/media/laura/laura_usb/qgis/calperumResearch/site2_1_DD0011/lidar/sfm_dense/flow_u.npy")
flow_v_path = Path("/media/laura/laura_usb/qgis/calperumResearch/site2_1_DD0011/lidar/sfm_dense/flow_v.npy")
dsm_path = Path("/media/laura/Extreme SSD/qgis/calperumResearch/site2_1_DD0011/lidar/cloud2453cf90763afd91_DSM_3cm.tif")

# === Load LAS
las = laspy.read(input_las)
x_affine, y_affine, z_affine = las.x.copy(), las.y.copy(), las.z.copy()

# === Load flow fields
flow_u = np.load(flow_u_path)
flow_v = np.load(flow_v_path)

# === Get DSM geotransform
with rasterio.open(dsm_path) as dsm:
    transform: Affine = dsm.transform
    res_x, res_y = transform.a, -transform.e
    width, height = dsm.width, dsm.height

# === Invert affine to go from (x,y) to (col,row)
inv_transform = ~transform
cols, rows = inv_transform * (x_affine, y_affine)

# === Clip to image bounds
cols_clipped = np.clip(cols, 0, flow_u.shape[1] - 1)
rows_clipped = np.clip(rows, 0, flow_u.shape[0] - 1)

# === Interpolate pixel shifts at LiDAR locations
dx_pix = map_coordinates(flow_u, [rows_clipped, cols_clipped], order=1, mode='nearest')
dy_pix = map_coordinates(flow_v, [rows_clipped, cols_clipped], order=1, mode='nearest')

# === Convert pixel shift to meters
dx_m = -dx_pix * res_x
dy_m = -dy_pix * res_y
z_offset = 0.0438  # median Z offset from previous KDTree alignment

# === Apply correction
x_new = x_affine + dx_m
y_new = y_affine + dy_m
z_new = z_affine + z_offset

# === Update LAS header scales/offsets
las.header.offsets = [float(np.min(x_new)), float(np.min(y_new)), float(np.min(z_new))]
las.header.scales = [0.001, 0.001, 0.001]

# === Write new coordinates
las.x = x_new
las.y = y_new
las.z = z_new

# === Save
las.write(output_las)
print(f"✅ GeFolki XY + Z shift applied and saved to: {output_las}")


# %%
##test 9
import laspy
import numpy as np
from pathlib import Path

# === Load point cloud
input_las = Path("/media/laura/laura_usb/uas_data/DD0011/lidar_xyz_gefolki_aligned_to_rgb.las")
output_las = Path("/media/laura/laura_usb/uas_data/DD0011/lidar_xyz_shifted_to_rgbCRS.las")

las = laspy.read(input_las)

# === Extract coordinates (they are still in local coords)
x_local, y_local, z_local = las.x, las.y, las.z

# === Apply global shift from CloudCompare (to return to original CRS)
x_global = x_local + 473243.69
y_global = y_local + 6235761.17
z_global = z_local + 35.58

# === BEFORE assigning coordinates, update header
las.header.offsets = [np.min(x_global), np.min(y_global), np.min(z_global)]
las.header.scales = [0.001, 0.001, 0.001]  # 1 mm precision

# === Now safely assign updated coordinates
las.x = x_global
las.y = y_global
las.z = z_global

# === Save
las.write(output_las)
print(f"✅ Global shift applied and saved to: {output_las}")


# %%
# test 10
import laspy
import numpy as np
from pathlib import Path

# === Load local-coords LAS
input_las = Path("/media/laura/laura_usb/uas_data/DD0011/lidar_xyz_gefolki_aligned_to_rgb.las")
output_las = Path("/media/laura/laura_usb/uas_data/DD0011/lidar_xyz_shifted_to_rgbCRS.las")

las = laspy.read(input_las)

# === Extract local coordinates
x_local, y_local, z_local = las.x, las.y, las.z

# === Apply global shift from CloudCompare (latest)
x_global = x_local + 946378.88
y_global = y_local + 12471378.01
z_global = z_local + 64.65

# === Update header BEFORE assigning new coordinates
las.header.offsets = [np.min(x_global), np.min(y_global), np.min(z_global)]
las.header.scales = [0.001, 0.001, 0.001]  # millimeter precision

# === Assign new coordinates
las.x = x_global
las.y = y_global
las.z = z_global

# === Save aligned LAS
las.write(output_las)
print(f"✅ Updated global shift applied and saved to: {output_las}")


# %%
#test 11
import numpy as np
import laspy
import rasterio
from pathlib import Path
import open3d as o3d
from scipy.interpolate import RegularGridInterpolator

# === INPUT PATHS ===
lidar_path = Path("/media/laura/laura_usb/uas_data/DD0011/lidar_xyz_gefolki_aligned_to_rgb.las")
rgb_path = Path("/media/laura/laura_usb/uas_data/DD0011/point_cloud_p1.laz")
flow_u_path = Path("/media/laura/laura_usb/qgis/calperumResearch/site2_1_DD0011/lidar/sfm_dense/flow_u.npy")
flow_v_path = Path("/media/laura/laura_usb/qgis/calperumResearch/site2_1_DD0011/lidar/sfm_dense/flow_v.npy")
rgb_dsm_path = Path("/media/laura/laura_usb/qgis/calperumResearch/site2_1_DD0011/lidar/sfm_dense/SASMDD0011_dsm_3cm.tif")
lidar_dsm_path = Path("/media/laura/laura_usb/qgis/calperumResearch/site2_1_DD0011/lidar/cloud2453cf90763afd91_dsm_3cm_resampled_to_sfm.tif")
output_las_path = Path("/media/laura/laura_usb/uas_data/DD0011/lidar_icp_final.las")

# === Load LAS and DSM ===
las = laspy.read(lidar_path)
x, y, z = las.x.copy(), las.y.copy(), las.z.copy()
flow_u, flow_v = np.load(flow_u_path), np.load(flow_v_path)

with rasterio.open(rgb_dsm_path) as src:
    profile = src.profile
    transform = src.transform
    width = src.width
    height = src.height

    col = ((x - transform.c) / transform.a).astype(int)
    row = ((y - transform.f) / transform.e).astype(int)

    valid_mask = (col >= 0) & (col < width) & (row >= 0) & (row < height)
    dx = np.zeros_like(x)
    dy = np.zeros_like(y)
    dx[valid_mask] = flow_u[row[valid_mask], col[valid_mask]]
    dy[valid_mask] = flow_v[row[valid_mask], col[valid_mask]]

    x_shifted = x + dx * transform.a
    y_shifted = y + dy * transform.e

# === Prepare for ICP ===
lidar_points = np.vstack((x_shifted, y_shifted, z)).T
rgb_las = laspy.read(rgb_path)
rgb_points = np.vstack((rgb_las.x, rgb_las.y, rgb_las.z)).T

def np_to_o3d(points, voxel_size=0.05):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    return pcd.voxel_down_sample(voxel_size=voxel_size)

pcd_lidar = np_to_o3d(lidar_points)
pcd_rgb = np_to_o3d(rgb_points)

pcd_lidar.estimate_normals()
pcd_rgb.estimate_normals()

print("🔄 Running ICP...")
icp_result = o3d.pipelines.registration.registration_icp(
    pcd_lidar,
    pcd_rgb,
    max_correspondence_distance=0.5,
    init=np.eye(4),
    estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint()
)

T_icp = icp_result.transformation
print("✅ ICP done. Transformation:\n", T_icp)

# === Apply ICP transform to shifted LiDAR ===
points_h = np.hstack((lidar_points, np.ones((lidar_points.shape[0], 1))))
aligned_points = (T_icp @ points_h.T).T[:, :3]

# === Optional Z correction ===
def compute_z_offset(rgb_dsm_path, lidar_dsm_path):
    with rasterio.open(rgb_dsm_path) as rgb:
        rgb_z = rgb.read(1)
    with rasterio.open(lidar_dsm_path) as lidar:
        lidar_z = lidar.read(1)
    mask = (rgb_z != rgb.nodata) & (lidar_z != lidar.nodata)
    delta = rgb_z[mask] - lidar_z[mask]
    return np.median(delta)

z_offset = compute_z_offset(rgb_dsm_path, lidar_dsm_path)
aligned_points[:, 2] += z_offset

# === Save final aligned LiDAR ===
las.x = aligned_points[:, 0]
las.y = aligned_points[:, 1]
las.z = aligned_points[:, 2]
las.write(output_las_path)
print(f"💾 Saved ICP + GeFolki aligned cloud to: {output_las_path}")


# %%
##test 12
import rasterio
import numpy as np
import laspy
from pathlib import Path

# === Input Paths ===
rgb_dsm_path = Path("/media/laura/laura_usb/qgis/calperumResearch/site2_1_DD0011/lidar/sfm_dense/SASMDD0011_dsm_3cm.tif")
lidar_dsm_path = Path("/media/laura/laura_usb/qgis/calperumResearch/site2_1_DD0011/lidar/cloud2453cf90763afd91_dsm_3cm_resampled_to_sfm_aligned_to_sfm.tif")

input_las = Path("/media/laura/laura_usb/uas_data/DD0011/lidar_icp_aligned.las")
output_las = Path("/media/laura/laura_usb/uas_data/DD0011/lidar_icp_zcorrected.las")

# === Load DSM and mask
def load_dsm(path):
    with rasterio.open(path) as src:
        dsm = src.read(1).astype(np.float32)
        nodata = src.nodata if src.nodata is not None else -9999
        mask = dsm != nodata
        dsm[~mask] = np.nan
        return dsm, mask

# === Load DSMs
rgb_dsm, rgb_mask = load_dsm(rgb_dsm_path)
lidar_dsm, lidar_mask = load_dsm(lidar_dsm_path)

# === Compute difference and apply mask
common_mask = rgb_mask & lidar_mask
dsm_diff = rgb_dsm - lidar_dsm
dsm_diff[~common_mask] = np.nan

# === Compute vertical offset statistics
median_offset = np.nanmedian(dsm_diff)
mean_offset   = np.nanmean(dsm_diff)
p95_offset    = np.nanpercentile(dsm_diff, 95)

print(f"🔎 DSM vertical offset estimates:")
print(f"   • Median: {median_offset:.4f} m")
print(f"   • Mean:   {mean_offset:.4f} m")
print(f"   • 95th %: {p95_offset:.4f} m")

# === Choose preferred offset method
z_offset = median_offset  # 👈 change this if needed
print(f"🟨 Applying Z offset: {z_offset:.4f} m")

# === Apply Z offset to LAS
las = laspy.read(input_las)
las.z += z_offset
las.write(output_las)

print(f"✅ Z-corrected LAS saved to: {output_las}")



# %%
##test 13
import numpy as np
import laspy
import rasterio
import open3d as o3d
from pathlib import Path

# === Paths ===
base_dir = Path("/media/laura/laura_usb/uas_data/DD0011/")
lidar_input_path = Path("/media/laura/Extreme SSD/qgis/calperumResearch/site2_1_DD0011/lidar/cloud2453cf90763afd91.las")
rgb_input_path = Path("/media/laura/laura_usb/uas_data/DD0011/point_cloud_p1.laz")
rgb_dsm_path = Path("/media/laura/laura_usb/qgis/calperumResearch/site2_1_DD0011/lidar/sfm_dense/SASMDD0011_dsm_3cm.tif")
lidar_dsm_aligned_path = Path("/media/laura/laura_usb/qgis/calperumResearch/site2_1_DD0011/lidar/cloud2453cf90763afd91_dsm_3cm_resampled_to_sfm_aligned_to_sfm.tif")
flow_u_path = Path("/media/laura/laura_usb/qgis/calperumResearch/site2_1_DD0011/lidar/sfm_dense/flow_u.npy")
flow_v_path = Path("/media/laura/laura_usb/qgis/calperumResearch/site2_1_DD0011/lidar/sfm_dense/flow_v.npy")

shifted_las_path = base_dir / "lidar_shifted_from_gefolki.las"
intermediate_las_path = base_dir / "lidar_icp_aligned.las"
final_output_las = base_dir / "lidar_icp_zcorrected_final.las"

# === Parameters ===
voxel_size = 0.1
z_offset_refinement = 0.0
rgb_global_shift = np.array([-473000.0, -6235000.0, 1000.0])  # CloudCompare local shift

# === Utility Functions ===
def las_to_np(las_path, apply_shift=None):
    las = laspy.read(las_path)
    points = np.vstack((las.x, las.y, las.z)).T
    if apply_shift is not None:
        points -= apply_shift
    return points, las

def np_to_o3d(points, voxel_size=0.1):
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(points)
    return cloud.voxel_down_sample(voxel_size)

def apply_gefolki_shift(las_path, flow_u_path, flow_v_path, output_path):
    las = laspy.read(las_path)
    u, v = np.load(flow_u_path), np.load(flow_v_path)
    dx, dy = u.mean(), v.mean()
    las.x += dx
    las.y += dy
    las.write(output_path)
    print(f"✅ Applied GeFolki XY shift → {output_path}\n   ΔX = {dx:.2f}, ΔY = {dy:.2f}")
    return output_path

def load_dsm(path):
    with rasterio.open(path) as src:
        dsm = src.read(1).astype(np.float32)
        nodata = src.nodata if src.nodata is not None else -9999
        mask = dsm != nodata
        dsm[~mask] = np.nan
        return dsm, mask

# === Step 1: Apply GeFolki shift to LiDAR
apply_gefolki_shift(lidar_input_path, flow_u_path, flow_v_path, shifted_las_path)

# === Step 2: Load point clouds
lidar_pts, lidar_las = las_to_np(shifted_las_path)
rgb_pts, _ = las_to_np(rgb_input_path, apply_shift=rgb_global_shift)

# === Step 3: Downsample and ICP
pcd_lidar = np_to_o3d(lidar_pts, voxel_size)
pcd_rgb = np_to_o3d(rgb_pts, voxel_size)

pcd_lidar.estimate_normals()
pcd_rgb.estimate_normals()

icp_result = o3d.pipelines.registration.registration_icp(
    source=pcd_lidar,
    target=pcd_rgb,
    max_correspondence_distance=1.0,
    init=np.eye(4),
    estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint()
)
T = icp_result.transformation
print(f"🔄 ICP transformation matrix:\n{T}")

# === Step 4: Apply ICP transformation to full-resolution LiDAR
pts_h = np.hstack((lidar_pts, np.ones((lidar_pts.shape[0], 1))))
aligned_pts = (T @ pts_h.T).T[:, :3]

lidar_las.x = aligned_pts[:, 0]
lidar_las.y = aligned_pts[:, 1]
lidar_las.z = aligned_pts[:, 2]
lidar_las.write(intermediate_las_path)
print(f"✅ ICP aligned LAS saved: {intermediate_las_path}")

# === Step 5: DSM Z-offset correction
rgb_dsm, rgb_mask = load_dsm(rgb_dsm_path)
lidar_dsm, lidar_mask = load_dsm(lidar_dsm_aligned_path)
common_mask = rgb_mask & lidar_mask
z_diff = rgb_dsm - lidar_dsm
z_diff[~common_mask] = np.nan
median_offset = np.nanmedian(z_diff)
print(f"🔍 DSM median Z-offset: {median_offset:.4f} m")

# === Step 6: Final Z correction
final_las = laspy.read(intermediate_las_path)
final_las.z += median_offset + z_offset_refinement
final_las.write(final_output_las)
print(f"✅ Final LAS written to {final_output_las}")


# %%
##test 14
import open3d as o3d
import numpy as np
from pathlib import Path
import laspy

# === CONFIG ===
lidar_path = Path("/media/laura/Extreme SSD/qgis/calperumResearch/site2_1_DD0011/lidar/cloud2453cf90763afd91.las")
rgb_path = Path("/media/laura/laura_usb/uas_data/DD0011/point_cloud_p1.laz")

shift_xy = np.array([46.16, -13.69])  # GeFolki XY shift
z_offset_initial = 35.0              # Approx. from CloudCompare
voxel_size = 0.05                    # Voxel size (~5 cm)
threshold = 3.0                      # ICP distance threshold
z_correction = 0.0546                # Final DSM offset

# === Load LAS as Open3D PointCloud ===
def load_las_as_o3d(las_path):
    las = laspy.read(str(las_path))
    xyz = np.vstack((las.x, las.y, las.z)).T
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(xyz))
    return pcd, las

# === Center PointClouds for more stable ICP ===
def center_pcd(pcd):
    centroid = np.mean(np.asarray(pcd.points), axis=0)
    pcd.translate(-centroid)
    return pcd, centroid

# === Apply initial XY + Z shift ===
def apply_shift(pcd, shift_xy, z_offset):
    pts = np.asarray(pcd.points)
    pts[:, 0] += shift_xy[0]
    pts[:, 1] += shift_xy[1]
    pts[:, 2] += z_offset
    pcd.points = o3d.utility.Vector3dVector(pts)
    return pcd

# === Load clouds ===
lidar_pcd, lidar_las = load_las_as_o3d(lidar_path)
rgb_pcd, _ = load_las_as_o3d(rgb_path)

# === Apply GeFolki shift ===
lidar_shifted = apply_shift(lidar_pcd, shift_xy, z_offset_initial)

# === Downsample (Octree-like step) ===
lidar_down = lidar_shifted.voxel_down_sample(voxel_size)
rgb_down = rgb_pcd.voxel_down_sample(voxel_size)

# === Center for ICP stability ===
lidar_down, lidar_centroid = center_pcd(lidar_down)
rgb_down, rgb_centroid = center_pcd(rgb_down)

# === Estimate Normals (needed for point-to-plane ICP) ===
lidar_down.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.5, max_nn=30))
rgb_down.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.5, max_nn=30))

# === Run ICP ===
trans_init = np.eye(4)
reg = o3d.pipelines.registration.registration_icp(
    lidar_down, rgb_down, threshold, trans_init,
    o3d.pipelines.registration.TransformationEstimationPointToPlane()
)

print(f"🔄 ICP fitness: {reg.fitness:.4f}")
print(f"🔍 RMSE: {reg.inlier_rmse:.4f}")
print("📌 ICP transformation matrix:\n", reg.transformation)

# === Apply ICP result ===
lidar_shifted.transform(reg.transformation)
lidar_shifted.translate(rgb_centroid)  # Recenter to RGB cloud space

# === Final Z correction (DSM vertical offset) ===
pts = np.asarray(lidar_shifted.points)
pts[:, 2] += z_correction
lidar_shifted.points = o3d.utility.Vector3dVector(pts)

# === Save debug copy in PLY format ===
ply_debug_path = Path("lidar_icp_debug.ply")
o3d.io.write_point_cloud(str(ply_debug_path), lidar_shifted)
print(f"📁 Saved intermediate PLY: {ply_debug_path}")

# === Save final LAS ===
final_output_path = Path("/media/laura/laura_usb/uas_data/DD0011/lidar_icp_zcorrected_finale.las")
final_las = laspy.create(point_format=lidar_las.header.point_format, file_version=lidar_las.header.version)
final_las.x = pts[:, 0]
final_las.y = pts[:, 1]
final_las.z = pts[:, 2]

# Copy all extra fields
for dim in lidar_las.point_format.dimension_names:
    if dim not in ['X', 'Y', 'Z']:
        final_las[dim] = lidar_las[dim]

final_las.write(str(final_output_path))
print(f"✅ Final LAS written to: {final_output_path}")

# %%
# import numpy as np
# from pathlib import Path

# # === CONFIG ===
# flow_u_path = Path("/media/laura/laura_usb/qgis/calperumResearch/site2_1_DD0011/lidar/sfm_dense/flow_u.npy")
# flow_v_path = Path("/media/laura/laura_usb/qgis/calperumResearch/site2_1_DD0011/lidar/sfm_dense/flow_v.npy")

# # === Load GeFolki flow fields ===
# flow_u = np.load(flow_u_path)
# flow_v = np.load(flow_v_path)

# # === Compute median flow shift (ignores NaNs or infs if present) ===
# valid_mask = np.isfinite(flow_u) & np.isfinite(flow_v)
# shift_u = np.median(flow_u[valid_mask])
# shift_v = np.median(flow_v[valid_mask])

# # === Output shift as numpy array for ICP translation ===
# shift_xy = np.array([shift_u, shift_v])
# print(f"📌 Computed GeFolki XY shift: {shift_xy}")

#%%
##TEST 15
import open3d as o3d
import numpy as np
from pathlib import Path
import laspy
import rasterio

# === CONFIG ===
lidar_path = Path("/media/laura/Extreme SSD/qgis/calperumResearch/site2_1_DD0011/lidar/cloud2453cf90763afd91.las")
rgb_path = Path("/media/laura/laura_usb/uas_data/DD0011/point_cloud_p1.laz")
dsm_lidar_path = Path("/media/laura/laura_usb/qgis/calperumResearch/site2_1_DD0011/lidar/cloud2453cf90763afd91_dsm_3cm_resampled_to_sfm.tif")
dsm_rgb_path = Path("/media/laura/laura_usb/qgis/calperumResearch/site2_1_DD0011/lidar/sfm_dense/SASMDD0011_dsm_3cm.tif")

# === Load DSMs and compute Z offsets ===
with rasterio.open(dsm_lidar_path) as dsm_lidar, rasterio.open(dsm_rgb_path) as dsm_rgb:
    if dsm_lidar.shape != dsm_rgb.shape:
        raise ValueError("❌ DSMs have different shapes. Please resample them to match resolution and extent.")
    lidar = dsm_lidar.read(1)
    rgb = dsm_rgb.read(1)
    mask = np.isfinite(lidar) & np.isfinite(rgb)
    dz = lidar[mask] - rgb[mask]
    z_offset_initial = np.median(dz)
    z_correction = -np.mean(dz)
    resolution = dsm_lidar.res[0]
    voxel_size = resolution
    print(f"📏 Voxel size set to: {voxel_size:.3f} m")
    print(f"⬆️ Initial Z offset (median): {z_offset_initial:.3f}")
    print(f"🔁 Z correction (to flatten): {z_correction:.3f}")

shift_xy = np.array([46.16, -13.69])  # Placeholder if needed
threshold = 3.0

# === Load LAS as Open3D PointCloud ===
def load_las_as_o3d(las_path):
    las = laspy.read(str(las_path))
    xyz = np.vstack((las.x, las.y, las.z)).T
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(xyz))
    return pcd, las

# === Center PointClouds ===
def center_pcd(pcd):
    centroid = np.mean(np.asarray(pcd.points), axis=0)
    pcd.translate(-centroid)
    return pcd, centroid

# === Shift point cloud ===
def apply_shift(pcd, shift_xy, z_offset):
    pts = np.asarray(pcd.points)
    pts[:, :2] += shift_xy
    pts[:, 2] += z_offset
    pcd.points = o3d.utility.Vector3dVector(pts)
    return pcd

# === Load and preprocess clouds ===
lidar_pcd, lidar_las = load_las_as_o3d(lidar_path)
rgb_pcd, _ = load_las_as_o3d(rgb_path)

lidar_shifted = apply_shift(lidar_pcd, shift_xy, z_offset_initial)
lidar_down = lidar_shifted.voxel_down_sample(voxel_size)
rgb_down = rgb_pcd.voxel_down_sample(voxel_size)

lidar_down, lidar_centroid = center_pcd(lidar_down)
rgb_down, rgb_centroid = center_pcd(rgb_down)

for cloud in [lidar_down, rgb_down]:
    cloud.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.5, max_nn=30))

reg = o3d.pipelines.registration.registration_icp(
    lidar_down, rgb_down, threshold, np.eye(4),
    o3d.pipelines.registration.TransformationEstimationPointToPlane()
)

print(f"🔄 ICP fitness: {reg.fitness:.4f}")
print(f"🔍 RMSE: {reg.inlier_rmse:.4f}")
print("📌 ICP transformation matrix:\n", reg.transformation)

lidar_shifted.transform(reg.transformation)
lidar_shifted.translate(rgb_centroid)

pts = np.asarray(lidar_shifted.points)
pts[:, 2] += z_correction
lidar_shifted.points = o3d.utility.Vector3dVector(pts)

ply_debug_path = Path("lidar_icp_debug.ply")
o3d.io.write_point_cloud(str(ply_debug_path), lidar_shifted)
print(f"📁 Saved intermediate PLY: {ply_debug_path}")

final_output_path = Path("/media/laura/laura_usb/uas_data/DD0011/lidar_icp_zcorrected_finale.las")
final_las = laspy.create(point_format=lidar_las.header.point_format, file_version=lidar_las.header.version)
final_las.x = pts[:, 0]
final_las.y = pts[:, 1]
final_las.z = pts[:, 2]

for dim in lidar_las.point_format.dimension_names:
    if dim not in ['X', 'Y', 'Z']:
        final_las[dim] = lidar_las[dim]

final_las.write(str(final_output_path))
print(f"✅ Final LAS written to: {final_output_path}")

# 📏 Voxel size set to: 0.030 m
# ⬆️ Initial Z offset (median): -0.015
# 🔁 Z correction (to flatten): 1502.174
# 🔄 ICP fitness: 0.9554
# 🔍 RMSE: 0.2464
# 📌 ICP transformation matrix:
#  [[ 9.99998649e-01 -1.62415456e-03  2.51350302e-04  1.99439567e-01]
#  [ 1.62416632e-03  9.99998680e-01 -4.65651222e-05  1.33372147e-01]
#  [-2.51274342e-04  4.69732940e-05  9.99999967e-01 -8.47788750e-02]
#  [ 0.00000000e+00  0.00000000e+00  0.00000000e+00  1.00000000e+00]]

#%%
##test 16
import numpy as np
import laspy
import rasterio
import open3d as o3d
from pathlib import Path

# === Paths ===
base_dir = Path("/media/laura/laura_usb/uas_data/DD0011/")
lidar_path = Path("/media/laura/Extreme SSD/qgis/calperumResearch/site2_1_DD0011/lidar/cloud2453cf90763afd91.las")
rgb_path = base_dir / "point_cloud_p1.laz"
flow_u_path = Path("/media/laura/laura_usb/qgis/calperumResearch/site2_1_DD0011/lidar/sfm_dense/flow_u.npy")
flow_v_path = Path("/media/laura/laura_usb/qgis/calperumResearch/site2_1_DD0011/lidar/sfm_dense/flow_v.npy")
rgb_dsm_path = Path("/media/laura/laura_usb/qgis/calperumResearch/site2_1_DD0011/lidar/sfm_dense/SASMDD0011_dsm_3cm.tif")
lidar_dsm_aligned_path = Path("/media/laura/laura_usb/qgis/calperumResearch/site2_1_DD0011/lidar/cloud2453cf90763afd91_dsm_3cm_resampled_to_sfm_aligned_to_sfm.tif")

# === Compute GeFolki XY shift ===
flow_u = np.load(flow_u_path)
flow_v = np.load(flow_v_path)
valid = np.isfinite(flow_u) & np.isfinite(flow_v)
shift_xy = np.array([np.median(flow_u[valid]), np.median(flow_v[valid])])
print(f"📌 GeFolki median XY shift: {shift_xy}")

# === Compute DSM Z offset ===
def load_dsm(path):
    with rasterio.open(path) as src:
        arr = src.read(1).astype(np.float32)
        nodata = src.nodata if src.nodata is not None else -9999
        arr[arr == nodata] = np.nan
        return arr

rgb_dsm = load_dsm(rgb_dsm_path)
lidar_dsm = load_dsm(lidar_dsm_aligned_path)
common_mask = np.isfinite(rgb_dsm) & np.isfinite(lidar_dsm)
z_diff = rgb_dsm[common_mask] - lidar_dsm[common_mask]
z_offset_initial = np.median(z_diff)
print(f"📏 DSM median Z-offset: {z_offset_initial:.4f} m")

# === Apply shift and ICP alignment ===
def load_las_as_o3d(las_path):
    las = laspy.read(str(las_path))
    xyz = np.vstack((las.x, las.y, las.z)).T
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(xyz))
    return pcd, las

def center_pcd(pcd):
    centroid = np.mean(np.asarray(pcd.points), axis=0)
    pcd.translate(-centroid)
    return pcd, centroid

def apply_shift(pcd, shift_xy, z_offset):
    pts = np.asarray(pcd.points)
    pts[:, 0] += shift_xy[0]
    pts[:, 1] += shift_xy[1]
    pts[:, 2] += z_offset
    pcd.points = o3d.utility.Vector3dVector(pts)
    return pcd

voxel_size = 0.03
threshold = 3.0
z_correction = z_offset_initial

lidar_pcd, lidar_las = load_las_as_o3d(lidar_path)
rgb_pcd, _ = load_las_as_o3d(rgb_path)

lidar_shifted = apply_shift(lidar_pcd, shift_xy, z_offset_initial)
lidar_down = lidar_shifted.voxel_down_sample(voxel_size)
rgb_down = rgb_pcd.voxel_down_sample(voxel_size)

lidar_down, lidar_centroid = center_pcd(lidar_down)
rgb_down, rgb_centroid = center_pcd(rgb_down)

for cloud in [lidar_down, rgb_down]:
    cloud.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.5, max_nn=30))

trans_init = np.eye(4)
reg = o3d.pipelines.registration.registration_icp(
    lidar_down, rgb_down, threshold, trans_init,
    o3d.pipelines.registration.TransformationEstimationPointToPlane()
)

print(f"🔄 ICP fitness: {reg.fitness:.4f}")
print(f"🔍 RMSE: {reg.inlier_rmse:.4f}")
print("📌 ICP transformation matrix:\n", reg.transformation)

lidar_shifted.transform(reg.transformation)
lidar_shifted.translate(rgb_centroid)
pts = np.asarray(lidar_shifted.points)
pts[:, 2] += z_correction
lidar_shifted.points = o3d.utility.Vector3dVector(pts)

ply_debug_path = base_dir / "lidar_icp_debug.ply"
o3d.io.write_point_cloud(str(ply_debug_path), lidar_shifted)
print(f"📁 Saved intermediate PLY: {ply_debug_path}")

final_output_path = base_dir / "lidar_icp_zcorrected_finales.las"
final_las = laspy.create(point_format=lidar_las.header.point_format, file_version=lidar_las.header.version)
final_las.x = pts[:, 0]
final_las.y = pts[:, 1]
final_las.z = pts[:, 2]

for dim in lidar_las.point_format.dimension_names:
    if dim not in ['X', 'Y', 'Z']:
        final_las[dim] = lidar_las[dim]

final_las.write(str(final_output_path))
print(f"✅ Final LAS written to: {final_output_path}")


#%%
# Combined and Updated ICP Alignment Script

import numpy as np
import laspy
import rasterio
import open3d as o3d
from pathlib import Path

# === Paths ===
base_dir = Path("/media/laura/laura_usb/uas_data/DD0011/")
lidar_path = Path("/media/laura/Extreme SSD/qgis/calperumResearch/site2_1_DD0011/lidar/cloud2453cf90763afd91.las")
rgb_path = base_dir / "point_cloud_p1.laz"
dsm_lidar_path = Path("/media/laura/laura_usb/qgis/calperumResearch/site2_1_DD0011/lidar/cloud2453cf90763afd91_dsm_3cm_resampled_to_sfm.tif")
dsm_rgb_path = Path("/media/laura/laura_usb/qgis/calperumResearch/site2_1_DD0011/lidar/sfm_dense/SASMDD0011_dsm_3cm.tif")

# === Load DSMs and compute Z offsets ===
with rasterio.open(dsm_lidar_path) as dsm_lidar, rasterio.open(dsm_rgb_path) as dsm_rgb:
    if dsm_lidar.shape != dsm_rgb.shape:
        raise ValueError("❌ DSMs have different shapes. Please resample them to match resolution and extent.")
    lidar = dsm_lidar.read(1)
    rgb = dsm_rgb.read(1)
    mask = np.isfinite(lidar) & np.isfinite(rgb)
    dz = lidar[mask] - rgb[mask]
    z_offset_initial = np.median(dz)
    z_correction = -np.mean(dz)
    voxel_size = dsm_lidar.res[0]
    print(f"📏 Voxel size set to: {voxel_size:.3f} m")
    print(f"⬆️ Initial Z offset (median): {z_offset_initial:.3f}")
    print(f"🔁 Z correction (to flatten): {z_correction:.3f}")

# Placeholder for shift, should be computed from GeFolki flow
shift_xy = np.array([46.16, -13.69])  # Example fallback value
threshold = 3.0

# === Load LAS as Open3D PointCloud ===
def load_las_as_o3d(las_path):
    las = laspy.read(str(las_path))
    xyz = np.vstack((las.x, las.y, las.z)).T
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(xyz))
    return pcd, las

# === Center PointClouds ===
def center_pcd(pcd):
    centroid = np.mean(np.asarray(pcd.points), axis=0)
    pcd.translate(-centroid)
    return pcd, centroid

# === Shift point cloud ===
def apply_shift(pcd, shift_xy, z_offset):
    pts = np.asarray(pcd.points)
    pts[:, :2] += shift_xy
    pts[:, 2] += z_offset
    pcd.points = o3d.utility.Vector3dVector(pts)
    return pcd

# === Load and preprocess clouds ===
lidar_pcd, lidar_las = load_las_as_o3d(lidar_path)
rgb_pcd, _ = load_las_as_o3d(rgb_path)

lidar_shifted = apply_shift(lidar_pcd, shift_xy, z_offset_initial)
lidar_down = lidar_shifted.voxel_down_sample(voxel_size)
rgb_down = rgb_pcd.voxel_down_sample(voxel_size)

lidar_down, lidar_centroid = center_pcd(lidar_down)
rgb_down, rgb_centroid = center_pcd(rgb_down)

for cloud in [lidar_down, rgb_down]:
    cloud.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.5, max_nn=30))

reg = o3d.pipelines.registration.registration_icp(
    lidar_down, rgb_down, threshold, np.eye(4),
    o3d.pipelines.registration.TransformationEstimationPointToPlane()
)

print(f"🔄 ICP fitness: {reg.fitness:.4f}")
print(f"🔍 RMSE: {reg.inlier_rmse:.4f}")
print("📌 ICP transformation matrix:\n", reg.transformation)

lidar_shifted.transform(reg.transformation)
lidar_shifted.translate(rgb_centroid)

pts = np.asarray(lidar_shifted.points)
pts[:, 2] += z_correction
lidar_shifted.points = o3d.utility.Vector3dVector(pts)

ply_debug_path = Path("lidar_icp_debug.ply")
o3d.io.write_point_cloud(str(ply_debug_path), lidar_shifted)
print(f"📁 Saved intermediate PLY: {ply_debug_path}")

final_output_path = base_dir / "lidar_icp_zcorrected_finale.las"
final_las = laspy.create(point_format=lidar_las.header.point_format, file_version=lidar_las.header.version)
final_las.x = pts[:, 0]
final_las.y = pts[:, 1]
final_las.z = pts[:, 2]

for dim in lidar_las.point_format.dimension_names:
    if dim not in ['X', 'Y', 'Z']:
        final_las[dim] = lidar_las[dim]

final_las.write(str(final_output_path))
print(f"✅ Final LAS written to: {final_output_path}")

# %%
##test 17
# === WHY APPLY XY FLOW SHIFTS (flow_u, flow_v) BEFORE ICP ===
# GeFolki's dense optical flow (flow_u, flow_v) provides per-pixel horizontal displacement fields (in meters)
# These fields describe how much the RGB-derived DSM should be shifted to align with the LiDAR DSM.
# If we apply the **inverse** of that shift to the LiDAR cloud, it aligns with the RGB cloud.
# This improves the initial alignment, helping ICP converge faster and better (reduces search space).

import numpy as np
import laspy
import open3d as o3d
import rasterio
from pathlib import Path

# === INPUT PATHS ===
base_dir = Path("/media/laura/laura_usb/uas_data/DD0011/")
lidar_path = Path("/media/laura/Extreme SSD/qgis/calperumResearch/site2_1_DD0011/lidar/cloud2453cf90763afd91.las")
rgb_path = base_dir / "point_cloud_p1.laz"
dsm_lidar_path = Path("/media/laura/laura_usb/qgis/calperumResearch/site2_1_DD0011/lidar/cloud2453cf90763afd91_dsm_3cm_resampled_to_sfm.tif")
dsm_rgb_path = Path("/media/laura/laura_usb/qgis/calperumResearch/site2_1_DD0011/lidar/sfm_dense/SASMDD0011_dsm_3cm.tif")
flow_u_path = Path("/media/laura/laura_usb/qgis/calperumResearch/site2_1_DD0011/lidar/sfm_dense/flow_u.npy")
flow_v_path = Path("/media/laura/laura_usb/qgis/calperumResearch/site2_1_DD0011/lidar/sfm_dense/flow_v.npy")
final_output_path = base_dir / "lidar_icp_zcorrected_final.las"

# === LOAD DSMs TO GET Z OFFSETS + VOXEL SIZE ===
with rasterio.open(dsm_lidar_path) as dsm_lidar, rasterio.open(dsm_rgb_path) as dsm_rgb:
    if dsm_lidar.shape != dsm_rgb.shape:
        raise ValueError("❌ DSMs have different shapes. Please resample to same extent/resolution.")
    lidar = dsm_lidar.read(1).astype(np.float32)
    rgb = dsm_rgb.read(1).astype(np.float32)
    nodata_lidar = dsm_lidar.nodata or -9999
    nodata_rgb = dsm_rgb.nodata or -9999
    lidar[lidar == nodata_lidar] = np.nan
    rgb[rgb == nodata_rgb] = np.nan
    mask = np.isfinite(lidar) & np.isfinite(rgb)
    dz = lidar[mask] - rgb[mask]
    z_offset_initial = np.median(dz)
    z_correction = -np.mean(dz)
    voxel_size = dsm_lidar.res[0]
    print(f"📏 Voxel size set to: {voxel_size:.3f} m")
    print(f"⬆️ Initial Z offset (median): {z_offset_initial:.3f}")
    print(f"🔁 Z correction (to flatten): {z_correction:.3f}")

# === COMPUTE MEAN FLOW SHIFT ===
flow_u = np.load(flow_u_path)
flow_v = np.load(flow_v_path)
valid = np.isfinite(flow_u) & np.isfinite(flow_v) & (flow_u != 0) & (flow_v != 0)
if np.any(valid):
    shift_xy = np.array([np.median(flow_u[valid]), np.median(flow_v[valid])])
else:
    shift_xy = np.array([0.0, 0.0])
print(f"📌 GeFolki XY shift from flow: {shift_xy}")

# === Load LAS as Open3D PointCloud ===
def load_las_as_o3d(las_path):
    las = laspy.read(str(las_path))
    pts = np.vstack((las.x, las.y, las.z)).T
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts))
    return pcd, las

# === Shift & Center ===
def apply_shift(pcd, shift_xy, z_offset):
    pts = np.asarray(pcd.points)
    pts[:, 0] -= shift_xy[0]  # Invert X shift
    pts[:, 1] -= shift_xy[1]  # Invert Y shift
    pts[:, 2] += z_offset     # Apply Z offset normally
    pcd.points = o3d.utility.Vector3dVector(pts)
    return pcd

def center_pcd(pcd):
    centroid = np.mean(np.asarray(pcd.points), axis=0)
    pcd.translate(-centroid)
    return pcd, centroid

# === MAIN ALIGNMENT LOGIC ===
lidar_pcd, lidar_las = load_las_as_o3d(lidar_path)
rgb_pcd, _ = load_las_as_o3d(rgb_path)

# Apply flow-derived XY shift and DSM-derived Z offset
lidar_shifted = apply_shift(lidar_pcd, shift_xy, z_offset_initial)

# Downsample and center
lidar_down = lidar_shifted.voxel_down_sample(voxel_size)
rgb_down = rgb_pcd.voxel_down_sample(voxel_size)
lidar_down, lidar_centroid = center_pcd(lidar_down)
rgb_down, rgb_centroid = center_pcd(rgb_down)

# Estimate normals (for point-to-plane ICP)
lidar_down.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.5, max_nn=30))
rgb_down.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.5, max_nn=30))

# ICP Registration
threshold = 3.0
reg = o3d.pipelines.registration.registration_icp(
    lidar_down, rgb_down, threshold, np.eye(4),
    o3d.pipelines.registration.TransformationEstimationPointToPlane())
print(f"🔄 ICP fitness: {reg.fitness:.4f}")
print(f"🔍 ICP RMSE: {reg.inlier_rmse:.4f}")
print("\ud83d\udccc ICP Transformation:\n", reg.transformation)

# Apply transformation + recenter to RGB cloud
lidar_shifted.transform(reg.transformation)
lidar_shifted.translate(rgb_centroid)

# Final Z correction
pts = np.asarray(lidar_shifted.points)
pts[:, 2] += z_correction

# Save LAS
final_las = laspy.create(point_format=lidar_las.header.point_format, file_version=lidar_las.header.version)
final_las.x = pts[:, 0]
final_las.y = pts[:, 1]
final_las.z = pts[:, 2]
for dim in lidar_las.point_format.dimension_names:
    if dim not in ['X', 'Y', 'Z']:
        final_las[dim] = lidar_las[dim]
final_las.write(str(final_output_path))
print(f"✅ Final LAS written to: {final_output_path}")

# %%
