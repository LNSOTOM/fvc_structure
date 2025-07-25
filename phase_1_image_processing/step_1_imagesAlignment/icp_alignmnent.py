#%%
import open3d as o3d
import laspy
import numpy as np
from pathlib import Path

# === Input paths ===
source_path = Path("/media/laura/Extreme SSD/qgis/calperumResearch/site2_1_DD0011/lidar/cloud2453cf90763afd91_displaced_by_efolki.las")
target_path = Path("/media/laura/laura_usb/qgis/calperumResearch/site2_1_DD0011/lidar/thinned_voxel/point_cloud_p1_thinned_3cm.las")
flow_u_path = Path("/media/laura/laura_usb/code/fvc_structure/gefolki/flow_u.npy")
flow_v_path = Path("/media/laura/laura_usb/code/fvc_structure/gefolki/flow_v.npy")

# === Output path ===
output_path = source_path.with_name(target_path.stem + "_icp_refined.las")

# === Load LAS as numpy arrays ===
def las_to_o3d_pointcloud(las_file):
    las = laspy.read(str(las_file))
    points = np.vstack((las.x, las.y, las.z)).T
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    return pcd, las

# Load source (GeFolki-aligned LiDAR)
source_pcd, source_las = las_to_o3d_pointcloud(source_path)

# Load target (thinned SfM dense point cloud)
target_pcd, _ = las_to_o3d_pointcloud(target_path)

# === Load GeFolki flow fields (for reference only, not needed here but you mentioned them) ===
flow_u = np.load(flow_u_path)
flow_v = np.load(flow_v_path)

# === ICP refinement ===
print("🔄 Running ICP refinement...")

threshold = 0.1  # Max correspondence distance in meters
init_transform = np.eye(4)  # EFolki already displaced LiDAR → identity init

icp_result = o3d.pipelines.registration.registration_icp(
    source=source_pcd,
    target=target_pcd,
    max_correspondence_distance=threshold,
    init=init_transform,
    estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint()
)

print("✅ ICP completed.")
print("📐 Transformation matrix:")
print(icp_result.transformation)

# === Apply transformation to source ===
source_pcd.transform(icp_result.transformation)

# === Save transformed source as new LAS ===
refined_points = np.asarray(source_pcd.points)
refined_las = laspy.create(point_format=source_las.header.point_format, file_version=source_las.header.version)

refined_las.x = refined_points[:, 0]
refined_las.y = refined_points[:, 1]
refined_las.z = refined_points[:, 2]

refined_las.write(str(output_path))

# Save transformation matrix (optional)
np.savetxt(output_path.with_suffix(".txt"), icp_result.transformation, fmt="%.8f")

print(f"✅ ICP-refined LAS saved to:\n   {output_path}")
print(f"📝 Transformation matrix saved to:\n   {output_path.with_suffix('.txt')}")

#%%
####test 2
import open3d as o3d
import numpy as np
import laspy

# === Paths ===
source_path = "/media/laura/laura_usb/qgis/calperumResearch/site2_1_DD0011/lidar/lidar_gefolki_shifted_xyz.las"  # shifted LiDAR
target_path = "/media/laura/laura_usb/qgis/calperumResearch/site2_1_DD0011/lidar/sfm_dense/point_cloud_p1.laz"  # RGB dense cloud
output_icp_path = "/media/laura/laura_usb/qgis/calperumResearch/site2_1_DD0011/lidar/lidar_icp_refined.las"

# === Load point clouds ===
def las_to_o3d(las_path):
    las = laspy.read(las_path)
    xyz = np.vstack((las.x, las.y, las.z)).T
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    return pcd, las

pcd_src, las_src = las_to_o3d(source_path)
pcd_tgt, _ = las_to_o3d(target_path)

# === Downsample (optional, speeds up ICP) ===
pcd_src_ds = pcd_src.voxel_down_sample(0.05)
pcd_tgt_ds = pcd_tgt.voxel_down_sample(0.05)

# === Estimate normals (required for point-to-plane ICP) ===
pcd_src_ds.estimate_normals()
pcd_tgt_ds.estimate_normals()

# === Run ICP ===
threshold = 1.0  # max distance in meters
reg_icp = o3d.pipelines.registration.registration_icp(
    pcd_src_ds, pcd_tgt_ds, threshold,
    np.eye(4),
    o3d.pipelines.registration.TransformationEstimationPointToPlane()
)

print("✅ ICP Transformation Matrix:")
print(reg_icp.transformation)

# === Apply ICP transform to original cloud ===
pcd_src.transform(reg_icp.transformation)
transformed_xyz = np.asarray(pcd_src.points)

# === Write ICP-adjusted LAS ===
las_src.x = transformed_xyz[:, 0]
las_src.y = transformed_xyz[:, 1]
las_src.z = transformed_xyz[:, 2]
las_src.write(output_icp_path)

print(f"✅ ICP applied and saved: {output_icp_path}")

# %%
##test 3
import numpy as np
import laspy
import open3d as o3d
from pathlib import Path

# === Paths
lidar_shifted_las = Path("/media/laura/laura_usb/qgis/calperumResearch/site2_1_DD0011/lidar/lidar_gefolki_shifted_xyz.las")
rgb_dense_las = Path("/media/laura/laura_usb/qgis/calperumResearch/site2_1_DD0011/lidar/sfm_dense/point_cloud_p1.laz")
output_las_path = Path("/media/laura/laura_usb/qgis/calperumResearch/site2_1_DD0011/lidar/lidar_icp_registered.las")

# === Function to convert LAS to Open3D point cloud
def las_to_o3d_cloud(las_path):
    las = laspy.read(las_path)
    points = np.vstack((las.x, las.y, las.z)).T
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    return pcd, las

# === Load both clouds
source_pcd, las_src = las_to_o3d_cloud(lidar_shifted_las)  # LiDAR (to align)
target_pcd, _ = las_to_o3d_cloud(rgb_dense_las)            # RGB (reference)

# === Preprocess (downsample for faster ICP)
source_down = source_pcd.voxel_down_sample(voxel_size=0.05)
target_down = target_pcd.voxel_down_sample(voxel_size=0.05)

# === Estimate normals (required)
source_down.estimate_normals()
target_down.estimate_normals()

# === Run ICP
threshold = 2.0  # max correspondence distance (meters)
reg = o3d.pipelines.registration.registration_icp(
    source_down, target_down, threshold,
    np.eye(4),
    o3d.pipelines.registration.TransformationEstimationPointToPoint()
)

print(f"ICP Fitness: {reg.fitness:.3f}, RMSE: {reg.inlier_rmse:.3f}")
T = reg.transformation
print("Transformation Matrix:\n", T)

# === Apply transformation to full-resolution cloud
source_pcd.transform(T)
transformed_xyz = np.asarray(source_pcd.points)

# === Save new transformed LAS
new_las = laspy.LasData(las_src.header)
new_las.x, new_las.y, new_las.z = transformed_xyz.T

# Preserve extra fields
for dim in las_src.point_format.dimensions:
    if dim.name not in ['X', 'Y', 'Z']:
        new_las[dim.name] = las_src[dim.name]

new_las.write(output_las_path)
print(f"✅ ICP-aligned LiDAR saved to: {output_las_path}")

# %%
##test 4
import numpy as np
import laspy
import rasterio
from rasterio.transform import rowcol
import open3d as o3d
from pathlib import Path

# === INPUTS ===
lidar_path = Path("/media/laura/Extreme SSD/qgis/calperumResearch/site2_1_DD0011/lidar/cloud2453cf90763afd91.las")
ref_dsm_path = Path("/media/laura/laura_usb/qgis/calperumResearch/site2_1_DD0011/lidar/sfm_dense/SASMDD0011_dsm_3cm.tif")
flow_u_path = Path("/media/laura/laura_usb/qgis/calperumResearch/site2_1_DD0011/lidar/sfm_dense/flow_u.npy")
flow_v_path = Path("/media/laura/laura_usb/qgis/calperumResearch/site2_1_DD0011/lidar/sfm_dense/flow_v.npy")
rgb_dense_path = Path("/media/laura/Extreme SSD/qgis/calperumResearch/site2_1_DD0011/lidar/point_cloud_p1.laz")

output_las_path = Path("/media/laura/laura_usb/qgis/calperumResearch/site2_1_DD0011/lidar/lidar_gefolki_icp_aligned.las")

# === Load flow fields ===
flow_u = np.load(flow_u_path)
flow_v = np.load(flow_v_path)

# === Load DSM ===
with rasterio.open(ref_dsm_path) as ref_dsm:
    ref_array = ref_dsm.read(1)
    transform = ref_dsm.transform
    height, width = ref_dsm.height, ref_dsm.width

# === Load original LiDAR ===
las = laspy.read(lidar_path)
x, y, z = las.x.copy(), las.y.copy(), las.z.copy()

# === Map XY to DSM pixel coords
cols, rows = rowcol(transform, x, y)
valid = (rows >= 0) & (rows < height) & (cols >= 0) & (cols < width)
cols, rows = cols[valid], rows[valid]
x_valid, y_valid, z_valid = x[valid], y[valid], z[valid]

# === Apply GeFolki XY shift (average)
dx = np.mean(flow_u[rows, cols]) * transform.a
dy = np.mean(flow_v[rows, cols]) * abs(transform.e)

# === Apply Z offset from DSM
ref_z = ref_array[rows, cols]
dz = np.nanmean(ref_z - z_valid)

print(f"📐 GeFolki Shift: dx={dx:.3f}, dy={dy:.3f}, dz={dz:.3f}")

# === Shift entire cloud
x_shifted = x + dx
y_shifted = y + dy
z_shifted = z + dz

# === Convert shifted to Open3D
shifted_points = np.vstack((x_shifted, y_shifted, z_shifted)).T
source_pcd = o3d.geometry.PointCloud()
source_pcd.points = o3d.utility.Vector3dVector(shifted_points)

# === Load RGB dense as reference
def load_las_as_o3d(las_path):
    ref_las = laspy.read(las_path)
    pts = np.vstack((ref_las.x, ref_las.y, ref_las.z)).T
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    return pcd

target_pcd = load_las_as_o3d(rgb_dense_path)

# === Downsample
voxel = 0.05  # 5cm
source_down = source_pcd.voxel_down_sample(voxel)
target_down = target_pcd.voxel_down_sample(voxel)

source_down.estimate_normals()
target_down.estimate_normals()

# === Run ICP
print("⚙️ Running ICP...")
icp = o3d.pipelines.registration.registration_icp(
    source_down, target_down, 2.0, np.eye(4),
    o3d.pipelines.registration.TransformationEstimationPointToPoint()
)
print(f"✅ ICP Done. Fitness: {icp.fitness:.4f}, RMSE: {icp.inlier_rmse:.4f}")
print("Transformation Matrix:\n", icp.transformation)

# === Apply ICP to full-shifted cloud
source_pcd.transform(icp.transformation)
transformed_xyz = np.asarray(source_pcd.points)

# === Save final aligned LAS
final_las = laspy.LasData(las.header)
final_las.x, final_las.y, final_las.z = transformed_xyz.T
for dim in las.point_format.dimensions:
    if dim.name not in ['X', 'Y', 'Z']:
        final_las[dim.name] = las[dim.name]

final_las.write(output_las_path)
print(f"💾 Final aligned LAS saved to: {output_las_path}")

#%%
##pract
import numpy as np
import laspy
import rasterio
from pathlib import Path
from scipy.interpolate import RegularGridInterpolator

# === Paths
las_path = Path("/media/laura/Extreme SSD/qgis/calperumResearch/site2_1_DD0011/lidar/cloud2453cf90763afd91.las")
lidar_dsm_aligned_path = Path("/media/laura/laura_usb/qgis/calperumResearch/site2_1_DD0011/lidar/cloud2453cf90763afd91_dsm_3cm_resampled_to_sfm_aligned_to_rgb.tif")
flow_u_path = Path("/media/laura/laura_usb/qgis/calperumResearch/site2_1_DD0011/lidar/sfm_dense/flow_u.npy")
flow_v_path = Path("/media/laura/laura_usb/qgis/calperumResearch/site2_1_DD0011/lidar/sfm_dense/flow_v.npy")
output_las_icp_init = lidar_dsm_aligned_path.with_name("cloud2453cf90763afd91_gefolki_xy_init_for_icp.las")

# === Load DSM and flow
with rasterio.open(lidar_dsm_aligned_path) as src:
    transform = src.transform
    width = src.width
    height = src.height
    xres, yres = transform.a, -transform.e
    x0, y0 = transform.c, transform.f

# === Flow (u = horizontal, v = vertical)
flow_u = np.load(flow_u_path)
flow_v = np.load(flow_v_path)

# === Coordinate grid for interpolation
x_coords = x0 + np.arange(width) * xres
y_coords = y0 - np.arange(height) * yres

interp_u = RegularGridInterpolator((y_coords, x_coords), flow_u, bounds_error=False, fill_value=0)
interp_v = RegularGridInterpolator((y_coords, x_coords), flow_v, bounds_error=False, fill_value=0)

# === Load LAS and apply 2D flow to XY
las = laspy.read(str(las_path))
x, y, z = las.x.copy(), las.y.copy(), las.z.copy()

# Interpolate displacements at each (y, x)
points = np.column_stack((y, x))  # raster order = (row, col) = (y, x)
dx = interp_u(points) * xres
dy = interp_v(points) * yres

# Apply only XY shift (ICP will refine XYZ)
las.x = x + dx
las.y = y + dy

# Optional: Keep original Z or later refine it
# las.z = z + dz_interpolated (if dz is available)

las.write(str(output_las_icp_init))
print(f"✅ LiDAR LAS file prealigned for ICP saved to:\n{output_las_icp_init}")

# %%
##
import numpy as np
import laspy
import rasterio
from pathlib import Path
from scipy.interpolate import RegularGridInterpolator

# === Paths ===
las_path = Path("/media/laura/Extreme SSD/qgis/calperumResearch/site2_1_DD0011/lidar/cloud2453cf90763afd91.las")
lidar_dsm_path = Path("/media/laura/laura_usb/qgis/calperumResearch/site2_1_DD0011/lidar/cloud2453cf90763afd91_dsm_3cm_resampled_to_sfm.tif")
flow_u_path = Path("/media/laura/laura_usb/qgis/calperumResearch/site2_1_DD0011/lidar/sfm_dense/flow_u.npy")
flow_v_path = Path("/media/laura/laura_usb/qgis/calperumResearch/site2_1_DD0011/lidar/sfm_dense/flow_v.npy")
output_las_path = las_path.with_name("cloud2453cf90763afd91_gefolki_xy_shift_only.las")

# === Load flow
flow_u = np.load(flow_u_path)
flow_v = np.load(flow_v_path)

# === Load raster transform
with rasterio.open(lidar_dsm_path) as src:
    transform = src.transform
    width = src.width
    height = src.height
    xres, yres = transform.a, -transform.e
    x0, y0 = transform.c, transform.f

# === Create grid for interpolation
x_coords = x0 + np.arange(width) * xres
y_coords = y0 - np.arange(height) * yres

interp_u = RegularGridInterpolator((y_coords, x_coords), flow_u, bounds_error=False, fill_value=0)
interp_v = RegularGridInterpolator((y_coords, x_coords), flow_v, bounds_error=False, fill_value=0)

# === Load original LAS
las = laspy.read(str(las_path))
x = las.x.copy()
y = las.y.copy()
z = las.z.copy()  # Preserve Z

# === Interpolate flow at (y, x)
points = np.column_stack((y, x))
dx = interp_u(points) * xres
dy = interp_v(points) * yres

x_shifted = x + dx
y_shifted = y + dy

# === Create new LAS and copy all attributes
las_out = laspy.LasData(las.header)
las_out.points = las.points.copy()

# Convert shifted XYZ to integer coordinates using header scaling
las_out.X = ((x_shifted - las.header.offsets[0]) / las.header.scales[0]).astype(np.int32)
las_out.Y = ((y_shifted - las.header.offsets[1]) / las.header.scales[1]).astype(np.int32)
las_out.Z = ((z - las.header.offsets[2]) / las.header.scales[2]).astype(np.int32)

# === Save output
las_out.write(str(output_las_path))
print(f"✅ LAS file with GeFolki XY shift and original Z saved to:\n{output_las_path}")


# %%
