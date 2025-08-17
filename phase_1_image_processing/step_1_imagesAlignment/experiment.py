#%%
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
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
from scipy.interpolate import RegularGridInterpolator
import matplotlib.pyplot as plt

# === INPUT PATHS ===
base_dir = Path("/media/laura/laura_usb/uas_data/DD0011/")
lidar_path = Path("/media/laura/Extreme SSD/qgis/calperumResearch/site2_1_DD0011/lidar/cloud2453cf90763afd91.las")
rgb_path = base_dir / "sfm_p1.las"
dsm_lidar_path = Path("/media/laura/laura_usb/qgis/calperumResearch/site2_1_DD0011/lidar/cloud2453cf90763afd91_dsm_3cm_resampled_to_sfm.tif")
dsm_rgb_path = Path("/media/laura/laura_usb/qgis/calperumResearch/site2_1_DD0011/lidar/sfm_dense/SASMDD0011_dsm_3cm.tif")
flow_u_path = Path("/media/laura/laura_usb/qgis/calperumResearch/site2_1_DD0011/lidar/sfm_dense/flow_u.npy")
flow_v_path = Path("/media/laura/laura_usb/qgis/calperumResearch/site2_1_DD0011/lidar/sfm_dense/flow_v.npy")
final_output_path = base_dir / "lidar_icp_zcorrected_final_bayes.las"


# === LOAD DSMs & COMPUTE OFFSETS ===
with rasterio.open(dsm_lidar_path) as dsm_lidar_src, rasterio.open(dsm_rgb_path) as dsm_rgb_src:
    dsm_lidar = dsm_lidar_src.read(1).astype(np.float32)
    dsm_rgb = dsm_rgb_src.read(1).astype(np.float32)

    nodata_lidar = dsm_lidar_src.nodata or -9999
    nodata_rgb = dsm_rgb_src.nodata or -9999
    dsm_lidar[dsm_lidar == nodata_lidar] = np.nan
    dsm_rgb[dsm_rgb == nodata_rgb] = np.nan

    mask = np.isfinite(dsm_lidar) & np.isfinite(dsm_rgb)
    dz = dsm_lidar[mask] - dsm_rgb[mask]
    z_offset_initial = np.median(dz)
    z_correction = -np.mean(dz)

    pixel_x, pixel_y = dsm_lidar_src.res
    if not np.isclose(pixel_x, pixel_y):
        print(f"⚠️ Warning: DSM pixels are not square! x: {pixel_x:.3f}, y: {pixel_y:.3f}")
    voxel_size = np.mean([pixel_x, pixel_y])  # use average as fallback

    print(f"📏 Voxel size set to: {voxel_size:.3f} m")
    print(f"⬆️ Initial Z offset (median): {z_offset_initial:.3f}")
    print(f"🔁 Z correction (to flatten): {z_correction:.3f}")


# === COMPUTE MEAN FLOW SHIFT ===
flow_u = np.load(flow_u_path)
flow_v = np.load(flow_v_path)
valid = np.isfinite(flow_u) & np.isfinite(flow_v) & (flow_u != 0) & (flow_v != 0)
shift_xy = np.array([np.median(flow_u[valid]), np.median(flow_v[valid])]) if np.any(valid) else np.array([0.0, 0.0])
print(f"📌 GeFolki XY shift: {shift_xy}")

# 📏 Voxel size set to: 0.030 m
# ⬆️ Initial Z offset (median): -0.020
# 🔁 Z correction (to flatten): 0.042
# 📌 GeFolki XY shift from flow: [ 1.40699484 -0.56138638]

#%%
# === Load LAS as Open3D PointCloud ===
def load_las_as_o3d(path):
    las = laspy.read(str(path))
    points = np.vstack((las.x, las.y, las.z)).T
    return o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points)), las

def apply_gefolki_shift(pcd, shift_xy, z_offset):
    points = np.asarray(pcd.points)
    points[:, 0] += shift_xy[0]
    points[:, 1] += shift_xy[1]
    points[:, 2] += z_offset
    pcd.points = o3d.utility.Vector3dVector(points)
    return pcd

def center(pcd):
    centroid = np.mean(np.asarray(pcd.points), axis=0)
    pcd.translate(-centroid)
    return pcd, centroid

# === Align & Transform ===
lidar_pcd, lidar_las = load_las_as_o3d(lidar_path)
rgb_pcd, _ = load_las_as_o3d(rgb_path)

lidar_shifted = apply_gefolki_shift(lidar_pcd, shift_xy, z_offset_initial)

# Downsample for ICP
lidar_down = lidar_shifted.voxel_down_sample(voxel_size)
rgb_down = rgb_pcd.voxel_down_sample(voxel_size)

# Center for stability
lidar_down, lidar_centroid = center(lidar_down)
rgb_down, rgb_centroid = center(rgb_down)

# Estimate normals for Point-to-Plane ICP
lidar_down.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.5, max_nn=30))
rgb_down.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.5, max_nn=30))


#%%
# === CRS & Centroid Check ===
print("\n🧭 LAS CRS and Coordinate Origin Diagnostic")

with laspy.open(lidar_path) as f_lidar, laspy.open(rgb_path) as f_rgb:
    print(f"📌 LiDAR LAS: point count = {f_lidar.header.point_count}, scale: {f_lidar.header.scales}, offset: {f_lidar.header.offsets}")
    print(f"📌 RGB LAS: point count = {f_rgb.header.point_count}, scale: {f_rgb.header.scales}, offset: {f_rgb.header.offsets}")

# Load centroids
lidar_pcd_tmp, _ = load_las_as_o3d(lidar_path)
rgb_pcd_tmp, _ = load_las_as_o3d(rgb_path)
lidar_centroid_diag = np.mean(np.asarray(lidar_pcd_tmp.points), axis=0)
rgb_centroid_diag = np.mean(np.asarray(rgb_pcd_tmp.points), axis=0)

print(f"📍 LiDAR centroid (XYZ): {lidar_centroid_diag}")
print(f"📍 RGB centroid (XYZ):   {rgb_centroid_diag}")

centroid_dist = np.linalg.norm(lidar_centroid_diag - rgb_centroid_diag)
print(f"📏 Centroid distance: {centroid_dist:.2f} units")

# Warn if RGB cloud is local
if centroid_dist > 100:
    print("⚠️ Warning: Large spatial separation between LiDAR and RGB centroids. Likely mismatch in coordinate reference systems.")
    print("💡 Suggestion: RGB point cloud appears to be in local SfM coordinates. Consider georeferencing it to match LiDAR CRS.")

# 🧭 LAS CRS and Coordinate Origin Diagnostic
# 📌 LiDAR LAS: point count = 66646512, scale: [0.0001 0.0001 0.0001], offset: [4.73243694e+05 6.23576117e+06 3.55845337e+01]
# 📌 RGB LAS: point count = 148730634, scale: [0.001 0.001 0.001], offset: [ 473000. 6235000.       0.]
# 📍 LiDAR centroid (XYZ): [4.73240059e+05 6.23576162e+06 3.35861021e+01]
# 📍 RGB centroid (XYZ):   [4.73233748e+05 6.23575743e+06 3.47635603e+01]
# 📏 Centroid distance: 7.67 units

#%%
# Run ICP
reg = o3d.pipelines.registration.registration_icp(
    lidar_down, rgb_down, max_correspondence_distance=3.0, init=np.eye(4),
    estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPlane())

print(f"🔄 ICP fitness: {reg.fitness:.4f}")
print(f"🔍 ICP RMSE: {reg.inlier_rmse:.4f}")
print("📌 ICP transformation:\n", reg.transformation)


# 🔄 ICP fitness: 0.9554
# 🔍 ICP RMSE: 0.2464
# 📌 ICP Transformation:
#  [[ 9.99998649e-01 -1.62430502e-03  2.51349173e-04  1.99445776e-01]
#  [ 1.62431678e-03  9.99998680e-01 -4.65712259e-05  1.33371020e-01]
#  [-2.51273195e-04  4.69794337e-05  9.99999967e-01 -8.47789608e-02]
#  [ 0.00000000e+00  0.00000000e+00  0.00000000e+00  1.00000000e+00]]

# 🔄 ICP fitness: 0.9487
# 🔍 ICP RMSE: 0.2243
# 📌 ICP transformation:
#  [[ 9.99999917e-01  4.04653005e-04  4.89428191e-05  3.13983857e-01]
#  [-4.04654455e-04  9.99999918e-01  2.96330719e-05  4.78166422e-01]
#  [-4.89308240e-05 -2.96528744e-05  9.99999998e-01 -1.58397574e+00]
#  [ 0.00000000e+00  0.00000000e+00  0.00000000e+00  1.00000000e+00]]

#%%
# === Apply transformation to original LiDAR points ===
# === Original points with GeFolki shift ===
original_pts = np.vstack((lidar_las.x + shift_xy[0],
                          lidar_las.y + shift_xy[1],
                          lidar_las.z + z_offset_initial)).T

# ✅ STEP 1: Re-center the full-resolution point cloud to match ICP
original_pts_centered = original_pts - lidar_Recentroid  # <- use lidar_centroid from earlier

# ✅ STEP 2: Apply ICP transformation
original_pts_centered_h = np.hstack((original_pts_centered, np.ones((original_pts.shape[0], 1))))
transformed_centered = (reg.transformation @ original_pts_centered_h.T).T[:, :3]

# ✅ STEP 3: De-center (bring back to real-world coordinates)
transformed_pts = transformed_centered + rgb_centroid

# ✅ STEP 4: Apply final Z flattening correction
transformed_pts[:, 2] += z_correction
print(f"✅ Final mean Z after all corrections: {np.mean(transformed_pts[:, 2]):.2f}")


# Quick sanity check
assert np.nanmean(transformed_pts[:, 2]) < 100, "⚠️ Z values seem too high – possible double shift!"

# === Write final LAS ===
final_las = laspy.create(point_format=lidar_las.header.point_format, file_version=lidar_las.header.version)
final_las.x = transformed_pts[:, 0]
final_las.y = transformed_pts[:, 1]
final_las.z = transformed_pts[:, 2]

# Preserve other attributes (intensity, classification, etc.)
for dim in lidar_las.point_format.dimension_names:
    if dim not in ['X', 'Y', 'Z']:
        final_las[dim] = lidar_las[dim]

final_las.write(str(final_output_path))
print(f"✅ Final aligned LAS written to:\n{final_output_path}")

#%%
print("📊 Z statistics after alignment:")
print(f"   Min Z: {np.min(transformed_pts[:, 2]):.2f}")
print(f"   Max Z: {np.max(transformed_pts[:, 2]):.2f}")
print(f"   Mean Z: {np.mean(transformed_pts[:, 2]):.2f}")
print(f"   Median Z: {np.median(transformed_pts[:, 2]):.2f}")


#%%
import json

def check_dsm_square(pixel_res):
    """Check and log whether DSM pixels are square."""
    pixel_x, pixel_y = pixel_res
    if not np.isclose(pixel_x, pixel_y):
        print(f"⚠️ Warning: DSM pixels are not square! x = {pixel_x:.3f}, y = {pixel_y:.3f}")
    else:
        print("✅ DSM pixel shape is square.")
    return pixel_x, pixel_y

def extract_icp_translation(transformation):
    """Extract XYZ translation vector from ICP 4x4 transformation matrix."""
    return transformation[:3, 3]

def save_alignment_summary(summary_dict, out_path):
    """Save alignment metadata as pretty-printed JSON."""
    with open(out_path, "w") as f:
        json.dump(summary_dict, f, indent=2)
    print(f"📝 Alignment summary saved to:\n{out_path}")

# === Recompute or reuse relevant metadata ===
pixel_res = (pixel_x, pixel_y)  # ensure these are defined earlier
pixel_x, pixel_y = check_dsm_square(pixel_res)
icp_translation = extract_icp_translation(reg.transformation)

# === Create structured metadata summary ===
alignment_summary = {
    "input_paths": {
        "lidar_las": str(lidar_path),
        "rgb_dense_las": str(rgb_path),
        "dsm_lidar": str(dsm_lidar_path),
        "dsm_rgb": str(dsm_rgb_path),
        "gefolki_flow_u": str(flow_u_path),
        "gefolki_flow_v": str(flow_v_path)
    },
    "alignment_parameters": {
        "voxel_size_m": round(voxel_size, 4),
        "pixel_resolution_m": [round(pixel_x, 4), round(pixel_y, 4)],
        "gefolki_xy_shift_m": [round(v, 4) for v in shift_xy],
        "z_offset_initial_median_m": round(z_offset_initial, 4),
        "z_correction_mean_m": round(z_correction, 4),
        "icp_translation_xyz_m": [round(v, 4) for v in icp_translation],
        "icp_fitness": round(reg.fitness, 4),
        "icp_rmse": round(reg.inlier_rmse, 4)
    },
    "output": {
        "aligned_las_path": str(final_output_path)
    }
}

# === Save the summary JSON ===
summary_path = base_dir / "alignment_summary.json"
save_alignment_summary(alignment_summary, summary_path)



# 4. Optional Open3D visualization
# try:
#     o3d.visualization.draw_geometries([lidar_shifted.paint_uniform_color([1, 0, 0]),
#                                        rgb_pcd.paint_uniform_color([0, 1, 0])])
# except Exception as e:
#     print("⚠️ Open3D visualization skipped:", e)


#%%
# Original LiDAR Z
print("Original Z stats:")
print(f"  min: {np.min(lidar_las.z):.2f}, max: {np.max(lidar_las.z):.2f}, mean: {np.mean(lidar_las.z):.2f}")

# After GeFolki shift
shifted_z = lidar_las.z + z_offset_initial
print("After GeFolki Z offset:")
print(f"  mean: {np.mean(shifted_z):.2f}, median offset: {z_offset_initial:.3f}")

# After ICP
transformed_pts_h = np.hstack((np.vstack((lidar_las.x + shift_xy[0], 
                                           lidar_las.y + shift_xy[1],
                                           shifted_z)).T, 
                               np.ones((lidar_las.z.shape[0], 1))))
transformed_pts = (reg.transformation @ transformed_pts_h.T).T[:, :3]
print("After ICP transformation:")
print(f"  mean Z: {np.mean(transformed_pts[:, 2]):.2f}")

# After final Z flattening
transformed_pts[:, 2] += z_correction
print("After final Z correction:")
print(f"  mean Z: {np.mean(transformed_pts[:, 2]):.2f}, Z correction applied: {z_correction:.3f}")



# %%
import matplotlib.pyplot as plt

# === 1. Extract Z height profiles ===
z_rgb = np.asarray(rgb_pcd.points)[:, 2]
z_lidar_orig = np.vstack((lidar_las.x, lidar_las.y, lidar_las.z))[-1]
z_lidar_aligned = transformed_pts[:, 2]

# === 2. Plot histograms of Z values ===
plt.figure(figsize=(10, 6))
plt.hist(z_rgb, bins=100, alpha=0.5, label='RGB Dense Cloud (Z)', color='green')
plt.hist(z_lidar_orig, bins=100, alpha=0.5, label='Original LiDAR (Z)', color='blue')
plt.hist(z_lidar_aligned, bins=100, alpha=0.5, label='Aligned LiDAR (Z)', color='red')
plt.title("Z Height Distribution Comparison")
plt.xlabel("Z (meters)")
plt.ylabel("Frequency")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# === 3. Optional: XY profile slice comparison ===
sample_indices = np.random.choice(len(z_lidar_aligned), size=100000, replace=False)

plt.figure(figsize=(12, 6))
plt.scatter(transformed_pts[sample_indices, 0], transformed_pts[sample_indices, 2],
            s=0.1, color='red', label='Aligned LiDAR')
plt.scatter(rgb_pcd.points[sample_indices][..., 0], rgb_pcd.points[sample_indices][..., 2],
            s=0.1, color='green', label='RGB Cloud')
plt.title("XZ Profile Slice Comparison")
plt.xlabel("X")
plt.ylabel("Z")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


# %%
