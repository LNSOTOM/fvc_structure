#%%
#step 1.1: Voxel Thinning of LiDAR and SfM Point Clouds
# LiDAR point cloud using voxel grid filtering at 3 cm resolution
import laspy
import open3d as o3d
import numpy as np
from pathlib import Path

# === INPUT LAS ===
las_path = Path("/media/laura/Extreme SSD/qgis/calperumResearch/site2_1_DD0011/lidar/cloud2453cf90763afd91.las")

# === OUTPUT DIRECTORY ===
output_dir = Path("/media/laura/laura_usb/qgis/calperumResearch/site2_1_DD0011/lidar/thinned_voxel")
output_dir.mkdir(parents=True, exist_ok=True)

# === Load LAS using laspy ===
las = laspy.read(str(las_path))
points = np.vstack((las.x, las.y, las.z)).T

# === Convert to Open3D PointCloud ===
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points)

# === Apply voxel downsampling (3 cm) ===
voxel_size = 0.03  # 3 cm
pcd_down = pcd.voxel_down_sample(voxel_size)

# === Extract downsampled points ===
down_points = np.asarray(pcd_down.points)

# === Create new LAS with thinned points ===
new_las = laspy.create(point_format=las.header.point_format, file_version=las.header.version)

# Only copy coordinates (X, Y, Z)
new_las.x = down_points[:, 0]
new_las.y = down_points[:, 1]
new_las.z = down_points[:, 2]

# === Save output LAS ===
output_path = output_dir / (las_path.stem + "_thinned_3cm.las")
new_las.write(str(output_path))

print(f"✅ Thinned point cloud saved to:\n   {output_path}")
print(f"📉 Original points: {len(points):,}")
print(f"📈 Thinned points : {len(down_points):,}")

#  Original points: 66,646,512
# Thinned points : 44,443,920


#%%
#step 1.2: Voxel Thinning of SfM Point Cloud
# Thin SfM Point Cloud
import laspy
import open3d as o3d
import numpy as np
from pathlib import Path

# === INPUT SfM Dense Cloud (RGB) ===
sfm_path = Path("/media/laura/laura_usb/qgis/calperumResearch/site2_1_DD0011/lidar/sfm_dense/point_cloud_p1.laz")

# === OUTPUT DIRECTORY ===
output_dir = Path("/media/laura/laura_usb/qgis/calperumResearch/site2_1_DD0011/lidar/thinned_voxel")
output_dir.mkdir(parents=True, exist_ok=True)

# === Load LAZ (or LAS) using laspy ===
sfm_las = laspy.read(str(sfm_path))
points = np.vstack((sfm_las.x, sfm_las.y, sfm_las.z)).T

# === Convert to Open3D PointCloud ===
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points)

# === Apply voxel downsampling (3 cm) ===
voxel_size = 0.03  # 3 cm
pcd_down = pcd.voxel_down_sample(voxel_size)

# === Extract downsampled points ===
down_points = np.asarray(pcd_down.points)

# === Create new LAS with thinned points ===
new_las = laspy.create(point_format=sfm_las.header.point_format, file_version=sfm_las.header.version)

# Only copy XYZ coordinates (intensity, classification, etc. can be handled if needed)
new_las.x = down_points[:, 0]
new_las.y = down_points[:, 1]
new_las.z = down_points[:, 2]

# === Save output LAS ===
output_path = output_dir / (sfm_path.stem + "_thinned_3cm.las")
new_las.write(str(output_path))

print(f"✅ Thinned SfM point cloud saved to:\n   {output_path}")
print(f"📉 Original points: {len(points):,}")
print(f"📈 Thinned points : {len(down_points):,}")

#  Original points: 36,705,937
#  Thinned points : 33,324,572

# %%
