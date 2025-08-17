

import laspy
import numpy as np
from pathlib import Path

# === Load the RGB LAS (local or SfM) ===
rgb_las_path = Path("/media/laura/laura_usb/uas_data/DD0011/point_cloud_p1.laz")
rgb_las = laspy.read(rgb_las_path)

# === Convert scaled coordinates to real (float) XYZ ===
xyz = np.vstack((rgb_las.x, rgb_las.y, rgb_las.z)).T

# === Define LiDAR CRS origin and scale ===
lidar_scale = np.array([0.0001, 0.0001, 0.0001])
lidar_offset = np.min(xyz, axis=0)  # or use LiDAR offset if known

# === Rebuild header with correct scale/offset ===
new_header = laspy.LasHeader(point_format=rgb_las.header.point_format.id,
                             version=rgb_las.header.version)
new_header.scales = lidar_scale
new_header.offsets = lidar_offset

# === Create new LAS with global CRS ===
rgb_fixed = laspy.LasData(new_header)
rgb_fixed.x = xyz[:, 0]
rgb_fixed.y = xyz[:, 1]
rgb_fixed.z = xyz[:, 2]

# === Copy other dimensions ===
for dim in rgb_las.point_format.dimension_names:
    if dim not in ["X", "Y", "Z"]:
        rgb_fixed[dim] = rgb_las[dim]

# === Save ===
rgb_fixed_path = rgb_las_path.with_name("point_cloud_p1_fixed.laz")
rgb_fixed.write(rgb_fixed_path)

print(f"✅ RGB LAS re-written with global CRS: {rgb_fixed_path}")
