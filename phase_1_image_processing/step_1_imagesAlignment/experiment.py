
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
