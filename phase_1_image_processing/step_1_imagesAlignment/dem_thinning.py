
#%%
# Step 2.1: DEM from thinned LiDAR point cloud
import laspy
import numpy as np
import rasterio
from rasterio.transform import from_origin
from scipy.stats import binned_statistic_2d
from pathlib import Path

# Input and output paths
las_path = Path("/media/laura/laura_usb/qgis/calperumResearch/site2_1_DD0011/lidar/thinned_voxel/cloud2453cf90763afd91_thinned_3cm.las")
output_path = las_path.with_name("dem_3cm_from_lidar.tif")

# Load LAS
las = laspy.read(str(las_path))
x, y, z = las.x, las.y, las.z

# Define grid parameters
voxel_size = 0.03  # 3 cm
x_min, y_max = np.min(x), np.max(y)
x_max, y_min = np.max(x), np.min(y)

ncols = int(np.ceil((x_max - x_min) / voxel_size))
nrows = int(np.ceil((y_max - y_min) / voxel_size))

# Create DEM as minimum Z per XY grid cell
stat, x_edge, y_edge, _ = binned_statistic_2d(
    x, y, z, statistic='min', bins=[ncols, nrows]
)

# Flip Y axis for rasterio
dem = np.flipud(stat.T)
dem = np.nan_to_num(dem, nan=-9999)

# Define transform
transform = from_origin(x_min, y_max, voxel_size, voxel_size)

# Save as GeoTIFF
with rasterio.open(
    output_path, "w",
    driver="GTiff",
    height=dem.shape[0],
    width=dem.shape[1],
    count=1,
    dtype='float32',
    crs="EPSG:7856",  # Adjust to match your data's CRS
    transform=transform,
    nodata=-9999
) as dst:
    dst.write(dem.astype('float32'), 1)

print(f"✅ DEM saved to: {output_path}")

# %%
# Step 2.2: DSM from thinned SfM dense cloud (RGB or multispectral)
import numpy as np
import laspy
import rasterio
from pathlib import Path
from rasterio.transform import from_origin
from scipy.stats import binned_statistic_2d

# === Input and output paths ===
las_path = Path("/media/laura/laura_usb/qgis/calperumResearch/site2_1_DD0011/lidar/thinned_voxel/point_cloud_p1_thinned_3cm.las")
output_path = las_path.with_name("dsm_3cm_from_sfm.tif")

# === Path to orthomosaic (to get CRS) ===
ortho_path = Path("/home/laura/Documents/uas_data/Calperum/site4_DD0011/othomosaic/20220517_SASMDD0011_p1_ortho_01.tif")

# === Load CRS from RGB orthomosaic ===
with rasterio.open(ortho_path) as src:
    crs = src.crs

# === Load LAS ===
las = laspy.read(str(las_path))
x, y, z = las.x, las.y, las.z

# === Define grid parameters ===
voxel_size = 0.03  # 3 cm resolution
x_min, y_max = np.min(x), np.max(y)
x_max, y_min = np.max(x), np.min(y)

ncols = int(np.ceil((x_max - x_min) / voxel_size))
nrows = int(np.ceil((y_max - y_min) / voxel_size))

# === Create DSM (max Z per XY cell) ===
stat, x_edge, y_edge, _ = binned_statistic_2d(
    x, y, z, statistic='max', bins=[ncols, nrows]
)

# === Flip and clean up array ===
dsm = np.flipud(stat.T)
dsm = np.nan_to_num(dsm, nan=-9999)

# === Define affine transform ===
transform = from_origin(x_min, y_max, voxel_size, voxel_size)

# === Save as GeoTIFF ===
with rasterio.open(
    output_path, "w",
    driver="GTiff",
    height=dsm.shape[0],
    width=dsm.shape[1],
    count=1,
    dtype='float32',
    crs=crs,
    transform=transform,
    nodata=-9999
) as dst:
    dst.write(dsm.astype('float32'), 1)

print(f"✅ DSM saved to: {output_path}")

# %%
