
#%%
##OPTION 1
from osgeo import gdal
import json
from pathlib import Path

# === INPUT FILES ===
orthomosaic_path = Path("/media/laura/Extreme SSD/qgis/calperumResearch/site2_1_DD0011/orthomosaic/20220517_SASMDD0011_p1_ortho_01.tif")
metadata_path = Path("/media/laura/laura_usb/qgis/calperumResearch/site2_1_DD0011/lidar/sfm_dense/sfm_dense_aligned/icp_metadata.json")

# === CUSTOM OUTPUT DIRECTORY ===
aligned_dir = Path("/media/laura/laura_usb/qgis/calperumResearch/site2_1_DD0011/lidar/orthomosaic_aligned")
aligned_dir.mkdir(parents=True, exist_ok=True)

# === OUTPUT FILE ===
aligned_orthomosaic_path = aligned_dir / (orthomosaic_path.stem + "_aligned.tif")

# === Extract ICP Transform Offsets ===
with open(metadata_path) as f:
    metadata = json.load(f)

transform_str = metadata["stages"]["filters.icp"]["transform"]
matrix = list(map(float, transform_str.split()))

offset_x = matrix[3]  # ICP X shift
offset_y = matrix[7]  # ICP Y shift
offset_z = matrix[11] # ICP Z shift

print(f"📐 Applying offset: X = {offset_x:.3f} m, Y = {offset_y:.3f} m, Z = {offset_z:.3f} m")

#%%
# === Open original orthomosaic ===
src_ds = gdal.Open(str(orthomosaic_path))
gt = list(src_ds.GetGeoTransform())

# === Modify top-left corner coordinates ===
gt[0] += offset_x  # top-left X
gt[3] += offset_y  # top-left Y

# === Create output with modified geotransform ===
driver = gdal.GetDriverByName("GTiff")
dst_ds = driver.CreateCopy(str(aligned_orthomosaic_path), src_ds, 0)

dst_ds.SetGeoTransform(gt)

# === Cleanup ===
dst_ds.FlushCache()
del dst_ds
del src_ds

print(f"✅ Aligned orthomosaic saved to:\n{aligned_orthomosaic_path}")

# %%
##OPTION 2 - full affine transformation extracted from the ICP metadata using GDAL Warp
#%%
from osgeo import gdal
import json
from pathlib import Path

# === INPUT FILES ===
orthomosaic_path = Path("/media/laura/Extreme SSD/qgis/calperumResearch/site2_1_DD0011/orthomosaic/20220517_SASMDD0011_p1_ortho_01.tif")
metadata_path = Path("/media/laura/laura_usb/qgis/calperumResearch/site2_1_DD0011/lidar/sfm_dense/sfm_dense_aligned/icp_metadata.json")

# === CUSTOM OUTPUT DIRECTORY ===
aligned_dir = Path("/media/laura/laura_usb/qgis/calperumResearch/site2_1_DD0011/lidar/orthomosaic_aligned")
aligned_dir.mkdir(parents=True, exist_ok=True)

# === OUTPUT FILE ===
aligned_orthomosaic_path = aligned_dir / (orthomosaic_path.stem + "_affine_aligned.tif")

# === Extract ICP Transform Matrix ===
with open(metadata_path) as f:
    metadata = json.load(f)

transform_str = metadata["stages"]["filters.icp"]["transform"]
matrix = list(map(float, transform_str.split()))

# Extract 2D affine matrix (from top-left 2x2 + translation)
a = matrix[0]   # R11
b = matrix[1]   # R12
d = matrix[4]   # R21
e = matrix[5]   # R22
xoff = matrix[3]  # T1 (X translation)
yoff = matrix[7]  # T2 (Y translation)

# GDAL expects a 3x3 warp matrix (flattened row-major)
warp_matrix = [a, b, 0, d, e, 0, xoff, yoff, 1]

print("📐 Applying full affine ICP transform:")
print(f"Matrix:\n [{a:.6f}, {b:.6f}, {xoff:.3f}]\n [{d:.6f}, {e:.6f}, {yoff:.3f}]")

# === Apply GDAL warp with full affine matrix ===
gdal.Warp(
    str(aligned_orthomosaic_path),
    str(orthomosaic_path),
    transformerOptions=["SRC_METHOD=NO_GEOTRANSFORM"],
    warpOptions=["INIT_DEST=NO_DATA"],
    options=["NUM_THREADS=ALL_CPUS"],
    transformer={'Geotransform': warp_matrix}
)

print(f"✅ Orthomosaic fully aligned using full affine transform:\n{aligned_orthomosaic_path}")
#%%
