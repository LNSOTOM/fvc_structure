

from osgeo import gdal
import json
from pathlib import Path

# === INPUT FILES ===
orthomosaic_path = Path("/path/to/orthomosaic_original.tif")
metadata_path = Path("/media/laura/laura_usb/qgis/calperumResearch/site2_1_DD0011/lidar/sfm_dense_aligned/icp_metadata.json")

# === OUTPUT FILE ===
aligned_orthomosaic_path = orthomosaic_path.with_name(orthomosaic_path.stem + "_aligned.tif")

# === Extract ICP Transform Offsets ===
with open(metadata_path) as f:
    metadata = json.load(f)

transform_str = metadata["stages"]["filters.icp"]["transform"]
matrix = list(map(float, transform_str.split()))

offset_x = matrix[3]  # ICP X shift
offset_y = matrix[7]  # ICP Y shift

print(f"📐 Applying offset: X = {offset_x:.3f} m, Y = {offset_y:.3f} m")

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
