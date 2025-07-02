#%%
##OPTION 1: PDAL
import pdal
import json
from pathlib import Path
import subprocess
import time

# === Step 1: Define Input Paths ===
lidar_path = Path('/media/laura/Extreme SSD/qgis/calperumResearch/site2_1_DD0011/lidar/cloud2453cf90763afd91.las')  # LiDAR (reference)
sfm_path = Path('/media/laura/laura_usb/qgis/calperumResearch/site2_1_DD0011/lidar/sfm_dense/point_cloud_p1.laz')  # SfM (to be aligned)
# sfm_path = Path('/media/laura/laura_usb/qgis/calperumResearch/site2_1_DD0011/lidar/sfm_dense/point_cloud_micasense.laz')  # SfM (to be aligned)


# === Step 2: Define Output Paths ===
output_dir = sfm_path.parent / "sfm_dense_aligned_p1"
# output_dir = sfm_path.parent / "sfm_dense_aligned_mica"
output_dir.mkdir(parents=True, exist_ok=True)

aligned_output = output_dir / f"{sfm_path.stem}_aligned.laz"
metadata_output = output_dir / "icp_metadata.json"
pipeline_path = output_dir / "icp_pipeline.json"

# === Step 3: Define PDAL Pipeline ===
pipeline = [
    { "type": "readers.las", "filename": str(sfm_path) },
    { "type": "filters.voxeldownsize", "cell": 0.05, "mode": "center" },
    { "type": "readers.las", "filename": str(lidar_path) },
    { "type": "filters.voxeldownsize", "cell": 0.05, "mode": "center" },
    {
        "type": "filters.icp",
        "max_iter": 20,
        "max_similar": 3,
        "mse_abs": 1e-5,
        "rt": 0.99999,
        "tt": 1e-5
    },
    {
        "type": "writers.las",
        "filename": str(aligned_output),
        "compression": "laszip", #ensures efficient .laz output format
        "extra_dims": "all", #ensures extra attributes like normals, colour...are preserved
        "forward": "all" #ensures scale, offset, VLRs, version, data format are inherited from reference
    }
]

# === Step 4: Save Pipeline to JSON ===
with open(pipeline_path, "w") as f:
    json.dump(pipeline, f, indent=2)

# === Step 5: Run PDAL Pipeline and Capture Time ===
print("📡 Running ICP alignment with PDAL...")
start_time = time.time()

result = subprocess.run(
    ["pdal", "pipeline", str(pipeline_path), "--metadata", str(metadata_output)],
    capture_output=True,
    text=True
)

end_time = time.time()
elapsed_time = end_time - start_time

# === Step 6: Handle Results ===
if result.returncode != 0:
    print("❌ PDAL pipeline failed!")
    print("STDOUT:\n", result.stdout)
    print("STDERR:\n", result.stderr)
    raise subprocess.CalledProcessError(result.returncode, result.args)
else:
    print("✅ PDAL pipeline completed successfully.")

print(f"✅ ICP completed in {elapsed_time:.2f} seconds.")
print(f"➡️ Aligned output: {aligned_output}")
print(f"🗂️ ICP metadata saved to: {metadata_output}")

# === Step 7: Parse ICP Transform from Metadata ===
with open(metadata_output) as f:
    metadata = json.load(f)

transform_str = metadata["stages"]["filters.icp"]["transform"]
matrix = list(map(float, transform_str.split()))
offset_x, offset_y, offset_z = matrix[3], matrix[7], matrix[11]

print("\n📐 ICP Translation Offset:")
print(f"  X: {offset_x:.3f} m")
print(f"  Y: {offset_y:.3f} m")
print(f"  Z: {offset_z:.3f} m")


##################################################
#%%
##OPTION 2: PDAL + LAStools
import pdal
import json
from pathlib import Path
import subprocess
import time

# === Step 1: Define Input Paths ===
lidar_path = Path('/media/laura/Extreme SSD/qgis/calperumResearch/site2_1_DD0011/lidar/cloud2453cf90763afd91.las')  # Reference LiDAR
sfm_path = Path('/media/laura/laura_usb/qgis/calperumResearch/site2_1_DD0011/lidar/sfm_dense/point_cloud_p1.laz')  # SfM to align
# sfm_path = Path('/media/laura/laura_usb/qgis/calperumResearch/site2_1_DD0011/lidar/sfm_dense/point_cloud_micasense.laz')  # SfM (to be aligned)


# === Step 2: Define Output Paths ===
output_dir = sfm_path.parent / "sfm_dense_aligned_p1"
# output_dir = sfm_path.parent / "sfm_dense_aligned_mica"
output_dir.mkdir(parents=True, exist_ok=True)

aligned_las = output_dir / f"{sfm_path.stem}_aligned.las"
compressed_laz = aligned_las.with_suffix('.laz')
metadata_output = output_dir / "icp_metadata.json"
pipeline_path = output_dir / "icp_pipeline.json"

# === Step 3: Define PDAL Pipeline ===
pipeline = [
    { "type": "readers.las", "filename": str(sfm_path) },
    { "type": "filters.voxeldownsize", "cell": 0.05, "mode": "center" },
    { "type": "readers.las", "filename": str(lidar_path) },
    { "type": "filters.voxeldownsize", "cell": 0.05, "mode": "center" },
    {
        "type": "filters.icp",
        "max_iter": 20,
        "max_similar": 3,
        "mse_abs": 1e-5,
        "rt": 0.99999,
        "tt": 1e-5
    },
    {
        "type": "writers.las",
        "filename": str(aligned_las),
        "extra_dims": "all",
        "forward": "all"
    }
]

# === Step 4: Save Pipeline to JSON ===
with open(pipeline_path, "w") as f:
    json.dump(pipeline, f, indent=2)

# === Step 5: Run PDAL Pipeline ===
print("📡 Running ICP alignment with PDAL...")
start_time = time.time()

result = subprocess.run(
    ["pdal", "pipeline", str(pipeline_path), "--metadata", str(metadata_output)],
    capture_output=True,
    text=True
)

end_time = time.time()
elapsed_time = end_time - start_time

if result.returncode != 0:
    print("❌ PDAL pipeline failed!")
    print("STDOUT:\n", result.stdout)
    print("STDERR:\n", result.stderr)
    raise subprocess.CalledProcessError(result.returncode, result.args)
else:
    print("✅ PDAL ICP completed in {:.2f} seconds.".format(elapsed_time))
    print(f"➡️ Aligned LAS: {aligned_las}")
    print(f"🗂️ Metadata: {metadata_output}")

# === Step 6: Parse ICP Transform from Metadata ===
with open(metadata_output) as f:
    metadata = json.load(f)

transform_str = metadata["stages"]["filters.icp"]["transform"]
matrix = list(map(float, transform_str.split()))
offset_x, offset_y, offset_z = matrix[3], matrix[7], matrix[11]

print("\n📐 ICP Translation Offset:")
print(f"  X: {offset_x:.3f} m")
print(f"  Y: {offset_y:.3f} m")
print(f"  Z: {offset_z:.3f} m")

# === Step 7: Compress using LAStools ===
print("🛠️ Compressing with LAStools (laszip)...")
lastools_cmd = [
    "laszip",
    "-i", str(aligned_las),
    "-o", str(compressed_laz)
]

compress_result = subprocess.run(lastools_cmd, capture_output=True, text=True)

if compress_result.returncode != 0:
    print("❌ LAStools compression failed!")
    print("STDOUT:\n", compress_result.stdout)
    print("STDERR:\n", compress_result.stderr)
else:
    print("✅ LAStools compression complete.")
    print(f"➡️ Compressed LAZ: {compressed_laz}")
    # Optionally delete the .las
    # aligned_las.unlink()


#%%
## Verify METADATA with LAStools in bash (lasinfo - Extracts metadata information and additional information from a LAZ/LAS file. )
# lasinfo -i yourfile.laz

import laspy

las = laspy.read("yourfile.laz")
print(las.header)
