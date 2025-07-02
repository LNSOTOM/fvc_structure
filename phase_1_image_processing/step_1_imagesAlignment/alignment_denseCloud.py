#%%
import pdal
import json
from pathlib import Path
import subprocess
import time

# === Step 1: Define Input Paths ===
lidar_path = Path('/media/laura/Extreme SSD/qgis/calperumResearch/site2_1_DD0011/lidar/cloud2453cf90763afd91.las')  # LiDAR (reference)
sfm_path = Path('/media/laura/laura_usb/qgis/calperumResearch/site2_1_DD0011/lidar/sfm_dense/point_cloud_p1.laz')  # SfM (to be aligned)

# === Step 2: Define Output Paths ===
output_dir = sfm_path.parent / "sfm_dense_aligned"
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
        "max_iter": 100,
        "max_similar": 3,
        "mse_abs": 1e-6,
        "rt": 0.99999,
        "tt": 1e-6
    },
    {
        "type": "writers.las",
        "filename": str(aligned_output),
        "compression": "laszip",
        "forward": "all"
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
print(f"✅ SfM dense point cloud aligned to LiDAR reference:")
print(f"   ➡️ Output aligned LAZ: {aligned_output}")
print(f"   🗂️ ICP metadata file: {metadata_output}")

# === Step 7: Parse ICP Transform from Metadata ===
with open(metadata_output) as f:
    metadata = json.load(f)

transform_str = metadata['stages']['filters.icp']['transform']
matrix = list(map(float, transform_str.split()))
offset_x, offset_y, offset_z = matrix[3], matrix[7], matrix[11]

print("\n📐 ICP Translation Offset:")
print(f"  X: {offset_x:.3f} m")
print(f"  Y: {offset_y:.3f} m")
print(f"  Z: {offset_z:.3f} m")
