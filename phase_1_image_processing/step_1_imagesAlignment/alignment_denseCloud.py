

import pdal
import json
from pathlib import Path
import subprocess
import time

# Step 1: ICP Registration Using PDAL
#INPUTS
lidar_path = Path('/media/laura/Extreme SSD/qgis/calperumResearch/site2_1_DD0011/lidar/cloud2453cf90763afd91.las') # LiDAR (reference)
sfm_path = Path('/media/laura/laura_usb/qgis/calperumResearch/site2_1_DD0011/lidar/sfm_dense/point_cloud_p1.laz') # SfM (to be aligned)

#OUTPUTS
output_dir = sfm_path.parent / "sfm_dense_aligned"
output_dir.mkdir(parents=True, exist_ok=True)

# === Aligned output and metadata path ===
aligned_output = output_dir / f"{sfm_path.stem}_aligned.las"
metadata_output = output_dir / "icp_metadata.json"
pipeline_path = output_dir / "icp_pipeline.json"

# === Step 2: Build PDAL ICP Pipeline (Array-based format) ===
# PDAL PIPELINE: SfM aligned to LiDAR with voxelgrid and ICP ===
pipeline = [
    {
        "type": "filters.icp",
        "max_iter": 20,
        "mse_abs": 1e-5,
        "max_similar": 3,
        "rt": 0.9999,
        "tt": 1e-5,
        "inputs": [
            {
                "type": "filters.voxeldownsize",
                "cell": 0.05,
                "mode": "center",
                "inputs": [
                    {
                        "type": "readers.las",
                        "filename": str(sfm_path)  # SfM (to be aligned)
                    }
                ]
            },
            {
                "type": "filters.voxeldownsize",
                "cell": 0.05,
                "mode": "center",
                "inputs": [
                    {
                        "type": "readers.las",
                        "filename": str(lidar_path)  # LiDAR (reference)
                    }
                ]
            }
        ]
    },
    {
        "type": "writers.las",
        "filename": str(aligned_output)
    }
]


# === Step 3: Write pipeline to file ===
with open(pipeline_path, "w") as f:
    json.dump(pipeline, f, indent=2)

# === Step 4: Run PDAL Pipeline with Metadata Output ===
print("📡 Running ICP alignment with PDAL...")
start_time = time.time()

subprocess.run(["pdal", "pipeline", str(pipeline_path), "--metadata", str(metadata_output)], check=True)
end_time = time.time()
elapsed_time = end_time - start_time

print(f"✅ ICP completed in {elapsed_time:.2f} seconds.")
print(f"✅ SfM dense point cloud aligned to LiDAR reference:")
print(f"   ➡️ Output aligned LAS: {aligned_output}")
print(f"   🗂️ ICP metadata file: {metadata_output}")


# === Step 5: Parse Offset from Metadata, ICP TRANSFORM ===
with open(metadata_output) as f:
    metadata = json.load(f)

transform_str = metadata['stages']['filters.icp']['transform']
matrix = list(map(float, transform_str.split()))
offset_x, offset_y, offset_z = matrix[3], matrix[7], matrix[11]

print("\n📐 ICP Translation Offset:")
print(f"  X: {offset_x:.3f} m")
print(f"  Y: {offset_y:.3f} m")
print(f"  Z: {offset_z:.3f} m")