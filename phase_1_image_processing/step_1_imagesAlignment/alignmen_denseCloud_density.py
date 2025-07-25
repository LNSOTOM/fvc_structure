#%%
import pdal
import json
from pathlib import Path
import subprocess
import time
import re

# === Step 1: Define Input Paths ===
lidar_path = Path('/media/laura/Extreme SSD/qgis/calperumResearch/site2_1_DD0011/lidar/cloud2453cf90763afd91.las')  # LiDAR
sfm_path = Path('/media/laura/laura_usb/qgis/calperumResearch/site2_1_DD0011/lidar/sfm_dense/point_cloud_p1.laz')  # SfM

output_dir = Path('/media/laura/laura_usb/qgis/calperumResearch/site2_1_DD0011/lidar/sfm_dense_aligned_p1')
output_dir.mkdir(parents=True, exist_ok=True)


# === Step 2: Define Output Paths ===
aligned_las = output_dir / f"{sfm_path.stem}_aligned.las"
metadata_output = output_dir / "icp_metadata.json"
pipeline_path = output_dir / "icp_pipeline.json"

# === Step 3: Compute Spacing and Density ===
def compute_pointcloud_stats(path):
    try:
        info = json.loads(subprocess.run(["pdal", "info", str(path)], capture_output=True, text=True).stdout)
        bbox = info.get("bounds", {}).get("bbox", {})
        minx, maxx = bbox["minx"], bbox["maxx"]
        miny, maxy = bbox["miny"], bbox["maxy"]
        num_points = info.get("num_points", 0)
    except:
        print(f"⚠️ PDAL failed for {path.name}. Falling back to lasinfo...")
        result = subprocess.run(["lasinfo", "-i", str(path)], capture_output=True, text=True)
        lines = result.stdout.splitlines()
        minx = maxx = miny = maxy = num_points = None
        for line in lines:
            line = line.lower()
            if "min x y z:" in line:
                parts = re.findall(r"[-+]?\d*\.\d+|\d+", line)
                minx, miny = float(parts[0]), float(parts[1])
            elif "max x y z:" in line:
                parts = re.findall(r"[-+]?\d*\.\d+|\d+", line)
                maxx, maxy = float(parts[0]), float(parts[1])
            elif "number of point records:" in line:
                parts = re.findall(r"\d+", line)
                if parts:
                    num_points = int(parts[0])

    if None in [minx, maxx, miny, maxy, num_points] or num_points == 0:
        raise RuntimeError(f"❌ Could not compute stats from {path.name}")

    area = (maxx - minx) * (maxy - miny)
    density = num_points / area
    spacing = (1 / density) ** 0.5
    return {
        "path": path,
        "points": num_points,
        "area_m2": area,
        "density": round(density, 2),
        "spacing": round(spacing, 4)
    }

# Get both point cloud stats
lidar_stats = compute_pointcloud_stats(lidar_path)
sfm_stats = compute_pointcloud_stats(sfm_path)
lidar_spacing = lidar_stats['spacing']

print("\n📏 Point Cloud Stats:")
print(f"🛰️ LiDAR: {lidar_stats['points']:,} pts | {lidar_stats['density']} pts/m² | spacing ≈ {lidar_spacing:.4f} m")
print(f"📷 SfM  : {sfm_stats['points']:,} pts | {sfm_stats['density']} pts/m² | spacing ≈ {sfm_stats['spacing']:.4f} m")

#%%
# === Step 4: Define PDAL Pipeline
pipeline = [
    { "type": "readers.las", "filename": str(sfm_path) },
    { "type": "filters.voxeldownsize", "cell": lidar_spacing, "mode": "center" },  # optional

    { "type": "readers.las", "filename": str(lidar_path) },
    { "type": "filters.voxeldownsize", "cell": lidar_spacing, "mode": "center" },

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
        "forward": "all",
        "minor_version": 4,
        "dataformat_id": 6
    }
]

# Save pipeline
with open(pipeline_path, "w") as f:
    json.dump(pipeline, f, indent=2)

# === Step 5: Run Pipeline
print("📡 Running ICP alignment with PDAL...")
start = time.time()
result = subprocess.run(
    ["pdal", "pipeline", str(pipeline_path), "--metadata", str(metadata_output)],
    capture_output=True, text=True
)
elapsed = time.time() - start

if result.returncode != 0:
    print("❌ PDAL failed")
    print(result.stderr)
    raise subprocess.CalledProcessError(result.returncode, result.args)

print(f"✅ ICP completed in {elapsed:.2f} s")
print(f"➡️ Output LAS: {aligned_las}")
print(f"🗂️ Metadata: {metadata_output}")

# === Step 6: Parse ICP Transform
with open(metadata_output) as f:
    transform = json.load(f)["stages"]["filters.icp"]["transform"]
matrix = list(map(float, transform.split()))
print("\n📐 ICP Translation Offset:")
print(f"  X: {matrix[3]:.3f} m")
print(f"  Y: {matrix[7]:.3f} m")
print(f"  Z: {matrix[11]:.3f} m")

#%%
# === Step 7: Compress with LAStools
print("🛠️ Compressing with laszip...")
cmd = ["laszip", "-i", str(aligned_las), "-o", str(compressed_laz)]
comp = subprocess.run(cmd, capture_output=True, text=True)

if comp.returncode != 0:
    print("❌ Compression failed:", comp.stderr)
else:
    print("✅ Compressed LAZ:", compressed_laz)
#%%
