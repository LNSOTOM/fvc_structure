#%%
import subprocess
import json
import re
from pathlib import Path

# === Input Paths ===
lidar_path = Path('/media/laura/Extreme SSD/qgis/calperumResearch/site2_1_DD0011/lidar/cloud2453cf90763afd91.las')
sfm_path = Path('/media/laura/laura_usb/qgis/calperumResearch/site2_1_DD0011/lidar/sfm_dense/point_cloud_p1.laz')

def parse_lasinfo_output(text):
    """Extract min/max X/Y and number of points from lasinfo output."""
    minx = maxx = miny = maxy = num_points = None
    for line in text.splitlines():
        line = line.strip().lower()

        if "min x y z:" in line:
            match = re.findall(r"[-+]?\d*\.\d+|\d+", line)
            if len(match) >= 2:
                minx, miny = float(match[0]), float(match[1])

        elif "max x y z:" in line:
            match = re.findall(r"[-+]?\d*\.\d+|\d+", line)
            if len(match) >= 2:
                maxx, maxy = float(match[0]), float(match[1])

        elif "number of point records:" in line:
            match = re.findall(r"\d+", line)
            if match:
                num_points = int(match[0])

    return minx, maxx, miny, maxy, num_points

def get_point_cloud_stats(path):
    print(f"\n📊 Calculating stats for: {path.name}")

    try:
        # First try pdal info
        info_cmd = ["pdal", "info", str(path)]
        info_result = subprocess.run(info_cmd, capture_output=True, text=True)
        parsed = json.loads(info_result.stdout)

        bbox = parsed.get("bounds", {}).get("bbox", {})
        minx = bbox.get("minx")
        maxx = bbox.get("maxx")
        miny = bbox.get("miny")
        maxy = bbox.get("maxy")
        num_points = parsed.get("num_points", 0)

        if None in [minx, maxx, miny, maxy] or num_points == 0:
            raise ValueError("Missing bounds or zero points in PDAL.")

    except Exception as e:
        print(f"⚠️ PDAL failed: {e}\n🔁 Falling back to lasinfo...")

        # Use lasinfo as fallback
        lasinfo_cmd = ["lasinfo", "-i", str(path)]
        info_result = subprocess.run(lasinfo_cmd, capture_output=True, text=True)
        if info_result.returncode != 0:
            raise RuntimeError(f"❌ lasinfo failed: {info_result.stderr}")

        minx, maxx, miny, maxy, num_points = parse_lasinfo_output(info_result.stdout)

        if None in [minx, maxx, miny, maxy, num_points]:
            raise ValueError("❌ Failed to parse lasinfo output properly.")

    x_extent = maxx - minx
    y_extent = maxy - miny
    area_m2 = x_extent * y_extent
    density = num_points / area_m2
    spacing = (1 / density) ** 0.5

    return {
        "file": str(path),
        "points": num_points,
        "area_m2": area_m2,
        "density_pts_per_m2": density,
        "spacing_m": spacing
    }

# === Compute Stats ===
lidar_stats = get_point_cloud_stats(lidar_path)
sfm_stats = get_point_cloud_stats(sfm_path)

# === Print Summary ===
print("\n📏 Point Cloud Stats:")
print(f"🛰️ LiDAR: {lidar_stats['points']:,} pts | {lidar_stats['density_pts_per_m2']:.2f} pts/m² | spacing ≈ {lidar_stats['spacing_m']:.4f} m")
print(f"📷 SfM  : {sfm_stats['points']:,} pts | {sfm_stats['density_pts_per_m2']:.2f} pts/m² | spacing ≈ {sfm_stats['spacing_m']:.4f} m")

# === Save to JSON ===
save_path = sfm_path.parent / "point_cloud_density_stats.json"
with open(save_path, 'w') as f:
    json.dump({'lidar': lidar_stats, 'sfm': sfm_stats}, f, indent=2)
print(f"\n💾 Stats saved to: {save_path}")


# 📏 Point Cloud Stats:
# 🛰️ LiDAR: 66,646,512 pts | 1177.84 pts/m² | spacing ≈ 0.0291 m
# 📷 SfM  : 36,705,937 pts | 436.51 pts/m² | spacing ≈ 0.0479 m
#%%
