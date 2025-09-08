#%%
<<<<<<< HEAD
# improve classification based on HAG
# HAG_DELAUNAY: keep only original points, clamp negatives, stronger ground trim
=======
# === Step 4: Normalize Z (Height Above Ground) using your ground LAS ===
# Uses nearest-neighbour to the provided ground points (fast & robust).
#%%
# === Step 4: Normalize Z (Height Above Ground) using your ground LAS ===
# Uses k-NN (k=3) weighted by 1/d for a smoother ground surface than single NN.

import os
import laspy
import numpy as np
from scipy.spatial import cKDTree

# INPUTS
ground_only_las = "/media/laura/Laura 102/uas_data/site1_1_DD0001/lidar/trim/csf/smrf/cloud1457eb47b07af181_clip_raster_extent_smrf_ground.las"
all_points_las  = "/media/laura/Laura 102/uas_data/site1_1_DD0001/lidar/trim/cloud1457eb47b07af181_clip_raster_extent.las"

# OUTPUT
normalised_output_las = "/media/laura/Laura 102/uas_data/site1_1_DD0001/lidar/trim/csf/smrf/cloud1457eb47b07af181_clip_raster_extent_HAG.las"
os.makedirs(os.path.dirname(normalised_output_las), exist_ok=True)

# Load ALL points (classified), not ground-only
las = laspy.read(all_points_las)
points = np.vstack([las.x, las.y, las.z]).T

# Load ground points from your SMRF output
g = laspy.read(ground_only_las)
ground_points = np.vstack([g.x, g.y, g.z]).T
if ground_points.size == 0:
    raise ValueError("❌ No ground points found in the provided ground LAS!")

# KD-tree of ground XY
tree = cKDTree(ground_points[:, :2])

# k-NN query for smoother ground surface
k = 3
dist, idx = tree.query(points[:, :2], k=k)

# Handle the case where k=1 returns 1D arrays
if k == 1 or np.ndim(idx) == 1:
    idx = idx[:, None]
    dist = dist[:, None]

# Inverse-distance weights (avoid div-by-zero)
dist = np.maximum(dist, 1e-6)
w = 1.0 / dist
w /= w.sum(axis=1, keepdims=True)

# Weighted ground Z at each XY
gz_neighbors = ground_points[idx, 2]
ground_z_at_xy = np.sum(gz_neighbors * w, axis=1)

# Height Above Ground
z_normalized = (points[:, 2] - ground_z_at_xy).astype(np.float32)

# Clamp tiny negatives from numerical noise
z_normalized = np.maximum(z_normalized, 0.0)

# Add normalized Z as extra dimension (your name)
if "Z_norm" not in las.point_format.dimension_names:
    las.add_extra_dim(laspy.ExtraBytesParams(name="Z_norm", type=np.float32))
las.Z_norm = z_normalized

las.write(normalised_output_las)
print(f"✅ Normalized LAS with Z_norm (HAG) saved to: {normalised_output_las}")
print(f"   HAG stats (m): min={z_normalized.min():.3f}, median={np.median(z_normalized):.3f}, max={z_normalized.max():.3f}")



#%%
###improving previous
# z_normalise_hag.py
import os
>>>>>>> 25d0e1f (phase1_data processing)
from pathlib import Path
import json, subprocess, tempfile, shutil, sys

ground_only_las = "/media/laura/laura_usb/qgis/calperumResearch/site1_1_DD0001/lidar/clipped/csf/triple_AusGeoid_tiles_multispectral.26-smrf_ground.las"
all_points_las  = "/media/laura/laura_usb/qgis/calperumResearch/site1_1_DD0001/lidar/clipped/20220519_SASMDD0001_triple_AusGeoid_tiles_multispectral.26.las"
normalised_output_las = "/media/laura/laura_usb/qgis/calperumResearch/site1_1_DD0001/lidar/clipped/z_norm/triple_AusGeoid_tiles_multispectral.26_HAG_delaunay.las"

if shutil.which("pdal") is None:
    raise RuntimeError("PDAL CLI not found in PATH.")
Path(normalised_output_las).parent.mkdir(parents=True, exist_ok=True)

NEAR_GROUND_CUTOFF = 0.05  # <-- CHANGED: raise from 0.02 to 0.05 m (try 0.07–0.10 if needed)
USE_OUTLIER_FILTER = True  # <-- ADD: removes isolated specks after thresholding

pipeline = [
    {"type": "readers.las", "tag": "full",       "filename": all_points_las},
    {"type": "readers.las", "tag": "ground_raw", "filename": ground_only_las},

    {"type": "filters.assign", "inputs": ["full"],       "assignment": "UserData[:]=0", "tag": "full_ud"},
    {"type": "filters.assign", "inputs": ["ground_raw"], "assignment": "UserData[:]=1", "tag": "ground_ud"},
    {"type": "filters.assign", "inputs": ["ground_ud"],  "assignment": "Classification[:]=2", "tag": "ground_cls"},

    # Merge so HAG sees ground + original
    {"type": "filters.merge", "inputs": ["full_ud","ground_cls"], "tag": "merged"},

    # HAG via Delaunay (use hag_nn if your PDAL lacks this stage)
    {"type": "filters.hag_delaunay", "inputs": ["merged"], "count": 10, "allow_extrapolation": True},
    # Copy HAG to Z_norm (keep Z)
    {"type": "filters.ferry", "dimensions": "HeightAboveGround=>Z_norm"},
    # Clamp tiny negatives
    {"type": "filters.assign", "assignment": "HeightAboveGround[:]=0", "where": "HeightAboveGround < 0"},
    {"type": "filters.assign", "assignment": "Z_norm[:]=0",            "where": "Z_norm < 0"},

    # Keep ONLY original points (drop the extra ground copy)
    {"type": "filters.range", "limits": "UserData[0:0]"},

    # --- CHANGED: remove near-ground by HAG cutoff ---
    {"type": "filters.range", "limits": f"HeightAboveGround[{NEAR_GROUND_CUTOFF}:]"},
]

if USE_OUTLIER_FILTER:
    # --- ADD: light statistical outlier removal to clean sparse low specks
    pipeline.append({"type": "filters.outlier", "method": "statistical", "mean_k": 12, "multiplier": 2.5})

pipeline.append({
    "type": "writers.las",
    "filename": normalised_output_las,
    "extra_dims": "HeightAboveGround=float32,Z_norm=float32"
})

def run(pobj):
    with tempfile.TemporaryDirectory() as td:
        j = Path(td)/"pipeline.json"
        j.write_text(json.dumps(pobj, indent=2))
        v = subprocess.run(["pdal","pipeline",str(j),"--validate"], capture_output=True, text=True)
        if v.returncode != 0:
            print("❌ validate STDOUT\n", v.stdout, "\n❌ validate STDERR\n", v.stderr, file=sys.stderr)
            raise SystemExit(1)
        r = subprocess.run(["pdal","pipeline",str(j)], capture_output=True, text=True)
        if r.returncode != 0:
            print("❌ run STDOUT\n", r.stdout, "\n❌ run STDERR\n", r.stderr, file=sys.stderr)
            raise SystemExit(r.returncode)
        print("✅ PDAL pipeline ran OK.")

run(pipeline)
print("✅ Wrote:", normalised_output_las)


#%%
# improve classification based on HAG and drop the extra attribute HAG --> GOOD
# HAG_DELAUNAY: keep only original points, clamp negatives on Z_norm, trim near-ground
from pathlib import Path
import json, subprocess, tempfile, shutil, sys

ground_only_las = "/media/laura/laura_usb/qgis/calperumResearch/site1_1_DD0001/lidar/clipped/csf/triple_AusGeoid_tiles_multispectral.26-smrf_ground.las"
# ground_only_las = "/media/laura/laura_usb/qgis/calperumResearch/site1_1_DD0001/lidar/clipped/z_norm/20220519_SASMDD0001_triple_AusGeoid_tiles_multispectral.26_lasground_large_steps.las"
all_points_las  = "/media/laura/laura_usb/qgis/calperumResearch/site1_1_DD0001/lidar/clipped/20220519_SASMDD0001_triple_AusGeoid_tiles_multispectral.26.las"
normalised_output_las = "/media/laura/laura_usb/qgis/calperumResearch/site1_1_DD0001/lidar/clipped/z_norm/triple_AusGeoid_tiles_multispectral.26_HAG_delaunay.las"

if shutil.which("pdal") is None:
    raise RuntimeError("PDAL CLI not found in PATH.")
Path(normalised_output_las).parent.mkdir(parents=True, exist_ok=True)

NEAR_GROUND_CUTOFF = 0.05
USE_OUTLIER_FILTER = True

pipeline = [
    {"type": "readers.las", "tag": "full",       "filename": all_points_las},
    {"type": "readers.las", "tag": "ground_raw", "filename": ground_only_las},

    {"type": "filters.assign", "inputs": ["full"],       "assignment": "UserData[:]=0", "tag": "full_ud"},
    {"type": "filters.assign", "inputs": ["ground_raw"], "assignment": "UserData[:]=1", "tag": "ground_ud"},
    {"type": "filters.assign", "inputs": ["ground_ud"],  "assignment": "Classification[:]=2", "tag": "ground_cls"},

    {"type": "filters.merge", "inputs": ["full_ud","ground_cls"], "tag": "merged"},

    {"type": "filters.hag_delaunay", "inputs": ["merged"], "count": 10, "allow_extrapolation": True, "tag": "hag"},

    {"type": "filters.ferry", "inputs": ["hag"], "dimensions": "HeightAboveGround=>Z_norm", "tag": "hag_f"},

    # ✅ Clamp negatives on Z_norm only
    {"type": "filters.assign", "inputs": ["hag_f"], "assignment": "Z_norm[:]=0", "where": "Z_norm < 0", "tag":"clamped"},

    {"type": "filters.range", "inputs": ["clamped"], "limits": "UserData[0:0]", "tag":"orig_only"},

    {"type": "filters.range", "inputs": ["orig_only"], "limits": f"Z_norm[{NEAR_GROUND_CUTOFF}:]", "tag":"trimmed"},
]

if USE_OUTLIER_FILTER:
    pipeline.append({"type": "filters.outlier", "inputs": ["trimmed"], "method": "statistical", "mean_k": 12, "multiplier": 2.5, "tag": "clean"})
    final_tag = "clean"
else:
    final_tag = "trimmed"

pipeline.append({
    "type": "writers.las",
    "inputs": [final_tag],
    "filename": normalised_output_las,
    "extra_dims": "Z_norm=float32"
})

def run(pobj):
    with tempfile.TemporaryDirectory() as td:
        j = Path(td)/"pipeline.json"
        j.write_text(json.dumps({"pipeline": pobj}, indent=2))
        v = subprocess.run(["pdal","pipeline",str(j),"--validate"], capture_output=True, text=True)
        if v.returncode != 0:
            print("❌ validate STDOUT\n", v.stdout, "\n❌ validate STDERR\n", v.stderr, file=sys.stderr)
            raise SystemExit(1)
        r = subprocess.run(["pdal","pipeline",str(j)], capture_output=True, text=True)
        if r.returncode != 0:
            print("❌ run STDOUT\n", r.stdout, "\n❌ run STDERR\n", r.stderr, file=sys.stderr)
            raise SystemExit(r.returncode)
        print("✅ PDAL pipeline ran OK.")

run(pipeline)
print("✅ Wrote:", normalised_output_las)

# %%
