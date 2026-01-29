
#%%
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
from pathlib import Path
import numpy as np
import laspy
from scipy.spatial import cKDTree

# ==== INPUTS ====
ground_only_las = Path("/media/laura/Laura 102/uas_data/site1_1_DD0001/lidar/trim/csf/smrf/cloud1457eb47b07af181_clip_raster_extent_smrf_ground.las")
all_points_las  = Path("/media/laura/Laura 102/uas_data/site1_1_DD0001/lidar/trim/cloud1457eb47b07af181_clip_raster_extent.las")

# ==== OUTPUT ====
normalised_output_las = Path("/media/laura/Laura 102/uas_data/site1_1_DD0001/lidar/trim/csf/smrf_v1/cloud1457eb47b07af181_clip_raster_extent_HAG.las")
os.makedirs(normalised_output_las.parent, exist_ok=True)

# ---- Load ALL points (classified), not ground-only ----
las_all = laspy.read(all_points_las)
pts_xyz = np.column_stack([las_all.x, las_all.y, las_all.z])

# ---- Load ground points from SMRF output ----
las_g = laspy.read(ground_only_las)
ground_xyz = np.column_stack([las_g.x, las_g.y, las_g.z])

if ground_xyz.size == 0:
    raise ValueError("❌ No ground points found in the provided ground LAS!")

# ---- Basic header/CRS sanity (units/scale/offset) ----
h_all, h_g = las_all.header, las_g.header
if not np.allclose(h_all.scales, h_g.scales) or not np.allclose(h_all.offsets, h_g.offsets):
    print("⚠️  Warning: header scales/offsets differ between files. Ensure both are in same CRS/units.")

# ---- Build KDTree on ground XY ----
tree = cKDTree(ground_xyz[:, :2])

# k-NN parameters
k = 3
dist, idx = tree.query(pts_xyz[:, :2], k=k, workers=-1)  # use multiple threads if available

# Ensure 2D shape for k==1
if k == 1 or dist.ndim == 1:
    dist = dist[:, None]
    idx  = idx[:, None]

# Optional radius fallback: if no neighbor within r, search with larger k/r or fall back to nearest
r_max = 1.5   # meters; tweak if needed
# when all k neighbors are farther than r_max, we’ll still use them but weight less
dist = np.maximum(dist, 1e-6)

# Inverse-distance weights (IDW)
w = 1.0 / dist
w /= w.sum(axis=1, keepdims=True)

# Weighted ground Z estimate at each XY
gz_neighbors = ground_xyz[idx, 2]                   # (N, k)
ground_z_at_xy = np.sum(gz_neighbors * w, axis=1)   # (N,)

# Height Above Ground (HAG)
hag = (pts_xyz[:, 2] - ground_z_at_xy).astype(np.float32)
hag = np.maximum(hag, 0.0)  # clamp tiny negatives from numeric noise

# ---- Write back as extra dimension ----
if "Z_norm" not in las_all.point_format.dimension_names:
    las_all.add_extra_dim(laspy.ExtraBytesParams(name="Z_norm", type=np.float32))
las_all.Z_norm = hag

las_all.write(normalised_output_las)

print(f"✅ Normalized LAS with Z_norm (HAG) saved to: {normalised_output_las}")
print(f"   HAG stats (m): min={hag.min():.3f}, median={np.median(hag):.3f}, max={hag.max():.3f}")
print(f"   Ground pts: {len(ground_xyz):,} | All pts: {len(pts_xyz):,} | k={k}")



# %%
import laspy
import numpy as np

# Read the file that actually contains HAG/Z_norm
hag_las = "/media/laura/Laura 102/uas_data/site1_1_DD0001/lidar/trim/smrf/cloud1457eb47b07af181_clip_raster_extent_HAG.las"
las_all = laspy.read(hag_las)

# Peek available dimensions if you want to confirm
# print(las_all.point_format.dimension_names)

# Grab HAG from the extra dimension (robust to either name)
if "Z_norm" in las_all.point_format.dimension_names:
    h = las_all["Z_norm"].astype(float)
elif "HAG" in las_all.point_format.dimension_names:
    h = las_all["HAG"].astype(float)
else:
    raise RuntimeError("No Z_norm/HAG dimension found in the HAG file. "
                       "Make sure you ran the normalization step and saved to this path.")

# Now your diagnostics:
thr = [0.02, 0.05, 0.10, 0.30, 1.0]
counts = {t: int((h > t).sum()) for t in thr}
total = len(h)
print("Counts above thresholds:", {f">{t}m": c for t, c in counts.items()})
print("Fractions:", {f">{t}m": round(c/total, 3) for t, c in counts.items()})

pcts = np.percentile(h, [50, 75, 90, 95, 99, 99.9])
print("HAG percentiles [p50,p75,p90,p95,p99,p99.9]:", [round(v,3) for v in pcts])

# %%
import os
import laspy
import numpy as np

hag_las = "/media/laura/Laura 102/uas_data/site1_1_DD0001/lidar/trim/csf/smrf/cloud1457eb47b07af181_clip_raster_extent_HAG.las"

out_reclass = "/media/laura/Laura 102/uas_data/site1_1_DD0001/lidar/trim/csf/smrf/cloud1457eb47b07af181_clip_raster_extent_HAG_reclass.las"
out_ground  = "/media/laura/Laura 102/uas_data/site1_1_DD0001/lidar/trim/csf/smrf/cloud1457eb47b07af181_ground_hag05.las"
out_veg     = "/media/laura/Laura 102/uas_data/site1_1_DD0001/lidar/trim/csf/smrf/cloud1457eb47b07af181_veg_hag30.las"
os.makedirs(os.path.dirname(out_reclass), exist_ok=True)

las = laspy.read(hag_las)
h = las["Z_norm"].astype(np.float32)

# thresholds (tweak if needed)
g_thr  = 0.05   # 5 cm
veg_thr = 0.30  # 30 cm

# start from existing classes if present, else zeros
cls = las.classification.copy() if "classification" in las.point_format.standard_dimension_names else np.zeros(len(h), np.uint8)

# reclassify
cls[h <= g_thr] = 2                         # ground
cls[(h > g_thr) & (h < veg_thr)] = 3        # low vegetation
cls[h >= veg_thr] = 5                        # high vegetation / shrubs

las.classification = cls
las.write(out_reclass)
print("✅ Wrote reclassified LAS:", out_reclass)

# optional: write ground-only and veg-only subsets
mask_ground = (cls == 2)
mask_veg    = (cls >= 3)

g_out = laspy.LasData(las.header)
g_out.points = las.points[mask_ground]
g_out.write(out_ground)
print("✅ Wrote ground-only:", out_ground, "(", mask_ground.sum(), "pts )")

v_out = laspy.LasData(las.header)
v_out.points = las.points[mask_veg]
v_out.write(out_veg)
print("✅ Wrote veg-only:", out_veg, "(", mask_veg.sum(), "pts )")


# %%
##mesh
# Ground TIN mesh from LAS (2.5D Delaunay) → PLY
import os
from pathlib import Path
import numpy as np
import laspy
from scipy.spatial import Delaunay

# ---------- inputs ----------
ground_only_las = Path("/media/laura/Laura 102/uas_data/site1_1_DD0001/lidar/trim/csf/smrf/cloud1457eb47b07af181_clip_raster_extent_smrf_ground.las")

# output mesh path
out_mesh_ply = Path("/media/laura/Laura 102/uas_data/site1_1_DD0001/lidar/trim/smrf/mesh/cloud1457eb47b07af181_clip_raster_extent_groundTIN_5cm.ply")

# meshing knobs
voxel_size = 0.05   # meters; 0.03–0.10 typical for UAS. Larger = fewer points, faster.
max_edge   = 0.50   # meters; drop triangles with any edge longer than this. Set None to disable.

# ---------- helpers ----------
def voxel_downsample_xyz(xyz: np.ndarray, leaf: float) -> np.ndarray:
    """Grid/voxel downsample by taking 1 point per cell."""
    if leaf is None or leaf <= 0:
        return xyz
    mins = xyz[:, :2].min(axis=0)
    keys = np.floor((xyz[:, :2] - mins) / leaf).astype(np.int64)
    # unique cells → keep first index
    _, keep_idx = np.unique(keys, axis=0, return_index=True)
    return xyz[np.sort(keep_idx)]

def tri_max_edge_len(verts: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Compute max edge length per triangle."""
    p0 = verts[faces[:,0]]
    p1 = verts[faces[:,1]]
    p2 = verts[faces[:,2]]
    e01 = np.linalg.norm(p0 - p1, axis=1)
    e12 = np.linalg.norm(p1 - p2, axis=1)
    e20 = np.linalg.norm(p2 - p0, axis=1)
    return np.maximum(e01, np.maximum(e12, e20))

def write_ply_ascii(vertices: np.ndarray, faces: np.ndarray, path: Path):
    """Minimal ASCII PLY writer for triangle meshes."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {len(vertices)}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write(f"element face {len(faces)}\n")
        f.write("property list uchar int vertex_indices\n")
        f.write("end_header\n")
        for v in vertices:
            f.write(f"{v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        for tri in faces:
            f.write(f"3 {tri[0]} {tri[1]} {tri[2]}\n")

# ---------- load ground points ----------
las = laspy.read(str(ground_only_las))
ground_xyz = np.vstack([las.x, las.y, las.z]).T.astype(np.float64)
if ground_xyz.shape[0] == 0:
    raise ValueError("No ground points found in the ground LAS.")

print(f"Ground points: {ground_xyz.shape[0]:,}")

# ---------- downsample (strongly recommended) ----------
ds_xyz = voxel_downsample_xyz(ground_xyz, voxel_size)
print(f"Downsampled to: {ds_xyz.shape[0]:,} points at voxel={voxel_size} m")

# ---------- Delaunay on XY (2.5D surface) ----------
xy = ds_xyz[:, :2]
tri = Delaunay(xy)       # may be heavy if too many points; increase voxel_size if it runs out of memory
faces = tri.simplices    # (M, 3) int indices into ds_xyz
print(f"Raw triangles: {faces.shape[0]:,}")

# ---------- optional: cull long/skinny triangles ----------
if max_edge is not None and max_edge > 0:
    maxlen = tri_max_edge_len(ds_xyz, faces)
    keep = maxlen <= max_edge
    faces = faces[keep]
    print(f"Kept triangles after edge≤{max_edge} m: {faces.shape[0]:,}")

# ---------- write PLY ----------
write_ply_ascii(ds_xyz, faces, out_mesh_ply)
print(f"✅ Mesh written: {out_mesh_ply}")

# %%
