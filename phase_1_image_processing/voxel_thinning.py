#%%
#check CRS
import laspy
from pathlib import Path

las_path = Path("/media/laura/laura_usb/qgis/calperumResearch/site2_1_DD0011/lidar/thinned_voxel/cloud2453cf90763afd91_thinned_3cm.las")
las = laspy.read(str(las_path))

# Try to parse CRS from GeoTIFF keys / EPSG in header
crs = las.header.parse_crs()  # returns a pyproj.CRS or None
if crs is None:
    print("❌ No CRS stored in LAS header.")
else:
    try:
        print("✅ CRS:", crs)
        print("   EPSG:", crs.to_epsg())  # may be None if not exactly mappable
    except Exception:
        print("✅ CRS present, but no direct EPSG code available.")


#%%
#step 1.1: Voxel Thinning of LiDAR and SfM Point Clouds
# LiDAR point cloud using voxel grid filtering at 3 cm resolution
import laspy
import open3d as o3d
import numpy as np
from pathlib import Path

# === INPUT LAS ===
las_path = Path("/media/laura/Extreme SSD/qgis/calperumResearch/site2_1_DD0011/lidar/cloud2453cf90763afd91.las")

# === OUTPUT DIRECTORY ===
output_dir = Path("/media/laura/laura_usb/qgis/calperumResearch/site2_1_DD0011/lidar/thinned_voxel")
output_dir.mkdir(parents=True, exist_ok=True)

# === Load LAS using laspy ===
las = laspy.read(str(las_path))
points = np.vstack((las.x, las.y, las.z)).T

# === Convert to Open3D PointCloud ===
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points)

# === Apply voxel downsampling (3 cm) ===
voxel_size = 0.03  # 3 cm
pcd_down = pcd.voxel_down_sample(voxel_size)

# === Extract downsampled points ===
down_points = np.asarray(pcd_down.points)

# === Create new LAS with thinned points ===
new_las = laspy.create(point_format=las.header.point_format, file_version=las.header.version)

# Only copy coordinates (X, Y, Z)
new_las.x = down_points[:, 0]
new_las.y = down_points[:, 1]
new_las.z = down_points[:, 2]

# === Save output LAS ===
output_path = output_dir / (las_path.stem + "_thinned_3cm.las")
new_las.write(str(output_path))

print(f"✅ Thinned point cloud saved to:\n   {output_path}")
print(f"📉 Original points: {len(points):,}")
print(f"📈 Thinned points : {len(down_points):,}")

#  Original points: 66,646,512
# Thinned points : 44,443,920


#%%
# OPTION B LIDAR
#!/usr/bin/env python3
# Voxel thin a LAS to 3 cm while preserving CRS + attributes, laspy without header.copy()

import numpy as np
import laspy
from laspy.header import LasHeader
from pathlib import Path
from copy import deepcopy

# === Config ===
voxel_size = 0.03  # 3 cm
in_path  = Path("/media/laura/Extreme SSD/qgis/calperumResearch/site2_1_DD0011/lidar/cloud2453cf90763afd91.las")
out_dir  = Path("/media/laura/laura_usb/qgis/calperumResearch/site2_1_DD0011/lidar/thinned_voxel")
out_dir.mkdir(parents=True, exist_ok=True)
out_path = out_dir / (in_path.stem + "_thinned_3cm.las")

# === Read input ===
las = laspy.read(str(in_path))
pts = np.vstack((las.x, las.y, las.z)).T

# CRS check (optional)
crs = las.header.parse_crs()
if crs is None:
    print("⚠️  Input LAS has no CRS in header.")
else:
    try:
        print(f"✅ Input CRS: {crs.to_epsg() or crs.to_string()}")
    except Exception:
        print("✅ Input CRS present (no direct EPSG mapping).")

# === Build voxel keys ===
origin = pts.min(axis=0)
keys = np.floor((pts - origin) / voxel_size).astype(np.int64)

uniq_keys, inv = np.unique(keys, axis=0, return_inverse=True)
centroids = (uniq_keys.astype(np.float64) + 0.5) * voxel_size + origin
d2 = np.sum((pts - centroids[inv])**2, axis=1)

order = np.lexsort((d2, inv))
sorted_groups = inv[order]
first_in_group = np.ones_like(sorted_groups, dtype=bool)
first_in_group[1:] = sorted_groups[1:] != sorted_groups[:-1]
kept_idx = order[first_in_group]
kept_idx.sort()

print(f"📉 Original points: {pts.shape[0]:,}")
print(f"📈 Thinned points : {kept_idx.size:,} (voxel={voxel_size} m)")

# === Rebuild header (no .copy() available) ===
# Use same LAS version & point format; copy scales/offsets too.
header = LasHeader(version=las.header.version, point_format=las.header.point_format)
header.scales = las.header.scales
header.offsets = las.header.offsets

# === Create output LAS and transfer CRS (VLRs/EVLRs) ===
out_las = laspy.LasData(header)
# Copy CRS/other VLRs
out_las.vlrs = deepcopy(las.vlrs)
try:
    out_las.evlrs = deepcopy(las.evlrs)
except AttributeError:
    pass

# === Coordinates ===
out_las.x = las.x[kept_idx]
out_las.y = las.y[kept_idx]
out_las.z = las.z[kept_idx]

# === Copy ALL other dimensions (Intensity, RGB, Classification, GPS time, etc.) ===
for dim in las.point_format.dimensions:
    name = dim.name
    if name in ("X", "Y", "Z"):
        continue
    try:
        arr = las[name]
        if arr is not None and len(arr) == len(las.x):
            out_las[name] = arr[kept_idx]
    except Exception:
        # Some optional dims may be missing or read-only; skip quietly
        pass

# === Write ===
out_las.write(str(out_path))
print(f"✅ Thinned point cloud saved:\n   {out_path}")

# ✅ Input CRS: 7854
# 📉 Original points: 66,646,512
# 📈 Thinned points : 44,442,206 (voxel=0.03 m)
# ✅ Thinned point cloud saved:
#    /media/laura/laura_usb/qgis/calperumResearch/site2_1_DD0011/lidar/thinned_voxel/cloud2453cf90763afd91_thinned_3cm.las

# %%
## FOR RGB SFM
#!/usr/bin/env python3
# Voxel thin a LAS to 3 cm while preserving CRS + attributes, laspy without header.copy()

import numpy as np
import laspy
from laspy.header import LasHeader
from pathlib import Path
from copy import deepcopy

# === Config ===
voxel_size = 0.03  # 3 cm
in_path  = Path("/media/laura/laura_usb/uas_data/DD0011/lidar/SASMDD0011_rgb_pc.las")
out_dir  = Path("/media/laura/laura_usb/qgis/calperumResearch/site2_1_DD0011/lidar/thinned_voxel")
out_dir.mkdir(parents=True, exist_ok=True)
out_path = out_dir / (in_path.stem + "_thinned_3cm.las")

# === Read input ===
las = laspy.read(str(in_path))
pts = np.vstack((las.x, las.y, las.z)).T

# CRS check (optional)
crs = las.header.parse_crs()
if crs is None:
    print("⚠️  Input LAS has no CRS in header.")
else:
    try:
        print(f"✅ Input CRS: {crs.to_epsg() or crs.to_string()}")
    except Exception:
        print("✅ Input CRS present (no direct EPSG mapping).")

# === Build voxel keys ===
origin = pts.min(axis=0)
keys = np.floor((pts - origin) / voxel_size).astype(np.int64)

uniq_keys, inv = np.unique(keys, axis=0, return_inverse=True)
centroids = (uniq_keys.astype(np.float64) + 0.5) * voxel_size + origin
d2 = np.sum((pts - centroids[inv])**2, axis=1)

order = np.lexsort((d2, inv))
sorted_groups = inv[order]
first_in_group = np.ones_like(sorted_groups, dtype=bool)
first_in_group[1:] = sorted_groups[1:] != sorted_groups[:-1]
kept_idx = order[first_in_group]
kept_idx.sort()

print(f"📉 Original points: {pts.shape[0]:,}")
print(f"📈 Thinned points : {kept_idx.size:,} (voxel={voxel_size} m)")

# === Rebuild header (no .copy() available) ===
# Use same LAS version & point format; copy scales/offsets too.
header = LasHeader(version=las.header.version, point_format=las.header.point_format)
header.scales = las.header.scales
header.offsets = las.header.offsets

# === Create output LAS and transfer CRS (VLRs/EVLRs) ===
out_las = laspy.LasData(header)
# Copy CRS/other VLRs
out_las.vlrs = deepcopy(las.vlrs)
try:
    out_las.evlrs = deepcopy(las.evlrs)
except AttributeError:
    pass

# === Coordinates ===
out_las.x = las.x[kept_idx]
out_las.y = las.y[kept_idx]
out_las.z = las.z[kept_idx]

# === Copy ALL other dimensions (Intensity, RGB, Classification, GPS time, etc.) ===
for dim in las.point_format.dimensions:
    name = dim.name
    if name in ("X", "Y", "Z"):
        continue
    try:
        arr = las[name]
        if arr is not None and len(arr) == len(las.x):
            out_las[name] = arr[kept_idx]
    except Exception:
        # Some optional dims may be missing or read-only; skip quietly
        pass

# === Write ===
out_las.write(str(out_path))
print(f"✅ Thinned point cloud saved:\n   {out_path}")

# %%
##imporve
#!/usr/bin/env python3
# Match LiDAR and RGB SfM point clouds to the SAME 3 cm resolution and grid

from pathlib import Path
from copy import deepcopy
import numpy as np
import laspy

# === Config ===
VOXEL = 0.03  # 3 cm
SAFETY_DENSITY = 1.0 / (VOXEL ** 2)  # ~1111.11 pts/m² required for 3 cm
LIDAR_IN = Path("/media/laura/Extreme SSD/qgis/calperumResearch/site2_1_DD0011/lidar/cloud2453cf90763afd91.las")
RGB_IN   = Path("/media/laura/laura_usb/uas_data/DD0011/lidar/SASMDD0011_rgb_pc.las")
OUT_DIR  = Path("/media/laura/laura_usb/qgis/calperumResearch/site2_1_DD0011/lidar/thinned_voxel_matched")
OUT_DIR.mkdir(parents=True, exist_ok=True)

def load_points(p):
    las = laspy.read(str(p))
    pts = np.vstack((las.x, las.y, las.z)).T
    return las, pts

def density_xy(pts):
    xmin, ymin = pts[:,0].min(), pts[:,1].min()
    xmax, ymax = pts[:,0].max(), pts[:,1].max()
    area = max((xmax - xmin) * (ymax - ymin), 1e-12)
    dens = len(pts) / area
    spacing = 1.0 / np.sqrt(dens)
    return dens, spacing, (xmin, ymin, xmax, ymax)

def snapped_origin(bbox_union, voxel):
    xmin, ymin, zmin = bbox_union[0], bbox_union[1], bbox_union[2]
    # snap to a stable grid so both clouds quantize identically
    ox = np.floor(xmin / voxel) * voxel
    oy = np.floor(ymin / voxel) * voxel
    oz = np.floor(zmin / voxel) * voxel
    return np.array([ox, oy, oz], dtype=np.float64)

def voxel_thin_with_origin(pts, voxel, origin):
    keys = np.floor((pts - origin) / voxel).astype(np.int64)
    uniq_keys, inv = np.unique(keys, axis=0, return_inverse=True)
    # pick representative per voxel: closest to voxel centroid
    centroids = (uniq_keys.astype(np.float64) + 0.5) * voxel + origin
    d2 = np.sum((pts - centroids[inv])**2, axis=1)
    order = np.lexsort((d2, inv))
    sorted_groups = inv[order]
    first = np.ones_like(sorted_groups, dtype=bool)
    first[1:] = sorted_groups[1:] != sorted_groups[:-1]
    kept_idx = order[first]
    kept_idx.sort()
    return kept_idx

def write_subset(las, kept_idx, out_path):
    header = laspy.LasHeader(version=las.header.version, point_format=las.header.point_format)
    header.scales  = las.header.scales
    header.offsets = las.header.offsets
    out = laspy.LasData(header)
    # copy CRS + other VLR/EVLR
    out.vlrs = deepcopy(las.vlrs)
    try:
        out.evlrs = deepcopy(las.evlrs)
    except Exception:
        pass
    # coordinates
    out.x = las.x[kept_idx]
    out.y = las.y[kept_idx]
    out.z = las.z[kept_idx]
    # copy all other dimensions
    for dim in las.point_format.dimensions:
        name = dim.name
        if name in ("X", "Y", "Z"):
            continue
        try:
            arr = las[name]
            if arr is not None and len(arr) == len(las.x):
                out[name] = arr[kept_idx]
        except Exception:
            pass
    out.write(str(out_path))
    return out_path

# === Load both ===
lidar_las, lidar_pts = load_points(LIDAR_IN)
rgb_las,   rgb_pts   = load_points(RGB_IN)

# === Densities + sanity ===
lidar_den, lidar_spacing, (lx0, ly0, lx1, ly1) = density_xy(lidar_pts)
rgb_den,   rgb_spacing,   (rx0, ry0, rx1, ry1) = density_xy(rgb_pts)
print(f"LiDAR: {lidar_den:.2f} pts/m² → spacing ~ {lidar_spacing*100:.2f} cm")
print(f"RGB  : {rgb_den:.2f} pts/m² → spacing ~ {rgb_spacing*100:.2f} cm")

if lidar_den < SAFETY_DENSITY:
    print(f"⚠️ LiDAR density < required for 3 cm ({SAFETY_DENSITY:.2f} pts/m²). "
          f"You may see gaps after thinning.")
if rgb_den < SAFETY_DENSITY:
    print(f"⚠️ RGB density < required for 3 cm ({SAFETY_DENSITY:.2f} pts/m²). "
          f"You may see gaps after thinning.")

# === Shared, snapped origin over the union bbox ===
xmin = min(lx0, rx0); ymin = min(ly0, ry0)
zmin = min(lidar_pts[:,2].min(), rgb_pts[:,2].min())
origin = snapped_origin((xmin, ymin, zmin), VOXEL)
print(f"Shared voxel grid: size={VOXEL:.3f} m, origin=({origin[0]:.3f}, {origin[1]:.3f}, {origin[2]:.3f})")

# === Voxel thin BOTH using the SAME grid ===
lidar_keep = voxel_thin_with_origin(lidar_pts, VOXEL, origin)
rgb_keep   = voxel_thin_with_origin(rgb_pts,   VOXEL, origin)

print(f"LiDAR: {len(lidar_pts):,} → {lidar_keep.size:,} points (3 cm matched)")
print(f"RGB  : {len(rgb_pts):,} → {rgb_keep.size:,} points (3 cm matched)")

# === Write outputs ===
lidar_out = OUT_DIR / f"{LIDAR_IN.stem}_thinned_3cm_matched.las"
rgb_out   = OUT_DIR / f"{RGB_IN.stem}_thinned_3cm_matched.las"
write_subset(lidar_las, lidar_keep, lidar_out)
write_subset(rgb_las,   rgb_keep,   rgb_out)

print("✅ Saved:")
print(f"   {lidar_out}")
print(f"   {rgb_out}")
print("👉 Downstream tip: rasterize DSMs at 0.03 m using the same grid/geotransform for perfect alignment.")

# %%
##other version
#!/usr/bin/env python3
# Auto-origin from raster → matched 3 cm voxel thinning (PDAL) + crop to raster extent (no rasterization)

from pathlib import Path
import json, math
import laspy
import rasterio
import pdal
import numpy as np

# ===== User config =====
CELL = 0.02  # 3 cm voxel size
REF_RASTER = Path("/media/laura/laura_usb/qgis/calperumResearch/site2_1_DD0011/orthomosaic/20220517_SASMDD0011_dual_ortho_06.tif")
LIDAR_IN   = Path("/media/laura/Extreme SSD/qgis/calperumResearch/site2_1_DD0011/lidar/cloud2453cf90763afd91.las")
RGB_IN     = Path("/media/laura/laura_usb/uas_data/DD0011/lidar/SASMDD0011_rgb_pc.las")
OUT_DIR    = Path("/media/laura/laura_usb/qgis/calperumResearch/site2_1_DD0011/lidar/thinned_voxel_3cm")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ===== Helpers =====
def zmin_union(*las_paths):
    zmins = []
    for p in las_paths:
        las = laspy.read(str(p))
        zmins.append(float(np.min(las.z)))
    return float(min(zmins))

def snap_floor_to_lattice(v, step, origin):
    return origin + math.floor((v - origin) / step) * step

def snap_ceil_to_lattice(v, step, origin):
    return origin + math.ceil((v - origin) / step) * step

def pdal_run(pipe):
    p = pdal.Pipeline(json.dumps(pipe))
    p.execute()

# ===== 1) Read raster georef & bounds → lattice origin =====
with rasterio.open(REF_RASTER) as src:
    left, bottom, right, top = src.bounds
    # We use raster's lower-left corner as lattice origin (matches common GeoTIFF conventions)
    origin_x = float(left)
    origin_y = float(bottom)
    print(f"Ref raster bounds: left={left:.3f}, right={right:.3f}, bottom={bottom:.3f}, top={top:.3f}")
    print(f"Using voxel origin XY from raster: origin_x={origin_x:.3f}, origin_y={origin_y:.3f}")

# Get Z-origin from union min Z (snapped down to 3 cm)
origin_z_raw = zmin_union(LIDAR_IN, RGB_IN)
origin_z = math.floor(origin_z_raw / CELL) * CELL
print(f"Union min Z: {origin_z_raw:.3f} → origin_z (snapped)={origin_z:.3f}")

# ===== 2) Build snapped crop bounds on the same lattice =====
xmin = snap_floor_to_lattice(left,  CELL, origin_x)
ymin = snap_floor_to_lattice(bottom, CELL, origin_y)
xmax = snap_ceil_to_lattice(right, CELL, origin_x)
ymax = snap_ceil_to_lattice(top,   CELL, origin_y)
BOUNDS = f"([{xmin:.3f},{xmax:.3f}],[{ymin:.3f},{ymax:.3f}])"
print(f"Crop bounds (snapped to lattice): {BOUNDS}")

# ===== 3) PDAL pipelines: thin (filters.sample) → crop → write LAS =====
def sample_then_crop(src_las: Path, dst_las: Path):
    pipe = {
        "pipeline": [
            str(src_las),
            {   # One-per-cell thinning anchored to raster-derived origin
                "type": "filters.sample",
                "cell": CELL,
                "origin_x": origin_x,
                "origin_y": origin_y,
                "origin_z": origin_z
            },
            {   # Crop to raster extent (snapped)
                "type": "filters.crop",
                "bounds": BOUNDS
            },
            {
                "type": "writers.las",
                "filename": str(dst_las)
            }
        ]
    }
    pdal_run(pipe)
    print(f"✅ Wrote: {dst_las}")
    
   # ---- Compute density & recommended resolution + 5 cm occupancy ----
    las = laspy.read(str(dst_las))
    x, y = las.x, las.y
    xmin, xmax = float(x.min()), float(x.max())
    ymin, ymax = float(y.min()), float(y.max())
    area = (xmax - xmin) * (ymax - ymin)
    density = len(x) / area if area > 0 else 0.0            # pts / m^2
    spacing = (1.0 / density) ** 0.5 if density > 0 else float("nan")
    pts_per_5cm = density * (0.05 ** 2)                      # expected pts per 5 cm pixel
    print(f"Point density: {density:.2f} pts/m² → Recommended spacing: {spacing:.4f} m")
    print(f"Estimated points per 5 cm pixel: {pts_per_5cm:.2f}")


LIDAR_OUT = OUT_DIR / f"{LIDAR_IN.stem}_thinned_3cm_from_ref.las"
RGB_OUT   = OUT_DIR / f"{RGB_IN.stem}_thinned_3cm_from_ref.las"

sample_then_crop(LIDAR_IN, LIDAR_OUT)
sample_then_crop(RGB_IN,   RGB_OUT)

print("Done.")

# voxel size= 0.03 cm
# Ref raster bounds: left=473069.739, right=473344.359, bottom=6235596.731, top=6235925.711
# Using voxel origin XY from raster: origin_x=473069.739, origin_y=6235596.731
# Union min Z: 29.026 → origin_z (snapped)=29.010
# Crop bounds (snapped to lattice): ([473069.739,473344.359],[6235596.731,6235925.741])
# ✅ Wrote: /media/laura/laura_usb/qgis/calperumResearch/site2_1_DD0011/lidar/thinned_voxel_ref/cloud2453cf90763afd91_thinned_3cm_from_ref.las
# Point density: 564.66 pts/m² → Recommended resolution: 0.0421 m
# ✅ Wrote: /media/laura/laura_usb/qgis/calperumResearch/site2_1_DD0011/lidar/thinned_voxel_ref/SASMDD0011_rgb_pc_thinned_3cm_from_ref.las
# Point density: 611.23 pts/m² → Recommended resolution: 0.0404 m
# Done.

# voxel size= 0.02 cm
# Ref raster bounds: left=473069.739, right=473344.359, bottom=6235596.731, top=6235925.711
# Using voxel origin XY from raster: origin_x=473069.739, origin_y=6235596.731
# Union min Z: 29.026 → origin_z (snapped)=29.020
# Crop bounds (snapped to lattice): ([473069.739,473344.359],[6235596.731,6235925.731])
# ✅ Wrote: /media/laura/laura_usb/qgis/calperumResearch/site2_1_DD0011/lidar/thinned_voxel_3cm/cloud2453cf90763afd91_thinned_3cm_from_ref.las
# Point density: 835.83 pts/m² → Recommended spacing: 0.0346 m
# Estimated points per 5 cm pixel: 2.09
# ✅ Wrote: /media/laura/laura_usb/qgis/calperumResearch/site2_1_DD0011/lidar/thinned_voxel_3cm/SASMDD0011_rgb_pc_thinned_3cm_from_ref.las
# Point density: 1152.05 pts/m² → Recommended spacing: 0.0295 m
# Estimated points per 5 cm pixel: 2.88
# Done.


# %%
## test improve

#!/usr/bin/env python3
# Auto-thin to ~3 cm if needed (tries 0.02, 0.025, 0.03), otherwise skip thinning.
# Uses PDAL filters.sample with raster-derived origin; crops to raster extent.
# Reports densities and implied spacings before/after.

from pathlib import Path
import json, math
import numpy as np
import laspy
import rasterio
import pdal
import shutil
import tempfile

# ========= USER CONFIG =========
REF_RASTER = Path("/media/laura/laura_usb/qgis/calperumResearch/site2_1_DD0011/orthomosaic/20220517_SASMDD0011_dual_ortho_06.tif")
LIDAR_IN   = Path("/media/laura/Extreme SSD/qgis/calperumResearch/site2_1_DD0011/lidar/cloud2453cf90763afd91.las")
RGB_IN     = Path("/media/laura/laura_usb/uas_data/DD0011/lidar/SASMDD0011_rgb_pc.las")

OUT_DIR    = Path("/media/laura/laura_usb/qgis/calperumResearch/site2_1_DD0011/lidar/thinned_voxel_auto_3cm")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# What “close to 3 cm” means
TARGET_SPACING = 0.030       # 3 cm
TOLERANCE      = 0.0015      # ±1.5 mm tolerance => <= 0.0315 m is “close enough”

# Candidate voxel cells to try if thinning is needed
CANDIDATE_CELLS = [0.02, 0.025, 0.03]   # 2cm, 2.5cm, 3cm

# ========= HELPERS =========
def read_raster_bounds(ref_raster: Path):
    with rasterio.open(ref_raster) as src:
        left, bottom, right, top = src.bounds
        crs = src.crs
    return left, bottom, right, top, crs

def union_min_z(*las_paths):
    vals = []
    for p in las_paths:
        las = laspy.read(str(p))
        vals.append(float(np.min(las.z)))
    return float(min(vals)) if vals else 0.0

def snap_floor_to_lattice(v, step, origin):
    return origin + math.floor((v - origin) / step) * step

def snap_ceil_to_lattice(v, step, origin):
    return origin + math.ceil((v - origin) / step) * step

def pdal_run(pipe_json):
    p = pdal.Pipeline(json.dumps(pipe_json))
    p.execute()

def measure_density_xy(las_path: Path):
    las = laspy.read(str(las_path))
    x, y = las.x, las.y
    if len(x) == 0:
        return 0.0, float("nan")
    xmin, xmax = float(x.min()), float(x.max())
    ymin, ymax = float(y.min()), float(y.max())
    area = max((xmax - xmin) * (ymax - ymin), 1e-12)
    dens = len(x) / area
    spacing = (1.0 / dens) ** 0.5 if dens > 0 else float("nan")
    return dens, spacing

def crop_only(src_las: Path, dst_las: Path, bounds_str: str):
    pipe = {
        "pipeline": [
            str(src_las),
            {"type": "filters.crop", "bounds": bounds_str},
            {"type": "writers.las", "filename": str(dst_las)}
        ]
    }
    pdal_run(pipe)

def sample_then_crop(src_las: Path, dst_las: Path, cell: float,
                     origin_x: float, origin_y: float, origin_z: float,
                     bounds_str: str):
    pipe = {
        "pipeline": [
            str(src_las),
            {"type": "filters.sample",
             "cell": cell,
             "origin_x": origin_x,
             "origin_y": origin_y,
             "origin_z": origin_z},
            {"type": "filters.crop", "bounds": bounds_str},
            {"type": "writers.las", "filename": str(dst_las)}
        ]
    }
    pdal_run(pipe)

def choose_best_cell(src_las: Path, tmp_dir: Path, cells, origin_x, origin_y, origin_z, bounds_str):
    """
    Try each cell, measure spacing, choose the one with spacing <= (TARGET+tol) closest to TARGET.
    If none meets the threshold, choose the smallest spacing achieved.
    Returns (best_cell, best_path, density, spacing).
    """
    results = []
    for cell in cells:
        tmp_out = tmp_dir / f"{src_las.stem}_thin_{int(cell*1000)}mm_tmp.las"
        sample_then_crop(src_las, tmp_out, cell, origin_x, origin_y, origin_z, bounds_str)
        dens, spacing = measure_density_xy(tmp_out)
        results.append((cell, tmp_out, dens, spacing))
        print(f"   Tried CELL={cell:.3f} → density={dens:.2f} pts/m², spacing={spacing:.4f} m")

    # First, candidates meeting spacing <= TARGET+TOL
    good = [r for r in results if r[3] <= (TARGET_SPACING + TOLERANCE)]
    if good:
        # Pick the one with spacing closest to TARGET (>=TARGET preferred). If tie, denser wins.
        good.sort(key=lambda r: (abs(r[3] - TARGET_SPACING), -r[2]))
        return good[0]

    # Else pick the best (smallest spacing) achieved
    results.sort(key=lambda r: r[3])
    return results[0]

# ========= MAIN =========
left, bottom, right, top, raster_crs = read_raster_bounds(REF_RASTER)
origin_x = float(left)
origin_y = float(bottom)

# Z-origin from union min Z, snapped to 3 cm so later 3 cm grids align well, but this origin works for any cell
zmin = union_min_z(LIDAR_IN, RGB_IN)
origin_z = math.floor(zmin / 0.03) * 0.03

# Crop bounds snapped to lattice with respect to origin_x/y and TARGET_SPACING (3 cm lattice)
xmin = snap_floor_to_lattice(left,  TARGET_SPACING, origin_x)
ymin = snap_floor_to_lattice(bottom, TARGET_SPACING, origin_y)
xmax = snap_ceil_to_lattice(right,  TARGET_SPACING, origin_x)
ymax = snap_ceil_to_lattice(top,    TARGET_SPACING, origin_y)
BOUNDS = f"([{xmin:.3f},{xmax:.3f}],[{ymin:.3f},{ymax:.3f}])"

print(f"Ref raster bounds: left={left:.3f}, right={right:.3f}, bottom={bottom:.3f}, top={top:.3f}")
print(f"Using voxel origin XY from raster: origin_x={origin_x:.3f}, origin_y={origin_y:.3f}")
print(f"Union min Z: {zmin:.3f} → origin_z (snapped to 3cm)={origin_z:.3f}")
print(f"Crop bounds (snapped to 3cm lattice): {BOUNDS}")

def process_one(src: Path, label: str) -> Path:
    print(f"\n— {label} —")
    d0, s0 = measure_density_xy(src)
    print(f"Original density: {d0:.2f} pts/m² → spacing ~ {s0:.4f} m")

    # If already within tolerance of 3 cm, skip thinning; just crop
    if s0 <= (TARGET_SPACING + TOLERANCE):
        out_path = OUT_DIR / f"{src.stem}_cropped_only.las"
        crop_only(src, out_path, BOUNDS)
        d1, s1 = measure_density_xy(out_path)
        print(f"✅ Close to 3cm: SKIP thinning → cropped only")
        print(f"Output: {out_path}")
        print(f"Post-crop density: {d1:.2f} pts/m² → spacing ~ {s1:.4f} m")
        return out_path

    # Otherwise, try cells and choose best
    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        best_cell, best_tmp, d1, s1 = choose_best_cell(
            src, td_path, CANDIDATE_CELLS,
            origin_x, origin_y, origin_z, BOUNDS
        )
        out_path = OUT_DIR / f"{src.stem}_thin{int(best_cell*1000)}mm_from_ref.las"
        shutil.move(str(best_tmp), str(out_path))
        print(f"✅ Chosen CELL={best_cell:.3f} for {label}")
        print(f"Output: {out_path}")
        print(f"Post-thin density: {d1:.2f} pts/m² → spacing ~ {s1:.4f} m")
        return out_path

lidar_out = process_one(LIDAR_IN, "LiDAR")
rgb_out   = process_one(RGB_IN,   "RGB SfM")

print("\nDone.")


#%%
##more imporved --> GOOD
#!/usr/bin/env python3
# Auto-thin to ~3 cm if needed (tries 0.02, 0.025, 0.03), otherwise skip thinning.
# Uses PDAL filters.sample with raster-derived origin; crops to raster extent.
# Reports densities, implied spacing, and estimated points per 5 cm pixel.

from pathlib import Path
import json, math, shutil, tempfile
import numpy as np
import laspy
import rasterio
import pdal

# ========= USER CONFIG =========
REF_RASTER = Path("/media/laura/laura_usb/qgis/calperumResearch/site2_1_DD0011/orthomosaic/20220517_SASMDD0011_dual_ortho_06.tif")
# LIDAR_IN   = Path("/media/laura/Extreme SSD/qgis/calperumResearch/site2_1_DD0011/lidar/cloud2453cf90763afd91.las")
LIDAR_IN = Path("/media/laura/laura_usb/qgis/calperumResearch/site2_1_DD0011/lidar_l2/cloud2d765c57b8471e2c.las")
RGB_IN     = Path("/media/laura/laura_usb/uas_data/DD0011/lidar/SASMDD0011_rgb_pc.las")

OUT_DIR    = Path("/media/laura/laura_usb/qgis/calperumResearch/site2_1_DD0011/lidar_l2/thinned_voxel_auto_3cm")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Target spacing & logic
TARGET_SPACING = 0.030        # 3 cm
UPPER_TOL      = 0.0015       # 1.5 mm tolerance above 3 cm (3.15 cm)
THREE          = 0.030
TWO            = 0.020
CANDIDATE_CELLS = [0.02, 0.025, 0.03]   # 2cm, 2.5cm, 3cm

# ========= HELPERS =========
def read_raster_bounds(ref_raster: Path):
    with rasterio.open(ref_raster) as src:
        left, bottom, right, top = src.bounds
        crs = src.crs
    return left, bottom, right, top, crs

def union_min_z(*las_paths):
    vals = []
    for p in las_paths:
        las = laspy.read(str(p))
        vals.append(float(np.min(las.z)))
    return float(min(vals)) if vals else 0.0

def snap_floor_to_lattice(v, step, origin):
    return origin + math.floor((v - origin) / step) * step

def snap_ceil_to_lattice(v, step, origin):
    return origin + math.ceil((v - origin) / step) * step

def pdal_run(pipe_json):
    p = pdal.Pipeline(json.dumps(pipe_json))
    p.execute()

def measure_density_xy(las_path: Path):
    las = laspy.read(str(las_path))
    x, y = las.x, las.y
    n = len(x)
    if n == 0:
        return 0.0, float("nan")
    xmin, xmax = float(x.min()), float(x.max())
    ymin, ymax = float(y.min()), float(y.max())
    area = max((xmax - xmin) * (ymax - ymin), 1e-12)
    dens = n / area
    spacing = (1.0 / dens) ** 0.5 if dens > 0 else float("nan")
    return dens, spacing

def pts_per_pixel(density_pts_per_m2, pixel_size_m=0.05):
    return density_pts_per_m2 * (pixel_size_m ** 2)

def crop_only(src_las: Path, dst_las: Path, bounds_str: str):
    pipe = {
        "pipeline": [
            str(src_las),
            {"type": "filters.crop", "bounds": bounds_str},
            {"type": "writers.las", "filename": str(dst_las)}
        ]
    }
    pdal_run(pipe)

def sample_then_crop(src_las: Path, dst_las: Path, cell: float,
                     origin_x: float, origin_y: float, origin_z: float,
                     bounds_str: str):
    pipe = {
        "pipeline": [
            str(src_las),
            {"type": "filters.sample",
             "cell": cell,
             "origin_x": origin_x,
             "origin_y": origin_y,
             "origin_z": origin_z},
            {"type": "filters.crop", "bounds": bounds_str},
            {"type": "writers.las", "filename": str(dst_las)}
        ]
    }
    pdal_run(pipe)

def choose_best_cell(src_las: Path, tmp_dir: Path, cells, origin_x, origin_y, origin_z, bounds_str):
    """
    Try each cell, measure spacing, choose the one with spacing <= (TARGET+tol) closest to TARGET.
    Prefer spacing >= TARGET (i.e., not overshooting below 3 cm) when possible; if tie, denser wins.
    If none meets <= (TARGET+tol), pick the smallest spacing achieved.
    Returns (best_cell, best_path, density, spacing).
    """
    results = []
    for cell in cells:
        tmp_out = tmp_dir / f"{src_las.stem}_thin_{int(cell*1000)}mm_tmp.las"
        sample_then_crop(src_las, tmp_out, cell, origin_x, origin_y, origin_z, bounds_str)
        dens, spacing = measure_density_xy(tmp_out)
        results.append((cell, tmp_out, dens, spacing))
        print(f"   Tried CELL={cell:.3f} → density={dens:.2f} pts/m², spacing={spacing:.4f} m, ~{pts_per_pixel(dens):.2f} pts/5cm px")

    # Candidates meeting spacing <= TARGET+TOL
    good = [r for r in results if r[3] <= (TARGET_SPACING + UPPER_TOL)]
    if good:
        # Prefer spacing >= TARGET (closer to 3 cm without going much below), then closeness, then density
        good.sort(key=lambda r: (0 if r[3] >= TARGET_SPACING else 1,
                                 abs(r[3] - TARGET_SPACING),
                                 -r[2]))
        return good[0]

    # Else pick best (smallest spacing) achieved
    results.sort(key=lambda r: r[3])
    return results[0]

def closer_to_3cm(spacing: float) -> bool:
    return abs(spacing - THREE) <= abs(spacing - TWO)

# ========= MAIN =========
left, bottom, right, top, raster_crs = read_raster_bounds(REF_RASTER)
origin_x = float(left)
origin_y = float(bottom)

# Z-origin from union min Z, snapped to 3 cm (works fine for any candidate cell)
zmin = union_min_z(LIDAR_IN, RGB_IN)
origin_z = math.floor(zmin / 0.03) * 0.03

# Crop bounds snapped to 3 cm lattice (so the rectangle aligns with the raster grid)
xmin = snap_floor_to_lattice(left,  TARGET_SPACING, origin_x)
ymin = snap_floor_to_lattice(bottom, TARGET_SPACING, origin_y)
xmax = snap_ceil_to_lattice(right,  TARGET_SPACING, origin_x)
ymax = snap_ceil_to_lattice(top,    TARGET_SPACING, origin_y)
BOUNDS = f"([{xmin:.3f},{xmax:.3f}],[{ymin:.3f},{ymax:.3f}])"

print(f"Ref raster bounds: left={left:.3f}, right={right:.3f}, bottom={bottom:.3f}, top={top:.3f}")
print(f"Using voxel origin XY from raster: origin_x={origin_x:.3f}, origin_y={origin_y:.3f}")
print(f"Union min Z: {zmin:.3f} → origin_z (snapped to 3cm)={origin_z:.3f}")
print(f"Crop bounds (snapped to 3cm lattice): {BOUNDS}")

def process_one(src: Path, label: str) -> Path:
    print(f"\n— {label} —")
    d0, s0 = measure_density_xy(src)
    print(f"Original density: {d0:.2f} pts/m² → spacing ~ {s0:.4f} m (~{pts_per_pixel(d0):.2f} pts/5cm px)")

    # Skip only if: closer to 3 cm than 2 cm AND within 3 cm + tolerance
    if closer_to_3cm(s0) and s0 <= (THREE + UPPER_TOL):
        out_path = OUT_DIR / f"{src.stem}_cropped_only.las"
        crop_only(src, out_path, BOUNDS)
        d1, s1 = measure_density_xy(out_path)
        print("✅ Closer to 3 cm → SKIP thinning (crop only)")
        print(f"Output: {out_path}")
        print(f"Post-crop density: {d1:.2f} pts/m² → spacing ~ {s1:.4f} m (~{pts_per_pixel(d1):.2f} pts/5cm px)")
        return out_path

    # Otherwise: thin using candidate cells and choose best towards 3 cm
    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        best_cell, best_tmp, d1, s1 = choose_best_cell(
            src, td_path, CANDIDATE_CELLS,
            origin_x, origin_y, origin_z, BOUNDS
        )
        out_path = OUT_DIR / f"{src.stem}_thin{int(best_cell*1000)}mm_from_ref.las"
        shutil.move(str(best_tmp), str(out_path))
        print(f"✅ Closer to 2 cm or outside tolerance → THIN (CELL={best_cell:.3f})")
        print(f"Output: {out_path}")
        print(f"Post-thin density: {d1:.2f} pts/m² → spacing ~ {s1:.4f} m (~{pts_per_pixel(d1):.2f} pts/5cm px)")
        return out_path

lidar_out = process_one(LIDAR_IN, "LiDAR")
rgb_out   = process_one(RGB_IN,   "RGB SfM")

print("\nDone.")


## L1
# Ref raster bounds: left=473069.739, right=473344.359, bottom=6235596.731, top=6235925.711
# Using voxel origin XY from raster: origin_x=473069.739, origin_y=6235596.731
# Union min Z: 29.026 → origin_z (snapped to 3cm)=29.010
# Crop bounds (snapped to 3cm lattice): ([473069.739,473344.359],[6235596.731,6235925.741])

# — LiDAR —
# Original density: 1177.84 pts/m² → spacing ~ 0.0291 m (~2.94 pts/5cm px)
# ✅ Closer to 3 cm → SKIP thinning (crop only)
# Output: /media/laura/laura_usb/qgis/calperumResearch/site2_1_DD0011/lidar/thinned_voxel_auto_3cm/cloud2453cf90763afd91_cropped_only.las
# Post-crop density: 1177.85 pts/m² → spacing ~ 0.0291 m (~2.94 pts/5cm px)

# — RGB SfM —
# Original density: 2016.89 pts/m² → spacing ~ 0.0223 m (~5.04 pts/5cm px)
#    Tried CELL=0.020 → density=1152.05 pts/m², spacing=0.0295 m, ~2.88 pts/5cm px
#    Tried CELL=0.025 → density=816.21 pts/m², spacing=0.0350 m, ~2.04 pts/5cm px
#    Tried CELL=0.030 → density=611.23 pts/m², spacing=0.0404 m, ~1.53 pts/5cm px
# ✅ Closer to 2 cm or outside tolerance → THIN (CELL=0.020)
# Output: /media/laura/laura_usb/qgis/calperumResearch/site2_1_DD0011/lidar/thinned_voxel_auto_3cm/SASMDD0011_rgb_pc_thin20mm_from_ref.las
# Post-thin density: 1152.05 pts/m² → spacing ~ 0.0295 m (~2.88 pts/5cm px)

## L2
# Ref raster bounds: left=473069.739, right=473344.359, bottom=6235596.731, top=6235925.711
# Using voxel origin XY from raster: origin_x=473069.739, origin_y=6235596.731
# Union min Z: 29.625 → origin_z (snapped to 3cm)=29.610
# Crop bounds (snapped to 3cm lattice): ([473069.739,473344.359],[6235596.731,6235925.741])

# — LiDAR —
# Original density: 1662.82 pts/m² → spacing ~ 0.0245 m (~4.16 pts/5cm px)

# %%
#!/usr/bin/env python3
# Auto-thin toward ~3 cm if needed (tries multiple cells), otherwise skip.
# PDAL: crop → sample (shared origin from raster). Reports density/spacing/pts-per-5cm.

from pathlib import Path
import json, math, shutil, tempfile
import numpy as np
import laspy
import rasterio
import pdal

# ========= USER CONFIG =========
REF_RASTER = Path(" /media/laura/laura_usb/qgis/calperumResearch/site2_1_DD0011/orthomosaic/dsm/RGB_dsm_3cm.tif")
LIDAR_IN = Path("/media/laura/laura_usb/qgis/calperumResearch/site2_1_DD0011/lidar_l2/cloud2d765c57b8471e2c.las")
RGB_IN     = Path("/media/laura/laura_usb/uas_data/DD0011/lidar/SASMDD0011_rgb_pc.las")

OUT_DIR    = Path("/media/laura/laura_usb/qgis/calperumResearch/site2_1_DD0011/lidar_l2/thinned_voxel_auto_3cm")
OUT_DIR.mkdir(parents=True, exist_ok=True)


# Target spacing logic
TARGET = 0.030      # 3 cm
TOL    = 0.0015     # accept up to 3.15 cm as "close enough"
THREE, TWO = 0.030, 0.020

# ========= Helpers =========
def read_raster_bounds(ref_raster: Path):
    with rasterio.open(ref_raster) as src:
        left, bottom, right, top = src.bounds
        crs = src.crs
    return left, bottom, right, top, crs

def snap_floor(v, step, origin):
    return origin + math.floor((v - origin) / step) * step

def snap_ceil(v, step, origin):
    return origin + math.ceil((v - origin) / step) * step

def pdal_run(pipe_json):
    p = pdal.Pipeline(json.dumps(pipe_json))
    p.execute()

def measure_density_xy(las_path: Path):
    las = laspy.read(str(las_path))
    x, y = las.x, las.y
    n = len(x)
    if n == 0:
        return 0.0, float("nan")
    xmin, xmax = float(x.min()), float(x.max())
    ymin, ymax = float(y.min()), float(y.max())
    area = max((xmax - xmin) * (ymax - ymin), 1e-12)
    dens = n / area
    spacing = (1.0 / dens) ** 0.5 if dens > 0 else float("nan")
    return dens, spacing

def pts_per_5cm(density):
    return density * (0.05**2)

def crop_only(src_las: Path, dst_las: Path, bounds_str: str):
    pipe = {
        "pipeline": [
            str(src_las),
            {"type": "filters.crop", "bounds": bounds_str},
            {"type": "writers.las", "filename": str(dst_las)}
        ]
    }
    pdal_run(pipe)

def crop_then_sample(src_las: Path, dst_las: Path, cell: float, origin_x: float, origin_y: float, bounds_str: str):
    pipe = {
        "pipeline": [
            str(src_las),
            {"type": "filters.crop", "bounds": bounds_str},
            {"type": "filters.sample",
             "cell": cell,
             "origin_x": origin_x,
             "origin_y": origin_y},
            {"type": "writers.las", "filename": str(dst_las)}
        ]
    }
    pdal_run(pipe)

def closer_to_3cm(spacing):
    return abs(spacing - THREE) <= abs(spacing - TWO)

def dynamic_cells_from_spacing(s0):
    """
    Build a candidate ladder around current spacing to land near 3 cm.
    Denser clouds (smaller s0) get smaller cells; sparser ones get larger.
    """
    # Base ladders by tier (can be tweaked)
    if s0 <= 0.0185:                      # super dense (~>3000–4000 pts/m²)
        ladder = [0.017, 0.018, 0.019, 0.020, 0.0225, 0.025, 0.0275, 0.030]
    elif s0 <= 0.022:                     # very dense (~2000–3000)
        ladder = [0.019, 0.020, 0.021, 0.0225, 0.025, 0.0275, 0.030]
    elif s0 <= 0.026:                     # dense (~1400–2000)
        ladder = [0.020, 0.0225, 0.025, 0.0275, 0.030]
    elif s0 <= 0.0315:                    # near target (~1000–1200)
        ladder = [0.030]                  # likely skip thinning anyway
    else:                                 # sparser than target
        ladder = [0.030, 0.032, 0.035]    # keep larger cells; mostly no-op
    # Ensure uniqueness and bounds
    ladder = sorted({c for c in ladder if 0.015 <= c <= 0.04})
    return ladder

def choose_best_cell(src_las: Path, tmp_dir: Path, cells, origin_x, origin_y, bounds_str):
    """
    Try cells, measure spacing, prefer spacing <= TARGET+TOL and >= TARGET,
    then closest to TARGET, then denser. Else pick smallest spacing achieved.
    Returns (best_cell, best_path, density, spacing).
    """
    results = []
    for cell in cells:
        tmp_out = tmp_dir / f"{src_las.stem}_thin_{int(round(cell*1000))}mm_tmp.las"
        crop_then_sample(src_las, tmp_out, cell, origin_x, origin_y, bounds_str)
        dens, spacing = measure_density_xy(tmp_out)
        # fallback if empty/NaN
        if (dens == 0) or (not np.isfinite(spacing)):
            crop_only(src_las, tmp_out, bounds_str)
            dens, spacing = measure_density_xy(tmp_out)
        results.append((cell, tmp_out, dens, spacing))
        print(f"   Tried CELL={cell:.3f} → density={dens:.2f} pts/m², spacing={spacing:.4f} m, ~{pts_per_5cm(dens):.2f} pts/5cm px")

    good = [r for r in results if r[3] <= (TARGET + TOL)]
    if good:
        good.sort(key=lambda r: (0 if r[3] >= TARGET else 1,
                                 abs(r[3] - TARGET),
                                 -r[2]))
        return good[0]
    results.sort(key=lambda r: r[3])  # otherwise smallest spacing
    return results[0]

# ========= MAIN =========
# Raster → anchor & bounds
left, bottom, right, top, _ = read_raster_bounds(REF_RASTER)
origin_x, origin_y = float(left), float(bottom)

# Snap crop rect to 3 cm lattice (aligns rectangle with raster grid)
xmin = snap_floor(left,  TARGET, origin_x)
ymin = snap_floor(bottom, TARGET, origin_y)
xmax = snap_ceil(right,   TARGET, origin_x)
ymax = snap_ceil(top,     TARGET, origin_y)
BOUNDS = f"([{xmin:.3f},{xmax:.3f}],[{ymin:.3f},{ymax:.3f}])"

print(f"Ref raster bounds: left={left:.3f}, right={right:.3f}, bottom={bottom:.3f}, top={top:.3f}")
print(f"Using voxel origin XY from raster: origin_x={origin_x:.3f}, origin_y={origin_y:.3f}")
print(f"Crop bounds (snapped to 3cm lattice): {BOUNDS}")

def process_one(src: Path, label: str) -> Path:
    print(f"\n— {label} —")
    d0, s0 = measure_density_xy(src)
    print(f"Original density: {d0:.2f} pts/m² → spacing ~ {s0:.4f} m (~{pts_per_5cm(d0):.2f} pts/5cm px)")

    # Skip only if: closer to 3 cm than 2 cm AND within 3 cm + tolerance
    if closer_to_3cm(s0) and s0 <= (THREE + TOL):
        out_path = OUT_DIR / f"{src.stem}_cropped_only.las"
        crop_only(src, out_path, BOUNDS)
        d1, s1 = measure_density_xy(out_path)
        print("✅ Closer to 3 cm → SKIP thinning (crop only)")
        print(f"Output: {out_path}")
        print(f"Post-crop density: {d1:.2f} pts/m² → spacing ~ {s1:.4f} m (~{pts_per_5cm(d1):.2f} pts/5cm px)")
        return out_path

    # Otherwise build a dynamic candidate list and choose best toward 3 cm
    cells = dynamic_cells_from_spacing(s0)
    print(f"Candidates based on current spacing ({s0:.4f} m): {', '.join(f'{c:.3f}' for c in cells)}")
    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        best_cell, best_tmp, d1, s1 = choose_best_cell(
            src, td_path, cells, origin_x, origin_y, BOUNDS
        )
        out_path = OUT_DIR / f"{src.stem}_thin{int(round(best_cell*1000))}mm_from_ref.las"
        shutil.move(str(best_tmp), str(out_path))
        print(f"✅ Thinned (CELL={best_cell:.3f})")
        print(f"Output: {out_path}")
        print(f"Post-thin density: {d1:.2f} pts/m² → spacing ~ {s1:.4f} m (~{pts_per_5cm(d1):.2f} pts/5cm px)")
        return out_path

lidar_out = process_one(LIDAR_IN, "LiDAR")
rgb_out   = process_one(RGB_IN,   "RGB SfM")

print("\nDone.")
# %%
