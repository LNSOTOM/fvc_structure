#%%
#!/usr/bin/env python3
from pathlib import Path
import numpy as np
import laspy
import rasterio
from rasterio.transform import rowcol
import pandas as pd
import matplotlib.pyplot as plt

# -----------------------
# Inputs
# -----------------------
las_file = Path(
    "/media/laura/laura_usb/qgis/calperumResearch/site1_1_DD0001/lidar/clipped/z_norm/reduced_noise/"
    "triple_AusGeoid_tiles_multispectral.26_HAG_delaunay_noiseout.laz"
)
mask_file = Path(
    "/media/laura/Extreme SSD/qgis/calperumResearch/site1_1_DD0001/inputs/masks/tiles_3072/raw/fvc_class/"
    "annotation_raster/mask_fvc_3072.26.tif"
)
IGNORE_LABELS = {255, -9999, -1}  # adjust to your mask's nodata

# -----------------------
# Helper to find Z_norm
# -----------------------
def find_z_norm_dim(las) -> tuple[str, np.ndarray]:
    extra = las.point_format.extra_dimensions or []
    info = []
    for ed in extra:
        name = ed.name
        desc = getattr(ed, "description", "") or ""
        vals = las[name].astype(np.float64)
        vmin, vmax = float(np.nanmin(vals)), float(np.nanmax(vals))
        info.append((name, desc, vmin, vmax, vals))

    if info:
        print("\nExtra dimensions in LAS:")
        for name, desc, vmin, vmax, _ in info:
            print(f"  - name='{name}', desc='{desc}', range=[{vmin:.3f}, {vmax:.3f}]")
    else:
        print("\nNo extra dimensions present in this LAS.")

    targets = ("z_norm", "hag", "heightaboveground", "hag_delaunay")
    for name, desc, vmin, vmax, vals in info:
        if any(t in name.lower() for t in targets) or any(t in desc.lower() for t in targets):
            return name, vals

    for name, desc, vmin, vmax, vals in info:
        if (vmax > 0.0) and (vmax <= 20.0) and (vmin >= -2.0):
            return name, vals

    print("WARNING: Could not find a Z_norm-like extra dimension; falling back to las.z.")
    return "las.z (fallback)", las.z.astype(np.float64)

# -----------------------
# Read LAS and pick Z_norm
# -----------------------
las = laspy.read(las_file)
z_name, z_norm = find_z_norm_dim(las)

xy = np.vstack((las.x, las.y)).T.astype(np.float64)
points = np.column_stack([xy, z_norm])   # [X, Y, Z_norm]

print(f"\nUsing Z source: {z_name}")
print("las.z range (raw):", float(las.z.min()), "→", float(las.z.max()))
print("Z_norm range     :", float(z_norm.min()), "→", float(z_norm.max()))

# -----------------------
# Load mask & clip to extent
# -----------------------
with rasterio.open(mask_file) as src:
    mask = src.read(1)
    transform = src.transform
    bounds = src.bounds

inside = (
    (points[:,0] >= bounds.left) & (points[:,0] <= bounds.right) &
    (points[:,1] >= bounds.bottom) & (points[:,1] <= bounds.top)
)
points_inside = points[inside]

rows, cols = rowcol(transform, points_inside[:,0], points_inside[:,1], op=float)
rows = rows.astype(int); cols = cols.astype(int)

valid = (
    (rows >= 0) & (rows < mask.shape[0]) &
    (cols >= 0) & (cols < mask.shape[1])
)
points_valid = points_inside[valid]
labels = mask[rows[valid], cols[valid]]

# Drop nodata labels if configured
if IGNORE_LABELS:
    keep = ~np.isin(labels, list(IGNORE_LABELS))
    points_valid = points_valid[keep]
    labels = labels[keep]

# -----------------------
# Save [X, Y, Z_norm, Label]
# -----------------------
xyz_labels = np.column_stack([points_valid, labels])
np.save("xyz_norm_with_labels.npy", xyz_labels)

print("\nTotals:")
print("  Total LAS points      :", len(las.x))
print("  Inside raster bounds  :", len(points_inside))
print("  With valid mask label :", len(points_valid))
print("  Unique labels (kept)  :", np.unique(labels))
print("  Sample rows:\n", xyz_labels[:5])

# =======================
# Build DataFrame for test 1
# =======================
df = pd.DataFrame(xyz_labels, columns=["X", "Y", "Z_norm", "Label"])

# Cast Label to integer (nullable) to avoid float NaNs
df["Label"] = df["Label"].astype("Int32")

# Inspect
print("\nDataFrame head:")
print(df.head())
print("\nUnique labels:", df["Label"].unique())

# =======================
# Stats
# =======================
print("\nCounts per label:")
print(df["Label"].value_counts(dropna=False))

print("\nZ_norm overall:")
print(df["Z_norm"].describe())

print("\nZ_norm by label:")
print(df.groupby("Label")["Z_norm"].describe())

# =======================
# Save outputs
# =======================
csv_path = Path("xyz_norm_with_labels.csv")
df.to_csv(csv_path, index=False)
print(f"\nSaved CSV: {csv_path}")

# =======================
# Optional Quick Plots
# =======================
plt.figure(figsize=(6,4))
df["Z_norm"].hist(bins=50, edgecolor="black")
plt.xlabel("Z_norm (m)")
plt.ylabel("Frequency")
plt.title("Distribution of normalised heights")
plt.show()

df.boxplot(column="Z_norm", by="Label", grid=False, figsize=(8,5))
plt.ylabel("Z_norm (m)")
plt.title("Height distribution per label")
plt.suptitle("")
plt.show()





# %%
##tets 2
#!/usr/bin/env python3
import numpy as np
import laspy
import rasterio
from rasterio.transform import rowcol
from pathlib import Path

# =======================
# Inputs
# =======================
las_file = Path("/media/laura/laura_usb/qgis/calperumResearch/site1_1_DD0001/lidar/clipped/z_norm/reduced_noise/triple_AusGeoid_tiles_multispectral.26_HAG_delaunay_noiseout.laz")

mask_file = Path("/media/laura/Extreme SSD/qgis/calperumResearch/site1_1_DD0001/inputs/masks/tiles_3072/raw/fvc_class/annotation_raster/mask_fvc_3072.26.tif")

voxel_size = 0.05  # 5 cm

# =======================
# Load LiDAR points
# =======================
las = laspy.read(las_file)
points = np.vstack((las.x, las.y, las.z)).T  # (N,3)

# =======================
# Load mask raster
# =======================
with rasterio.open(mask_file) as src:
    mask = src.read(1)
    transform = src.transform
    bounds = src.bounds

# =======================
# Keep only points inside raster bounds
# =======================
inside = (
    (points[:,0] >= bounds.left) & (points[:,0] <= bounds.right) &
    (points[:,1] >= bounds.bottom) & (points[:,1] <= bounds.top)
)
points_inside = points[inside]

# =======================
# Map LiDAR XY to raster indices
# =======================
rows, cols = rowcol(transform, points_inside[:,0], points_inside[:,1], op=float)
rows = rows.astype(int)
cols = cols.astype(int)

valid = (
    (rows >= 0) & (rows < mask.shape[0]) &
    (cols >= 0) & (cols < mask.shape[1])
)
points_valid = points_inside[valid]
rows_valid = rows[valid]
cols_valid = cols[valid]

# =======================
# Extract mask labels
# =======================
labels = mask[rows_valid, cols_valid]

# =======================
# Build array: [X, Y, Z_norm, Label]
# =======================
xyz_labels = np.column_stack([points_valid, labels])
np.save("xyz_norm_with_labels.npy", xyz_labels)

# =======================
# Voxel grid occupancy
# =======================
x_min, y_min, z_min = points_valid.min(axis=0)
x_max, y_max, z_max = points_valid.max(axis=0)

nx = int(np.ceil((x_max - x_min) / voxel_size))
ny = int(np.ceil((y_max - y_min) / voxel_size))
nz = int(np.ceil((z_max - z_min) / voxel_size))

ix = ((points_valid[:,0] - x_min) / voxel_size).astype(int)
iy = ((points_valid[:,1] - y_min) / voxel_size).astype(int)
iz = ((points_valid[:,2] - z_min) / voxel_size).astype(int)

occupancy = np.zeros((nx, ny, nz), dtype=np.uint8)
occupancy[ix, iy, iz] = 1

# =======================
# Print stats
# =======================
print("Total LiDAR points:", len(points))
print("Inside raster bounds:", len(points_inside))
print("With valid mask label:", len(points_valid))
print("Unique labels in overlap:", np.unique(labels))

print("Z range (norm):", points_valid[:,2].min(), "→", points_valid[:,2].max())
print("Voxel grid shape:", occupancy.shape)
print("Occupied voxels:", int(occupancy.sum()))

# Small preview
print("\nSample [X, Y, Z_norm, Label]:\n", xyz_labels[:10])


#%%
### build datramfe for test 2
import numpy as np
import pandas as pd

# Load the npy file (point-level table)
xyz_labels = np.load("xyz_norm_with_labels.npy")

# Build DataFrame
df = pd.DataFrame(xyz_labels, columns=["X", "Y", "Z_norm", "Label"])

# Inspect
print(df.head())
print("\nUnique labels:", df["Label"].unique())

# Quick stats
print("\nCounts per label:")
print(df["Label"].value_counts(dropna=False))

# Save to CSV if needed
df.to_csv("xyz_norm_with_labels.csv", index=False)



# %%
##test 3
#!/usr/bin/env python3
import numpy as np
import laspy
import rasterio
from rasterio.transform import rowcol
from pathlib import Path
import json

# ---------------------------
# Inputs & params
# ---------------------------
las_file  = Path("/media/laura/laura_usb/qgis/calperumResearch/site1_1_DD0001/lidar/clipped/z_norm/reduced_noise/triple_AusGeoid_tiles_multispectral.26_HAG_delaunay_noiseout.laz")
mask_file = Path("/media/laura/Extreme SSD/qgis/calperumResearch/site1_1_DD0001/inputs/masks/tiles_3072/raw/fvc_class/annotation_raster/mask_fvc_3072.26.tif")

voxel_size = 0.05                 # 5 cm
min_points_per_voxel = 1          # ≥N points -> occupied
ignore_labels = {255, -9999, -1}  # mask nodata codes (edit as needed)
use_tie_break = "highest_count"    # "highest_count" | "max_z" | "min_z"
snap_to_raster_origin = True       # make voxel grid align to raster pixels

# ---------------------------
# Load LiDAR and mask
# ---------------------------
las = laspy.read(las_file)
pts = np.vstack((las.x, las.y, las.z)).T  # (N,3) already HAG

with rasterio.open(mask_file) as src:
    mask = src.read(1)
    tfm = src.transform
    b = src.bounds
    pix_w = tfm.a     # pixel width (positive)
    pix_h = -tfm.e    # pixel height (positive)
    ras_origin_x = tfm.c
    ras_origin_y = tfm.f

# Keep only points inside raster bounds
inside = ((pts[:,0] >= b.left) & (pts[:,0] <= b.right) &
          (pts[:,1] >= b.bottom) & (pts[:,1] <= b.top))
pts = pts[inside]

# Map LiDAR XY to raster row/col, pull labels
rows, cols = rowcol(tfm, pts[:,0], pts[:,1], op=float)
rows = rows.astype(np.int64); cols = cols.astype(np.int64)
valid = ((rows >= 0) & (rows < mask.shape[0]) &
         (cols >= 0) & (cols < mask.shape[1]))
pts   = pts[valid]
rows  = rows[valid]; cols = cols[valid]
labels = mask[rows, cols]

# Drop nodata labels if desired
if ignore_labels:
    ok = ~np.isin(labels, list(ignore_labels))
    pts = pts[ok]; labels = labels[ok]

if pts.size == 0:
    raise RuntimeError("No valid points overlap the raster with usable labels.")

# ---------------------------
# Build voxel grid
# ---------------------------
# Optional: snap XY grid origin to raster origin so tiles interlock
if snap_to_raster_origin:
    # snap to a multiple of voxel_size relative to raster origin
    x_min_raw = pts[:,0].min()
    y_min_raw = pts[:,1].min()
    x_min = ras_origin_x + np.floor((x_min_raw - ras_origin_x) / voxel_size) * voxel_size
    y_min = ras_origin_y + np.floor((y_min_raw - ras_origin_y) / voxel_size) * voxel_size
else:
    x_min, y_min = pts[:,0].min(), pts[:,1].min()

z_min = pts[:,2].min()
x_max, y_max, z_max = pts[:,0].max(), pts[:,1].max(), pts[:,2].max()

nx = int(np.ceil((x_max - x_min) / voxel_size))
ny = int(np.ceil((y_max - y_min) / voxel_size))
nz = int(np.ceil((z_max - z_min) / voxel_size))

ix = ((pts[:,0] - x_min) / voxel_size).astype(np.int64)
iy = ((pts[:,1] - y_min) / voxel_size).astype(np.int64)
iz = ((pts[:,2] - z_min) / voxel_size).astype(np.int64)

# Clamp (numerical safety)
ix = np.clip(ix, 0, nx-1); iy = np.clip(iy, 0, ny-1); iz = np.clip(iz, 0, nz-1)

# ---------------------------
# Occupancy & dominant label
# ---------------------------
# Flattened voxel id in Fortran order (ix + nx*(iy + ny*iz))
linear = ix + nx * (iy + ny * iz)

# Unique voxels and inverse index per point -> voxel
unique_vox, inv = np.unique(linear, return_inverse=True)
inv = inv.astype(np.int64)  # << FIX: ensure integer dtype

# counts per voxel
voxel_counts = np.bincount(inv)
V = int(unique_vox.size)

# Prepare volumes
occ        = np.zeros((nx, ny, nz), dtype=np.uint8)
count_vol  = np.zeros((nx, ny, nz), dtype=np.uint32)
label_vol  = np.full((nx, ny, nz), fill_value=-1, dtype=np.int32)  # -1 for empty

# Mark occupancy with threshold
occupied_mask = voxel_counts >= min_points_per_voxel
occ_flat_idx  = unique_vox[occupied_mask]
occ[np.unravel_index(occ_flat_idx, (nx, ny, nz), order="F")] = 1

# Fill point counts
count_vol[np.unravel_index(unique_vox, (nx, ny, nz), order="F")] = voxel_counts

# -------- Majority label per voxel (robust integer mapping) --------
uniq_labels = np.unique(labels)                         # sorted ascending
label_idx = np.searchsorted(uniq_labels, labels)       # int positions
label_idx = label_idx.astype(np.int64)                 # << FIX: ensure integer dtype
K = int(uniq_labels.size)

# Build class counts [V, K]
class_counts = np.zeros((V, K), dtype=np.uint32)
np.add.at(class_counts, (inv, label_idx), 1)           # << both indices are ints now

# Choose dominant class per voxel
dom_idx = class_counts.argmax(axis=1)

if use_tie_break in ("max_z", "min_z"):
    maxc = class_counts.max(axis=1)
    ties = (class_counts == maxc[:, None])
    if np.any(ties.sum(axis=1) > 1):
        z_stat = np.full((V, K), np.nan, dtype=np.float64)
        # per-class z stat (max/min per voxel)
        for k in range(K):
            k_mask = (label_idx == k)
            if not np.any(k_mask): continue
            inv_k  = inv[k_mask]
            z_k    = pts[k_mask, 2]
            if use_tie_break == "max_z":
                order = np.argsort(z_k)           # last is max
            else:
                order = np.argsort(-z_k)          # last is min
            inv_k  = inv_k[order]
            z_k    = z_k[order]
            z_stat[inv_k, k] = z_k                # last write wins
        # resolve ties
        for v in np.where(ties.sum(axis=1) > 1)[0]:
            tie_classes = np.where(ties[v])[0]
            if use_tie_break == "max_z":
                best = tie_classes[np.nanargmax(z_stat[v, tie_classes])]
            else:
                best = tie_classes[np.nanargmin(z_stat[v, tie_classes])]
            dom_idx[v] = int(best)

dom_label = uniq_labels[dom_idx].astype(np.int32)

# Write labels only to voxels that exist; keep -1 elsewhere
label_vol[np.unravel_index(unique_vox, (nx, ny, nz), order="F")] = dom_label
label_vol[occ == 0] = -1  # enforce nodata on empty

# ---------------------------
# Stats & save
# ---------------------------
print("Total LiDAR points:", las.header.point_count)
print("Used points (inside & valid labels):", pts.shape[0])
print("Unique labels considered:", uniq_labels.tolist())
print("Z range (norm):", float(pts[:,2].min()), "→", float(pts[:,2].max()))
print("Voxel grid shape:", (nx, ny, nz))
print("Occupied voxels:", int(occ.sum()))
if occupied_mask.any():
    print("Min/median/mean points per occupied voxel:",
          int(voxel_counts[occupied_mask].min()), "/",
          float(np.median(voxel_counts[occupied_mask])), "/",
          float(voxel_counts[occupied_mask].mean()))
else:
    print("Min/median/mean points per occupied voxel: 0 / 0 / 0")

# Save arrays
np.save("voxel_occupancy.npy", occ)               # uint8 [nx,ny,nz]
np.save("voxel_label_dominant.npy", label_vol)    # int32 [-1 empty]
np.save("voxel_point_counts.npy", count_vol)      # uint32 [nx,ny,nz]
np.save("xyz_norm_with_labels.npy", np.column_stack([pts, labels]))  # [x,y,z,label]

# Save tiny metadata for world coords reconstruction
meta = {
    "x_min": float(x_min), "y_min": float(y_min), "z_min": float(z_min),
    "voxel_size": float(voxel_size),
    "shape": [nx, ny, nz],
    "crs_like_mask": True,
    "mask_pixel_size": [float(pix_w), float(pix_h)],
    "snapped_to_raster_origin": bool(snap_to_raster_origin),
}
Path("voxel_meta.json").write_text(json.dumps(meta, indent=2))

# %%
##test 4
#!/usr/bin/env python3
import numpy as np
import laspy
import rasterio
from rasterio.transform import rowcol
from pathlib import Path
import json

# ---------------------------
# Inputs & params
# ---------------------------
las_file  = Path("/media/laura/laura_usb/qgis/calperumResearch/site1_1_DD0001/lidar/clipped/z_norm/reduced_noise/triple_AusGeoid_tiles_multispectral.26_HAG_delaunay_noiseout.laz")
mask_file = Path("/media/laura/Extreme SSD/qgis/calperumResearch/site1_1_DD0001/inputs/masks/tiles_3072/raw/fvc_class/annotation_raster/mask_fvc_3072.26.tif")

voxel_size = 0.05     # 5 cm
# 👉 Adjust to reduce noise: e.g. set to 3 so a voxel is "occupied" only if it has ≥3 points
min_points_per_voxel = 1  

# 👉 Adjust to match your mask’s nodata values
ignore_labels = {255, -9999, -1}  

# "highest_count" | "max_z" | "min_z" (tie-break rules for dominant class)
use_tie_break = "highest_count"    

# 👉 If True, voxel grid is snapped to raster origin
# Replace with a fixed (x0,y0,z0) if you want EPSG:7854-aligned 5 cm global grid
snap_to_raster_origin = True       

# ---------------------------
# Load LiDAR and mask
# ---------------------------
las = laspy.read(las_file)
pts = np.vstack((las.x, las.y, las.z)).T  # (N,3) already HAG

with rasterio.open(mask_file) as src:
    mask = src.read(1)
    tfm = src.transform
    b = src.bounds
    pix_w = tfm.a
    pix_h = -tfm.e
    ras_origin_x = tfm.c
    ras_origin_y = tfm.f

# Keep only points inside raster bounds
inside = ((pts[:,0] >= b.left) & (pts[:,0] <= b.right) &
          (pts[:,1] >= b.bottom) & (pts[:,1] <= b.top))
pts = pts[inside]

# Map LiDAR XY to raster row/col, pull labels
rows, cols = rowcol(tfm, pts[:,0], pts[:,1], op=float)
rows = rows.astype(np.int64); cols = cols.astype(np.int64)
valid = ((rows >= 0) & (rows < mask.shape[0]) &
         (cols >= 0) & (cols < mask.shape[1]))
pts   = pts[valid]
rows  = rows[valid]; cols = cols[valid]
labels = mask[rows, cols]

# Drop nodata labels if desired
if ignore_labels:
    ok = ~np.isin(labels, list(ignore_labels))
    pts = pts[ok]; labels = labels[ok]

if pts.size == 0:
    raise RuntimeError("No valid points overlap the raster with usable labels.")

# ---------------------------
# Build voxel grid
# ---------------------------
if snap_to_raster_origin:
    # Snap XY grid origin to raster origin for interlocking tiles
    x_min_raw = pts[:,0].min()
    y_min_raw = pts[:,1].min()
    x_min = ras_origin_x + np.floor((x_min_raw - ras_origin_x) / voxel_size) * voxel_size
    y_min = ras_origin_y + np.floor((y_min_raw - ras_origin_y) / voxel_size) * voxel_size
else:
    x_min, y_min = pts[:,0].min(), pts[:,1].min()

# 👉 If you want a fixed global origin, replace x_min, y_min, z_min here:
z_min = pts[:,2].min()
x_max, y_max, z_max = pts[:,0].max(), pts[:,1].max(), pts[:,2].max()

nx = int(np.ceil((x_max - x_min) / voxel_size))
ny = int(np.ceil((y_max - y_min) / voxel_size))
nz = int(np.ceil((z_max - z_min) / voxel_size))

ix = ((pts[:,0] - x_min) / voxel_size).astype(np.int64)
iy = ((pts[:,1] - y_min) / voxel_size).astype(np.int64)
iz = ((pts[:,2] - z_min) / voxel_size).astype(np.int64)

# Clamp (numerical safety)
ix = np.clip(ix, 0, nx-1); iy = np.clip(iy, 0, ny-1); iz = np.clip(iz, 0, nz-1)

# ---------------------------
# Occupancy & dominant label
# ---------------------------
linear = ix + nx * (iy + ny * iz)
unique_vox, inv = np.unique(linear, return_inverse=True)
inv = inv.astype(np.int64)
voxel_counts = np.bincount(inv)
V = int(unique_vox.size)

occ        = np.zeros((nx, ny, nz), dtype=np.uint8)
count_vol  = np.zeros((nx, ny, nz), dtype=np.uint32)
label_vol  = np.full((nx, ny, nz), fill_value=-1, dtype=np.int32)

occupied_mask = voxel_counts >= min_points_per_voxel
occ_flat_idx  = unique_vox[occupied_mask]
occ[np.unravel_index(occ_flat_idx, (nx, ny, nz), order="F")] = 1

count_vol[np.unravel_index(unique_vox, (nx, ny, nz), order="F")] = voxel_counts

# Majority label per voxel
uniq_labels = np.unique(labels)
label_idx = np.searchsorted(uniq_labels, labels).astype(np.int64)
K = int(uniq_labels.size)
class_counts = np.zeros((V, K), dtype=np.uint32)
np.add.at(class_counts, (inv, label_idx), 1)
dom_idx = class_counts.argmax(axis=1)

dom_label = uniq_labels[dom_idx].astype(np.int32)
label_vol[np.unravel_index(unique_vox, (nx, ny, nz), order="F")] = dom_label
label_vol[occ == 0] = -1

# ---------------------------
# Stats & save
# ---------------------------
print("Total LiDAR points:", las.header.point_count)
print("Used points (inside & valid labels):", pts.shape[0])
print("Unique labels considered:", uniq_labels.tolist())
print("Z range (norm):", float(pts[:,2].min()), "→", float(pts[:,2].max()))
print("Voxel grid shape:", (nx, ny, nz))
print("Occupied voxels:", int(occ.sum()))
if occupied_mask.any():
    print("Min/median/mean points per occupied voxel:",
          int(voxel_counts[occupied_mask].min()), "/",
          float(np.median(voxel_counts[occupied_mask])), "/",
          float(voxel_counts[occupied_mask].mean()))
else:
    print("Min/median/mean points per occupied voxel: 0 / 0 / 0")

np.save("voxel_occupancy.npy", occ)
np.save("voxel_label_dominant.npy", label_vol)
np.save("voxel_point_counts.npy", count_vol)
np.save("xyz_norm_with_labels.npy", np.column_stack([pts, labels]))

meta = {
    "x_min": float(x_min), "y_min": float(y_min), "z_min": float(z_min),
    "voxel_size": float(voxel_size),
    "shape": [nx, ny, nz],
    "crs_like_mask": True,
    "mask_pixel_size": [float(pix_w), float(pix_h)],
    "snapped_to_raster_origin": bool(snap_to_raster_origin),
    "min_points_per_voxel": int(min_points_per_voxel),
    "ignore_labels": list(ignore_labels),
}
Path("voxel_meta.json").write_text(json.dumps(meta, indent=2))

# %%
###test 5
#!/usr/bin/env python3
import numpy as np
import laspy
import rasterio
from rasterio.transform import rowcol, Affine
from rasterio.crs import CRS
from pathlib import Path
import json

# ---------------------------
# Inputs & params
# ---------------------------
las_file  = Path("/media/laura/laura_usb/qgis/calperumResearch/site1_1_DD0001/lidar/clipped/z_norm/reduced_noise/triple_AusGeoid_tiles_multispectral.26_HAG_delaunay_noiseout.laz")
mask_file = Path("/media/laura/Extreme SSD/qgis/calperumResearch/site1_1_DD0001/inputs/masks/tiles_3072/raw/fvc_class/annotation_raster/mask_fvc_3072.26.tif")

voxel_size = 0.05     # 5 cm XY & Z voxel
min_points_per_voxel = 1
ignore_labels = {255, -9999, -1}
use_tie_break = "highest_count"    # "highest_count" | "max_z" | "min_z"
snap_to_raster_origin = True
# --- Z-slice export settings ---
slice_dz = 0.5  # meters per slice
occ_slices_tif   = Path("voxel_occ_slices.tif")
label_slices_tif = Path("voxel_label_slices.tif")

# ---------------------------
# Load LiDAR and mask
# ---------------------------
las = laspy.read(las_file)
pts = np.vstack((las.x, las.y, las.z)).T  # (N,3) already HAG

with rasterio.open(mask_file) as src:
    mask = src.read(1)
    tfm = src.transform
    b = src.bounds
    crs = src.crs if src.crs else CRS.from_epsg(7854)  # fallback if missing
    pix_w = tfm.a
    pix_h = -tfm.e
    ras_origin_x = tfm.c
    ras_origin_y = tfm.f

inside = ((pts[:,0] >= b.left) & (pts[:,0] <= b.right) &
          (pts[:,1] >= b.bottom) & (pts[:,1] <= b.top))
pts = pts[inside]

rows, cols = rowcol(tfm, pts[:,0], pts[:,1], op=float)
rows = rows.astype(np.int64); cols = cols.astype(np.int64)
valid = ((rows >= 0) & (rows < mask.shape[0]) &
         (cols >= 0) & (cols < mask.shape[1]))
pts   = pts[valid]
rows  = rows[valid]; cols = cols[valid]
labels = mask[rows, cols]

if ignore_labels:
    ok = ~np.isin(labels, list(ignore_labels))
    pts = pts[ok]; labels = labels[ok]

if pts.size == 0:
    raise RuntimeError("No valid points overlap the raster with usable labels.")

# ---------------------------
# Build voxel grid
# ---------------------------
if snap_to_raster_origin:
    x_min_raw = pts[:,0].min()
    y_min_raw = pts[:,1].min()
    x_min = ras_origin_x + np.floor((x_min_raw - ras_origin_x) / voxel_size) * voxel_size
    y_min = ras_origin_y + np.floor((y_min_raw - ras_origin_y) / voxel_size) * voxel_size
else:
    x_min, y_min = pts[:,0].min(), pts[:,1].min()

z_min = pts[:,2].min()
x_max, y_max, z_max = pts[:,0].max(), pts[:,1].max(), pts[:,2].max()

nx = int(np.ceil((x_max - x_min) / voxel_size))
ny = int(np.ceil((y_max - y_min) / voxel_size))
nz = int(np.ceil((z_max - z_min) / voxel_size))

ix = ((pts[:,0] - x_min) / voxel_size).astype(np.int64)
iy = ((pts[:,1] - y_min) / voxel_size).astype(np.int64)
iz = ((pts[:,2] - z_min) / voxel_size).astype(np.int64)

ix = np.clip(ix, 0, nx-1); iy = np.clip(iy, 0, ny-1); iz = np.clip(iz, 0, nz-1)

# ---------------------------
# Occupancy & dominant label
# ---------------------------
linear = ix + nx * (iy + ny * iz)
unique_vox, inv = np.unique(linear, return_inverse=True)
inv = inv.astype(np.int64)
voxel_counts = np.bincount(inv)
V = int(unique_vox.size)

occ        = np.zeros((nx, ny, nz), dtype=np.uint8)
count_vol  = np.zeros((nx, ny, nz), dtype=np.uint32)
label_vol  = np.full((nx, ny, nz), fill_value=-1, dtype=np.int32)

occupied_mask = voxel_counts >= min_points_per_voxel
occ_flat_idx  = unique_vox[occupied_mask]
occ[np.unravel_index(occ_flat_idx, (nx, ny, nz), order="F")] = 1

count_vol[np.unravel_index(unique_vox, (nx, ny, nz), order="F")] = voxel_counts

uniq_labels = np.unique(labels)
label_idx = np.searchsorted(uniq_labels, labels).astype(np.int64)
K = int(uniq_labels.size)
class_counts = np.zeros((V, K), dtype=np.uint32)
np.add.at(class_counts, (inv, label_idx), 1)
dom_idx = class_counts.argmax(axis=1)

# Optional tie-break by Z
if use_tie_break in ("max_z", "min_z"):
    maxc = class_counts.max(axis=1)
    ties = (class_counts == maxc[:, None])
    if np.any(ties.sum(axis=1) > 1):
        z_stat = np.full((V, K), np.nan, dtype=np.float64)
        for k in range(K):
            k_mask = (label_idx == k)
            if not np.any(k_mask): continue
            inv_k  = inv[k_mask]
            z_k    = pts[k_mask, 2]
            order = np.argsort(z_k) if use_tie_break == "max_z" else np.argsort(-z_k)
            inv_k  = inv_k[order]
            z_k    = z_k[order]
            z_stat[inv_k, k] = z_k
        for v in np.where(ties.sum(axis=1) > 1)[0]:
            tie_classes = np.where(ties[v])[0]
            best = tie_classes[np.nanargmax(z_stat[v, tie_classes])] if use_tie_break == "max_z" \
                   else tie_classes[np.nanargmin(z_stat[v, tie_classes])]
            dom_idx[v] = int(best)

dom_label = uniq_labels[dom_idx].astype(np.int32)
label_vol[np.unravel_index(unique_vox, (nx, ny, nz), order="F")] = dom_label
label_vol[occ == 0] = -1

# ---------------------------
# Stats
# ---------------------------
print("Total LiDAR points:", las.header.point_count)
print("Used points (inside & valid labels):", pts.shape[0])
print("Unique labels considered:", uniq_labels.tolist())
print("Z range (norm):", float(pts[:,2].min()), "→", float(pts[:,2].max()))
print("Voxel grid shape:", (nx, ny, nz))
print("Occupied voxels:", int(occ.sum()))
if occupied_mask.any():
    print("Min/median/mean points per occupied voxel:",
          int(voxel_counts[occupied_mask].min()), "/",
          float(np.median(voxel_counts[occupied_mask])), "/",
          float(voxel_counts[occupied_mask].mean()))
else:
    print("Min/median/mean points per occupied voxel: 0 / 0 / 0")

# ---------------------------
# Save NPYs (+ tiny meta)
# ---------------------------
np.save("voxel_occupancy.npy", occ)
np.save("voxel_label_dominant.npy", label_vol)
np.save("voxel_point_counts.npy", count_vol)
np.save("xyz_norm_with_labels.npy", np.column_stack([pts, labels]))
meta = {
    "x_min": float(x_min), "y_min": float(y_min), "z_min": float(z_min),
    "voxel_size": float(voxel_size),
    "shape": [nx, ny, nz],
    "crs_epsg_or_wkt": crs.to_string(),
    "snapped_to_raster_origin": bool(snap_to_raster_origin),
    "min_points_per_voxel": int(min_points_per_voxel),
    "ignore_labels": list(map(int, ignore_labels)),
}
Path("voxel_meta.json").write_text(json.dumps(meta, indent=2))

print("\nSaved NPYs:")
for p in ["voxel_occupancy.npy","voxel_label_dominant.npy","voxel_point_counts.npy","xyz_norm_with_labels.npy","voxel_meta.json"]:
    print(" -", p)

# ---------------------------
# GeoTIFF Z-slice export (0.5 m bands)
# ---------------------------
# Reorder to raster image order: rows=ny, cols=nx
occ_xyz   = np.transpose(occ, (1,0,2))        # (ny, nx, nz)
label_xyz = np.transpose(label_vol, (1,0,2))  # (ny, nx, nz)

# Map physical slice thickness to number of z-voxels
vz = voxel_size
vox_per_slice = max(1, int(round(slice_dz / vz)))
n_slices = int(np.ceil(nz / vox_per_slice))

# GeoTransform: top-left at (x_min, y_max)
y_max_world = y_min + ny * voxel_size
transform = Affine(voxel_size, 0.0, x_min,
                   0.0, -voxel_size, y_max_world)

# Build and write occupancy slices
with rasterio.open(
    occ_slices_tif,
    "w",
    driver="GTiff",
    height=ny,
    width=nx,
    count=n_slices,
    dtype=rasterio.uint8,
    crs=crs,
    transform=transform,
    nodata=0,
    compress="LZW",
) as dst_occ:

    for s in range(n_slices):
        z0 = s * vox_per_slice
        z1 = min((s+1) * vox_per_slice, nz)
        band_occ = (occ_xyz[:, :, z0:z1].any(axis=2)).astype(np.uint8)
        dst_occ.write(band_occ, s+1)

print(f"\nWrote occupancy slices GeoTIFF: {occ_slices_tif} (bands: {n_slices}, dz≈{vox_per_slice*voxel_size:.2f} m)")

# Build and write label slices (majority across z within each slice)
def majority_label_2d(lbl_cube_slice: np.ndarray, nodata_val: int = -1) -> np.ndarray:
    """
    lbl_cube_slice: (ny, nx, Zk) int labels; -1 = empty
    Returns a 2D array (ny, nx) with majority label per (y,x),
    ignoring -1. If all -1, returns -1 at that pixel.
    """
    ny_, nx_, zk = lbl_cube_slice.shape
    out = np.full((ny_, nx_), nodata_val, dtype=np.int32)

    # Flatten per pixel stacks
    flat = lbl_cube_slice.reshape(ny_*nx_, zk)
    # Mask out -1
    valid_mask = flat >= 0
    any_valid = valid_mask.any(axis=1)
    if not np.any(any_valid):
        return out

    # For pixels with any valid labels, compute per-pixel mode
    idxs = np.where(any_valid)[0]
    sub = flat[idxs]                # (M, zk)
    sub_mask = valid_mask[idxs]     # (M, zk)

    # Collect unique labels present in this slice to keep memory small
    labs_present = np.unique(sub[sub_mask])
    # Map to [0..L-1]
    lab_map = {lab:i for i,lab in enumerate(labs_present)}
    mapped = np.full_like(sub, fill_value=-1)
    # map only valid entries
    for i, lab in enumerate(labs_present):
        mapped[sub == lab] = i

    # Count per pixel
    L = labs_present.size
    counts = np.zeros((sub.shape[0], L), dtype=np.uint16)
    # For each column (z plane), add counts
    for k in range(sub.shape[1]):
        col = mapped[:, k]
        ok = col >= 0
        if not np.any(ok): 
            continue
        np.add.at(counts, (np.where(ok)[0], col[ok]), 1)

    # Mode index
    mode_idx = counts.argmax(axis=1)
    mode_lab = labs_present[mode_idx]

    # Write back to 2D out
    yx = np.unravel_index(idxs, (ny_, nx_))
    out[yx] = mode_lab.astype(np.int32)
    return out

with rasterio.open(
    label_slices_tif,
    "w",
    driver="GTiff",
    height=ny,
    width=nx,
    count=n_slices,
    dtype=rasterio.int32,
    crs=crs,
    transform=transform,
    nodata=-1,
    compress="LZW",
) as dst_lab:

    for s in range(n_slices):
        z0 = s * vox_per_slice
        z1 = min((s+1) * vox_per_slice, nz)
        cube = label_xyz[:, :, z0:z1]  # (ny, nx, zk)
        band_lab = majority_label_2d(cube, nodata_val=-1)
        dst_lab.write(band_lab, s+1)

print(f"Wrote label slices GeoTIFF: {label_slices_tif} (bands: {n_slices}, dz≈{vox_per_slice*voxel_size:.2f} m)")

# %%
