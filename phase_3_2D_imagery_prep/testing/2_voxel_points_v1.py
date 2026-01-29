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
    "/media/laura/laura_usb/qgis/calperumResearch/site1_1_DD0001/inputs/masks/tiles_3072/raw/fvc_class/annotation_raster/"
    "dowsample_5cm_nearest/mask_fvc_3072.26.tif"
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


#%%
# -----------------------
# Voxelization aligned to mask pixel size (5 cm inferred from mask)
# -----------------------
# 1) derive voxel size from mask pixels
voxel_xy = float(abs(transform.a))          # e.g., 0.05 m if mask is 5 cm
voxel_z  = voxel_xy                          # use same 5 cm in Z

# 2) snap XY origin to raster origin so tiles interlock
ras_x0, ras_y0 = transform.c, transform.f
X, Y, Z = points_valid[:,0], points_valid[:,1], points_valid[:,2]

x_min_raw, y_min_raw, z_min = X.min(), Y.min(), Z.min()
x_min = ras_x0 + np.floor((x_min_raw - ras_x0) / voxel_xy) * voxel_xy
y_min = ras_y0 + np.floor((y_min_raw - ras_y0) / voxel_xy) * voxel_xy

x_max, y_max, z_max = X.max(), Y.max(), Z.max()
nx = int(np.ceil((x_max - x_min) / voxel_xy))
ny = int(np.ceil((y_max - y_min) / voxel_xy))
nz = int(np.ceil((z_max - z_min) / voxel_z))

# 3) voxel indices (clipped for safety)
ix = np.clip(((X - x_min) / voxel_xy).astype(np.int64), 0, nx-1)
iy = np.clip(((Y - y_min) / voxel_xy).astype(np.int64), 0, ny-1)
iz = np.clip(((Z - z_min) / voxel_z ).astype(np.int64), 0, nz-1)

# 4) linearize and aggregate
lin = ix + nx * (iy + ny * iz)
uniq_vox, inv = np.unique(lin, return_inverse=True)
inv = inv.astype(np.int64)

# occupancy & counts
voxel_counts = np.bincount(inv)
occ = np.zeros((nx, ny, nz), dtype=np.uint8)
occ[np.unravel_index(uniq_vox, (nx,ny,nz), order="F")] = 1

count_vol = np.zeros((nx, ny, nz), dtype=np.uint32)
count_vol[np.unravel_index(uniq_vox, (nx,ny,nz), order="F")] = voxel_counts

# dominant label per voxel (majority vote)
uniq_labels = np.unique(labels)
lab_idx = np.searchsorted(uniq_labels, labels).astype(np.int64)
K = int(uniq_labels.size)
V = int(uniq_vox.size)

class_counts = np.zeros((V, K), dtype=np.uint32)
np.add.at(class_counts, (inv, lab_idx), 1)
dom_idx = class_counts.argmax(axis=1)
dom_label_flat = uniq_labels[dom_idx].astype(np.int32)

label_vol = np.full((nx, ny, nz), -1, dtype=np.int32)
label_vol[np.unravel_index(uniq_vox, (nx,ny,nz), order="F")] = dom_label_flat
label_vol[occ == 0] = -1

# 5) stats + save
print("\n--- Voxelization (aligned to mask) ---")
print(f"Voxel size XY: {voxel_xy:.3f} m, Z: {voxel_z:.3f} m")
print("Z range (norm):", float(Z.min()), "→", float(Z.max()))
print("Voxel grid shape:", (nx, ny, nz))
print("Occupied voxels:", int(occ.sum()))
print("Mean points / occupied voxel:", float(voxel_counts.mean()))

np.save("voxel_occupancy.npy", occ)
np.save("voxel_label_dominant.npy", label_vol)
np.save("voxel_point_counts.npy", count_vol)

# keep your point table too
np.save("xyz_norm_with_labels.npy", np.column_stack([points_valid, labels]))
print("Saved: voxel_occupancy.npy, voxel_label_dominant.npy, voxel_point_counts.npy, xyz_norm_with_labels.npy")

# %%
### imprve before - test 2
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
    "/media/laura/laura_usb/qgis/calperumResearch/site1_1_DD0001/inputs/masks/tiles_3072/raw/fvc_class/annotation_raster/"
    "dowsample_5cm_nearest/mask_fvc_3072.26.tif"
)

# Nodata + classes to exclude
IGNORE_LABELS = {255, -9999, -1}      # nodata codes
EXCLUDE_CLASSES = {0, 3}              # <- you asked to drop these

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
    for name, desc, *_rest in info:
        if any(t in name.lower() for t in targets) or any(t in (getattr(las.point_format, 'description', '') or '').lower() for t in targets):
            # prefer exact/substring name match
            if any(t in name.lower() for t in targets):
                return name, las[name].astype(np.float64)

    # heuristic: plausible HAG range
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
    nodata = src.nodata

# Clip to raster bounds
inside = (
    (points[:,0] >= bounds.left) & (points[:,0] <= bounds.right) &
    (points[:,1] >= bounds.bottom) & (points[:,1] <= bounds.top)
)
points_inside = points[inside]

# -----------------------
# Label sampling (nearest pixel center) — better than floor indexing
# -----------------------
with rasterio.open(mask_file) as src:
    samples = list(src.sample(points_inside[:, :2]))  # [(val,), ...]
labels = np.array([s[0] for s in samples])

# -----------------------
# Robust filtering: remove NaN, nodata, and classes 0 & 3
# -----------------------
to_exclude = set(IGNORE_LABELS)
if nodata is not None:
    to_exclude.add(nodata)
to_exclude |= EXCLUDE_CLASSES

if np.issubdtype(labels.dtype, np.floating):
    keep = (~np.isnan(labels)) & (~np.isin(labels, list(to_exclude)))
else:
    keep = ~np.isin(labels, list(to_exclude))

points_valid = points_inside[keep]
labels = labels[keep]

# Ensure labels are integers for downstream work
labels = labels.astype(np.int32, copy=False)

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
# Build DataFrame for QA
# =======================
df = pd.DataFrame(xyz_labels, columns=["X", "Y", "Z_norm", "Label"])
df["Label"] = df["Label"].astype("Int32")  # nullable ints for convenience

print("\nCounts per label:")
print(df["Label"].value_counts(dropna=False).sort_index())
print("\nZ_norm overall:")
print(df["Z_norm"].describe())
print("\nZ_norm by label:")
print(df.groupby("Label")["Z_norm"].describe())

csv_path = Path("xyz_norm_with_labels.csv")
df.to_csv(csv_path, index=False)
print(f"\nSaved CSV: {csv_path}")

# =======================
# Optional quick plots
# =======================
plt.figure(figsize=(6,4))
df["Z_norm"].hist(bins=50, edgecolor="black")
plt.xlabel("Z_norm (m)"); plt.ylabel("Frequency")
plt.title("Distribution of normalised heights")
plt.show()

df.boxplot(column="Z_norm", by="Label", grid=False, figsize=(8,5))
plt.ylabel("Z_norm (m)")
plt.title("Height distribution per label"); plt.suptitle("")
plt.show()

#%% -----------------------
# Voxelization aligned to mask pixel size (5 cm inferred from mask)
# -----------------------
voxel_xy = float(abs(transform.a))  # e.g. 0.05 m if mask is 5 cm
voxel_z  = voxel_xy

# Snap XY origin to raster origin so tiles interlock
ras_x0, ras_y0 = transform.c, transform.f
X, Y, Z = points_valid[:,0], points_valid[:,1], points_valid[:,2]

x_min = ras_x0 + np.floor((X.min() - ras_x0) / voxel_xy) * voxel_xy
y_min = ras_y0 + np.floor((Y.min() - ras_y0) / voxel_xy) * voxel_xy
z_min = Z.min()

nx = int(np.ceil((X.max() - x_min) / voxel_xy))
ny = int(np.ceil((Y.max() - y_min) / voxel_xy))
nz = int(np.ceil((Z.max() - z_min) / voxel_z))

ix = np.clip(((X - x_min) / voxel_xy).astype(np.int64), 0, nx-1)
iy = np.clip(((Y - y_min) / voxel_xy).astype(np.int64), 0, ny-1)
iz = np.clip(((Z - z_min) / voxel_z ).astype(np.int64), 0, nz-1)

lin = ix + nx * (iy + ny * iz)
uniq_vox, inv = np.unique(lin, return_inverse=True)
inv = inv.astype(np.int64)

voxel_counts = np.bincount(inv)
occ = np.zeros((nx, ny, nz), dtype=np.uint8)
occ[np.unravel_index(uniq_vox, (nx,ny,nz), order="F")] = 1

count_vol = np.zeros((nx, ny, nz), dtype=np.uint32)
count_vol[np.unravel_index(uniq_vox, (nx,ny,nz), order="F")] = voxel_counts

# --- dominant label per voxel (no NaNs now, and 0/3 already excluded) ---
uniq_lbls = np.unique(labels)
lab_idx = np.searchsorted(uniq_lbls, labels).astype(np.int64)
K = int(uniq_lbls.size); V = int(uniq_vox.size)

class_counts = np.zeros((V, K), dtype=np.uint32)
np.add.at(class_counts, (inv, lab_idx), 1)
dom_idx = class_counts.argmax(axis=1)
dom_label_flat = uniq_lbls[dom_idx].astype(np.int32)

label_vol = np.full((nx, ny, nz), -1, dtype=np.int32)
label_vol[np.unravel_index(uniq_vox, (nx,ny,nz), order="F")] = dom_label_flat
label_vol[occ == 0] = -1

print("\n--- Voxelization (aligned to mask) ---")
print(f"Voxel size XY: {voxel_xy:.3f} m, Z: {voxel_z:.3f} m")
print("Z range (norm):", float(Z.min()), "→", float(Z.max()))
print("Voxel grid shape:", (nx, ny, nz))
print("Occupied voxels:", int(occ.sum()))
print("Mean points / occupied voxel:", float(voxel_counts.mean()))
print("Unique labels used in voxels:", uniq_lbls.tolist())

np.save("voxel_occupancy.npy", occ)
np.save("voxel_label_dominant.npy", label_vol)
np.save("voxel_point_counts.npy", count_vol)
np.save("xyz_norm_with_labels.npy", np.column_stack([points_valid, labels]))
print("Saved: voxel_occupancy.npy, voxel_label_dominant.npy, voxel_point_counts.npy, xyz_norm_with_labels.npy")


#%%
# vis test 2
import numpy as np
import matplotlib.pyplot as plt

occ    = np.load("voxel_occupancy.npy")          # [nx,ny,nz] uint8
labels = np.load("voxel_label_dominant.npy")     # [nx,ny,nz] int32 (-1 empty)

# only voxels that are occupied and labeled
mask = (occ.astype(bool)) & (labels != -1)
nx, ny, nz = occ.shape

# ---------- tweak these ----------
# downsample steps (bigger = faster, coarser)
step_x, step_y, step_z = 6, 6, 3
# show full volume -> set z_slab=None; or a slice like (10, 60)
z_slab = None
# ---------------------------------

xs = slice(0, nx, step_x)
ys = slice(0, ny, step_y)
zs = slice(0, nz, step_z) if z_slab is None else slice(z_slab[0], min(z_slab[1], nz), step_z)

mask_ds   = mask[xs, ys, zs]
labels_ds = labels[xs, ys, zs]

if not np.any(mask_ds):
    print("Nothing to show after filtering/downsampling.")
else:
    # colors per label (tab20)
    uniq = np.unique(labels_ds[mask_ds])
    cmap = plt.cm.get_cmap("tab20", uniq.size)
    lut  = {lab: cmap(i) for i, lab in enumerate(uniq)}

    face = np.zeros((*labels_ds.shape, 4), float)
    face[..., 3] = 0.0
    for lab, rgba in lut.items():
        face[labels_ds == lab] = rgba

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection="3d")
    ax.voxels(mask_ds, facecolors=face, edgecolor="k", linewidth=0.1)
    ax.set_xlabel("X (voxel)")
    ax.set_ylabel("Y (voxel)")
    ax.set_zlabel("Z (voxel)")
    ax.set_title("Voxel grid (downsampled) — colored by dominant label")

    from matplotlib.patches import Patch
    ax.legend([Patch(facecolor=lut[l], edgecolor='k', label=f"Label {l}") for l in uniq],
              loc="upper left", bbox_to_anchor=(1.05, 1))
    plt.tight_layout(); plt.show()

# %%
# b) 
import json, numpy as np
import plotly.graph_objects as go

occ    = np.load("voxel_occupancy.npy")
labels = np.load("voxel_label_dominant.npy")
mask = (occ.astype(bool)) & (labels != -1)

# world coords (if available)
try:
    meta = json.loads(open("voxel_meta.json").read())
    x0, y0, z0 = meta["x_min"], meta["y_min"], meta["z_min"]
    vs = meta["voxel_size"]
except:
    x0 = y0 = z0 = 0.0; vs = 1.0

xs, ys, zs = np.where(mask)   # indices of visible voxels
labs = labels[xs, ys, zs]

# light downsampling for speed
keep = np.linspace(0, xs.size-1, min(xs.size, 120000)).astype(int)
xs, ys, zs, labs = xs[keep], ys[keep], zs[keep], labs[keep]

# cube corners
X = x0 + (xs + 0.5) * vs
Y = y0 + (ys + 0.5) * vs
Z = z0 + (zs + 0.5) * vs

fig = go.Figure()
uniq = np.unique(labs)
for i, lab in enumerate(uniq):
    sel = labs == lab
    fig.add_trace(go.Scatter3d(
        x=X[sel], y=Y[sel], z=Z[sel],
        mode="markers",
        marker=dict(size=2, opacity=0.35),
        name=f"Label {lab}"
    ))

fig.update_layout(scene=dict(
    xaxis_title="X (m)", yaxis_title="Y (m)", zaxis_title="Z (m)"
), title="Voxel centroids (colored by label)")
fig.show()

# %%
# c)
import numpy as np, matplotlib.pyplot as plt

labels = np.load("voxel_label_dominant.npy")  # [nx,ny,nz]
occ    = np.load("voxel_occupancy.npy")

# re-order to image layout (rows, cols, bands)
lab = np.transpose(labels, (1,0,2))
occ2 = np.transpose(occ, (1,0,2))

def show_slice(k):
    img = lab[:,:,k].copy()
    img[occ2[:,:,k] == 0] = -1
    plt.figure(figsize=(6,6))
    plt.imshow(img, interpolation="nearest")
    plt.title(f"Z slice {k} / {lab.shape[2]-1}")
    plt.axis("off"); plt.show()

# example: view a middle slice
show_slice(lab.shape[2]//2)

# %%
