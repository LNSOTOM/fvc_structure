#%%
#!/usr/bin/env python3
from pathlib import Path
import numpy as np
import laspy
import rasterio
from rasterio.transform import rowcol
import pandas as pd
import matplotlib.pyplot as plt

#%%
# -----------------------
# Inputs
# -----------------------
las_file = Path(
    "/media/laura/laura_usb/qgis/calperumResearch/site1_1_DD0001/lidar/clipped/z_norm/reduced_noise/"
    "triple_AusGeoid_tiles_multispectral.26_HAG_delaunay_noiseout.laz"
)
mask_file = Path(
    "/media/laura/laura_usb/qgis/calperumResearch/site1_1_DD0001/inputs/masks/tiles_3072/raw/fvc_class/"
    "annotation_raster/dowsample_5cm_bilinear/mask_fvc_3072.26.tif"
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
