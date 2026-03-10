
#%%
#!/usr/bin/env python3
# voxel_occupancy_by_cluster.py
#
# Compute 3D voxel occupancy per cluster_id from a clustered LAS/LAZ.
# Requires: laspy, numpy, pandas

from pathlib import Path
import numpy as np
import pandas as pd
import laspy

# ========= INPUT / OUTPUT =========
IN_LAS  = Path("/media/laura/laura_usb/qgis/calperumResearch/site1_1_DD0001/lidar/clipped/z_norm/triple_AusGeoid_tiles_multispectral.26_HAG_delaunay_HDBSCAN.las")
OUT_DIR = IN_LAS.parent / "voxel"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_VOXELS = OUT_DIR / (IN_LAS.stem + "_voxels.csv")
OUT_SUMMARY = OUT_DIR / (IN_LAS.stem + "_voxel_summary.csv")

# ========= PARAMS =========
VOXEL_SIZE = (0.25, 0.25, 0.25)  # metres (dx, dy, dz). Change to your needs.
INCLUDE_NOISE = False            # include cluster_id == -1?
HAG_FILTER_MIN = 0.0             # optional: only count points with HAG/Z_norm >= this
HAG_DIM_CANDIDATES = ("Z_norm", "HeightAboveGround")

# ========= LOAD =========
las = laspy.read(str(IN_LAS))
dims = set(las.point_format.dimension_names)

# cluster_id is required
if "cluster_id" not in dims:
    raise RuntimeError("Input LAS has no 'cluster_id' dimension. Run your clustering step first.")

cluster_id = np.asarray(las["cluster_id"], dtype=np.int32)

# coordinates (ScaledArrayView -> ndarray)
x = np.asarray(las.x, dtype=np.float64)
y = np.asarray(las.y, dtype=np.float64)
z = np.asarray(las.z, dtype=np.float64)

# optional HAG filter
hag = None
for dn in HAG_DIM_CANDIDATES:
    if dn in dims:
        hag = np.asarray(las[dn], dtype=np.float32)
        break

mask = np.ones(cluster_id.size, dtype=bool)
if not INCLUDE_NOISE:
    mask &= (cluster_id >= 0)
if hag is not None and HAG_FILTER_MIN > 0:
    mask &= (hag >= HAG_FILTER_MIN)

if not mask.any():
    raise RuntimeError("No points left after filters (noise and/or HAG). Loosen filters.")

cid = cluster_id[mask]
xx, yy, zz = x[mask], y[mask], z[mask]

# ========= VOXEL INDEXING =========
dx, dy, dz = VOXEL_SIZE
xmin, ymin, zmin = xx.min(), yy.min(), zz.min()

ix = np.floor((xx - xmin) / dx).astype(np.int64)
iy = np.floor((yy - ymin) / dy).astype(np.int64)
iz = np.floor((zz - zmin) / dz).astype(np.int64)

# Combine (cluster_id, ix, iy, iz) to count points per voxel per cluster
stack = np.column_stack([cid.astype(np.int64), ix, iy, iz])
# unique rows + counts
uniq, counts = np.unique(stack, axis=0, return_counts=True)

# Split back
cid_u = uniq[:, 0].astype(np.int32)
ix_u, iy_u, iz_u = uniq[:, 1], uniq[:, 2], uniq[:, 3]

# ========= PER-VOXEL CSV =========
df_vox = pd.DataFrame({
    "cluster_id": cid_u,
    "ix": ix_u, "iy": iy_u, "iz": iz_u,
    "points_in_voxel": counts
})

# add voxel-space bounding box in metres (optional, handy for mapping back)
df_vox["x_min"] = xmin + df_vox["ix"] * dx
df_vox["y_min"] = ymin + df_vox["iy"] * dy
df_vox["z_min"] = zmin + df_vox["iz"] * dz
df_vox["x_max"] = df_vox["x_min"] + dx
df_vox["y_max"] = df_vox["y_min"] + dy
df_vox["z_max"] = df_vox["z_min"] + dz

df_vox.to_csv(OUT_VOXELS, index=False)
print(f"✅ Wrote per-voxel occupancy: {OUT_VOXELS}  (rows={len(df_vox):,})")

# ========= PER-CLUSTER SUMMARY =========
g = df_vox.groupby("cluster_id", sort=True)
summary = pd.DataFrame({
    "cluster_id": g.size().index.astype(int),
    "occupied_voxels": g.size().values,
    "points_total": g["points_in_voxel"].sum().values,
})
# points per occupied voxel stats
summary["ppv_mean"] = g["points_in_voxel"].mean().values
summary["ppv_median"] = g["points_in_voxel"].median().values
summary["ppv_p90"] = g["points_in_voxel"].quantile(0.90).values

# approximate physical extents per cluster in metres (from voxel mins/max)
summary["x_extent_m"] = (g["x_max"].max().values - g["x_min"].min().values)
summary["y_extent_m"] = (g["y_max"].max().values - g["y_min"].min().values)
summary["z_extent_m"] = (g["z_max"].max().values - g["z_min"].min().values)

# optional: add HAG coverage if HAG present
if hag is not None:
    # compute HAG stats over points kept, per cluster, without recomputing per voxel:
    # more accurate way: reuse point-level mask
    df_pts = pd.DataFrame({
        "cluster_id": cid,
        "hag": hag[mask] if hag is not None else np.nan
    })
    g2 = df_pts.groupby("cluster_id", sort=True)
    summary["hag_p50"] = g2["hag"].quantile(0.50).values
    summary["hag_p90"] = g2["hag"].quantile(0.90).values
    summary["hag_max"] = g2["hag"].max().values

summary = summary.sort_values("occupied_voxels", ascending=False).reset_index(drop=True)
summary.to_csv(OUT_SUMMARY, index=False)
print(f"✅ Wrote per-cluster summary:  {OUT_SUMMARY}")
print(summary.head(10))


#####################################
# %%
##test 1 - Z bins are from absolute z (ellipsoidal/AHD height).
from pathlib import Path
import numpy as np
import pandas as pd
import laspy

# ========= INPUTS =========
LAS_IN = Path("/media/laura/laura_usb/qgis/calperumResearch/site1_1_DD0001/lidar/clipped/z_norm/triple_AusGeoid_tiles_multispectral.26_HAG_delaunay.las")   # has HeightAboveGround or Z_norm
LABEL_RASTER = Path("/media/laura/Extreme SSD/qgis/calperumResearch/site1_1_DD0001/inputs/masks/tiles_3072/raw/fvc_class/annotation_raster/mask_fvc_3072.26.tif")      # only needed if you want raster-aligned outputs
OUT_DIR = LAS_IN.parent / "voxel"; OUT_DIR.mkdir(parents=True, exist_ok=True)

# ========= PARAMS =========
MIN_CLUSTER_SIZE = 200        # HDBSCAN
MIN_SAMPLES = None
USE_HAG_NAME_ORDER = ("HeightAboveGround", "Z_norm")
INCLUDE_NOISE = False         # keep -1?
HAG_FILTER_MIN = 0.0          # e.g. 0.05 to drop sub-5 cm returns
VOXEL_SIZE = (0.25, 0.25, 0.25)  # metres for per-cluster CSV/summary
MAKE_RASTER_ALIGNED = False   # set True if you want Z×H×W tensors aligned to LABEL_RASTER
OCC_MIN_POINTS = 1            # per-voxel occupancy threshold (raster-aligned)
Z_MAX = 6.0                   # raster-aligned vertical extent (m)
HEIGHT_AGG = "p95"            # 'max' or 'p95' per XY column

# ========= STEP 1: read LAS & choose HAG =========
las = laspy.read(str(LAS_IN))
dims = set(las.point_format.dimension_names)
if "HeightAboveGround" in dims:
    hag = np.asarray(las["HeightAboveGround"], dtype=np.float32)
elif "Z_norm" in dims:
    hag = np.asarray(las["Z_norm"], dtype=np.float32)
else:
    raise RuntimeError("No HAG-like dimension found (expected HeightAboveGround or Z_norm).")

x = np.asarray(las.x, dtype=np.float64)
y = np.asarray(las.y, dtype=np.float64)

# ========= STEP 2: HDBSCAN on (x, y, HAG) =========
import hdbscan
print("[HDBSCAN] clustering...")
pts = np.column_stack([x, y, hag])
clusterer = hdbscan.HDBSCAN(min_cluster_size=MIN_CLUSTER_SIZE,
                            min_samples=MIN_SAMPLES,
                            cluster_selection_epsilon=0.0)
cluster_id = clusterer.fit_predict(pts).astype(np.int32)
print(f"[HDBSCAN] clusters: {len(np.unique(cluster_id))} (incl. -1 noise)")

# attach to LAS and save
from laspy import ExtraBytesParams
out_las_hdb = OUT_DIR / (LAS_IN.stem + "_HDBSCAN.las")
if "cluster_id" not in dims:
    las.add_extra_dim(ExtraBytesParams(name="cluster_id", type=np.int32))
las["cluster_id"] = cluster_id
las.write(str(out_las_hdb))
print(f"✅ wrote {out_las_hdb}")

# ========= STEP 3: per-cluster voxel CSV + summary =========
print("[Voxel] per-cluster occupancy in local voxel space...")
cid = cluster_id.copy()
mask = np.ones(cid.size, dtype=bool)
if not INCLUDE_NOISE:
    mask &= (cid >= 0)
if HAG_FILTER_MIN > 0:
    mask &= (hag >= HAG_FILTER_MIN)
if not mask.any():
    raise RuntimeError("No points left after filters.")

cid = cid[mask]
xx, yy, zz = x[mask], y[mask], np.asarray(las.z, dtype=np.float64)[mask]

dx, dy, dz = VOXEL_SIZE
xmin, ymin, zmin = xx.min(), yy.min(), zz.min()
ix = np.floor((xx - xmin)/dx).astype(np.int64)
iy = np.floor((yy - ymin)/dy).astype(np.int64)
iz = np.floor((zz - zmin)/dz).astype(np.int64)

stack = np.column_stack([cid.astype(np.int64), ix, iy, iz])
uniq, counts = np.unique(stack, axis=0, return_counts=True)

cid_u = uniq[:, 0].astype(np.int32)
ix_u, iy_u, iz_u = uniq[:, 1], uniq[:, 2], uniq[:, 3]

df_vox = pd.DataFrame({
    "cluster_id": cid_u,
    "ix": ix_u, "iy": iy_u, "iz": iz_u,
    "points_in_voxel": counts
})
df_vox["x_min"] = xmin + df_vox["ix"] * dx
df_vox["y_min"] = ymin + df_vox["iy"] * dy
df_vox["z_min"] = zmin + df_vox["iz"] * dz
df_vox["x_max"] = df_vox["x_min"] + dx
df_vox["y_max"] = df_vox["y_min"] + dy
df_vox["z_max"] = df_vox["z_min"] + dz

out_csv = OUT_DIR / (LAS_IN.stem + "_HDBSCAN_voxels.csv")
df_vox.to_csv(out_csv, index=False)
print(f"✅ wrote per-voxel CSV: {out_csv}  (rows={len(df_vox):,})")

g = df_vox.groupby("cluster_id", sort=True)
summary = pd.DataFrame({
    "cluster_id": g.size().index.astype(int),
    "occupied_voxels": g.size().values,
    "points_total": g["points_in_voxel"].sum().values,
})
summary["ppv_mean"] = g["points_in_voxel"].mean().values
summary["ppv_median"] = g["points_in_voxel"].median().values
summary["ppv_p90"] = g["points_in_voxel"].quantile(0.90).values
summary["x_extent_m"] = (g["x_max"].max().values - g["x_min"].min().values)
summary["y_extent_m"] = (g["y_max"].max().values - g["y_min"].min().values)
summary["z_extent_m"] = (g["z_max"].max().values - g["z_min"].min().values)

if hag is not None:
    df_pts = pd.DataFrame({"cluster_id": cluster_id[mask], "hag": hag[mask]})
    g2 = df_pts.groupby("cluster_id", sort=True)
    summary["hag_p50"] = g2["hag"].quantile(0.50).values
    summary["hag_p90"] = g2["hag"].quantile(0.90).values
    summary["hag_max"] = g2["hag"].max().values

out_sum = OUT_DIR / (LAS_IN.stem + "_HDBSCAN_voxel_summary.csv")
summary = summary.sort_values("occupied_voxels", ascending=False).reset_index(drop=True)
summary.to_csv(out_sum, index=False)
print(f"✅ wrote summary: {out_sum}")
print(summary.head(10))

# ========= (Optional) Raster-aligned occupancy/height/cluster tensors =========
if MAKE_RASTER_ALIGNED:
    import rasterio
    from rasterio.transform import rowcol
    print("[Raster-aligned] building Z×H×W tensors...")
    with rasterio.open(LABEL_RASTER) as ds:
        H, W = ds.height, ds.width
        transform = ds.transform
    voxel = float(VOXEL_SIZE[2])
    z_bins = int(np.ceil(Z_MAX / voxel))

    rows, cols = rowcol(transform, x, y)
    ok = (rows >= 0) & (rows < H) & (cols >= 0) & (cols < W) & (hag >= 0)
    rows, cols, hh, cids = rows[ok], cols[ok], hag[ok], cluster_id[ok]
    z_idx = np.floor(hh / voxel).astype(int)
    ok2 = (z_idx >= 0) & (z_idx < z_bins)
    rows, cols, z_idx, cids = rows[ok2], cols[ok2], z_idx[ok2], cids[ok2]

    occ = np.zeros((z_bins, H, W), dtype=np.uint8)
    col_height = np.full((H, W), np.nan, dtype=np.float32)
    cluster_vol = -np.ones((z_bins, H, W), dtype=np.int32)

    # column height
    from collections import defaultdict, Counter
    col_heights = defaultdict(list)
    for r, c, h in zip(rows, cols, hh[ok2]):
        col_heights[(r, c)].append(h)
    for (r, c), vals in col_heights.items():
        v = np.asarray(vals, dtype=np.float32)
        col_height[r, c] = np.nanmax(v) if HEIGHT_AGG == "max" else np.nanpercentile(v, 95)

    # occupancy
    lin = z_idx * (H * W) + rows * W + cols
    counts = np.bincount(lin, minlength=z_bins * H * W).reshape(z_bins, H, W)
    occ[counts >= OCC_MIN_POINTS] = 1

    # cluster majority per voxel (ignoring -1 if there’s ≥1 non-noise)
    voxel_clusters = {}
    for l, cl in zip(lin, cids):
        voxel_clusters.setdefault(l, []).append(cl)
    for l, lst in voxel_clusters.items():
        z = l // (H * W); rem = l % (H * W); r = rem // W; c = rem % W
        arr = np.asarray(lst, dtype=np.int32)
        if np.any(arr >= 0):
            arr = arr[arr >= 0]
        if arr.size > 0:
            cluster_vol[z, r, c] = Counter(arr).most_common(1)[0][0]

    np.save(OUT_DIR / "occupancy_ZHW_uint8.npy", occ)
    np.save(OUT_DIR / "height_HxW_float32.npy", col_height)
    np.save(OUT_DIR / "cluster_ZHW_int32.npy", cluster_vol)
    print("✅ wrote raster-aligned tensors (occ, height, cluster)")

# %%
## test 2 - HAG CSV: Z bins are from height-above-ground (HAG/Z_norm)
# ========= STEP 3: per-cluster voxel CSV + summary (USING HAG/Z_norm on Z axis) =========
print("[Voxel] per-cluster occupancy in local voxel space (Z = HAG/Z_norm)...")
cid = cluster_id.copy()

# filters
mask = np.ones(cid.size, dtype=bool)
if not INCLUDE_NOISE:
    mask &= (cid >= 0)
if HAG_FILTER_MIN > 0:
    mask &= (hag >= HAG_FILTER_MIN)

if not mask.any():
    raise RuntimeError("No points left after filters.")

cid  = cid[mask]
xx   = x[mask]
yy   = y[mask]
hh   = hag[mask]                        # <--- use HAG (Z_norm) as vertical coordinate

# voxel index (X,Y,Z) where Z is discretised HAG
dx, dy, dz = VOXEL_SIZE
xmin, ymin, hmin = xx.min(), yy.min(), hh.min()
ix = np.floor((xx - xmin)/dx).astype(np.int64)
iy = np.floor((yy - ymin)/dy).astype(np.int64)
iz = np.floor((hh - hmin)/dz).astype(np.int64)

# build a small point table so we can aggregate HAG stats per (cluster, ix, iy, iz)
pt_df = pd.DataFrame({
    "cluster_id": cid.astype(np.int32),
    "ix": ix, "iy": iy, "iz": iz,
    "x": xx, "y": yy, "hag": hh
})

# aggregate per voxel: count + hag_min, hag_mean, hag_max
agg = pt_df.groupby(["cluster_id", "ix", "iy", "iz"], sort=True).agg(
    points_in_voxel=("hag", "size"),
    hag_min=("hag", "min"),
    hag_mean=("hag", "mean"),
    hag_max=("hag", "max"),
).reset_index()

# add voxel-space bounds in metres (XY from mins, Z from hmin)
agg["x_min"] = xmin + agg["ix"] * dx
agg["y_min"] = ymin + agg["iy"] * dy
agg["z_min_hag"] = hmin + agg["iz"] * dz        # note name: z_min in HAG units
agg["x_max"] = agg["x_min"] + dx
agg["y_max"] = agg["y_min"] + dy
agg["z_max_hag"] = agg["z_min_hag"] + dz

# write per-voxel CSV
out_csv = OUT_DIR / (LAS_IN.stem + "_HDBSCAN_voxels_HAG.csv")
agg.to_csv(out_csv, index=False)
print(f"✅ wrote per-voxel CSV: {out_csv}  (rows={len(agg):,})")

# --------- per-cluster summary (from the HAG-based voxels) ----------
g = agg.groupby("cluster_id", sort=True)
summary = pd.DataFrame({
    "cluster_id": g.size().index.astype(int),
    "occupied_voxels": g.size().values,
    "points_total": g["points_in_voxel"].sum().values,
})

# points-per-voxel stats
summary["ppv_mean"]   = g["points_in_voxel"].mean().values
summary["ppv_median"] = g["points_in_voxel"].median().values
summary["ppv_p90"]    = g["points_in_voxel"].quantile(0.90).values

# physical extents in metres (XY) and in HAG metres (Z)
summary["x_extent_m"]   = (g["x_max"].max().values - g["x_min"].min().values)
summary["y_extent_m"]   = (g["y_max"].max().values - g["y_min"].min().values)
summary["z_extent_hag"] = (g["z_max_hag"].max().values - g["z_min_hag"].min().values)

# HAG stats across points per cluster (independent of voxelisation)
g2 = pt_df.groupby("cluster_id", sort=True)
summary["hag_p50"] = g2["hag"].quantile(0.50).values
summary["hag_p90"] = g2["hag"].quantile(0.90).values
summary["hag_max"] = g2["hag"].max().values

out_sum = OUT_DIR / (LAS_IN.stem + "_HDBSCAN_voxel_summary_HAG.csv")
summary = summary.sort_values("occupied_voxels", ascending=False).reset_index(drop=True)
summary.to_csv(out_sum, index=False)
print(f"✅ wrote summary: {out_sum}")
print(summary.head(10))

# %%
### test 3
#!/usr/bin/env python3
from pathlib import Path
import numpy as np
import pandas as pd
import laspy

# ========= INPUTS =========
LAS_IN = Path("/media/laura/laura_usb/qgis/calperumResearch/site1_1_DD0001/lidar/clipped/z_norm/triple_AusGeoid_tiles_multispectral.26_HAG_delaunay.las")
LABEL_RASTER = Path("/media/laura/Extreme SSD/qgis/calperumResearch/site1_1_DD0001/inputs/masks/tiles_3072/raw/fvc_class/annotation_raster/mask_fvc_3072.26.tif")
OUT_DIR = LAS_IN.parent / "voxel_hag"; OUT_DIR.mkdir(parents=True, exist_ok=True)

# ========= PARAMS =========
# HDBSCAN (optional: set to 0 to skip clustering)
MIN_CLUSTER_SIZE = 200
MIN_SAMPLES = None

# HAG / voxelisation
HAG_NAME_ORDER = ("HeightAboveGround", "Z_norm")
INCLUDE_NOISE = True          # keep cluster -1 in voxel CSV
HAG_FILTER_MIN = 0.0          # e.g. 0.05 to drop sub-5cm returns
VOXEL_SIZE = (0.25, 0.25, 0.25)   # (dx, dy, dz) in metres; dz is HAG bin size

# Raster-aligned products
MAKE_RASTER_ALIGNED = True
OCC_MIN_POINTS = 1
Z_MAX = 6.0
HEIGHT_AGG = "p95"            # 'max' or 'p95'
# Strata edges in HAG metres (e.g., herb 0.5–2 m, shrub 2–5 m, canopy >5 m)
STRATA_EDGES = [0.0, 0.5, 2.0, 5.0, 10.0]  # N bins => N-1 strata

# ========= LOAD LAS & HAG =========
las = laspy.read(str(LAS_IN))
dims = set(las.point_format.dimension_names)

hag = None
for nm in HAG_NAME_ORDER:
    if nm in dims:
        hag = np.asarray(las[nm], dtype=np.float32)
        break
if hag is None:
    raise RuntimeError("No HAG-like dimension found (expected one of: %s)" % (HAG_NAME_ORDER,))

x = np.asarray(las.x, dtype=np.float64)
y = np.asarray(las.y, dtype=np.float64)

# ========= HDBSCAN (optional) =========
if MIN_CLUSTER_SIZE and MIN_CLUSTER_SIZE > 0:
    import hdbscan
    print("[HDBSCAN] clustering on (x,y,HAG)…")
    pts = np.column_stack([x, y, hag])
    clusterer = hdbscan.HDBSCAN(min_cluster_size=MIN_CLUSTER_SIZE,
                                min_samples=MIN_SAMPLES,
                                cluster_selection_epsilon=0.0)
    cluster_id = clusterer.fit_predict(pts).astype(np.int32)
    from laspy import ExtraBytesParams
    if "cluster_id" not in dims:
        las.add_extra_dim(ExtraBytesParams(name="cluster_id", type=np.int32))
    las["cluster_id"] = cluster_id
else:
    # cluster_id optional; fill with zeros
    cluster_id = np.zeros_like(hag, dtype=np.int32)

out_las = OUT_DIR / (LAS_IN.stem + "_HAGready_HDBSCAN.las")
las.write(str(out_las))
print(f"✅ wrote {out_las}")

# ========= HAG-VOXEL CSV (min/mean/max) =========
print("[Voxel-HAG] computing per-voxel occupancy with HAG on Z…")
mask = np.ones(hag.size, dtype=bool)
if not INCLUDE_NOISE and "cluster_id" in las.point_format.dimension_names:
    mask &= (cluster_id >= 0)
if HAG_FILTER_MIN > 0:
    mask &= (hag >= HAG_FILTER_MIN)
if not mask.any():
    raise RuntimeError("No points left after filters.")

cid = cluster_id[mask]
xx, yy, hh = x[mask], y[mask], hag[mask]

dx, dy, dz = VOXEL_SIZE
xmin, ymin, hmin = xx.min(), yy.min(), hh.min()
ix = np.floor((xx - xmin)/dx).astype(np.int64)
iy = np.floor((yy - ymin)/dy).astype(np.int64)
iz = np.floor((hh - hmin)/dz).astype(np.int64)

pt_df = pd.DataFrame({
    "cluster_id": cid.astype(np.int32),
    "ix": ix, "iy": iy, "iz": iz,
    "x": xx, "y": yy, "hag": hh
})

agg = pt_df.groupby(["cluster_id","ix","iy","iz"], sort=True).agg(
    points_in_voxel=("hag","size"),
    hag_min=("hag","min"),
    hag_mean=("hag","mean"),
    hag_max=("hag","max"),
).reset_index()

agg["x_min"] = xmin + agg["ix"] * dx
agg["y_min"] = ymin + agg["iy"] * dy
agg["z_min_hag"] = hmin + agg["iz"] * dz
agg["x_max"] = agg["x_min"] + dx
agg["y_max"] = agg["y_min"] + dy
agg["z_max_hag"] = agg["z_min_hag"] + dz

csv_vox = OUT_DIR / (LAS_IN.stem + "_voxels_HAG.csv")
agg.to_csv(csv_vox, index=False)
print(f"✅ wrote HAG voxel CSV: {csv_vox}  (rows={len(agg):,})")

# ========= Per-cluster summary (HAG) =========
g = agg.groupby("cluster_id", sort=True)
summary = pd.DataFrame({
    "cluster_id": g.size().index.astype(int),
    "occupied_voxels": g.size().values,
    "points_total": g["points_in_voxel"].sum().values,
})
summary["ppv_mean"]   = g["points_in_voxel"].mean().values
summary["ppv_median"] = g["points_in_voxel"].median().values
summary["ppv_p90"]    = g["points_in_voxel"].quantile(0.90).values
summary["x_extent_m"]   = (g["x_max"].max().values - g["x_min"].min().values)
summary["y_extent_m"]   = (g["y_max"].max().values - g["y_min"].min().values)
summary["z_extent_hag"] = (g["z_max_hag"].max().values - g["z_min_hag"].min().values)

g2 = pt_df.groupby("cluster_id", sort=True)
summary["hag_p50"] = g2["hag"].quantile(0.50).values
summary["hag_p90"] = g2["hag"].quantile(0.90).values
summary["hag_max"] = g2["hag"].max().values

csv_sum = OUT_DIR / (LAS_IN.stem + "_voxel_summary_HAG.csv")
summary = summary.sort_values("occupied_voxels", ascending=False).reset_index(drop=True)
summary.to_csv(csv_sum, index=False)
print(f"✅ wrote summary: {csv_sum}")

# ========= Raster-aligned (for 2D back-projection + 3D U-Net Z) =========
if MAKE_RASTER_ALIGNED:
    import rasterio
    from rasterio.transform import rowcol
    from collections import defaultdict, Counter

    with rasterio.open(LABEL_RASTER) as ds:
        H, W = ds.height, ds.width
        transform = ds.transform
        meta = ds.meta.copy()

    voxel_z = float(VOXEL_SIZE[2])
    z_bins = int(np.ceil(Z_MAX / voxel_z))

    rows, cols = rowcol(transform, x, y)
    ok = (rows >= 0) & (rows < H) & (cols >= 0) & (cols < W) & (hag >= 0)
    rows, cols, hh, cids = rows[ok], cols[ok], hag[ok], cluster_id[ok]

    # per-column (XY) height: HAG95 or max
    col_height = np.full((H, W), np.nan, dtype=np.float32)
    col_heights = defaultdict(list)
    for r, c, h in zip(rows, cols, hh):
        col_heights[(r, c)].append(h)
    for (r, c), vals in col_heights.items():
        v = np.asarray(vals, dtype=np.float32)
        col_height[r, c] = np.nanmax(v) if HEIGHT_AGG == "max" else np.nanpercentile(v, 95)

    # 3D occupancy (Z × H × W) in HAG bins
    z_idx = np.floor(hh / voxel_z).astype(int)
    ok2 = (z_idx >= 0) & (z_idx < z_bins)
    rows, cols, z_idx, cids = rows[ok2], cols[ok2], z_idx[ok2], cids[ok2]

    occ = np.zeros((z_bins, H, W), dtype=np.uint8)
    lin = z_idx * (H * W) + rows * W + cols
    counts = np.bincount(lin, minlength=z_bins * H * W).reshape(z_bins, H, W)
    occ[counts >= OCC_MIN_POINTS] = 1

    # majority cluster per voxel (ignore -1 if any non-noise exists)
    cluster_vol = -np.ones((z_bins, H, W), dtype=np.int32)
    voxel_clusters = {}
    for l, cl in zip(lin, cids):
        voxel_clusters.setdefault(l, []).append(cl)
    for l, lst in voxel_clusters.items():
        z = l // (H * W); rem = l % (H * W); r = rem // W; c = rem % W
        arr = np.asarray(lst, dtype=np.int32)
        if np.any(arr >= 0):  # drop noise
            arr = arr[arr >= 0]
        if arr.size > 0:
            cluster_vol[z, r, c] = Counter(arr).most_common(1)[0][0]

    # Save tensors for training
    np.save(OUT_DIR / "occupancy_ZHW_uint8.npy", occ)
    np.save(OUT_DIR / f"height_{HEIGHT_AGG}_HxW_float32.npy", col_height)
    np.save(OUT_DIR / "cluster_ZHW_int32.npy", cluster_vol)

    # Export height as GeoTIFF (the Z channel for your 3D U-Net)
    meta.update(dtype="float32", count=1, nodata=np.nan)
    out_tif = OUT_DIR / f"height_{HEIGHT_AGG}.tif"
    with rasterio.open(out_tif, "w", **meta) as dst:
        dst.write(col_height, 1)

    print("✅ wrote raster-aligned tensors (occ, height, cluster)")
    print(f"   Z-channel raster: {out_tif}")

    # ========= Per-pixel strata occupancy (for 3D vegetation structure) =========
    # Build strata index for each point, then per-pixel fraction per stratum.
    bins = np.asarray(STRATA_EDGES, dtype=np.float32)
    # strata_id in [0..len(bins)-2]
    strata_id = np.clip(np.digitize(hh, bins) - 1, 0, len(bins) - 2)

    # count points per pixel per strata
    S = len(bins) - 1
    strata_counts = np.zeros((S, H, W), dtype=np.int32)
    for r, c, s in zip(rows, cols, strata_id):
        strata_counts[s, r, c] += 1

    # convert to occupancy fraction by normalising each pixel by its max counts across strata or total
    total_counts = strata_counts.sum(axis=0).astype(np.float32)
    with np.errstate(divide='ignore', invalid='ignore'):
        strata_frac = np.where(total_counts[None, ...] > 0,
                               strata_counts / total_counts[None, ...],
                               0.0).astype(np.float32)

    # Save strata fractions (GeoTIFF with one band per strata)
    meta.update(count=S, dtype="float32", nodata=np.nan)
    strata_tif = OUT_DIR / "strata_fraction.tif"
    with rasterio.open(strata_tif, "w", **meta) as dst:
        for b in range(S):
            dst.write(strata_frac[b], b+1)
            # Optional: set band descriptions
            dst.set_band_description(b+1, f"{bins[b]}–{bins[b+1]} m HAG")

    print(f"✅ wrote strata fractions: {strata_tif}")



# %%
###test 4
#!/usr/bin/env python3
from pathlib import Path
import json
import numpy as np
import pandas as pd
import laspy

# ========= INPUTS =========
LAS_IN = Path("/media/laura/laura_usb/qgis/calperumResearch/site1_1_DD0001/lidar/clipped/z_norm/triple_AusGeoid_tiles_multispectral.109_HAG_delaunay.las")
LABEL_RASTER = Path("/media/laura/Extreme SSD/qgis/calperumResearch/site1_1_DD0001/inputs/masks/tiles_3072/raw/fvc_class/annotation_raster/mask_fvc_3072.109.tif")
OUT_DIR = LAS_IN.parent / "voxel_hag_109"; OUT_DIR.mkdir(parents=True, exist_ok=True)

# ========= PARAMS =========
# HDBSCAN (optional: set to 0 to skip clustering)
MIN_CLUSTER_SIZE = 200
MIN_SAMPLES = None  # use None for HDBSCAN default; or an int >= 1

# HAG / voxelisation
HAG_NAME_ORDER = ("HeightAboveGround", "Z_norm")
INCLUDE_NOISE = True           # keep cluster -1 rows in voxel CSV
HAG_FILTER_MIN = 0.0           # e.g. 0.05 to drop sub-5cm returns
VOXEL_SIZE = (0.25, 0.25, 0.25)  # (dx, dy, dz) metres; dz is HAG bin size

# Raster-aligned products
MAKE_RASTER_ALIGNED = True
OCC_MIN_POINTS = 1
Z_MAX = 6.0                    # maximum HAG to consider (metres)
HEIGHT_AGG = "p95"             # 'max' or 'p95'
# Strata edges in HAG metres (e.g., herb 0.5–2 m, shrub 2–5 m, canopy >5 m)
STRATA_EDGES = [0.0, 0.5, 2.0, 5.0, 10.0]  # len-1 == #bands

# ========= HELPERS =========
def pick_hag_dim(las, candidates):
    dims = set(las.point_format.dimension_names)
    for nm in candidates:
        if nm in dims:
            return np.asarray(las[nm], dtype=np.float32), nm
    raise RuntimeError(f"No HAG-like dimension found (expected one of: {candidates})")

def write_meta(path: Path, obj: dict):
    path.write_text(json.dumps(obj, indent=2))

# ========= LOAD LAS & HAG =========
las = laspy.read(str(LAS_IN))
hag, hag_name = pick_hag_dim(las, HAG_NAME_ORDER)
x = np.asarray(las.x, dtype=np.float64)
y = np.asarray(las.y, dtype=np.float64)

# ========= HDBSCAN (optional) =========
if MIN_CLUSTER_SIZE and MIN_CLUSTER_SIZE > 0:
    try:
        import hdbscan
    except Exception as e:
        raise RuntimeError("HDBSCAN not installed. `pip install hdbscan`") from e
    print(f"[HDBSCAN] clustering on (x,y,{hag_name})…")
    pts = np.column_stack([x, y, hag])
    clusterer = hdbscan.HDBSCAN(min_cluster_size=MIN_CLUSTER_SIZE,
                                min_samples=MIN_SAMPLES,
                                cluster_selection_epsilon=0.0)
    cluster_id = clusterer.fit_predict(pts).astype(np.int32)
    from laspy import ExtraBytesParams
    if "cluster_id" not in las.point_format.dimension_names:
        las.add_extra_dim(ExtraBytesParams(name="cluster_id", type=np.int32))
    las["cluster_id"] = cluster_id
    n_clusters = len(np.unique(cluster_id))
    print(f"[HDBSCAN] clusters found: {n_clusters} (incl. -1)")
else:
    # cluster_id optional; fill with zeros
    cluster_id = np.zeros_like(hag, dtype=np.int32)

out_las = OUT_DIR / (LAS_IN.stem + "_HAGready_HDBSCAN.las")
las.write(str(out_las))
print(f"✅ wrote {out_las}")

# ========= HAG-VOXEL CSV (min/mean/max) =========
print("[Voxel-HAG] computing per-voxel occupancy with HAG on Z…")
mask = np.ones(hag.size, dtype=bool)
if not INCLUDE_NOISE:
    mask &= (cluster_id >= 0)
if HAG_FILTER_MIN > 0:
    mask &= (hag >= HAG_FILTER_MIN)
if not mask.any():
    raise RuntimeError("No points left after filters (noise/HAG).")

cid = cluster_id[mask]
xx, yy, hh = x[mask], y[mask], hag[mask]

dx, dy, dz = VOXEL_SIZE
xmin, ymin, hmin = float(xx.min()), float(yy.min()), float(hh.min())
ix = np.floor((xx - xmin)/dx).astype(np.int64)
iy = np.floor((yy - ymin)/dy).astype(np.int64)
iz = np.floor((hh - hmin)/dz).astype(np.int64)

pt_df = pd.DataFrame({
    "cluster_id": cid.astype(np.int32),
    "ix": ix, "iy": iy, "iz": iz,
    "x": xx, "y": yy, "hag": hh
})

# aggregate per voxel
agg = pt_df.groupby(["cluster_id","ix","iy","iz"], sort=True).agg(
    points_in_voxel=("hag","size"),
    hag_min=("hag","min"),
    hag_mean=("hag","mean"),
    hag_max=("hag","max"),
).reset_index()

# voxel bounds
agg["x_min"] = xmin + agg["ix"] * dx
agg["y_min"] = ymin + agg["iy"] * dy
agg["z_min_hag"] = hmin + agg["iz"] * dz
agg["x_max"] = agg["x_min"] + dx
agg["y_max"] = agg["y_min"] + dy
agg["z_max_hag"] = agg["z_min_hag"] + dz

# handy voxel_id for joins (dense int)
agg["voxel_id"] = (
    (agg["ix"].values.astype(np.int64) << 42) ^
    (agg["iy"].values.astype(np.int64) << 21) ^
    (agg["iz"].values.astype(np.int64))
)

csv_vox = OUT_DIR / (LAS_IN.stem + "_voxels_HAG.csv")
agg.to_csv(csv_vox, index=False)
print(f"✅ wrote HAG voxel CSV: {csv_vox}  (rows={len(agg):,})")

# ========= Per-cluster summary (HAG) =========
g = agg.groupby("cluster_id", sort=True)
summary = pd.DataFrame({
    "cluster_id": g.size().index.astype(int),
    "occupied_voxels": g.size().values,
    "points_total": g["points_in_voxel"].sum().values,
})
summary["ppv_mean"]   = g["points_in_voxel"].mean().values
summary["ppv_median"] = g["points_in_voxel"].median().values
summary["ppv_p90"]    = g["points_in_voxel"].quantile(0.90).values
summary["x_extent_m"]   = (g["x_max"].max().values - g["x_min"].min().values)
summary["y_extent_m"]   = (g["y_max"].max().values - g["y_min"].min().values)
summary["z_extent_hag"] = (g["z_max_hag"].max().values - g["z_min_hag"].min().values)

# extra HAG stats per cluster (point-level, independent of voxelisation)
g2 = pt_df.groupby("cluster_id", sort=True)
summary["hag_p50"] = g2["hag"].quantile(0.50).values
summary["hag_p90"] = g2["hag"].quantile(0.90).values
summary["hag_max"] = g2["hag"].max().values

csv_sum = OUT_DIR / (LAS_IN.stem + "_voxel_summary_HAG.csv")
summary = summary.sort_values("occupied_voxels", ascending=False).reset_index(drop=True)
summary.to_csv(csv_sum, index=False)
print(f"✅ wrote summary: {csv_sum}")

# small metadata sidecar so future steps know dz etc.
write_meta(csv_sum.with_suffix(".json"), {
    "hag_dim": hag_name, "dx": dx, "dy": dy, "dz": dz,
    "filters": {"include_noise": INCLUDE_NOISE, "hag_min": HAG_FILTER_MIN},
})

# ========= Raster-aligned (for 2D back-projection + 3D U-Net Z) =========
if MAKE_RASTER_ALIGNED:
    import rasterio
    from rasterio.transform import rowcol

    # 1) load grid
    with rasterio.open(LABEL_RASTER) as ds:
        H, W = ds.height, ds.width
        transform = ds.transform
        meta = ds.meta.copy()

    voxel_z = float(VOXEL_SIZE[2])
    z_bins = int(np.ceil(Z_MAX / voxel_z))
    if z_bins <= 0:
        raise ValueError("z_bins computed <= 0; check Z_MAX and VOXEL_SIZE[2].")

    # 2) map points -> pixels, keep inside; hag >=0 only
    rows, cols = rowcol(transform, x, y)
    ok = (rows >= 0) & (rows < H) & (cols >= 0) & (cols < W) & np.isfinite(hag) & (hag >= 0)
    rows, cols, hh, cids = rows[ok], cols[ok], hag[ok], cluster_id[ok]

    # 3) per-pixel column height (p95 or max)
    from collections import defaultdict, Counter
    col_height = np.full((H, W), np.nan, dtype=np.float32)
    bucket = defaultdict(list)
    for r, c, h in zip(rows, cols, hh):
        bucket[(r, c)].append(h)
    for (r, c), vals in bucket.items():
        v = np.asarray(vals, dtype=np.float32)
        col_height[r, c] = np.nanmax(v) if HEIGHT_AGG == "max" else np.nanpercentile(v, 95)

    # 4) occupancy (Z × H × W) in HAG bins
    z_idx = np.floor(hh / voxel_z).astype(np.int64)
    ok2 = (z_idx >= 0) & (z_idx < z_bins)
    rows, cols, z_idx, cids = rows[ok2], cols[ok2], z_idx[ok2], cids[ok2]

    occ = np.zeros((z_bins, H, W), dtype=np.uint8)
    lin = z_idx * (H * W) + rows * W + cols
    counts = np.bincount(lin, minlength=z_bins * H * W).reshape(z_bins, H, W)
    occ[counts >= OCC_MIN_POINTS] = 1

    # 5) majority cluster per voxel (safe w/ noise = -1)
    cluster_vol = -np.ones((z_bins, H, W), dtype=np.int32)

    # build per-voxel list of cluster ids
    from collections import defaultdict
    vdict = defaultdict(list)
    for L, cl in zip(lin, cids):
        vdict[L].append(cl)

    for L, lst in vdict.items():
        arr = np.asarray(lst, dtype=np.int32)

        # remove negatives; if nothing left, keep -1
        pos = arr[arr >= 0]
        maj = -1 if pos.size == 0 else int(np.bincount(pos).argmax())

        z = L // (H * W)
        rem = L % (H * W)
        r = rem // W
        c = rem % W
        cluster_vol[z, r, c] = maj


    # 6) save tensors + height GeoTIFF
    np.save(OUT_DIR / "occupancy_ZHW_uint8.npy", occ)
    np.save(OUT_DIR / f"height_{HEIGHT_AGG}_HxW_float32.npy", col_height)
    np.save(OUT_DIR / "cluster_ZHW_int32.npy", cluster_vol)

    meta.update(dtype="float32", count=1, nodata=np.nan)
    out_tif = OUT_DIR / f"height_{HEIGHT_AGG}.tif"
    with rasterio.open(out_tif, "w", **meta) as dst:
        dst.write(col_height, 1)

    print("✅ wrote raster-aligned tensors (occ, height, cluster)")
    print(f"   Z-channel raster: {out_tif}")

    # ========= Per-pixel strata occupancy =========
    # sanity: edges strictly increasing
    edges = np.asarray(STRATA_EDGES, dtype=np.float32)
    if not np.all(edges[1:] > edges[:-1]):
        raise ValueError("STRATA_EDGES must be strictly increasing.")
    strata_id = np.clip(np.searchsorted(edges, hh, side="right") - 1, 0, len(edges) - 2)

    S = len(edges) - 1
    strata_counts = np.zeros((S, H, W), dtype=np.int32)
    for r, c, s in zip(rows, cols, strata_id):
        strata_counts[s, r, c] += 1

    total_counts = strata_counts.sum(axis=0).astype(np.float32)
    with np.errstate(divide='ignore', invalid='ignore'):
        strata_frac = np.where(total_counts[None, ...] > 0,
                               strata_counts / total_counts[None, ...],
                               0.0).astype(np.float32)

    meta.update(count=S, dtype="float32", nodata=np.nan)
    strata_tif = OUT_DIR / "strata_fraction.tif"
    with rasterio.open(strata_tif, "w", **meta) as dst:
        for b in range(S):
            dst.write(strata_frac[b], b+1)
            dst.set_band_description(b+1, f"{edges[b]}–{edges[b+1]} m HAG")

    print(f"✅ wrote strata fractions: {strata_tif}")


# %%
###test 5
#!/usr/bin/env python3
from pathlib import Path
import json
import numpy as np
import pandas as pd
import laspy

# ========= INPUTS =========
LAS_IN = Path("/media/laura/laura_usb/qgis/calperumResearch/site1_1_DD0001/lidar/clipped/z_norm/triple_AusGeoid_tiles_multispectral.109_HAG_delaunay.las")
LABEL_RASTER = Path("/media/laura/Extreme SSD/qgis/calperumResearch/site1_1_DD0001/inputs/masks/tiles_3072/raw/fvc_class/annotation_raster/mask_fvc_3072.109.tif")
OUT_DIR = LAS_IN.parent / "voxel_hag_109"; OUT_DIR.mkdir(parents=True, exist_ok=True)

# ========= PARAMS =========
MIN_CLUSTER_SIZE = 200
MIN_SAMPLES = None  # or an int >= 1

HAG_NAME_ORDER = ("HeightAboveGround", "Z_norm")
INCLUDE_NOISE = True
HAG_FILTER_MIN = 0.0
VOXEL_SIZE = (0.25, 0.25, 0.25)   # (dx, dy, dz)

MAKE_RASTER_ALIGNED = True
OCC_MIN_POINTS = 1
Z_MAX = 6.0
HEIGHT_AGG = "p95"                 # 'max' or 'p95'
STRATA_EDGES = [0.0, 0.5, 2.0, 5.0, 10.0]  # strictly increasing

# ========= HELPERS =========
def pick_hag_dim(las, candidates):
    dims = set(las.point_format.dimension_names)
    for nm in candidates:
        if nm in dims:
            return np.asarray(las[nm], dtype=np.float32), nm
    raise RuntimeError(f"No HAG-like dimension found (expected one of: {candidates})")

def write_meta(path: Path, obj: dict):
    path.write_text(json.dumps(obj, indent=2))

# ========= LOAD LAS & HAG =========
las = laspy.read(str(LAS_IN))
hag, hag_name = pick_hag_dim(las, HAG_NAME_ORDER)
x = np.asarray(las.x, dtype=np.float64)
y = np.asarray(las.y, dtype=np.float64)

# ========= HDBSCAN (optional) =========
if MIN_CLUSTER_SIZE and MIN_CLUSTER_SIZE > 0:
    try:
        import hdbscan
    except Exception as e:
        raise RuntimeError("HDBSCAN not installed. `pip install hdbscan`") from e
    print(f"[HDBSCAN] clustering on (x,y,{hag_name})…")
    pts = np.column_stack([x, y, hag])
    clusterer = hdbscan.HDBSCAN(min_cluster_size=MIN_CLUSTER_SIZE,
                                min_samples=MIN_SAMPLES,
                                cluster_selection_epsilon=0.0)
    cluster_id = clusterer.fit_predict(pts).astype(np.int32)
    from laspy import ExtraBytesParams
    if "cluster_id" not in las.point_format.dimension_names:
        las.add_extra_dim(ExtraBytesParams(name="cluster_id", type=np.int32))
    las["cluster_id"] = cluster_id
    print(f"[HDBSCAN] clusters found: {len(np.unique(cluster_id))} (incl. -1)")
else:
    cluster_id = np.zeros_like(hag, dtype=np.int32)

out_las = OUT_DIR / (LAS_IN.stem + "_HAGready_HDBSCAN.las")
las.write(str(out_las))
print(f"✅ wrote {out_las}")

# ========= HAG-VOXEL CSV (min/mean/max) =========
print("[Voxel-HAG] per-voxel stats with HAG on Z…")
mask = np.ones(hag.size, dtype=bool)
if not INCLUDE_NOISE: mask &= (cluster_id >= 0)
if HAG_FILTER_MIN > 0: mask &= (hag >= HAG_FILTER_MIN)
if not mask.any(): raise RuntimeError("No points left after filters (noise/HAG).")

cid = cluster_id[mask]
xx, yy, hh = x[mask], y[mask], hag[mask]

dx, dy, dz = VOXEL_SIZE
xmin, ymin, hmin = float(xx.min()), float(yy.min()), float(hh.min())
ix = np.floor((xx - xmin)/dx).astype(np.int64)
iy = np.floor((yy - ymin)/dy).astype(np.int64)
iz = np.floor((hh - hmin)/dz).astype(np.int64)

pt_df = pd.DataFrame({
    "cluster_id": cid.astype(np.int32),
    "ix": ix, "iy": iy, "iz": iz,
    "x": xx, "y": yy, "hag": hh
})

agg = pt_df.groupby(["cluster_id","ix","iy","iz"], sort=True).agg(
    points_in_voxel=("hag","size"),
    hag_min=("hag","min"),
    hag_mean=("hag","mean"),
    hag_max=("hag","max"),
).reset_index()

agg["x_min"] = xmin + agg["ix"] * dx
agg["y_min"] = ymin + agg["iy"] * dy
agg["z_min_hag"] = hmin + agg["iz"] * dz
agg["x_max"] = agg["x_min"] + dx
agg["y_max"] = agg["y_min"] + dy
agg["z_max_hag"] = agg["z_min_hag"] + dz

# stable voxel_id for joins
agg["voxel_id"] = (
    (agg["ix"].values.astype(np.int64) << 42) ^
    (agg["iy"].values.astype(np.int64) << 21) ^
    (agg["iz"].values.astype(np.int64))
)

csv_vox = OUT_DIR / (LAS_IN.stem + "_voxels_HAG.csv")
agg.to_csv(csv_vox, index=False)
print(f"✅ HAG voxel CSV: {csv_vox}  (rows={len(agg):,})")

# ========= Per-cluster summary =========
g = agg.groupby("cluster_id", sort=True)
summary = pd.DataFrame({
    "cluster_id": g.size().index.astype(int),
    "occupied_voxels": g.size().values,
    "points_total": g["points_in_voxel"].sum().values,
})
summary["ppv_mean"]   = g["points_in_voxel"].mean().values
summary["ppv_median"] = g["points_in_voxel"].median().values
summary["ppv_p90"]    = g["points_in_voxel"].quantile(0.90).values
summary["x_extent_m"]   = (g["x_max"].max().values - g["x_min"].min().values)
summary["y_extent_m"]   = (g["y_max"].max().values - g["y_min"].min().values)
summary["z_extent_hag"] = (g["z_max_hag"].max().values - g["z_min_hag"].min().values)

g2 = pt_df.groupby("cluster_id", sort=True)
summary["hag_p50"] = g2["hag"].quantile(0.50).values
summary["hag_p90"] = g2["hag"].quantile(0.90).values
summary["hag_max"] = g2["hag"].max().values

csv_sum = OUT_DIR / (LAS_IN.stem + "_voxel_summary_HAG.csv")
summary = summary.sort_values("occupied_voxels", ascending=False).reset_index(drop=True)
summary.to_csv(csv_sum, index=False)
print(f"✅ summary: {csv_sum}")

write_meta(csv_sum.with_suffix(".json"), {
    "hag_dim": hag_name, "dx": dx, "dy": dy, "dz": dz,
    "filters": {"include_noise": INCLUDE_NOISE, "hag_min": HAG_FILTER_MIN},
})

# ========= Raster-aligned tensors & extras =========
if MAKE_RASTER_ALIGNED:
    import rasterio
    from rasterio.transform import rowcol

    with rasterio.open(LABEL_RASTER) as ds:
        H, W = ds.height, ds.width
        transform = ds.transform
        meta = ds.meta.copy()

    voxel_z = float(VOXEL_SIZE[2])
    z_bins = int(np.ceil(Z_MAX / voxel_z))
    if z_bins <= 0: raise ValueError("z_bins <= 0; check Z_MAX and VOXEL_SIZE[2].")

    # points -> pixels, keep inside; hag >= 0
    rows, cols = rowcol(transform, x, y)
    ok = (rows >= 0) & (rows < H) & (cols >= 0) & (cols < W) & np.isfinite(hag) & (hag >= 0)
    rows, cols, hh, cids = rows[ok], cols[ok], hag[ok], cluster_id[ok]

    # per-pixel height (p95 or max)
    from collections import defaultdict
    col_height = np.full((H, W), np.nan, dtype=np.float32)
    bucket = defaultdict(list)
    for r, c, h in zip(rows, cols, hh): bucket[(r, c)].append(h)
    for (r, c), vals in bucket.items():
        v = np.asarray(vals, dtype=np.float32)
        col_height[r, c] = np.nanmax(v) if HEIGHT_AGG == "max" else np.nanpercentile(v, 95)

    # occupancy (Z,H,W)
    z_idx = np.floor(hh / voxel_z).astype(np.int64)
    ok2 = (z_idx >= 0) & (z_idx < z_bins)
    rows, cols, z_idx, cids = rows[ok2], cols[ok2], z_idx[ok2], cids[ok2]

    occ = np.zeros((z_bins, H, W), dtype=np.uint8)
    lin = z_idx * (H * W) + rows * W + cols
    counts = np.bincount(lin, minlength=z_bins * H * W).reshape(z_bins, H, W)
    occ[counts >= OCC_MIN_POINTS] = 1

    # majority cluster per voxel (noise-safe)
    cluster_vol = -np.ones((z_bins, H, W), dtype=np.int32)
    vdict = defaultdict(list)
    for L, cl in zip(lin, cids):
        vdict[L].append(cl)
    for L, lst in vdict.items():
        arr = np.asarray(lst, dtype=np.int32)
        pos = arr[arr >= 0]
        maj = -1 if pos.size == 0 else int(np.bincount(pos).argmax())
        z = L // (H * W); rem = L % (H * W); r = rem // W; c = rem % W
        cluster_vol[z, r, c] = maj

    # extras: top-of-column index & height (handy for slicing / ML features)
    has_any = occ.any(axis=0)
    top_z_idx = np.where(has_any, occ.argmax(axis=0), -1).astype(np.int16)
    top_height_m = np.full((H, W), np.nan, dtype=np.float32)
    good = top_z_idx >= 0
    top_height_m[good] = (top_z_idx[good] + 0.5) * voxel_z  # bin center

    # save tensors
    np.save(OUT_DIR / "occupancy_ZHW_uint8.npy", occ)
    np.save(OUT_DIR / f"height_{HEIGHT_AGG}_HxW_float32.npy", col_height)
    np.save(OUT_DIR / "cluster_ZHW_int32.npy", cluster_vol)
    np.save(OUT_DIR / "top_z_index_HxW_int16.npy", top_z_idx)
    np.save(OUT_DIR / "top_height_HxW_float32.npy", top_height_m)

    # rasters: height (Z-channel), top_z_index, top_height_m
    meta_z = meta.copy(); meta_z.update(dtype="float32", count=1, nodata=np.nan)
    with rasterio.open(OUT_DIR / f"height_{HEIGHT_AGG}.tif", "w", **meta_z) as dst:
        dst.write(col_height, 1)
    meta_i = meta.copy(); meta_i.update(dtype="int16", count=1, nodata=-1)
    with rasterio.open(OUT_DIR / "top_z_index.tif", "w", **meta_i) as dst:
        dst.write(top_z_idx, 1)
    with rasterio.open(OUT_DIR / "top_height_m.tif", "w", **meta_z) as dst:
        dst.write(top_height_m, 1)

    write_meta(OUT_DIR / "raster_tensors_meta.json", {
        "dz": voxel_z, "z_bins": z_bins, "zmax": Z_MAX,
        "occ_min_points": OCC_MIN_POINTS, "height_agg": HEIGHT_AGG
    })

    print("✅ raster-aligned tensors (occ, height, cluster, top_z_index, top_height_m)")

    # ========= Strata occupancy (fast, vectorised) =========
    edges = np.asarray(STRATA_EDGES, dtype=np.float32)
    if not np.all(edges[1:] > edges[:-1]):
        raise ValueError("STRATA_EDGES must be strictly increasing.")
    strata_id = np.clip(np.searchsorted(edges, hh, side="right") - 1, 0, len(edges) - 2)

    S = len(edges) - 1
    strata_counts = np.zeros((S, H, W), dtype=np.int32)
    # vectorised increment via linear indices
    lin_pix = rows * W + cols
    for s in range(S):
        m = strata_id == s
        if np.any(m):
            np.add.at(strata_counts[s].ravel(), lin_pix[m], 1)

    total = strata_counts.sum(axis=0).astype(np.float32)
    with np.errstate(divide='ignore', invalid='ignore'):
        strata_frac = np.where(total[None, ...] > 0, strata_counts / total[None, ...], 0.0).astype(np.float32)

    meta_s = meta.copy(); meta_s.update(count=S, dtype="float32", nodata=np.nan)
    strata_tif = OUT_DIR / "strata_fraction.tif"
    import rasterio
    with rasterio.open(strata_tif, "w", **meta_s) as dst:
        for b in range(S):
            dst.write(strata_frac[b], b+1)
            dst.set_band_description(b+1, f"{edges[b]}–{edges[b+1]} m HAG")
    print(f"✅ strata fractions: {strata_tif}")

# %%
###test 6
# --- CLIP 2D MASK TO 3D HAG-VOXEL OCCUPANCY ---
from pathlib import Path
import numpy as np
import rasterio
from rasterio.enums import Resampling

# ===== inputs =====
OUT_DIR = Path("/media/laura/laura_usb/qgis/calperumResearch/site1_1_DD0001/lidar/clipped/z_norm/voxel_hag_109")  # same folder as your tensors
OCC_NPY = OUT_DIR / "occupancy_ZHW_uint8.npy"                    # (Z,H,W)
LABEL_TIF = Path("/media/laura/Extreme SSD/qgis/calperumResearch/site1_1_DD0001/inputs/masks/tiles_3072/raw/fvc_class/annotation_raster/mask_fvc_3072.109.tif")  # 2D labels
DZ = 0.25                                    # must match how you built occ (VOXEL_SIZE[2])
LABELS_TO_KEEP = [1, 2, 3]                   # which label codes define “valid voxels”
EXPORT_PER_CLASS = True                      # also write one 3D volume per label
EXPORT_FIXED_SLICES_M = [0.5, 1.0, 2.0]      # fixed-height slices to export after masking (m)
SLICE_INDEX_MODE = "nearest"                 # 'nearest' or 'floor'

# ===== load occupancy =====
occ = np.load(OCC_NPY)  # uint8, shape (Z,H,W) with 0/1
Z, H, W = occ.shape

# ===== load / resample label raster to (H,W) =====
with rasterio.open(LABEL_TIF) as ds:
    label = ds.read(1)
    meta = ds.meta.copy()
    if label.shape != (H, W):
        label = ds.read(1, out_shape=(H, W), resampling=Resampling.nearest)
    transform = ds.transform

# ===== build 2D mask and broadcast to Z =====
valid2d = np.isin(label, np.asarray(LABELS_TO_KEEP))
valid3d = valid2d[None, :, :]               # (1,H,W) for broadcasting

# ===== apply mask =====
occ_masked = (occ.astype(bool) & valid3d).astype(np.uint8)  # (Z,H,W)
np.save(OUT_DIR / "occ_masked_ZHW_uint8.npy", occ_masked)

# ===== top-of-column AFTER masking (handy for maps/slicing) =====
has_any = occ_masked.any(axis=0)            # (H,W)
top_z_idx = np.where(has_any, occ_masked.argmax(axis=0), -1).astype(np.int16)
top_h_m = np.full((H, W), np.nan, dtype=np.float32)
good = top_z_idx >= 0
top_h_m[good] = (top_z_idx[good] + 0.5) * DZ

# write rasters (aligned to label grid)
meta_h = meta.copy(); meta_h.update(dtype="float32", count=1, nodata=np.nan)
meta_i = meta.copy(); meta_i.update(dtype="int16", count=1, nodata=-1)
with rasterio.open(OUT_DIR / "top_z_index_masked.tif", "w", **meta_i) as dst:
    dst.write(top_z_idx, 1)
with rasterio.open(OUT_DIR / "top_height_m_masked.tif", "w", **meta_h) as dst:
    dst.write(top_h_m, 1)

# ===== optional: per-class 3D volumes =====
if EXPORT_PER_CLASS:
    classes = np.unique(label[np.isfinite(label)]).astype(int)
    for c in classes:
        c2d = (label == c)
        if not c2d.any(): 
            continue
        c3d = c2d[None, :, :]
        occ_c = (occ.astype(bool) & c3d).astype(np.uint8)
        np.save(OUT_DIR / f"occ_label{c}_ZHW_uint8.npy", occ_c)

# ===== optional: export fixed-height Z-slices AFTER masking =====
heights = [float(h) for h in EXPORT_FIXED_SLICES_M if np.isfinite(h) and h >= 0]
if heights:
    # map requested heights to z indices
    z_bins = Z
    z_idx_list = []
    for h in heights:
        if SLICE_INDEX_MODE == "floor":
            zi = int(np.floor(h / DZ))
        else:
            zi = int(np.round(h / DZ))
        zi = int(np.clip(zi, 0, z_bins - 1))
        z_idx_list.append(zi)

    # stack bands [B,H,W] (binary 0/1)
    slices = np.stack([occ_masked[zi].astype(np.uint8) for zi in z_idx_list], axis=0)

    meta_s = meta.copy()
    meta_s.update(count=len(z_idx_list), dtype="uint8", nodata=0)
    out_slices = OUT_DIR / "z_slices_masked_fixed_heights.tif"
    with rasterio.open(out_slices, "w", **meta_s) as dst:
        for b, (zi, h) in enumerate(zip(z_idx_list, heights), start=1):
            dst.write(slices[b-1], b)
            dst.set_band_description(b, f"Z≈{h:.2f} m (idx {zi})")

# %%
