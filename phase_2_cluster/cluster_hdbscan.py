#!/usr/bin/env python3
# HDBSCAN on XY + Z_norm(HAG), write cluster_id to LAS/LAZ + per-cluster stats CSV

#%%
from pathlib import Path
import numpy as np
import pandas as pd
import laspy

# ---- Guard: don't shadow the package ----
if Path(__file__).stem == "hdbscan":
    raise SystemExit("This file is named 'hdbscan.py' and shadows the 'hdbscan' package. "
                     "Rename it (e.g., 'cluster_hdbscan.py') and run again.")

# pip install hdbscan scikit-learn shapely pandas
from hdbscan import HDBSCAN
try:
    from hdbscan.prediction import approximate_predict
    _HAS_APPROX = True
except Exception:
    approximate_predict = None
    _HAS_APPROX = False

from sklearn.preprocessing import StandardScaler
from shapely.geometry import MultiPoint

# ====== INPUTS / OUTPUTS ======
IN_LAS  = Path("/media/laura/Laura 102/uas_data/site1_1_DD0001/lidar/trim/smrf/cloud1457eb47b07af181_clip_raster_extent_HAG.las")
OUT_LAZ = IN_LAS.with_name(IN_LAS.stem + "_hdbscan_Znorm.laz")
OUT_CSV = IN_LAS.with_name(IN_LAS.stem + "_hdbscan_Znorm_stats.csv")

# ====== CLUSTERING PARAMS ======
HAG_MIN = 0.30
XY_WEIGHT = 1.0
Z_WEIGHT  = 0.8

USE_SAMPLE     = True
MAX_FIT_POINTS = 300_000

MIN_CLUSTER_SIZE = 800
MIN_SAMPLES      = 120
EPSILON          = 0.0
ALLOW_SINGLE     = False

Z_THRESHOLDS  = [0.02, 0.05, 0.10, 0.30, 1.0]
Z_PERCENTILES = [50, 75, 90, 95, 99, 99.9]

# ====== LOAD ======
las = laspy.read(str(IN_LAS))
dims = set(las.point_format.dimension_names)

if "Z_norm" in dims:
    z = np.asarray(las["Z_norm"], dtype=np.float32)
elif "HAG" in dims:
    z = np.asarray(las["HAG"], dtype=np.float32)
else:
    raise RuntimeError("No Z_norm/HAG dimension found in LAS.")

x = np.asarray(las.x, dtype=np.float32)
y = np.asarray(las.y, dtype=np.float32)

keep = z > HAG_MIN
idx  = np.where(keep)[0]
if idx.size == 0:
    raise RuntimeError("No points above HAG_MIN; lower the threshold.")

xv, yv, zv = x[idx], y[idx], z[idx]

# ====== FEATURES ======
xv_c = xv - np.median(xv)
yv_c = yv - np.median(yv)

feat = np.stack([xv_c * XY_WEIGHT, yv_c * XY_WEIGHT, zv * Z_WEIGHT], axis=1)
X = StandardScaler().fit_transform(feat).astype(np.float32, copy=False)

# ====== HDBSCAN ======
def fit_hdbscan(Xfit):
    return HDBSCAN(
        min_cluster_size=MIN_CLUSTER_SIZE,
        min_samples=MIN_SAMPLES,
        cluster_selection_method="eom",
        cluster_selection_epsilon=EPSILON,
        allow_single_cluster=ALLOW_SINGLE,
        core_dist_n_jobs=-1,
        prediction_data=True,
    ).fit(Xfit)

# If approximate_predict isn't available, disable sampling
if USE_SAMPLE and not _HAS_APPROX:
    print("ℹ️ hdbscan.prediction.approximate_predict not available — running without sampling.")
    USE_SAMPLE = False

if USE_SAMPLE and len(X) > MAX_FIT_POINTS:
    rng = np.random.default_rng(42)
    samp = rng.choice(len(X), size=MAX_FIT_POINTS, replace=False)
    model = fit_hdbscan(X[samp])
    labels = np.full(len(X), -1, np.int32)
    labels[samp] = model.labels_.astype(np.int32)

    rest = np.ones(len(X), bool); rest[samp] = False
    if rest.any():
        _, lab_rest = approximate_predict(model, X[rest])  # we know it's available here
        labels[rest] = lab_rest.astype(np.int32)
else:
    model = fit_hdbscan(X)
    labels = model.labels_.astype(np.int32)

labels01 = labels + 1
n_clusters = int(labels01.max())
noise_frac = float((labels01 == 0).sum()) / len(labels01)
print(f"Clusters: {n_clusters} | noise fraction: {noise_frac:.3f}")

# ====== WRITE cluster_id ======
n_points = len(las.X)
cluster_full = np.zeros(n_points, dtype=np.int32)
cluster_full[idx] = labels01

from laspy import ExtraBytesParams
if "cluster_id" not in las.point_format.extra_dimension_names:
    las.add_extra_dim(ExtraBytesParams(name="cluster_id", type=np.int32))
las["cluster_id"] = cluster_full

try:
    las.write(str(OUT_LAZ))
except Exception as e:
    print(f"LAZ write failed ({e}); writing LAS.")
    OUT_LAZ = OUT_LAZ.with_suffix(".las")
    las.write(str(OUT_LAZ))
print(f"✅ Wrote clustered point cloud: {OUT_LAZ}")

# ====== STATS ======
def cluster_stats(idx_keep, labels01, xv, yv, zv):
    df = pd.DataFrame({"label": labels01, "x": xv, "y": yv, "z": zv})
    dfc = df[df["label"] > 0].copy()
    if dfc.empty:
        return pd.DataFrame()

    g = dfc.groupby("label", sort=True)
    out = pd.DataFrame({"n_points": g.size()})

    for p in Z_PERCENTILES:
        out[f"z_p{str(p).replace('.','_')}"] = g["z"].quantile(p/100)

    for t in Z_THRESHOLDS:
        out[f"frac_z_gt_{t}m"] = g.apply(lambda s, tt=t: (s["z"] > tt).mean())

    out["z_min"]   = g["z"].min()
    out["z_max"]   = g["z"].max()
    out["z_range"] = out["z_max"] - out["z_min"]

    out["minx"] = g["x"].min(); out["maxx"] = g["x"].max()
    out["miny"] = g["y"].min(); out["maxy"] = g["y"].max()

    areas = []
    for _, grp in g:
        P = grp[["x","y"]].to_numpy()
        if len(P) < 3:
            areas.append(0.0); continue
        if len(P) > 10000:
            sel = np.random.default_rng(0).choice(len(P), 10000, replace=False)
            P = P[sel]
        hull = MultiPoint(P).convex_hull
        areas.append(float(hull.area))
    out["hull_area_m2"] = areas

    return out.reset_index().rename(columns={"label":"cluster_id"}).sort_values("n_points", ascending=False)

stats = cluster_stats(idx, labels01, xv, yv, zv)
if not stats.empty:
    stats.to_csv(OUT_CSV, index=False)
    print(f"✅ Wrote stats: {OUT_CSV}")
    print(stats.head(10))
else:
    print("All noise. Try lowering MIN_CLUSTER_SIZE / MIN_SAMPLES or raising Z_WEIGHT.")


# %% TEST 2
#!/usr/bin/env python3
# Cluster LiDAR using XY + Z_norm(HAG) + Color (RGB/Lab), write cluster_id to LAS/LAZ + stats CSV

from pathlib import Path
import numpy as np
import pandas as pd
import laspy
from sklearn.preprocessing import StandardScaler
from hdbscan import HDBSCAN
try:
    from hdbscan.prediction import approximate_predict
    _HAS_APPROX = True
except Exception:
    approximate_predict = None
    _HAS_APPROX = False

# Optional: perceptual color space
try:
    from skimage.color import rgb2lab   # pip install scikit-image
    _HAS_LAB = True
except Exception:
    _HAS_LAB = False

# ====== INPUTS / OUTPUTS ======
IN_LAS  = Path("/media/laura/Laura 102/uas_data/site1_1_DD0001/lidar/trim/smrf/cloud1457eb47b07af181_clip_raster_extent_HAG.las")
OUT_LAZ = IN_LAS.with_name(IN_LAS.stem + "_hdbscan_Znorm_COLOR.laz")
OUT_CSV = IN_LAS.with_name(IN_LAS.stem + "_hdbscan_Znorm_COLOR_stats.csv")

# ====== CLUSTERING PARAMS ======
HAG_MIN       = 0.30   # ignore near-ground
XY_WEIGHT     = 1.0
Z_WEIGHT      = 0.8
COLOR_WEIGHT  = 0.6    # overall weight for color block (Lab or RGB)

# If using Lab, you can bias chroma vs lightness:
LAB_L_WEIGHT  = 1.0
LAB_AB_WEIGHT = 1.0

USE_SAMPLE     = True
MAX_FIT_POINTS = 300_000
MIN_CLUSTER_SIZE = 800
MIN_SAMPLES      = 120
EPSILON          = 0.0
ALLOW_SINGLE     = False

# Z_norm stats config
Z_THRESHOLDS  = [0.02, 0.05, 0.10, 0.30, 1.0]
Z_PERCENTILES = [50, 75, 90, 95, 99, 99.9]

# Fallback if RGB is missing: derive “fake RGB” from Z_norm (⚠️ redundant with Z_norm)
ALLOW_FAKE_RGB_FROM_ZNORM = True
FAKE_CMAP = "viridis"  # only if deriving RGB from Z_norm
from matplotlib import cm  # for fallback mapping; pip install matplotlib

# ====== LOAD ======
las = laspy.read(str(IN_LAS))
dims = set(las.point_format.dimension_names)

# Z_norm / HAG
if "Z_norm" in dims:
    z = np.asarray(las["Z_norm"], dtype=np.float32)
elif "HAG" in dims:
    z = np.asarray(las["HAG"], dtype=np.float32)
else:
    raise RuntimeError("No Z_norm/HAG dimension found in LAS.")

# Coordinates
x = np.asarray(las.x, dtype=np.float32)
y = np.asarray(las.y, dtype=np.float32)

# Filter by height
keep = z > HAG_MIN
idx  = np.where(keep)[0]
if idx.size == 0:
    raise RuntimeError("No points above HAG_MIN; lower the threshold.")

xv, yv, zv = x[idx], y[idx], z[idx]

# ====== COLOR FEATURES ======
def get_rgb_colors(las_obj: laspy.LasData):
    """Return float32 RGB in [0,1] if LAS has true color; else None."""
    if {"red","green","blue"}.issubset(las_obj.point_format.dimension_names):
        r16 = np.asarray(las_obj.red,   dtype=np.float32)
        g16 = np.asarray(las_obj.green, dtype=np.float32)
        b16 = np.asarray(las_obj.blue,  dtype=np.float32)
        # Normalize 16-bit (0..65535) → 0..1
        # Some cameras store up to ~4095/16383; dividing by 65535 is still safe.
        m = 65535.0
        rgb = np.stack([r16/m, g16/m, b16/m], axis=1)
        # Clamp & replace NaNs/inf
        rgb = np.nan_to_num(np.clip(rgb, 0.0, 1.0), nan=0.0, posinf=1.0, neginf=0.0)
        return rgb
    return None

rgb_full = get_rgb_colors(las)

if rgb_full is None and ALLOW_FAKE_RGB_FROM_ZNORM:
    # Derive pseudo-RGB from Z_norm for visualization-consistent clustering (redundant info)
    t = np.clip((z - np.percentile(z, 1.0)) /
                max(np.percentile(z, 99.9) - np.percentile(z, 1.0), 1e-9), 0, 1)
    rgb8 = (cm.get_cmap(FAKE_CMAP)(t)[:, :3] * 255 + 0.5).astype(np.uint8)
    rgb_full = (rgb8 / 255.0).astype(np.float32)

if rgb_full is None:
    raise RuntimeError("No RGB in LAS and fake-from-Z_norm disabled. Enable ALLOW_FAKE_RGB_FROM_ZNORM or add RGB.")

# take subset for kept points
rgb = rgb_full[idx, :].astype(np.float32)

# Convert to Lab if available (perceptual color distance)
if _HAS_LAB:
    # skimage expects RGB in 0..1 (sRGB), returns Lab with L* in [0,100], a*,b* ~[-128,127]
    lab = rgb2lab(rgb.reshape(-1,1,3)).reshape(-1,3).astype(np.float32)
    # Optionally weight L vs a/b before scaling
    lab[:, 0] *= LAB_L_WEIGHT
    lab[:, 1:] *= LAB_AB_WEIGHT
    color_feats = lab
else:
    # fallback: normalized RGB
    color_feats = rgb

# ====== FEATURE STACK ======
# Center XY for conditioning; Z & color get scaled later by StandardScaler
xv_c = xv - np.median(xv)
yv_c = yv - np.median(yv)

# Apply block weights before scaling
xy_block  = np.stack([xv_c * XY_WEIGHT, yv_c * XY_WEIGHT], axis=1)
z_block   = (zv * Z_WEIGHT)[:, None]
col_block = color_feats * COLOR_WEIGHT

feat = np.concatenate([xy_block, z_block, col_block], axis=1)

X = StandardScaler(with_mean=True, with_std=True).fit_transform(feat).astype(np.float32, copy=False)

# ====== HDBSCAN ======
def fit_hdbscan(Xfit):
    return HDBSCAN(
        min_cluster_size=MIN_CLUSTER_SIZE,
        min_samples=MIN_SAMPLES,
        cluster_selection_method="eom",
        cluster_selection_epsilon=EPSILON,
        allow_single_cluster=ALLOW_SINGLE,
        core_dist_n_jobs=-1,
        prediction_data=True,
    ).fit(Xfit)

if USE_SAMPLE and not _HAS_APPROX:
    print("ℹ️ approximate_predict unavailable — running without sampling.")
    USE_SAMPLE = False

if USE_SAMPLE and len(X) > MAX_FIT_POINTS:
    rng = np.random.default_rng(42)
    samp = rng.choice(len(X), size=MAX_FIT_POINTS, replace=False)
    model = fit_hdbscan(X[samp])
    labels = np.full(len(X), -1, np.int32)
    labels[samp] = model.labels_.astype(np.int32)
    rest = np.ones(len(X), bool); rest[samp] = False
    if rest.any():
        _, lab_rest = approximate_predict(model, X[rest])
        labels[rest] = lab_rest.astype(np.int32)
else:
    model = fit_hdbscan(X)
    labels = model.labels_.astype(np.int32)

labels01 = labels + 1
n_clusters = int(labels01.max())
noise_frac = float((labels01 == 0).sum()) / len(labels01)
print(f"Clusters: {n_clusters} | noise fraction: {noise_frac:.3f}")

# ====== WRITE cluster_id back to LAS/LAZ ======
n_points = len(las.X)
cluster_full = np.zeros(n_points, dtype=np.int32)
cluster_full[idx] = labels01

from laspy import ExtraBytesParams
if "cluster_id" not in las.point_format.extra_dimension_names:
    las.add_extra_dim(ExtraBytesParams(name="cluster_id", type=np.int32))
las["cluster_id"] = cluster_full

try:
    las.write(str(OUT_LAZ))
except Exception as e:
    print(f"LAZ write failed ({e}); writing LAS.")
    OUT_LAZ = OUT_LAZ.with_suffix(".las")
    las.write(str(OUT_LAZ))
print(f"✅ Wrote clustered point cloud: {OUT_LAZ}")

# ====== STATS (pandas warning-safe) ======
def cluster_stats(idx_keep, labels01, xv, yv, zv):
    df = pd.DataFrame({"label": labels01, "x": xv, "y": yv, "z": zv})
    dfc = df[df["label"] > 0].copy()
    if dfc.empty:
        return pd.DataFrame()
    g = dfc.groupby("label", sort=True)
    out = pd.DataFrame({"n_points": g.size()})
    for p in Z_PERCENTILES:
        out[f"z_p{str(p).replace('.','_')}"] = g["z"].quantile(p/100)
    for t in Z_THRESHOLDS:
        out[f"frac_z_gt_{t}m"] = g["z"].apply(lambda s, tt=t: (s > tt).mean())
    out["z_min"]   = g["z"].min()
    out["z_max"]   = g["z"].max()
    out["z_range"] = out["z_max"] - out["z_min"]
    out["minx"] = g["x"].min(); out["maxx"] = g["x"].max()
    out["miny"] = g["y"].min(); out["maxy"] = g["y"].max()
    # (add color stats if you want, e.g., median L*, a*, b*)
    return out.reset_index().rename(columns={"label":"cluster_id"}).sort_values("n_points", ascending=False)

stats = cluster_stats(idx, labels01, xv, yv, zv)
if not stats.empty:
    stats.to_csv(OUT_CSV, index=False)
    print(f"✅ Wrote stats: {OUT_CSV}")
    print(stats.head(10))
else:
    print("All noise. Try lowering MIN_CLUSTER_SIZE / MIN_SAMPLES, or increase COLOR_WEIGHT/Z_WEIGHT.")

# %%
###test 3
#!/usr/bin/env python3
# Cluster LiDAR using XY + Z_norm(HAG) + Color (RGB/Lab), write cluster_id to LAS/LAZ + stats CSV

from pathlib import Path
import numpy as np
import pandas as pd
import laspy
from sklearn.preprocessing import StandardScaler
from hdbscan import HDBSCAN
try:
    from hdbscan.prediction import approximate_predict
    _HAS_APPROX = True
except Exception:
    approximate_predict = None
    _HAS_APPROX = False

# Optional: perceptual color space
try:
    from skimage.color import rgb2lab   # pip install scikit-image
    _HAS_LAB = True
except Exception:
    _HAS_LAB = False

from matplotlib import cm  # for fallback mapping; pip install matplotlib

# ====== INPUTS / OUTPUTS ======
IN_LAS  = Path("/media/laura/Laura 102/uas_data/site1_1_DD0001/lidar/trim/smrf/cloud1457eb47b07af181_clip_raster_extent_HAG.las")
OUT_LAZ = IN_LAS.with_name(IN_LAS.stem + "_hdbscan_Znorm_COLOR_nonground.laz")
OUT_CSV = IN_LAS.with_name(IN_LAS.stem + "_hdbscan_Znorm_COLOR_stats_nonground.csv")

# ====== CLUSTERING PARAMS ======
HAG_MIN       = 0.30   # ignore near-ground
XY_WEIGHT     = 1.0
Z_WEIGHT      = 0.8
COLOR_WEIGHT  = 0.6    # overall weight for color block (Lab or RGB)

# If using Lab, you can bias chroma vs lightness:
LAB_L_WEIGHT  = 1.0
LAB_AB_WEIGHT = 1.0

USE_SAMPLE       = True
MAX_FIT_POINTS   = 300_000
MIN_CLUSTER_SIZE = 800
MIN_SAMPLES      = 120
EPSILON          = 0.0
ALLOW_SINGLE     = False

# Z_norm stats config
Z_THRESHOLDS  = [0.02, 0.05, 0.10, 0.30, 1.0]
Z_PERCENTILES = [50, 75, 90, 95, 99, 99.9]

# Fallback if RGB is missing: derive “fake RGB” from Z_norm (⚠️ redundant with Z_norm)
ALLOW_FAKE_RGB_FROM_ZNORM = True
FAKE_CMAP = "viridis"

# ====== LOAD ======
las = laspy.read(str(IN_LAS))
dims = set(las.point_format.dimension_names)

# Z_norm / HAG
if "Z_norm" in dims:
    z = np.asarray(las["Z_norm"], dtype=np.float32)
elif "HAG" in dims:
    z = np.asarray(las["HAG"], dtype=np.float32)
else:
    raise RuntimeError("No Z_norm/HAG dimension found in LAS.")

# Coordinates
x = np.asarray(las.x, dtype=np.float32)
y = np.asarray(las.y, dtype=np.float32)

# ---------- ground filtering (class OR HAG) ----------
DROP_GROUND_BY_CLASS = True          # drop LAS classification 2 if present
GROUND_CLASSES       = {2}
HAG_GROUND_MAX       = 0.20          # meters: treat <= this as ground/near-ground
PERSIST_GROUND_CLASS = False         # set True to save a copy with class=2 tagged from HAG

# class-based ground
if DROP_GROUND_BY_CLASS and "classification" in dims:
    cls = np.asarray(las.classification, dtype=np.uint8)
    is_ground_cls = np.isin(cls, list(GROUND_CLASSES))
    print(f"class-2 points: {int(is_ground_cls.sum())}")
else:
    is_ground_cls = np.zeros(len(z), dtype=bool)
    print("no 'classification' field or not dropping by class")

# HAG-based ground
is_ground_hag = z <= HAG_GROUND_MAX
print(f"HAG<= {HAG_GROUND_MAX} m points: {int(is_ground_hag.sum())}")

# combine: ground if either says ground
is_ground = is_ground_cls | is_ground_hag
not_ground = ~is_ground

# (optional) persist class=2 for HAG-ground in a side file
if PERSIST_GROUND_CLASS:
    if "classification" in dims:
        cls_out = np.asarray(las.classification, dtype=np.uint8).copy()
    else:
        # most LAS formats have classification; if not, create zeros
        cls_out = np.zeros(len(z), dtype=np.uint8)
    cls_out[is_ground_hag] = 2
    las["classification"] = cls_out
    out_tagged = IN_LAS.with_name(IN_LAS.stem + "_groundTagged.laz")
    try:
        las.write(str(out_tagged))
    except Exception:
        out_tagged = out_tagged.with_suffix(".las")
        las.write(str(out_tagged))
    print(f"✅ wrote ground-tagged copy: {out_tagged}")

# final keep set for clustering (height + not ground)
HAG_MIN = 0.30  # keep your existing value, or bump to 0.6 to focus on shrubs/crowns
keep = (z > HAG_MIN) & not_ground
idx  = np.where(keep)[0]
if idx.size == 0:
    raise RuntimeError("No points left after ground filters. Lower HAG_GROUND_MAX/HAG_MIN.")
xv, yv, zv = x[idx], y[idx], z[idx]
# -----------------------------------------------------


# ====== COLOR FEATURES ======
def get_rgb_colors(las_obj: laspy.LasData):
    """Return float32 RGB in [0,1] if LAS has true color; else None."""
    if {"red","green","blue"}.issubset(las_obj.point_format.dimension_names):
        r16 = np.asarray(las_obj.red,   dtype=np.float32)
        g16 = np.asarray(las_obj.green, dtype=np.float32)
        b16 = np.asarray(las_obj.blue,  dtype=np.float32)
        m = 65535.0
        rgb = np.stack([r16/m, g16/m, b16/m], axis=1)
        return np.nan_to_num(np.clip(rgb, 0.0, 1.0), nan=0.0, posinf=1.0, neginf=0.0)
    return None

rgb_full = get_rgb_colors(las)

if rgb_full is None and ALLOW_FAKE_RGB_FROM_ZNORM:
    lo, hi = np.percentile(z, [1.0, 99.9])
    t = np.clip((z - lo) / max(hi - lo, 1e-9), 0, 1)
    rgb8 = (cm.get_cmap(FAKE_CMAP)(t)[:, :3] * 255 + 0.5).astype(np.uint8)
    rgb_full = (rgb8 / 255.0).astype(np.float32)

if rgb_full is None:
    raise RuntimeError("No RGB in LAS and fake-from-Z_norm disabled. Enable ALLOW_FAKE_RGB_FROM_ZNORM or add RGB.")

# take subset for kept points
rgb = rgb_full[idx, :].astype(np.float32)

# Convert to Lab if available (perceptual color distance)
if _HAS_LAB:
    from skimage.color import rgb2lab
    lab = rgb2lab(rgb.reshape(-1,1,3)).reshape(-1,3).astype(np.float32)
    lab[:, 0] *= LAB_L_WEIGHT
    lab[:, 1:] *= LAB_AB_WEIGHT
    color_feats = lab
else:
    color_feats = rgb

# ====== FEATURE STACK ======
xv_c = xv - np.median(xv)
yv_c = yv - np.median(yv)
xy_block  = np.stack([xv_c * XY_WEIGHT, yv_c * XY_WEIGHT], axis=1)
z_block   = (zv * Z_WEIGHT)[:, None]
col_block = color_feats * COLOR_WEIGHT
feat = np.concatenate([xy_block, z_block, col_block], axis=1)
X = StandardScaler(with_mean=True, with_std=True).fit_transform(feat).astype(np.float32, copy=False)

# ====== HDBSCAN ======
def fit_hdbscan(Xfit):
    return HDBSCAN(
        min_cluster_size=MIN_CLUSTER_SIZE,
        min_samples=MIN_SAMPLES,
        cluster_selection_method="eom",
        cluster_selection_epsilon=EPSILON,
        allow_single_cluster=ALLOW_SINGLE,
        core_dist_n_jobs=-1,
        prediction_data=True,
    ).fit(Xfit)

if USE_SAMPLE and not _HAS_APPROX:
    print("ℹ️ approximate_predict unavailable — running without sampling.")
    USE_SAMPLE = False

if USE_SAMPLE and len(X) > MAX_FIT_POINTS:
    rng = np.random.default_rng(42)
    samp = rng.choice(len(X), size=MAX_FIT_POINTS, replace=False)
    model = fit_hdbscan(X[samp])
    labels = np.full(len(X), -1, np.int32)
    labels[samp] = model.labels_.astype(np.int32)
    rest = np.ones(len(X), bool); rest[samp] = False
    if rest.any():
        _, lab_rest = approximate_predict(model, X[rest])
        labels[rest] = lab_rest.astype(np.int32)
else:
    model = fit_hdbscan(X)
    labels = model.labels_.astype(np.int32)

labels01 = labels + 1
n_clusters = int(labels01.max())
noise_frac = float((labels01 == 0).sum()) / len(labels01)
print(f"Clusters: {n_clusters} | noise fraction: {noise_frac:.3f}")

# ====== WRITE cluster_id back to LAS/LAZ ======
n_points = len(las.X)
cluster_full = np.zeros(n_points, dtype=np.int32)
cluster_full[idx] = labels01

from laspy import ExtraBytesParams
if "cluster_id" not in las.point_format.extra_dimension_names:
    las.add_extra_dim(ExtraBytesParams(name="cluster_id", type=np.int32))
las["cluster_id"] = cluster_full

try:
    las.write(str(OUT_LAZ))
except Exception as e:
    print(f"LAZ write failed ({e}); writing LAS.")
    OUT_LAZ = OUT_LAZ.with_suffix(".las")
    las.write(str(OUT_LAZ))
print(f"✅ Wrote clustered point cloud: {OUT_LAZ}")

# ====== STATS (pandas warning-safe) ======
def cluster_stats(idx_keep, labels01, xv, yv, zv):
    df = pd.DataFrame({"label": labels01, "x": xv, "y": yv, "z": zv})
    dfc = df[df["label"] > 0].copy()
    if dfc.empty:
        return pd.DataFrame()
    g = dfc.groupby("label", sort=True)
    out = pd.DataFrame({"n_points": g.size()})
    for p in Z_PERCENTILES:
        out[f"z_p{str(p).replace('.','_')}"] = g["z"].quantile(p/100)
    for t in Z_THRESHOLDS:
        out[f"frac_z_gt_{t}m"] = g["z"].apply(lambda s, tt=t: (s > tt).mean())
    out["z_min"]   = g["z"].min()
    out["z_max"]   = g["z"].max()
    out["z_range"] = out["z_max"] - out["z_min"]
    out["minx"] = g["x"].min(); out["maxx"] = g["x"].max()
    out["miny"] = g["y"].min(); out["maxy"] = g["y"].max()
    return out.reset_index().rename(columns={"label":"cluster_id"}).sort_values("n_points", ascending=False)

stats = cluster_stats(idx, labels01, xv, yv, zv)
if not stats.empty:
    stats.to_csv(OUT_CSV, index=False)
    print(f"✅ Wrote stats: {OUT_CSV}")
    print(stats.head(10))
else:
    print("All noise. Try lowering MIN_CLUSTER_SIZE / MIN_SAMPLES, or increase COLOR_WEIGHT/Z_WEIGHT.")

# %%
def write_subset_with_cluster(las_src, mask_keep, out_path: Path):
    # build a new header with same format/version/scale/offset
    hdr = laspy.LasHeader(point_format=las_src.header.point_format, version=las_src.header.version)
    hdr.scales  = las_src.header.scales
    hdr.offsets = las_src.header.offsets
    sub = laspy.LasData(hdr)

    # copy all standard dims present
    for dim in las_src.point_format.dimension_names:
        sub[dim] = las_src[dim][mask_keep].copy()

    # ensure cluster_id exists & uint16
    from laspy import ExtraBytesParams
    if "cluster_id" not in sub.point_format.extra_dimension_names:
        sub.add_extra_dim(ExtraBytesParams(name="cluster_id", type=np.uint16))
    # slice your full cluster array
    sub["cluster_id"] = cluster_full[mask_keep].astype(np.uint16, copy=False)

    try:
        sub.write(str(out_path))
    except Exception:
        out_path = out_path.with_suffix(".las")
        sub.write(str(out_path))
    return out_path

mask_keep_full = np.zeros(len(las.X), dtype=bool)
mask_keep_full[idx] = True  # same kept set you used for clustering
OUT_LAZ_KEEP = IN_LAS.with_name(IN_LAS.stem + "_hdbscan_keptOnly.laz")
print("✅ wrote kept-only file:", write_subset_with_cluster(las, mask_keep_full, OUT_LAZ_KEEP))

# %%
