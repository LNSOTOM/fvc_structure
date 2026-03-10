
#%%
from pathlib import Path
import numpy as np
import laspy

# Try real HDBSCAN; otherwise fallback to sklearn.DBSCAN
try:
    from hdbscan import HDBSCAN
    USE_HDBSCAN = True
except Exception:
    from sklearn.cluster import DBSCAN as HDBSCAN
    USE_HDBSCAN = False
    print("⚠️ Using sklearn.DBSCAN fallback. Install 'hdbscan' for better results.")

# ====== INPUT / OUTPUT ======
normalised_output_las = Path("/media/laura/laura_usb/qgis/calperumResearch/site1_1_DD0001/lidar/clipped/z_norm/triple_AusGeoid_tiles_multispectral.26_HAG_delaunay.las")
out_clusters_all = normalised_output_las.with_name(normalised_output_las.stem + "_HDBSCAN.las")
out_clusters_canopy = normalised_output_las.with_name(normalised_output_las.stem + "_HDBSCAN_canopy.las")

# ====== PARAMS ======
cutoff_m = 0.05                 # HAG cutoff to remove residual ground
use_dims = ("x", "y", "Z_norm") # features for clustering
min_cluster_size = 50
min_samples = 20
metric = "euclidean"
cluster_selection_epsilon = 0.0

# ====== LOAD LAS ======
las = laspy.read(str(normalised_output_las))

# HAG dim
if "Z_norm" in las.point_format.dimension_names:
    hag = np.asarray(las["Z_norm"], dtype=np.float32)
elif "HeightAboveGround" in las.point_format.dimension_names:
    hag = np.asarray(las["HeightAboveGround"], dtype=np.float32)
else:
    raise RuntimeError("No Z_norm / HeightAboveGround dimension found in LAS.")

# Basic arrays (✅ wrap ScaledArrayView)
x = np.asarray(las.x, dtype=np.float32)
y = np.asarray(las.y, dtype=np.float32)

# Can optionally normalise XY so distances are well scaled (helps HDBSCAN/DBSCAN)
x0, y0 = np.median(x), np.median(y)
x_scaled = (x - x0).astype(np.float32)
y_scaled = (y - y0).astype(np.float32)

# Build features
feat_map = {
    "x": x_scaled,
    "y": y_scaled,
    "Z_norm": hag,
    "HeightAboveGround": hag,
}
mask = hag >= cutoff_m
if not mask.any():
    raise RuntimeError("Cutoff removed all points; lower 'cutoff_m' or check HAG values.")
X = np.column_stack([feat_map[d][mask] for d in use_dims]).astype(np.float32)

# ====== CLUSTER ======
if USE_HDBSCAN:
    model = HDBSCAN(min_cluster_size=min_cluster_size,
                    min_samples=min_samples,
                    metric=metric,
                    cluster_selection_epsilon=cluster_selection_epsilon,
                    core_dist_n_jobs=-1)
    labels = model.fit_predict(X)
    probs = model.probabilities_.astype(np.float32)
else:
    # crude fallback params—tune eps for your scale
    model = HDBSCAN(eps=0.5, min_samples=min_samples, metric=metric, n_jobs=-1)
    labels = model.fit_predict(X)
    probs = np.ones_like(labels, dtype=np.float32)
    probs[labels == -1] = 0.0

# Expand labels to full length (noise for filtered-out points)
n = las.header.point_count
cluster_id_full = np.full(n, -1, dtype=np.int32)
cluster_prob_full = np.zeros(n, dtype=np.float32)
cluster_id_full[mask] = labels.astype(np.int32)
cluster_prob_full[mask] = probs

# ====== WRITE back to LAS ======
def ensure_extra_dim(lobj, name, dtype):
    if name not in lobj.point_format.dimension_names:
        lobj.add_extra_dim(laspy.ExtraBytesParams(name=name, type=dtype))

ensure_extra_dim(las, "cluster_id", np.int32)
ensure_extra_dim(las, "cluster_prob", np.float32)
las["cluster_id"] = cluster_id_full
las["cluster_prob"] = cluster_prob_full
las.write(str(out_clusters_all))
print(f"✅ Wrote clustered LAS (all points): {out_clusters_all}")
print(f"   clusters: {len(set(labels) - {-1})}, noise: {(labels==-1).sum()} / {mask.sum()} kept")

# ====== OPTIONAL: canopy-only export ======
canopy = laspy.LasData(las.header)
canopy.points = las.points[mask]
# Make sure the ExtraBytes exist on the canopy header too
ensure_extra_dim(canopy, "cluster_id", np.int32)
ensure_extra_dim(canopy, "cluster_prob", np.float32)
canopy["cluster_id"] = labels.astype(np.int32)
canopy["cluster_prob"] = probs.astype(np.float32)
canopy.write(str(out_clusters_canopy))
print(f"✅ Wrote canopy-only clustered LAS: {out_clusters_canopy}")

# %%
## TRYIN IMPRVING PREVIOUS
#!/usr/bin/env python3
# hdbscan_cluster_hag_xy_color.py
from pathlib import Path
import numpy as np
import pandas as pd
import laspy
from sklearn.preprocessing import StandardScaler

# Try real HDBSCAN (pip install hdbscan); otherwise fallback to sklearn.DBSCAN.
try:
    from hdbscan import HDBSCAN
    from hdbscan.prediction import approximate_predict
    USE_HDBSCAN = True
    HAS_APPROX = True
except Exception:
    from sklearn.cluster import DBSCAN as HDBSCAN
    approximate_predict = None
    USE_HDBSCAN = False
    HAS_APPROX = False
    print("⚠️ Using sklearn.DBSCAN fallback. Install 'hdbscan' for best results.")

# Optional: perceptual Lab color (pip install scikit-image)
try:
    from skimage.color import rgb2lab
    HAS_LAB = True
except Exception:
    HAS_LAB = False

# ====== INPUT / OUTPUT ======
IN_LAS  = Path("/media/laura/laura_usb/qgis/calperumResearch/site1_1_DD0001/lidar/clipped/z_norm/triple_AusGeoid_tiles_multispectral.26_HAG_delaunay.las")
OUT_ALL = IN_LAS.with_name(IN_LAS.stem + "_HDBSCAN.las")
OUT_CAN = IN_LAS.with_name(IN_LAS.stem + "_HDBSCAN_canopy.las")
OUT_CSV = IN_LAS.with_name(IN_LAS.stem + "_HDBSCAN_stats.csv")
OUT_MAP = IN_LAS.with_name(IN_LAS.stem + "_HDBSCAN_label_map.csv")

# ====== FILTER / FEATURE PARAMS ======
HAG_DIM_NAME = None  # auto: "Z_norm" or "HeightAboveGround"
HAG_MIN = 0.05       # keep ≥ this HAG (m). Try 0.07–0.10 to trim more ground.
USE_RGB = True       # use true RGB if available
USE_LAB = True       # convert RGB→Lab if skimage available
XY_WEIGHT = 1.0
Z_WEIGHT  = 0.8
COL_WEIGHT = 0.6     # overall weight for (Lab or RGB)
LAB_L_WEIGHT = 1.0   # within-Lab weights
LAB_AB_WEIGHT = 1.0

# ====== HDBSCAN PARAMS ======
USE_SAMPLE = True          # sample + approximate_predict for speed on big sets
MAX_FIT_POINTS = 300_000   # sample size for model fit
MIN_CLUSTER_SIZE = 800
MIN_SAMPLES = 120
EPSILON = 0.0              # keep 0 unless you know you need it
ALLOW_SINGLE = False

# ====== POST-CLUSTER ID REMAP (0..10) ======
MAX_LABELS = 11                  # results in IDs 0..10
COLLAPSE_EXTRAS_TO_OTHER = True  # map all extra clusters into ID 10
COLLAPSE_NOISE_TO_OTHER  = False # set True to also map noise (-1) to 10

# ====== STATS CONFIG ======
Z_THRESHOLDS  = [0.02, 0.05, 0.10, 0.30, 1.0]
Z_PERCENTILES = [50, 75, 90, 95, 99, 99.9]

# ====== LOAD ======
las = laspy.read(str(IN_LAS))
dims = set(las.point_format.dimension_names)

# HAG
if "Z_norm" in dims:
    HAG_DIM_NAME = "Z_norm"
elif "HeightAboveGround" in dims:
    HAG_DIM_NAME = "HeightAboveGround"
else:
    raise RuntimeError("No Z_norm / HeightAboveGround in LAS.")
hag = np.asarray(las[HAG_DIM_NAME], dtype=np.float32)

# XY (ScaledArrayView → np array)
x = np.asarray(las.x, dtype=np.float32)
y = np.asarray(las.y, dtype=np.float32)

# Keep canopy by HAG
keep = hag >= HAG_MIN
if not keep.any():
    raise RuntimeError("No points left after HAG_MIN filter. Lower HAG_MIN.")
xk, yk, zk = x[keep], y[keep], hag[keep]

# Optional color features (only if true RGB exists)
def get_rgb(las_obj):
    if {"red","green","blue"}.issubset(las_obj.point_format.dimension_names):
        r = np.asarray(las_obj.red, dtype=np.float32) / 65535.0
        g = np.asarray(las_obj.green, dtype=np.float32) / 65535.0
        b = np.asarray(las_obj.blue, dtype=np.float32) / 65535.0
        rgb = np.stack([r,g,b], axis=1)
        return np.clip(rgb, 0.0, 1.0)
    return None

rgb_full = get_rgb(las) if USE_RGB else None
color_feats = None
if rgb_full is not None:
    cf = rgb_full[keep].astype(np.float32)
    if USE_LAB and HAS_LAB:
        lab = rgb2lab(cf.reshape(-1,1,3)).reshape(-1,3).astype(np.float32)
        lab[:,0] *= LAB_L_WEIGHT
        lab[:,1:] *= LAB_AB_WEIGHT
        color_feats = lab
    else:
        color_feats = cf

# ====== FEATURE STACK ======
# Center XY to make distance scales nicer; then apply weights and standardize.
x0, y0 = np.median(xk), np.median(yk)
xy_block  = np.stack([(xk - x0) * XY_WEIGHT, (yk - y0) * XY_WEIGHT], axis=1)
z_block   = (zk * Z_WEIGHT)[:, None]
blocks = [xy_block, z_block]
if color_feats is not None:
    blocks.append(color_feats * COL_WEIGHT)
feat = np.concatenate(blocks, axis=1)
X = StandardScaler(with_mean=True, with_std=True).fit_transform(feat).astype(np.float32, copy=False)

# ====== CLUSTER ======
def fit_hdbscan(Xfit):
    return HDBSCAN(min_cluster_size=MIN_CLUSTER_SIZE,
                   min_samples=MIN_SAMPLES,
                   cluster_selection_method="eom",
                   cluster_selection_epsilon=EPSILON,
                   allow_single_cluster=ALLOW_SINGLE,
                   core_dist_n_jobs=-1,
                   prediction_data=True).fit(Xfit)

if USE_HDBSCAN and USE_SAMPLE and X.shape[0] > MAX_FIT_POINTS and HAS_APPROX:
    rng = np.random.default_rng(42)
    samp = rng.choice(X.shape[0], size=MAX_FIT_POINTS, replace=False)
    model = fit_hdbscan(X[samp])
    labels = np.full(X.shape[0], -1, np.int32)
    labels[samp] = model.labels_.astype(np.int32)
    rest_mask = np.ones(X.shape[0], bool); rest_mask[samp] = False
    if rest_mask.any():
        probs_rest, labs_rest = approximate_predict(model, X[rest_mask])
        labels[rest_mask] = labs_rest.astype(np.int32)
        probs = np.zeros(X.shape[0], np.float32)
        probs[samp] = model.probabilities_.astype(np.float32)
        probs[rest_mask] = probs_rest.astype(np.float32)
    else:
        probs = model.probabilities_.astype(np.float32)
elif USE_HDBSCAN:
    model = fit_hdbscan(X)
    labels = model.labels_.astype(np.int32)
    probs = model.probabilities_.astype(np.float32)
else:
    # DBSCAN fallback — tune eps to your scaled feature space (0.3–1.0 common)
    model = HDBSCAN(eps=0.5, min_samples=MIN_SAMPLES, n_jobs=-1)
    labels = model.fit_predict(X).astype(np.int32)
    probs = np.ones_like(labels, dtype=np.float32)
    probs[labels == -1] = 0.0

# ====== REMAP LABELS TO 0..10 ======
# Keep top 11 clusters by size → map to 0..10; extras → 10 (if enabled); noise → stays -1 (or to 10 if enabled)
orig_labels = labels.copy()
valid = labels >= 0
label_map_rows = []

if valid.any():
    vals, counts = np.unique(labels[valid], return_counts=True)
    order = vals[np.argsort(-counts)]
    keep_vals = order[:MAX_LABELS]
    remap = {v: i for i, v in enumerate(keep_vals)}

    new_labels = np.full(labels.shape, -1, dtype=np.int32)
    keep_mask = np.isin(labels, keep_vals)
    if keep_mask.any():
        new_labels[keep_mask] = np.vectorize(remap.get)(labels[keep_mask]).astype(np.int32)

    if COLLAPSE_EXTRAS_TO_OTHER:
        extra_mask = valid & ~keep_mask
        if extra_mask.any():
            new_labels[extra_mask] = MAX_LABELS - 1  # ID 10

    if COLLAPSE_NOISE_TO_OTHER:
        new_labels[labels == -1] = MAX_LABELS - 1  # send noise to 10

    # Save mapping table (original -> remapped)
    for v, c in zip(vals, counts):
        label_map_rows.append({"orig_label": int(v),
                               "count": int(c),
                               "remapped_to": int(remap.get(v, MAX_LABELS - 1 if COLLAPSE_EXTRAS_TO_OTHER else -1))})
    labels = new_labels
else:
    # all noise
    pass

# ====== MAP BACK TO FULL LENGTH ======
n = las.header.point_count
cluster_id_full = np.full(n, -1, np.int32)
cluster_prob_full = np.zeros(n, np.float32)
cluster_id_full[keep] = labels
cluster_prob_full[keep] = probs

# ====== WRITE EXTRAS ======
def ensure_extra(lobj, name, dtype):
    if name not in lobj.point_format.dimension_names:
        lobj.add_extra_dim(laspy.ExtraBytesParams(name=name, type=dtype))

ensure_extra(las, "cluster_id", np.int32)
ensure_extra(las, "cluster_prob", np.float32)
las["cluster_id"] = cluster_id_full
las["cluster_prob"] = cluster_prob_full
las.write(str(OUT_ALL))
print(f"✅ Wrote clustered (all points): {OUT_ALL}")

# canopy-only slice
can = laspy.LasData(las.header)
can.points = las.points[keep]
ensure_extra(can, "cluster_id", np.int32)
ensure_extra(can, "cluster_prob", np.float32)
can["cluster_id"] = labels.astype(np.int32)
can["cluster_prob"] = probs.astype(np.float32)
can.write(str(OUT_CAN))
print(f"✅ Wrote canopy-only: {OUT_CAN}")

# ====== STATS ======
def cluster_stats(xv, yv, zv, labels_arr):
    df = pd.DataFrame({"label": labels_arr, "x": xv, "y": yv, "z": zv})
    df = df[df["label"] >= 0]  # exclude noise
    if df.empty:
        return pd.DataFrame()
    g = df.groupby("label", sort=True)
    out = pd.DataFrame({"n_points": g.size()})
    # percentiles
    for p in Z_PERCENTILES:
        out[f"z_p{str(p).replace('.','_')}"] = g["z"].quantile(p/100)
    # thresholds
    for t in Z_THRESHOLDS:
        out[f"frac_z_gt_{t}m"] = g["z"].apply(lambda s, tt=t: (s > tt).mean())
    # extents
    out["z_min"] = g["z"].min(); out["z_max"] = g["z"].max()
    out["z_range"] = out["z_max"] - out["z_min"]
    out["minx"] = g["x"].min(); out["maxx"] = g["x"].max()
    out["miny"] = g["y"].min(); out["maxy"] = g["y"].max()
    return out.sort_values("n_points", ascending=False).reset_index().rename(columns={"label":"cluster_id"})

stats = cluster_stats(xk, yk, zk, labels)
if not stats.empty:
    stats.to_csv(OUT_CSV, index=False)
    print(f"✅ Wrote stats: {OUT_CSV}")
    print(stats.head(10))
else:
    print("All noise after remap. Consider lowering MIN_CLUSTER_SIZE / MIN_SAMPLES.")

# ====== SAVE LABEL MAP (original → remapped) ======
if label_map_rows:
    pd.DataFrame(label_map_rows).to_csv(OUT_MAP, index=False)
    print(f"📄 Saved label map: {OUT_MAP}")

# %%
