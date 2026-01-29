#%%
"""
Merge Z_norm LAS with structure LAS using:
1) Cartesian voxel majority vote (classes 64–68)
2) Polar refinement for 65↔68
3) Cylindrical (r, θ) + Z_norm fill for class 0
4) Final height sanity rule (demote tall 68 → 65)
5) HDBSCAN cluster smoothing (post-processing)

Batch mode over all tiles in a folder, writing:
- 3d_structure_cart_polar/
- 3d_structure_mask_cart_polar/
"""

import laspy
import numpy as np
import pandas as pd
from pathlib import Path
from collections import Counter

# --- HDBSCAN for post-processing clustering ----------------
try:
    import hdbscan
except ImportError as e:
    raise ImportError(
        "This script requires 'hdbscan' for cluster smoothing.\n"
        "Install with, e.g.:  micromamba install -c conda-forge hdbscan\n"
        "or: pip install hdbscan"
    ) from e

# -----------------------------------------------------------
# Class names
# -----------------------------------------------------------
CLASS_NAMES = {
    64: "Forb",
    65: "Woody Debris",
    66: "Hummock",
    67: "Shrub",
    68: "Mallee",
    0:  "Unmatched",
}

# -----------------------------------------------------------
# LAS 1.4 / Point Format 6 helper
# -----------------------------------------------------------
def ensure_las14_pf6(las_in: laspy.LasData) -> laspy.LasData:
    """
    Ensure LAS is in version 1.4 / point format 6 so that
    extended class codes (64–68) are valid.
    """
    if las_in.header.version == "1.4" and las_in.header.point_format.id >= 6:
        return las_in

    hdr = laspy.LasHeader(version="1.4", point_format=6)
    hdr.scales = las_in.header.scales
    hdr.offsets = las_in.header.offsets

    out = laspy.LasData(hdr)
    out.points = laspy.ScaleAwarePointRecord.zeros(len(las_in.points), hdr.point_format)

    # Copy coordinates
    out.X = las_in.X
    out.Y = las_in.Y
    out.Z = las_in.Z

    # Copy overlapping standard dims
    src_dims = set(las_in.point_format.dimension_names)
    dst_dims = set(out.point_format.dimension_names)
    for dim in (src_dims & dst_dims):
        if dim not in ("X", "Y", "Z"):
            out[dim] = las_in[dim]

    # Copy ExtraBytes
    for eb in las_in.header.extra_dimensions:
        if eb.name not in out.point_format.dimension_names:
            out.add_extra_dim(eb)
        out[eb.name] = las_in[eb.name]

    return out


# -----------------------------------------------------------
# Height ranges / priors
# -----------------------------------------------------------
HEIGHT_RANGES = {
    64: (0.00, 0.40),   # Forb
    65: (0.05, 1.20),   # Woody debris
    66: (0.40, 3.00),   # Hummock
    67: (0.00, 0.80),   # Shrub
    68: (1.50, 8.00),   # Mallee
}

def soft_height_prior(z_norm: float, cls: int) -> float:
    """
    Soft prior based on Z_norm and expected height ranges for each class.
    Returns 1.0 if within the range, 0.1 otherwise.
    """
    if cls not in HEIGHT_RANGES:
        return 1.0
    zmin, zmax = HEIGHT_RANGES[cls]
    return 1.0 if (zmin <= z_norm <= zmax) else 0.1


# -----------------------------------------------------------
# Cartesian voxel merge (base labels)
# -----------------------------------------------------------
def cartesian_merge(
    las_z: laspy.LasData,
    las_s: laspy.LasData,
    voxel_size_xy: float = 0.05,
    voxel_size_z: float = 0.05,
) -> np.ndarray:
    """
    Merge structure classes (64–68) into Z_norm LAS
    via 3D voxel majority vote.
    """
    x_min = min(las_z.x.min(), las_s.x.min())
    y_min = min(las_z.y.min(), las_s.y.min())
    z_min = min(las_z.z.min(), las_s.z.min())
    origin = np.array([x_min, y_min, z_min], dtype=np.float64)

    def vox(x, y, z):
        ix = np.floor((x - origin[0]) / voxel_size_xy).astype(np.int32)
        iy = np.floor((y - origin[1]) / voxel_size_xy).astype(np.int32)
        iz = np.floor((z - origin[2]) / voxel_size_z).astype(np.int32)
        return np.stack([ix, iy, iz], axis=-1)

    # Voxel indices for structure LAS
    vox_s = vox(las_s.x, las_s.y, las_s.z)
    raw_struct = las_s.classification.astype(np.uint16)
    raw_struct_masked = np.where((raw_struct >= 64) & (raw_struct <= 68), raw_struct, 0)

    uniq_vox_s, inv_s = np.unique(vox_s, axis=0, return_inverse=True)
    voxel_class = np.zeros(len(uniq_vox_s), dtype=np.uint16)

    for k in range(len(uniq_vox_s)):
        vals = raw_struct_masked[inv_s == k]
        vals = vals[vals != 0]
        if vals.size == 0:
            continue
        voxel_class[k] = Counter(vals).most_common(1)[0][0]

    voxel_struct_map = {
        tuple(idx): c
        for idx, c in zip(uniq_vox_s, voxel_class)
        if c != 0
    }
    print(f"✅ Voxels with non-zero structure class: {len(voxel_struct_map):,}")

    # Voxel indices for Z_norm LAS
    vox_z = vox(las_z.x, las_z.y, las_z.z)
    classification = np.zeros(len(las_z.points), dtype=np.uint8)

    matched = 0
    for i, v in enumerate(vox_z):
        lbl = voxel_struct_map.get(tuple(v), 0)
        classification[i] = lbl
        if lbl != 0:
            matched += 1

    print(f"✅ Matched labels via Cartesian voxels: {matched:,} / {len(las_z.points):,}")
    return classification


# -----------------------------------------------------------
# Polar refinement only for 65 ↔ 68
#   *skips points with Z_norm <= ground_thresh*
# -----------------------------------------------------------
def refine_mallee_with_polar(
    las_z: laspy.LasData,
    las_s: laspy.LasData,
    classification: np.ndarray,
    r_bin: float = 0.10,
    theta_bin_deg: float = 5.0,
    min_conf: float = 0.3,
    min_flip_ratio: float = 1.5,
    ground_thresh: float = 0.01,
) -> np.ndarray:
    """
    Refine confusion between 65 (Woody debris) and 68 (Mallee)
    using polar (r, θ) columns and Z_norm-based priors.
    """
    cls_out = classification.copy()

    if not np.any(las_s.classification == 68):
        print("⚪ No mallee (68) in structure LAS → skip polar refinement.")
        return cls_out

    # Polar origin at center of Z_norm tile
    x0 = 0.5 * (las_z.x.min() + las_z.x.max())
    y0 = 0.5 * (las_z.y.min() + las_z.y.max())
    origin = np.array([x0, y0], dtype=float)

    def cart_to_polar(x, y):
        dx = x - origin[0]
        dy = y - origin[1]
        r = np.sqrt(dx * dx + dy * dy)
        theta = np.mod(np.arctan2(dy, dx), 2 * np.pi)
        return r, theta

    def polar_bins(r, theta):
        ir = np.floor(r / r_bin).astype(np.int32)
        it = np.floor(theta / np.deg2rad(theta_bin_deg)).astype(np.int32)
        return ir, it

    # Build column probabilities for 65 & 68 from structure LAS
    s_r, s_th = cart_to_polar(las_s.x, las_s.y)
    s_ir, s_it = polar_bins(s_r, s_th)
    mask_6568 = (las_s.classification == 65) | (las_s.classification == 68)

    df_s = pd.DataFrame({
        "ir":  s_ir[mask_6568],
        "it":  s_it[mask_6568],
        "cls": las_s.classification[mask_6568],
    })
    col_counts = df_s.groupby(["ir", "it", "cls"]).size().unstack(fill_value=0)
    col_probs = col_counts.div(col_counts.sum(axis=1), axis=0)

    # Optionally filter low-confidence columns
    if min_conf > 0:
        conf = col_probs.max(axis=1)
        col_probs = col_probs[conf >= min_conf]

    if col_probs.empty:
        print("⚪ No confident polar columns → skip refinement.")
        return cls_out

    col_map = {idx: col_probs.loc[idx] for idx in col_probs.index}
    print(f"📦 Polar columns with 65/68 evidence: {len(col_map):,}")

    # Apply to Z_norm LAS
    z_r, z_th = cart_to_polar(las_z.x, las_z.y)
    z_ir, z_it = polar_bins(z_r, z_th)
    zn = las_z["Z_norm"].astype(float)

    flips_65_to_68 = 0
    flips_68_to_65 = 0

    idx_candidates = np.where((cls_out == 65) | (cls_out == 68))[0]

    for idx in idx_candidates:
        # Do NOT touch near-ground points
        if zn[idx] <= ground_thresh:
            continue

        key = (z_ir[idx], z_it[idx])
        if key not in col_map:
            continue

        base = col_map[key]
        classes = base.index.to_numpy()
        probs = base.to_numpy()

        post = np.array([
            p * soft_height_prior(zn[idx], int(c))
            for p, c in zip(probs, classes)
        ])
        if post.sum() <= 0:
            continue

        best_cls = int(classes[np.argmax(post)])
        current = int(cls_out[idx])
        if best_cls == current:
            continue

        best_val = post[classes == best_cls][0]
        cur_val = post[classes == current][0]

        if best_val / (cur_val + 1e-9) >= min_flip_ratio:
            if current == 65 and best_cls == 68:
                flips_65_to_68 += 1
            elif current == 68 and best_cls == 65:
                flips_68_to_65 += 1
            cls_out[idx] = best_cls

    print(f"🔁 Polar flips: 65→68 = {flips_65_to_68}, 68→65 = {flips_68_to_65}")
    return cls_out


# -----------------------------------------------------------
# Cylindrical (r,θ) + Z_norm fill for class 0
# -----------------------------------------------------------
def fill_missing_with_cyl_height(
    las_z: laspy.LasData,
    las_s: laspy.LasData,
    classification: np.ndarray,
    r_bin: float = 0.10,
    theta_bin_deg: float = 5.0,
    min_conf: float = 0.3,
    ground_thresh: float = 0.01,
) -> np.ndarray:
    """
    For points still class 0, infer a class using cylindrical (r, θ)
    column probabilities from structure LAS (64–68) + Z_norm height priors.
    """
    cls_out = classification.copy()

    # Origin at center of Z_norm tile
    x0 = 0.5 * (las_z.x.min() + las_z.x.max())
    y0 = 0.5 * (las_z.y.min() + las_z.y.max())
    origin_xy = np.array([x0, y0], float)

    def cart_to_polar(x, y):
        dx = x - origin_xy[0]
        dy = y - origin_xy[1]
        r = np.sqrt(dx * dx + dy * dy)
        theta = np.arctan2(dy, dx)
        theta = np.mod(theta, 2 * np.pi)
        return r, theta

    def polar_bins(r, theta):
        ir = np.floor(r / r_bin).astype(np.int32)
        it = np.floor(theta / np.deg2rad(theta_bin_deg)).astype(np.int32)
        return ir, it

    # 1) Column probabilities from ALL extended classes 64–68
    s_r, s_th = cart_to_polar(las_s.x, las_s.y)
    s_ir, s_it = polar_bins(s_r, s_th)

    mask_ext = (las_s.classification >= 64) & (las_s.classification <= 68)
    if not np.any(mask_ext):
        print("⚪ No extended classes (64–68) in structure LAS → skip cylindrical fill.")
        return cls_out

    df_s = pd.DataFrame({
        "ir":  s_ir[mask_ext],
        "it":  s_it[mask_ext],
        "cls": las_s.classification[mask_ext],
    })
    col_counts = df_s.groupby(["ir", "it", "cls"]).size().unstack(fill_value=0)
    col_probs = col_counts.div(col_counts.sum(axis=1), axis=0)

    # Filter low-confidence columns
    if min_conf > 0.0:
        best_conf = col_probs.max(axis=1)
        col_probs = col_probs[best_conf >= min_conf]

    if col_probs.empty:
        print("⚪ No confident columns after min_conf → skip cylindrical fill.")
        return cls_out

    col_map = {idx: col_probs.loc[idx] for idx in col_probs.index}
    print(f"📦 Cylindrical columns with class evidence: {len(col_map):,}")

    # 2) Assign ONLY to currently-unlabelled, non-ground points
    z_r, z_th = cart_to_polar(las_z.x, las_z.y)
    z_ir, z_it = polar_bins(z_r, z_th)
    zn = las_z["Z_norm"].astype(float)

    zero_idx = np.where((cls_out == 0) & (zn > ground_thresh))[0]
    filled = 0

    for idx in zero_idx:
        key = (z_ir[idx], z_it[idx])
        if key not in col_map:
            continue

        base = col_map[key]
        classes = base.index.to_numpy()
        p_base = base.to_numpy()

        post = np.array([
            p * soft_height_prior(zn[idx], int(cls))
            for p, cls in zip(p_base, classes)
        ])
        if post.sum() <= 0:
            continue

        best_cls = int(classes[np.argmax(post)])
        cls_out[idx] = best_cls
        if best_cls != 0:
            filled += 1

    print(f"🧩 Cylindrical fill: labelled {filled:,} previously-unlabelled points "
          f"(only Z_norm > {ground_thresh})")
    return cls_out


# -----------------------------------------------------------
#  HDBSCAN majority-cluster smoothing (post-processing)
# -----------------------------------------------------------
def hdbscan_cluster_smoothing(
    las_z: laspy.LasData,
    classification: np.ndarray,
    min_cluster_size: int = 50,
    min_samples: int = 10,
    xy_weight: float = 1.0,
    z_weight: float = 0.5,
    noise_to_zero: bool = True,
) -> np.ndarray:
    """
    Apply HDBSCAN clustering on vegetation points (classes 64–68) to
    spatially regularise labels.

    For each cluster:
      - compute the majority class
      - assign this class to all points in the cluster

    Noise points (cluster label -1):
      - optionally set to 0 (noise_to_zero=True)
      - otherwise keep original label.

    This mimics the post-processing step in recent 3D segmentation
    pipelines where clustering is used to smooth label noise.
    """
    cls_out = classification.copy()

    # Only consider vegetation-labelled points (64–68)
    mask_veg = (cls_out >= 64) & (cls_out <= 68)
    if not np.any(mask_veg):
        print("⚪ No vegetation classes (64–68) → skip HDBSCAN smoothing.")
        return cls_out

    x = las_z.x[mask_veg]
    y = las_z.y[mask_veg]
    z = las_z.z[mask_veg]

    # Feature scaling to balance XY vs Z
    X = np.stack([
        x * xy_weight,
        y * xy_weight,
        z * z_weight,
    ], axis=1)

    print(f"🔍 HDBSCAN on {X.shape[0]:,} vegetation points ...")
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric="euclidean",
        cluster_selection_method="eom",
    )
    cluster_labels = clusterer.fit_predict(X)

    unique_clusters, counts = np.unique(cluster_labels, return_counts=True)
    n_clusters = np.sum(unique_clusters >= 0)
    n_noise = counts[unique_clusters == -1].sum() if np.any(unique_clusters == -1) else 0
    print(f"📦 HDBSCAN clusters: {n_clusters} (noise points: {n_noise:,})")

    veg_indices = np.where(mask_veg)[0]

    # Majority vote per cluster (cluster_id >= 0)
    for cid in unique_clusters:
        if cid < 0:
            continue  # skip noise here; handled below

        local_idx = np.where(cluster_labels == cid)[0]
        global_idx = veg_indices[local_idx]

        cluster_classes = cls_out[global_idx]
        majority_cls = Counter(cluster_classes).most_common(1)[0][0]

        cls_out[global_idx] = majority_cls

    # Handle noise points
    if noise_to_zero and n_noise > 0:
        noise_local = np.where(cluster_labels == -1)[0]
        noise_global = veg_indices[noise_local]
        cls_out[noise_global] = 0
        print(f"🚮 HDBSCAN: set {len(noise_global):,} noisy vegetation points → class 0")

    return cls_out


# -----------------------------------------------------------
# Full pipeline for ONE FILE
# -----------------------------------------------------------
def merge_znorm_structure_one_file(
    z_norm_las: str | Path,
    structure_las: str | Path,
    output_las: str | Path,
    output_las_no0: str | Path | None = None,
    voxel_size_xy: float = 0.05,
    voxel_size_z: float = 0.05,
    ground_thresh: float = 0.01,
):
    """
    Run full merge & refinement pipeline for a single pair of LAS files.
    """
    z_norm_las = Path(z_norm_las)
    structure_las = Path(structure_las)
    output_las = Path(output_las)
    if output_las_no0 is not None:
        output_las_no0 = Path(output_las_no0)

    las_z = laspy.read(z_norm_las)
    las_s = laspy.read(structure_las)
    print(f"🔎 Z_norm points: {len(las_z.points):,}, Structure points: {len(las_s.points):,}")

    if "Z_norm" not in las_z.point_format.dimension_names:
        raise ValueError("Missing 'Z_norm' in z_norm LAS.")

    # 1) Cartesian voxel merge
    classification = cartesian_merge(las_z, las_s, voxel_size_xy, voxel_size_z)

    # 2) Polar refinement for 65 ↔ 68 (skip near-ground)
    classification = refine_mallee_with_polar(
        las_z, las_s, classification,
        r_bin=0.10,
        theta_bin_deg=5.0,
        min_conf=0.3,
        min_flip_ratio=1.5,
        ground_thresh=ground_thresh,
    )

    # 3) Cylindrical fill for remaining class 0 points (skip near-ground)
    classification = fill_missing_with_cyl_height(
        las_z, las_s, classification,
        r_bin=0.10,
        theta_bin_deg=5.0,
        min_conf=0.3,
        ground_thresh=ground_thresh,
    )

    # 4) Final height sanity check:
    #    If Z_norm >= 3 m AND class == 68 (Mallee) → demote to 65 (Woody Debris)
    def enforce_high_veg_height_rule(classification_arr: np.ndarray, zn_arr: np.ndarray) -> np.ndarray:
        demoted = 0
        cls = classification_arr.copy()

        mask_bad = (cls == 68) & (zn_arr >= 3.0)
        for i in np.where(mask_bad)[0]:
            cls[i] = 65   # tall mallee → Woody debris
            demoted += 1

        print(f"🌲 Height rule: demoted {demoted} mallee (68) points above 3 m to woody debris (65).")
        return cls

    classification = enforce_high_veg_height_rule(
        classification,
        las_z["Z_norm"].astype(float),
    )

    # 5) HDBSCAN cluster smoothing (post-processing)
    classification = hdbscan_cluster_smoothing(
        las_z,
        classification,
        min_cluster_size=50,
        min_samples=10,
        xy_weight=1.0,
        z_weight=0.5,
        noise_to_zero=True,
    )

    # ---- write LAS ----
    las_out = laspy.LasData(las_z.header)
    las_out.points = las_z.points.copy()
    las_out.classification = classification

    # Preserve Z_norm (ensure exists as ExtraBytes)
    if "Z_norm" not in las_out.point_format.dimension_names:
        las_out.add_extra_dim(laspy.ExtraBytesParams(name="Z_norm", type=np.float32))
    las_out["Z_norm"] = las_z["Z_norm"].astype(np.float32)

    # Tag software_id (ASCII only)
    las_out.header.software_id = "Cart+Polar+Cyl+HDBSCAN"
    las_out.header.software_id = las_out.header.software_id.encode("ascii", "ignore").decode("ascii")

    # Ensure LAS 1.4 / PF6 for extended classes
    las_out = ensure_las14_pf6(las_out)

    # Write full file
    output_las.parent.mkdir(parents=True, exist_ok=True)
    las_out.write(str(output_las))

    # Optionally write "no class 0" file
    if output_las_no0 is not None:
        keep = classification != 0
        las_no0 = laspy.LasData(las_out.header)
        las_no0.points = las_out.points[keep].copy()
        output_las_no0.parent.mkdir(parents=True, exist_ok=True)
        las_no0.write(str(output_las_no0))

    # Histogram
    cnt = Counter(classification)
    print("\n📊 Final class distribution:")
    for cid in sorted(cnt):
        print(f"{cnt[cid]:>10,}  {CLASS_NAMES.get(cid, 'Unknown')} ({cid})")

    print(f"\n✅ Wrote: {output_las}")
    if output_las_no0 is not None:
        print(f"✅ Wrote (no class 0): {output_las_no0}")


# -----------------------------------------------------------
# Batch processing for all tiles in a folder
# -----------------------------------------------------------
def merge_folder(
    z_norm_dir,
    structure_dir,
    out_dir_full,
    out_dir_no0,
    voxel_size_xy=0.05,
    voxel_size_z=0.05,
    ground_thresh=0.15,
):
    """
    Batch process all Z_norm tiles in a folder. Expects:
      z_norm_dir   : z_norm_csf_tiles_multispectral.X.las
      structure_dir: mask_tiles_multispectral.X.las
    Writes:
      out_dir_full : z_norm_struct_X_cart_polar.las
      out_dir_no0  : z_norm_struct_X_cart_polar_no0.las
    """
    z_norm_dir = Path(z_norm_dir)
    structure_dir = Path(structure_dir)
    out_dir_full = Path(out_dir_full)
    out_dir_no0 = Path(out_dir_no0)

    out_dir_full.mkdir(parents=True, exist_ok=True)
    out_dir_no0.mkdir(parents=True, exist_ok=True)

    # Find z_norm LAS tiles
    z_files = sorted(z_norm_dir.glob("z_norm_csf_tiles_multispectral.*.las"))
    print(f"📁 Found {len(z_files)} Z_norm LAS tiles")

    for zf in z_files:
        stem = zf.stem
        tile_id = stem.split(".")[-1]

        # Corresponding structure LAS
        struct_file = structure_dir / f"mask_tiles_multispectral.{tile_id}.las"
        if not struct_file.exists():
            print(f"⚠️ Missing structure LAS → skip tile {tile_id}")
            continue

        # Output file names
        out_full = out_dir_full / f"z_norm_struct_{tile_id}_cart_polar.las"
        out_no0 = out_dir_no0 / f"z_norm_struct_{tile_id}_cart_polar_no0.las"

        print("\n" + "=" * 70)
        print(f"▶️ Processing tile {tile_id}")
        print("=" * 70)

        merge_znorm_structure_one_file(
            z_norm_las=zf,
            structure_las=struct_file,
            output_las=out_full,
            output_las_no0=out_no0,
            voxel_size_xy=voxel_size_xy,
            voxel_size_z=voxel_size_z,
            ground_thresh=ground_thresh,
        )

    print("\n🎉 DONE — all tiles processed!")


#%%  Run for all files
if __name__ == "__main__":
    merge_folder(
        z_norm_dir    = "/data/calperumResearch/site1_1_DD0001/lidar/clipped/raster/z_norm/csf/",
        structure_dir = "/data/calperumResearch/site1_1_DD0001/lidar/clipped/mask/coreg/icp/",
        out_dir_full  = "/data/calperumResearch/site1_1_DD0001/lidar/clipped/3d_structure_cart_polar_hdbscan/",
        out_dir_no0   = "/data/calperumResearch/site1_1_DD0001/lidar/clipped/3d_structure_mask_cart_polar_hdbscan/",
        voxel_size_xy = 0.05,
        voxel_size_z  = 0.05,
        ground_thresh = 0.15,
    )
# %%
