#%%
"""
Merge Z_norm LAS with structure LAS using a 3-stage fusion:

1) Cartesian voxel majority vote (classes 64–68)
2) Polar refinement for 65 ↔ 68 (Mallee vs Woody debris)
3) Cylindrical (r, θ) + Z_norm fill for remaining class 0
4) Final height sanity rule (demote tall 68 → 65)

Batch mode over all tiles in a folder, writing:
- 3d_structure_cart_polar/        (all points, including class 0)
- 3d_structure_mask_cart_polar/   (only points with class != 0)

Voxel size and occlusion mapping
--------------------------------
A crucial consideration when mapping occlusion in 3D space is the spatial
resolution of the voxel grid. The voxel size represents a trade-off between
the effective LiDAR footprint and the level of detail required to support
meaningful crown-scale structural analysis (e.g. Béland et al., 2014;
Schneider et al., 2019; Brede et al., 2022).

Although individual LiDAR beams originate from a sub-millimetre laser diameter,
the projected footprint expands with range and does not by itself define an
appropriate neighbourhood for crown morphology or occlusion analysis. In this
study we therefore adopt a 0.10 m (10 cm) voxel size, following previous work
showing that 10 cm voxels are well-suited to distinguishing occluded regions
from unobserved, filled, and empty space, while remaining coarse enough to
support robust neighbourhood voting and crown-scale reasoning.

Within each 10 cm voxel, we first assign structure classes via 3D Cartesian
majority vote, and then refine these assignments in cylindrical (r, θ)
columns using polar statistics and soft height-based priors on Z_norm. This
combination of voxel neighbourhood support and column-wise refinement provides
an occlusion-aware 3D label fusion that is more robust to canopy shadowing and
confusion between woody debris and mallee canopy than a pure 3D voxel merge.
"""

from __future__ import annotations

from pathlib import Path
from collections import Counter
from typing import Dict, Tuple, Mapping

import laspy
import numpy as np
import pandas as pd

# -----------------------------------------------------------
# Global constants (easy to tweak for experiments)
# -----------------------------------------------------------

# Extended class names
CLASS_NAMES: Dict[int, str] = {
    64: "Forb",
    65: "Woody Debris",
    66: "Hummock",
    67: "Shrub",
    68: "Mallee",
    0:  "Unmatched",
}

# Expected Z_norm ranges for each class
HEIGHT_RANGES: Dict[int, Tuple[float, float]] = {
    64: (0.00, 0.40),   # Forb
    65: (0.05, 5.00),   # Woody debris
    66: (0.00, 0.80),   #  Hummock- 0.00, 0.80
    67: (0.40, 3.00),   # Shrub-0.40, 3.00
    68: (1.50, 8.00),   # Mallee
}

# Default hyperparameters for polar / cylindrical refinement
DEFAULT_R_BIN = 0.10          # m
DEFAULT_THETA_BIN_DEG = 5.0   # degrees
DEFAULT_MIN_CONF = 0.3
DEFAULT_MIN_FLIP_RATIO = 1.5
DEFAULT_GROUND_TH = 0.01      # m


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

    # Copy overlapping standard dims except XYZ (already set)
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
# Height prior helper
# -----------------------------------------------------------

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
# Small helpers (used by polar / cylindrical stages)
# -----------------------------------------------------------

def tile_center_xy(las_z: laspy.LasData) -> np.ndarray:
    """Return (x0, y0) centre of the Z_norm tile as float64 array."""
    x0 = 0.5 * (las_z.x.min() + las_z.x.max())
    y0 = 0.5 * (las_z.y.min() + las_z.y.max())
    return np.array([x0, y0], dtype=float)


def cart_to_polar_xy(
    x: np.ndarray, y: np.ndarray, origin_xy: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Cartesian → polar (r, θ) in the XY plane (θ in [0, 2π))."""
    dx = x - origin_xy[0]
    dy = y - origin_xy[1]
    r = np.sqrt(dx * dx + dy * dy)
    theta = np.mod(np.arctan2(dy, dx), 2 * np.pi)
    return r, theta


def polar_bins(
    r: np.ndarray,
    theta: np.ndarray,
    r_bin: float,
    theta_bin_deg: float
) -> Tuple[np.ndarray, np.ndarray]:
    """Discretise r, θ into integer bin indices."""
    ir = np.floor(r / r_bin).astype(np.int32)
    it = np.floor(theta / np.deg2rad(theta_bin_deg)).astype(np.int32)
    return ir, it


# -----------------------------------------------------------
# Common polar column probability builder (DRY)
# -----------------------------------------------------------

def build_column_probabilities(
    las_s: laspy.LasData,
    class_mask: np.ndarray,
    origin: np.ndarray,
    r_bin: float,
    theta_bin_deg: float,
    min_conf: float,
) -> Mapping[Tuple[int, int], pd.Series]:
    """
    Build column-wise class probabilities in (r, θ) space.

    Returns:
        dict[(ir, it)] -> pd.Series indexed by class label with probabilities.
    """
    if not np.any(class_mask):
        return {}

    s_r, s_th = cart_to_polar_xy(las_s.x, las_s.y, origin)
    s_ir, s_it = polar_bins(s_r, s_th, r_bin, theta_bin_deg)

    df_s = pd.DataFrame({
        "ir":  s_ir[class_mask],
        "it":  s_it[class_mask],
        "cls": las_s.classification[class_mask],
    })

    col_counts = df_s.groupby(["ir", "it", "cls"]).size().unstack(fill_value=0)
    col_probs = col_counts.div(col_counts.sum(axis=1), axis=0)

    if min_conf > 0.0:
        best_conf = col_probs.max(axis=1)
        col_probs = col_probs[best_conf >= min_conf]

    if col_probs.empty:
        return {}

    return {idx: col_probs.loc[idx] for idx in col_probs.index}


# -----------------------------------------------------------
# Cartesian voxel merge (base labels)
# -----------------------------------------------------------

def cartesian_merge(
    las_z: laspy.LasData,
    las_s: laspy.LasData,
    voxel_size_xy: float = 0.10,
    voxel_size_z: float = 0.10,
) -> np.ndarray:
    """
    Merge structure classes (64–68) into Z_norm LAS via 3D voxel majority vote.
    """
    print(
        f"   [cartesian_merge] voxel_size_xy={voxel_size_xy:.3f}, "
        f"voxel_size_z={voxel_size_z:.3f}"
    )
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
# Polar refinement for 65 / 66 / 68
#   *skips points with Z_norm <= ground_thresh*
# -----------------------------------------------------------

def refine_mallee_with_polar(
    las_z: laspy.LasData,
    las_s: laspy.LasData,
    classification: np.ndarray,
    r_bin: float = DEFAULT_R_BIN,
    theta_bin_deg: float = DEFAULT_THETA_BIN_DEG,
    min_conf: float = DEFAULT_MIN_CONF,
    min_flip_ratio: float = DEFAULT_MIN_FLIP_RATIO,
    ground_thresh: float = DEFAULT_GROUND_TH,
) -> np.ndarray:
    """
    Refine confusion between 65 (Woody debris), 66 (Hummock) and
    68 (Mallee) using polar (r, θ) columns and Z_norm-based priors.
    """
    cls_out = classification.copy()

    # If no mallee at all, polar evidence is weakly defined → skip
    if not np.any(las_s.classification == 68):
        print("⚪ No mallee (68) in structure LAS → skip polar refinement.")
        return cls_out

    origin = tile_center_xy(las_z)

    # Column probabilities for 65 / 66 / 68 from structure LAS
    mask_65_66_68 = (
        (las_s.classification == 65) |
        (las_s.classification == 66) |
        (las_s.classification == 68)
    )
    col_map = build_column_probabilities(
        las_s=las_s,
        class_mask=mask_65_66_68,
        origin=origin,
        r_bin=r_bin,
        theta_bin_deg=theta_bin_deg,
        min_conf=min_conf,
    )

    if not col_map:
        print("⚪ No confident polar columns → skip refinement.")
        return cls_out

    print(f"📦 Polar columns with 65/66/68 evidence: {len(col_map):,}")

    # Precompute polar indices and height for Z_norm LAS
    z_r, z_th = cart_to_polar_xy(las_z.x, las_z.y, origin)
    z_ir, z_it = polar_bins(z_r, z_th, r_bin, theta_bin_deg)
    zn = las_z["Z_norm"].astype(float)

    # Track flip statistics
    flips_65_to_68 = 0
    flips_68_to_65 = 0
    flips_66_to_68 = 0
    flips_68_to_66 = 0
    flips_65_to_66 = 0
    flips_66_to_65 = 0

    # Only consider points currently in {65, 66, 68}
    idx_candidates = np.where(
        (cls_out == 65) | (cls_out == 66) | (cls_out == 68)
    )[0]

    for idx in idx_candidates:
        if zn[idx] <= ground_thresh:
            continue  # ignore near-ground noise

        key = (z_ir[idx], z_it[idx])
        base = col_map.get(key, None)
        if base is None:
            continue

        classes = base.index.to_numpy()
        probs = base.to_numpy()

        # Apply height prior per class
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

        # Only flip if the new class is clearly better
        if best_val / (cur_val + 1e-9) >= min_flip_ratio:
            if   current == 65 and best_cls == 68: flips_65_to_68 += 1
            elif current == 68 and best_cls == 65: flips_68_to_65 += 1
            elif current == 66 and best_cls == 68: flips_66_to_68 += 1
            elif current == 68 and best_cls == 66: flips_68_to_66 += 1
            elif current == 65 and best_cls == 66: flips_65_to_66 += 1
            elif current == 66 and best_cls == 65: flips_66_to_65 += 1

            cls_out[idx] = best_cls

    print(
        f"🔁 Polar flips:"
        f" 65→68={flips_65_to_68}, 68→65={flips_68_to_65},"
        f" 66→68={flips_66_to_68}, 68→66={flips_68_to_66},"
        f" 65→66={flips_65_to_66}, 66→65={flips_66_to_65}"
    )
    return cls_out


# -----------------------------------------------------------
# Fill ONLY zero points with cylindrical (r,θ) + Z_norm fusion
#   *skips points with Z_norm <= ground_thresh*
# -----------------------------------------------------------

def fill_missing_with_cyl_height(
    las_z: laspy.LasData,
    las_s: laspy.LasData,
    classification: np.ndarray,
    r_bin: float = DEFAULT_R_BIN,
    theta_bin_deg: float = DEFAULT_THETA_BIN_DEG,
    min_conf: float = DEFAULT_MIN_CONF,
    ground_thresh: float = DEFAULT_GROUND_TH,
    max_fill_height: float | None = 2.0,   # NEW: only fill below this Z_norm
) -> np.ndarray:
    """
    For points still class 0, infer a class using cylindrical (r, θ)
    column probabilities from structure LAS (64–68) + Z_norm height priors.

    IMPORTANT: if max_fill_height is not None, we ONLY fill zero points with
    ground_thresh < Z_norm <= max_fill_height. Higher points stay 0 to
    preserve canopy occlusion.
    """
    cls_out = classification.copy()
    origin = tile_center_xy(las_z)

    # Column probabilities from ALL extended classes 64–68
    mask_ext = (las_s.classification >= 64) & (las_s.classification <= 68)
    col_map = build_column_probabilities(
        las_s=las_s,
        class_mask=mask_ext,
        origin=origin,
        r_bin=r_bin,
        theta_bin_deg=theta_bin_deg,
        min_conf=min_conf,
    )

    if not col_map:
        print("⚪ No confident columns after min_conf → skip cylindrical fill.")
        return cls_out

    print(f"📦 Cylindrical columns with class evidence: {len(col_map):,}")

    # Assign ONLY to currently-unlabelled, non-ground points
    z_r, z_th = cart_to_polar_xy(las_z.x, las_z.y, origin)
    z_ir, z_it = polar_bins(z_r, z_th, r_bin, theta_bin_deg)
    zn = las_z["Z_norm"].astype(float)

    # base condition: non-ground zeros
    cond = (cls_out == 0) & (zn > ground_thresh)
    # if limiting fill by height, enforce that here
    if max_fill_height is not None:
        cond &= (zn <= max_fill_height)

    zero_idx = np.where(cond)[0]
    filled = 0

    for idx in zero_idx:
        key = (z_ir[idx], z_it[idx])
        base = col_map.get(key, None)
        if base is None:
            continue

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

    print(
        f"🧩 Cylindrical fill: labelled {filled:,} previously-unlabelled points "
        f"(Z_norm ∈ ({ground_thresh}, "
        f"{'∞' if max_fill_height is None else max_fill_height}] )"
    )
    return cls_out



# -----------------------------------------------------------
# Full pipeline for ONE FILE
# -----------------------------------------------------------

def merge_znorm_structure_one_file(
    z_norm_las: str | Path,
    structure_las: str | Path,
    output_las: str | Path,
    output_las_no0: str | Path | None = None,
    voxel_size_xy: float = 0.10,
    voxel_size_z: float = 0.10,
    ground_thresh: float = 0.01,
    r_bin: float = DEFAULT_R_BIN,
    theta_bin_deg: float = DEFAULT_THETA_BIN_DEG,
    min_conf: float = DEFAULT_MIN_CONF,
    min_flip_ratio: float = DEFAULT_MIN_FLIP_RATIO,
    reuse_existing_cartesian: bool = True,   # NEW: optionally reuse existing 64–68 labels
) -> None:
    """
    Run full merge & refinement pipeline for a single pair of LAS files.

    If `reuse_existing_cartesian` is True and the Z_norm LAS already contains
    extended classes (64–68) in its classification field, those labels are
    used as the base "cartesian" classification. Otherwise, a fresh 3D voxel
    merge is computed from `structure_las`.

    Hyperparameters (voxel_size_xy, voxel_size_z, ground_thresh, r_bin,
    theta_bin_deg, min_conf, min_flip_ratio) are exposed so different
    configurations can be tested in experiments.
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

    # -------------------------------------------------------
    # 1) Base classification (cartesian)
    #    - Reuse existing 64–68 labels if present
    #    - Otherwise compute voxel majority vote
    # -------------------------------------------------------
    has_ext_classes = np.any((las_z.classification >= 64) & (las_z.classification <= 68))

    if reuse_existing_cartesian and has_ext_classes:
        classification = las_z.classification.astype(np.uint8)
        print("✅ Using existing extended classes (64–68) in Z_norm LAS as cartesian base.")
    else:
        classification = cartesian_merge(las_z, las_s, voxel_size_xy, voxel_size_z)

    # 2) Polar refinement for 65/66/68 (skip near-ground)
    classification = refine_mallee_with_polar(
        las_z, las_s, classification,
        r_bin=r_bin,
        theta_bin_deg=theta_bin_deg,
        min_conf=min_conf,
        min_flip_ratio=min_flip_ratio,
        ground_thresh=ground_thresh,
    )

    # 3) Cylindrical fill for remaining class 0 points (skip near-ground)
    classification = fill_missing_with_cyl_height(
        las_z, las_s, classification,
        r_bin=r_bin,
        theta_bin_deg=theta_bin_deg,
        min_conf=min_conf,
        ground_thresh=ground_thresh,
        max_fill_height=2.0,   # NEW: do not fill zeros above 2 m
    )

    # 4) Final height sanity rules
    zn_arr = las_z["Z_norm"].astype(float)

    HUMMOCK_TO_MALLEE_TH = 2.0  # m
    SHRUB_TO_WOODY_TH    = 2.0  # m
    MALLEE_TO_WOODY_TH   = 3.0  # m

    # 66 (Hummock) too tall → 68 (Mallee)
    mask_hummock_too_tall = (classification == 66) & (zn_arr >= HUMMOCK_TO_MALLEE_TH)
    n_hummock_to_mallee = int(mask_hummock_too_tall.sum())
    classification[mask_hummock_too_tall] = 68

    # 67 (Shrub) too tall → 65 (Woody debris / generic high woody)
    mask_shrub_too_tall = (classification == 67) & (zn_arr >= SHRUB_TO_WOODY_TH)
    n_shrub_to_woody = int(mask_shrub_too_tall.sum())
    classification[mask_shrub_too_tall] = 65

    # 68 (Mallee) very tall → 65 (Woody debris / generic high woody)
    mask_mallee_too_tall = (classification == 68) & (zn_arr >= MALLEE_TO_WOODY_TH)
    n_mallee_to_woody = int(mask_mallee_too_tall.sum())
    classification[mask_mallee_too_tall] = 65

    print(
        f"🌲 Height rule: reassigned {n_hummock_to_mallee} hummock (66) points "
        f"with Z_norm ≥ {HUMMOCK_TO_MALLEE_TH:.1f} m to mallee (68)."
    )
    print(
        f"🌲 Height rule: reassigned {n_shrub_to_woody} shrub (67) points "
        f"with Z_norm ≥ {SHRUB_TO_WOODY_TH:.1f} m to woody debris (65)."
    )
    print(
        f"🌲 Height rule: demoted {n_mallee_to_woody} mallee (68) points "
        f"with Z_norm ≥ {MALLEE_TO_WOODY_TH:.1f} m to woody debris (65)."
    )

    # ---- write LAS ----
    las_out = laspy.LasData(las_z.header)
    las_out.points = las_z.points.copy()
    las_out.classification = classification

    # Preserve Z_norm (ensure exists as ExtraBytes)
    if "Z_norm" not in las_out.point_format.dimension_names:
        las_out.add_extra_dim(laspy.ExtraBytesParams(name="Z_norm", type=np.float32))
    las_out["Z_norm"] = zn_arr.astype(np.float32)

    # Tag software_id (ASCII only)
    software_tag = "Cart+Polar65-68+CylFill_groundSafe"
    las_out.header.software_id = software_tag.encode("ascii", "ignore").decode("ascii")

    # Ensure LAS 1.4 / PF6 for extended classes
    las_out = ensure_las14_pf6(las_out)

    # Write full file (all points)
    output_las.parent.mkdir(parents=True, exist_ok=True)
    las_out.write(str(output_las))

        # ---------------------------------------------
    # "Mask" file: structure classes + promoted high class-0 → 65
    # ---------------------------------------------
    if output_las_no0 is not None:
        HIGH_Z_KEEP = 1.3  # m – promote unlabelled points above this height to class 65

        zn_arr = las_out["Z_norm"].astype(float)

        # Start from the final classification of the full file
        cls_mask = classification.copy()

        # 🔼 Promote high, unlabeled points to class 65 (Woody debris / generic high veg)
        high_unlabelled = (cls_mask == 0) & (zn_arr >= HIGH_Z_KEEP)
        n_promoted = int(high_unlabelled.sum())
        cls_mask[high_unlabelled] = 65

        print(f"🌲 Promoted {n_promoted:,} high unlabelled points (Z_norm ≥ {HIGH_Z_KEEP} m) to class 65 in mask file.")

        # For the mask file, we now keep ALL non-zero classes (64–68 + promoted 65)
        keep = (cls_mask != 0)

        las_no0 = laspy.LasData(las_out.header)
        las_no0.points = las_out.points[keep].copy()
        las_no0.classification = cls_mask[keep]

        # Preserve Z_norm in mask file too
        if "Z_norm" not in las_no0.point_format.dimension_names:
            las_no0.add_extra_dim(
                laspy.ExtraBytesParams(name="Z_norm", type=np.float32)
            )
        las_no0["Z_norm"] = zn_arr[keep].astype(np.float32)

        output_las_no0.parent.mkdir(parents=True, exist_ok=True)
        las_no0.write(str(output_las_no0))

    # Histogram
    cnt = Counter(classification)
    print("\n📊 Final class distribution:")
    for cid in sorted(cnt):
        print(f"{cnt[cid]:>10,}  {CLASS_NAMES.get(cid, 'Unknown')} ({cid})")

    print(f"\n✅ Wrote: {output_las}")
    if output_las_no0 is not None:
        print(f"✅ Wrote (no class 0 + high canopy): {output_las_no0}")


# -----------------------------------------------------------
# Batch processing for all tiles in a folder
# -----------------------------------------------------------

def merge_folder(
    z_norm_dir,
    structure_dir,
    out_dir_full,
    out_dir_no0,
    voxel_size_xy: float = 0.10,
    voxel_size_z: float = 0.10,
    ground_thresh: float = 0.15,
    r_bin: float = DEFAULT_R_BIN,
    theta_bin_deg: float = DEFAULT_THETA_BIN_DEG,
    min_conf: float = DEFAULT_MIN_CONF,
    min_flip_ratio: float = DEFAULT_MIN_FLIP_RATIO,
    reuse_existing_cartesian: bool = True,   # NEW
) -> None:
    """
    Batch process all Z_norm tiles in a folder.

    Typical usage:

      - RAW mode:
          z_norm_dir   : z_norm_csf_tiles_multispectral.X.las
          reuse_existing_cartesian=False  (force fresh voxel merge)

      - REFINEMENT mode (recommended once cartesian merge is good):
          z_norm_dir   : z_norm_struct_X_cartesian.las
          reuse_existing_cartesian=True   (reuse existing 64–68 labels)

      structure_dir: mask_tiles_multispectral.X.las
      out_dir_full : z_norm_struct_X_cart_polar.las
      out_dir_no0  : z_norm_struct_X_cart_polar_no0.las
    """
    z_norm_dir = Path(z_norm_dir)
    structure_dir = Path(structure_dir)
    out_dir_full = Path(out_dir_full)
    out_dir_no0 = Path(out_dir_no0)

    out_dir_full.mkdir(parents=True, exist_ok=True)
    out_dir_no0.mkdir(parents=True, exist_ok=True)

    z_files = sorted(z_norm_dir.glob("z_norm_csf_tiles_multispectral.*.las"))
    print(f"📁 Found {len(z_files)} Z_norm LAS tiles")

    for zf in z_files:
        stem = zf.stem
        tile_id = stem.split(".")[-1]

        struct_file = structure_dir / f"mask_tiles_multispectral.{tile_id}.las"
        if not struct_file.exists():
            print(f"⚠️ Missing structure LAS → skip tile {tile_id}")
            continue

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
            r_bin=r_bin,
            theta_bin_deg=theta_bin_deg,
            min_conf=min_conf,
            min_flip_ratio=min_flip_ratio,
            reuse_existing_cartesian=reuse_existing_cartesian,
        )

    print(
        f"⚙️  voxel_xy={voxel_size_xy}, voxel_z={voxel_size_z}, "
        f"ground_th={ground_thresh}, r_bin={r_bin}, "
        f"theta_bin_deg={theta_bin_deg}, min_conf={min_conf}, "
        f"min_flip_ratio={min_flip_ratio}, "
        f"reuse_existing_cartesian={reuse_existing_cartesian}"
    )
    print("\n🎉 DONE — all tiles processed!")


#%%  Run for all files
if __name__ == "__main__":
    merge_folder(
        # z_norm_dir    = "/data/calperumResearch/site1_1_DD0001/lidar/clipped/raster/z_norm/csf/",
        # structure_dir = "/data/calperumResearch/site1_1_DD0001/lidar/clipped/mask/coreg/icp/",
        # out_dir_full  = "/data/calperumResearch/site1_1_DD0001/lidar/clipped/3d_structure_cart_polar_v1/",
        # out_dir_no0   = "/data/calperumResearch/site1_1_DD0001/lidar/clipped/3d_structure_mask_cart_polar_v1/",
        
        #  site 2_1_DD0011
        z_norm_dir = Path("/data/calperumResearch/site2_1_DD0011/lidar/clipped/raster/z_norm/csf"),
        structure_dir = Path("/data/calperumResearch/site2_1_DD0011/lidar/clipped/mask/coreg/icp"),
        out_dir_full = Path("/data/calperumResearch/site2_1_DD0011/lidar/clipped/3d_structure_density"),
        out_dir_no0 = Path("/data/calperumResearch/site2_1_DD0011/lidar/clipped/3d_structure_mask_density"),
        voxel_size_xy = 0.10,
        voxel_size_z  = 0.10,
        ground_thresh = 0.15,
        r_bin         = 0.10,
        theta_bin_deg = 5.0,
        min_conf      = 0.3,
        min_flip_ratio= 1.5,
        # When you switch to using the previous cartesian output as input,
        # just update `z_norm_dir` and leave this as True.
        reuse_existing_cartesian=True,
    )

#%%
import laspy
from collections import Counter
from pathlib import Path

f = Path("/data/calperumResearch/site1_1_DD0001/lidar/clipped/3d_structure_mask_cart_polar_v1/z_norm_struct_109_cart_polar_no0.las")
las = laspy.read(f)

cls = las.classification
print("Unique classes:", sorted(set(cls)))
print("Counts:", Counter(cls))


#%% 3D STRUCTURE QA (reuse previous cartesian labels + apply HIGH_Z_KEEP logic to stats)

import laspy
import numpy as np
from collections import Counter
from pathlib import Path

# --- Inputs ---
f = Path("/data/calperumResearch/site1_1_DD0001/lidar/clipped/3d_structure_mask_cart_polar_v1/z_norm_struct_109_cart_polar_no0.las")

HIGH_Z_KEEP = 1.5  # m – keep unlabeled class 0 above this height (your original logic)

# --- Safe read ---
try:
    las = laspy.read(f)
except Exception as e:
    print(f"❌ Failed to read LAS tile: {e}")
    raise

cls = las.classification.astype(np.uint8)
zn = las["Z_norm"].astype(float) if "Z_norm" in las.point_format.dimension_names else None

print("\n🔎 Tile:", f.name)
print("================================================================")

# --- Count raw ---
cnt_raw = Counter(cls)
unique_raw = sorted(cnt_raw)
print("Unique classes (RAW):", unique_raw)
print("Counts (RAW):", cnt_raw)

# --- Apply canopy keep mask to stats (analysis stage only, doesn't write anything) ---
if zn is not None:
    keep_struct = (cls != 0)
    keep_high_zero = (cls == 0) & (zn >= HIGH_Z_KEEP)
    keep = keep_struct | keep_high_zero
    cls_keep = cls[keep]
    cnt_keep = Counter(cls_keep)

    print("\n📦 After applying HIGH_Z_KEEP canopy mask to stats:")
    for cid, c in cnt_keep.items():
        pct = 100 * c / len(cls_keep)
        print(f"{c:>10,}  {CLASS_NAMES.get(cid, 'Unknown')} ({cid})  {pct:5.2f}%")

    print(f"\nTotal kept points (for stats): {len(cls_keep):,}")
else:
    print("\n⚠️ Z_norm missing → cannot apply height mask to stats, showing raw only")

print("================================================================\n")
