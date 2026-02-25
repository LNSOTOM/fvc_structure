#%%
## BEST
import laspy
import numpy as np
from pathlib import Path
from collections import Counter

CLASS_NAMES = {
    64: "Forb",
    65: "Hummock",
    66: "Shrub",
    67: "Woody Debris",
    68: "Mallee",
    0:  "Unmatched",
}

def ensure_las14_pf6(las_in: laspy.LasData) -> laspy.LasData:
    """
    Return a copy as LAS 1.4 / point format 6 so PDAL accepts classes 64–68.
    Works with laspy 2.x: allocate points, copy raw integer coords X/Y/Z,
    copy shared standard dims, then copy ExtraBytes (e.g., Z_norm).
    """
    if las_in.header.version == "1.4" and las_in.header.point_format.id >= 6:
        return las_in

    # New header: LAS 1.4, PF6
    new_hdr = laspy.LasHeader(version="1.4", point_format=6)
    new_hdr.scales = las_in.header.scales
    new_hdr.offsets = las_in.header.offsets

    out = laspy.LasData(new_hdr)
    # allocate empty points for the new format
    out.points = laspy.ScaleAwarePointRecord.zeros(len(las_in.points), new_hdr.point_format)

    # --- copy raw integer coordinates
    out.X = las_in.X
    out.Y = las_in.Y
    out.Z = las_in.Z

    # --- copy overlapping standard dimensions (skip X/Y/Z which are done)
    src_dims = set(las_in.point_format.dimension_names)
    dst_dims = set(out.point_format.dimension_names)
    for dim in (src_dims & dst_dims):
        if dim in ("X", "Y", "Z"):
            continue
        out[dim] = las_in[dim]

    # --- copy ExtraBytes (if any)
    for eb in las_in.header.extra_dimensions:
        if eb.name not in out.point_format.dimension_names:
            out.add_extra_dim(eb)
        out[eb.name] = las_in[eb.name]

    return out


def estimate_voxel_sizes_from_density(
    las_z: laspy.LasData,
    target_points_per_voxel: float = 20.0,
    z_anisotropy: float = 0.5,
):
    """
    Estimate voxel sizes (XY, Z) from point density.

    - target_points_per_voxel: how many points you *roughly* want per voxel.
    - z_anisotropy: factor to scale Z relative to XY (e.g., 0.5 → Z voxels 2x finer).

    Returns
    -------
    voxel_size_xy, voxel_size_z
    """
    x_range = float(las_z.x.max() - las_z.x.min())
    y_range = float(las_z.y.max() - las_z.y.min())
    z_range = float(las_z.z.max() - las_z.z.min())

    # Avoid zero-volume edge cases
    eps = 1e-6
    x_range = max(x_range, eps)
    y_range = max(y_range, eps)
    z_range = max(z_range, eps)

    volume = x_range * y_range * z_range
    n_points = len(las_z.points)

    density = n_points / volume  # points per m³

    # voxel_volume = target_points / density
    voxel_volume = target_points_per_voxel / density
    edge = voxel_volume ** (1.0 / 3.0)

    voxel_size_xy = edge
    voxel_size_z = edge * z_anisotropy

    print("📏 Automatic voxel size estimation:")
    print(f"   Bounding box: X={x_range:.3f} m, Y={y_range:.3f} m, Z={z_range:.3f} m")
    print(f"   Points: {n_points:,}, density: {density:.2f} pts/m³")
    print(f"   Target ~{target_points_per_voxel} pts/voxel → edge ≈ {edge:.3f} m")
    print(f"   → voxel_size_xy={voxel_size_xy:.3f} m, voxel_size_z={voxel_size_z:.3f} m")

    return voxel_size_xy, voxel_size_z


def merge_znorm_structure_by_coords(
    z_norm_las: str,
    structure_las: str,
    output_las: str,
    output_las_no0: str | None = None,   # optional: write file without class 0
    structure_field: str = "classification",
    voxel_size_xy: float | None = None,  # if None → auto estimate
    voxel_size_z: float | None = None,   # if None → auto estimate
    target_points_per_voxel: float = 5.0,
    z_anisotropy: float = 0.5,
):
    # --- Load
    las_z = laspy.read(z_norm_las)
    las_s = laspy.read(structure_las)
    print(f"🔎 Z_norm points: {len(las_z.points):,}, Structure points: {len(las_s.points):,}")

    # --- Checks
    if "Z_norm" not in las_z.point_format.dimension_names:
        raise ValueError("Missing 'Z_norm' in z_norm LAS.")
    if structure_field not in las_s.point_format.dimension_names:
        raise ValueError(f"'{structure_field}' not found in structure LAS.")

    # --- Voxel sizes: auto or manual
    if voxel_size_xy is None or voxel_size_z is None:
        voxel_size_xy, voxel_size_z = estimate_voxel_sizes_from_density(
            las_z,
            target_points_per_voxel=target_points_per_voxel,
            z_anisotropy=z_anisotropy,
        )
    else:
        print(f"📐 Using user-defined voxel sizes: XY={voxel_size_xy} m, Z={voxel_size_z} m")

    # --- Define voxel grid origin (covers both clouds)
    x_min = min(las_z.x.min(), las_s.x.min())
    y_min = min(las_z.y.min(), las_s.y.min())
    z_min = min(las_z.z.min(), las_s.z.min())
    origin = np.array([x_min, y_min, z_min], dtype=np.float64)
    print(f"🧱 Voxel grid origin (x_min, y_min, z_min): {origin}")

    def compute_voxel_indices(x, y, z, origin, vs_xy, vs_z):
        ix = np.floor((x - origin[0]) / vs_xy).astype(np.int32)
        iy = np.floor((y - origin[1]) / vs_xy).astype(np.int32)
        iz = np.floor((z - origin[2]) / vs_z).astype(np.int32)
        return np.stack([ix, iy, iz], axis=-1)

    # === 1️⃣ Structure LAS → majority extended class per voxel (64–68) ===
    print("📦 Building voxel → structure-class map (original mask codes 64–68)...")

    vox_s = compute_voxel_indices(las_s.x, las_s.y, las_s.z, origin, voxel_size_xy, voxel_size_z)
    raw_struct = las_s[structure_field].astype(np.uint16)

    # keep only extended classes 64..68 (others treated as 0)
    raw_struct_masked = np.where((raw_struct >= 64) & (raw_struct <= 68), raw_struct, 0)

    uniq_vox_s, inv_s = np.unique(vox_s, axis=0, return_inverse=True)
    voxel_class = np.zeros(len(uniq_vox_s), dtype=np.uint16)

    for k in range(len(uniq_vox_s)):
        vals = raw_struct_masked[inv_s == k]
        if vals.size == 0:
            continue

        # Ignore 0 when choosing majority (0 = background / no class)
        nonzero_vals = vals[vals != 0]
        if nonzero_vals.size == 0:
            voxel_class[k] = 0
            continue

        cnt = Counter(nonzero_vals)
        voxel_class[k] = cnt.most_common(1)[0][0]

    # Map: voxel_index -> original class code (64–68)
    voxel_struct_map = {
        (ix, iy, iz): c
        for (ix, iy, iz), c in zip(uniq_vox_s, voxel_class)
        if c != 0
    }

    print(f"✅ Voxels with non-zero structure class: {len(voxel_struct_map):,}")

    # === 2️⃣ Z_norm LAS → assign original mask class from voxel ===
    print("🧩 Assigning voxel-based classes to Z_norm points (original codes 64–68)...")

    vox_z = compute_voxel_indices(las_z.x, las_z.y, las_z.z, origin, voxel_size_xy, voxel_size_z)
    classification = np.zeros(len(las_z.points), dtype=np.uint8)

    matched = 0
    for i, (ix, iy, iz) in enumerate(vox_z):
        lbl = voxel_struct_map.get((ix, iy, iz), 0)
        classification[i] = lbl
        if lbl != 0:
            matched += 1

    print(f"✅ Matched structure labels via voxels: {matched:,} / {len(las_z.points):,}")

    # --- Build output from z_norm; keep Z_norm and write classification
    las_out = laspy.LasData(las_z.header)
    las_out.points = las_z.points.copy()

    if "Z_norm" not in las_out.point_format.dimension_names:
        las_out.add_extra_dim(laspy.ExtraBytesParams(name="Z_norm", type=np.float32))
    las_out["Z_norm"] = las_z["Z_norm"].astype(np.float32)

    las_out.classification = classification
    
    # --- Software ID, ASCII-safe
    las_out.header.software_id = "Merged Structure (64-68) + Z_norm (Laura)"
    las_out.header.software_id = las_out.header.software_id.encode("ascii", errors="ignore").decode("ascii")

    # --- Force LAS 1.4 / PF6 for PDAL compatibility
    las_out = ensure_las14_pf6(las_out)

    out_path = Path(output_las)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    las_out.write(str(out_path))

    # --- Optional: save file without class 0
    if output_las_no0:
        keep = classification != 0

        las_no0 = laspy.LasData(las_out.header)
        las_no0.points = las_out.points[keep].copy()

        las_no0.header.software_id = "Merged Structure (64-68) + Z_norm (Laura) no0"
        las_no0.header.software_id = las_no0.header.software_id.encode("ascii", errors="ignore").decode("ascii")

        Path(output_las_no0).parent.mkdir(parents=True, exist_ok=True)
        las_no0.write(output_las_no0)

    # --- Histogram summary
    counts = Counter(classification)
    print("\n📊 Histogram of extended classification of points:")
    for cid in sorted(counts):
        print(f"{counts[cid]:>12,}  {CLASS_NAMES.get(cid,'Unknown'):<15} ({cid})")

    print(f"\n✅ Wrote: {output_las}")
    if output_las_no0:
        print(f"✅ Wrote (no class 0): {output_las_no0}")



#%%
if __name__ == "__main__":
    # --- Directories ---
    #  site 1_1_DD0001
    # Z_NORM_DIR = Path("/data/calperumResearch/site1_1_DD0001/lidar/clipped/raster/z_norm/csf")
    # STRUCT_DIR = Path("/data/calperumResearch/site1_1_DD0001/lidar/clipped/mask/coreg/icp")
    # OUT_DIR = Path("/data/calperumResearch/site1_1_DD0001/lidar/clipped/3d_structure_density_polar")
    # OUT_NO0_DIR = Path("/data/calperumResearch/site1_1_DD0001/lidar/clipped/3d_structure_mask_density_polar")
    
    #  site 2_1_DD0011
    Z_NORM_DIR = Path("/data/calperumResearch/site2_1_DD0011/lidar/clipped/raster/z_norm/csf")
    STRUCT_DIR = Path("/data/calperumResearch/site2_1_DD0011/lidar/clipped/mask/coreg/icp")
    OUT_DIR = Path("/data/calperumResearch/site2_1_DD0011/lidar/clipped/3d_structure_density")
    OUT_NO0_DIR = Path("/data/calperumResearch/site2_1_DD0011/lidar/clipped/3d_structure_mask_density")

    # pattern matching your example:
    #   z_norm_csf_tiles_multispectral.55.las
    #   mask_tiles_multispectral.55.las
    z_norm_files = sorted(Z_NORM_DIR.glob("z_norm_csf_tiles_multispectral.*.las"))

    print(f"🔁 Found {len(z_norm_files)} z_norm tiles")

    for z_path in z_norm_files:
        tile_id = z_path.stem.split(".")[-1]  # '55' in z_norm_csf_tiles_multispectral.55

        struct_path = STRUCT_DIR / f"mask_tiles_multispectral.{tile_id}.las"
        if not struct_path.exists():
            print(f"⚠️  Missing structure LAS for tile {tile_id}: {struct_path}")
            continue

        out_path = OUT_DIR / f"z_norm_struct_tiles_multispectral.{tile_id}.las"
        out_no0_path = OUT_NO0_DIR / f"z_norm_mask_tiles_multispectral.{tile_id}.las"

        print("\n" + "=" * 80)
        print(f"🧩 Processing tile {tile_id}")
        print(f"   Z_norm:     {z_path}")
        print(f"   Structure:  {struct_path}")
        print(f"   Output:     {out_path}")
        print(f"   Output no0: {out_no0_path}")
        print("=" * 80)

        try:
            merge_znorm_structure_by_coords(
                z_norm_las=str(z_path),
                structure_las=str(struct_path),
                output_las=str(out_path),
                output_las_no0=str(out_no0_path),
                # use fixed voxel sizes (your current choice)
                voxel_size_xy=0.05,
                voxel_size_z=0.05,
                # or set these to None to auto-estimate per tile:
                # voxel_size_xy=None,
                # voxel_size_z=None,
            )
        except Exception as e:
            print(f"❌ Error processing tile {tile_id}: {e}")


#%%
## BEST + POLAR 65↔68 REFINEMENT
import laspy
import numpy as np
import pandas as pd
from pathlib import Path
from collections import Counter

CLASS_NAMES = {
    64: "Forb",
    65: "Hummock",
    66: "Shrub",
    67: "Woody Debris",
    68: "Mallee",
    0:  "Unmatched",
}

# -----------------------------------------------------------
# Ensure LAS 1.4 / PF6
# -----------------------------------------------------------
def ensure_las14_pf6(las_in: laspy.LasData) -> laspy.LasData:
    if las_in.header.version == "1.4" and las_in.header.point_format.id >= 6:
        return las_in

    new_hdr = laspy.LasHeader(version="1.4", point_format=6)
    new_hdr.scales = las_in.header.scales
    new_hdr.offsets = las_in.header.offsets

    out = laspy.LasData(new_hdr)
    out.points = laspy.ScaleAwarePointRecord.zeros(len(las_in.points), new_hdr.point_format)

    out.X = las_in.X
    out.Y = las_in.Y
    out.Z = las_in.Z

    src_dims = set(las_in.point_format.dimension_names)
    dst_dims = set(out.point_format.dimension_names)
    for dim in (src_dims & dst_dims):
        if dim not in ("X","Y","Z"):
            out[dim] = las_in[dim]

    for eb in las_in.header.extra_dimensions:
        if eb.name not in out.point_format.dimension_names:
            out.add_extra_dim(eb)
        out[eb.name] = las_in[eb.name]

    return out


# -----------------------------------------------------------
# Height ranges used in polar refinement
# -----------------------------------------------------------
HEIGHT_RANGES = {
    64: (0.00, 0.40),
    65: (0.05, 1.20),
    66: (0.40, 3.00),
    67: (0.00, 0.80),
    68: (1.50, 8.00),
}

def soft_height_prior(z_norm: float, cls: int) -> float:
    if cls not in HEIGHT_RANGES:
        return 1.0
    zmin, zmax = HEIGHT_RANGES[cls]
    return 1.0 if (zmin <= z_norm <= zmax) else 0.1


# -----------------------------------------------------------
# POLAR REFINEMENT (65 ↔ 68 ONLY)
# -----------------------------------------------------------
def refine_mallee_with_polar(
    las_z: laspy.LasData,
    las_s: laspy.LasData,
    classification: np.ndarray,
    r_bin: float = 0.10,
    theta_bin_deg: float = 5.0,
    min_conf: float = 0.3,
    min_flip_ratio: float = 1.5,
):
    cls_out = classification.copy()

    # Only refine if structure LAS actually contains mallee
    if not np.any(las_s.classification == 68):
        print("⚪ No mallee (68) found in structure LAS → skip polar refinement.")
        return cls_out

    # --- Polar origin = tile centre ---
    x0 = 0.5 * (las_z.x.min() + las_z.x.max())
    y0 = 0.5 * (las_z.y.min() + las_z.y.max())
    origin = np.array([x0, y0])

    def cart_to_polar(x, y):
        dx = x - origin[0]
        dy = y - origin[1]
        r = np.sqrt(dx*dx + dy*dy)
        theta = np.mod(np.arctan2(dy, dx), 2*np.pi)
        return r, theta

    def polar_bins(r, theta):
        ir = np.floor(r / r_bin).astype(np.int32)
        it = np.floor(theta / np.deg2rad(theta_bin_deg)).astype(np.int32)
        return ir, it

    # ======================================================
    # 1) Build P(65|r,θ) and P(68|r,θ) from structure LAS
    # ======================================================
    s_r, s_th = cart_to_polar(las_s.x, las_s.y)
    s_ir, s_it = polar_bins(s_r, s_th)

    mask = (las_s.classification == 65) | (las_s.classification == 68)
    df_s = pd.DataFrame({
        "ir": s_ir[mask],
        "it": s_it[mask],
        "cls": las_s.classification[mask]
    })

    col_counts = df_s.groupby(["ir","it","cls"]).size().unstack(fill_value=0)
    col_probs = col_counts.div(col_counts.sum(axis=1), axis=0)

    if min_conf > 0:
        conf = col_probs.max(axis=1)
        col_probs = col_probs[conf >= min_conf]

    if col_probs.empty:
        print("⚪ Polar: no confident r/θ columns → skip refinement.")
        return cls_out

    col_map = {idx: col_probs.loc[idx] for idx in col_probs.index}
    print(f"📦 Polar columns with 65/68 evidence: {len(col_map):,}")

    # ======================================================
    # 2) For points currently 65 or 68 → maybe flip
    # ======================================================
    z_r, z_th = cart_to_polar(las_z.x, las_z.y)
    z_ir, z_it = polar_bins(z_r, z_th)
    zn = las_z["Z_norm"].astype(float)

    flips_65_to_68 = 0
    flips_68_to_65 = 0

    idxs = np.where((cls_out == 65) | (cls_out == 68))[0]

    for idx in idxs:
        key = (z_ir[idx], z_it[idx])
        if key not in col_map:
            continue

        base = col_map[key]
        classes = base.index.to_numpy()      # subset of {65,68}
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

        # require strong evidence (posterior ratio)
        best_val = post[classes == best_cls][0]
        cur_val  = post[classes == current][0]

        if best_val / (cur_val + 1e-9) >= min_flip_ratio:
            if current == 65 and best_cls == 68:
                flips_65_to_68 += 1
            if current == 68 and best_cls == 65:
                flips_68_to_65 += 1
            cls_out[idx] = best_cls

    print(f"🔁 Polar flips: 65→68 = {flips_65_to_68}, 68→65 = {flips_68_to_65}")
    return cls_out


# -----------------------------------------------------------
# MAIN MERGE (Cartesian voxel) + POLAR 65↔68 refinement
# -----------------------------------------------------------
def merge_znorm_structure_by_coords(
    z_norm_las: str,
    structure_las: str,
    output_las: str,
    output_las_no0: str | None = None,
    structure_field: str = "classification",
    voxel_size_xy: float = 0.10,
    voxel_size_z: float = 0.10,
):
    # --- Load
    las_z = laspy.read(z_norm_las)
    las_s = laspy.read(structure_las)

    if "Z_norm" not in las_z.point_format.dimension_names:
        raise ValueError("Missing Z_norm.")

    # --- Cartesian voxelisation
    x_min = min(las_z.x.min(), las_s.x.min())
    y_min = min(las_z.y.min(), las_s.y.min())
    z_min = min(las_z.z.min(), las_s.z.min())
    origin = np.array([x_min, y_min, z_min])

    def vox(x,y,z):
        return np.stack([
            np.floor((x-origin[0])/voxel_size_xy).astype(np.int32),
            np.floor((y-origin[1])/voxel_size_xy).astype(np.int32),
            np.floor((z-origin[2])/voxel_size_z).astype(np.int32)
        ], axis=-1)

    vox_s = vox(las_s.x, las_s.y, las_s.z)
    raw = las_s[structure_field].astype(np.uint16)
    raw = np.where((raw>=64)&(raw<=68), raw, 0)

    uniq, inv = np.unique(vox_s, axis=0, return_inverse=True)
    classes = np.zeros(len(uniq), dtype=np.uint16)

    for i in range(len(uniq)):
        vals = raw[inv==i]
        vals = vals[vals!=0]
        classes[i] = Counter(vals).most_common(1)[0][0] if len(vals)>0 else 0

    voxel_map = {(ix,iy,iz):c for (ix,iy,iz),c in zip(uniq,classes) if c!=0}

    vox_z = vox(las_z.x, las_z.y, las_z.z)
    class_cart = np.array([voxel_map.get(tuple(v),0) for v in vox_z], dtype=np.uint8)

    # ----------------------------------------------------
    # 🔥 POLAR REFINEMENT ADDITION HERE
    # ----------------------------------------------------
    class_refined = refine_mallee_with_polar(
        las_z, las_s, class_cart,
        r_bin=0.10,
        theta_bin_deg=5.0,
        min_conf=0.3,
        min_flip_ratio=1.5
    )

    # ----------------------------------------------------
    # Save LAS
    # ----------------------------------------------------
    las_out = laspy.LasData(las_z.header)
    las_out.points = las_z.points.copy()
    las_out.classification = class_refined

    if "Z_norm" not in las_out.point_format.dimension_names:
        las_out.add_extra_dim(laspy.ExtraBytesParams(name="Z_norm", type=np.float32))
    las_out["Z_norm"] = las_z["Z_norm"].astype(np.float32)

    las_out = ensure_las14_pf6(las_out)
    Path(output_las).parent.mkdir(parents=True, exist_ok=True)
    las_out.write(output_las)

    if output_las_no0:
        keep = class_refined != 0
        las_no0 = laspy.LasData(las_out.header)
        las_no0.points = las_out.points[keep].copy()
        Path(output_las_no0).parent.mkdir(parents=True, exist_ok=True)
        las_no0.write(output_las_no0)

    print("✔ Done:", output_las)
    return class_refined


#%%
if __name__ == "__main__":
    #  site 1_1_DD0001
    # Z_NORM_DIR = Path("/data/calperumResearch/site1_1_DD0001/lidar/clipped/raster/z_norm/csf")
    # STRUCT_DIR = Path("/data/calperumResearch/site1_1_DD0001/lidar/clipped/mask/coreg/icp")
    # OUT_DIR = Path("/data/calperumResearch/site1_1_DD0001/lidar/clipped/3d_structure_density_polar")
    # OUT_NO0_DIR = Path("/data/calperumResearch/site1_1_DD0001/lidar/clipped/3d_structure_mask_density_polar")
    
    #  site 2_1_DD0011
    Z_NORM_DIR = Path("/data/calperumResearch/site2_1_DD0011/lidar/clipped/raster/z_norm/csf")
    STRUCT_DIR = Path("/data/calperumResearch/site2_1_DD0011/lidar/clipped/mask/coreg/icp")
    OUT_DIR = Path("/data/calperumResearch/site2_1_DD0011/lidar/clipped/3d_structure_density")
    OUT_NO0_DIR = Path("/data/calperumResearch/site2_1_DD0011/lidar/clipped/3d_structure_mask_density")

    OUT_DIR.mkdir(exist_ok=True)
    OUT_NO0_DIR.mkdir(exist_ok=True)

    files = sorted(Z_NORM_DIR.glob("z_norm_csf_tiles_multispectral.*.las"))
    print(f"Found {len(files)} tiles")

    for zf in files:
        tid = zf.stem.split(".")[-1]
        sf = STRUCT_DIR / f"mask_tiles_multispectral.{tid}.las"
        if not sf.exists():
            print("Missing structure:", sf); continue

        merge_znorm_structure_by_coords(
            str(zf),
            str(sf),
            str(OUT_DIR / f"z_norm_struct.{tid}.las"),
            str(OUT_NO0_DIR / f"z_norm_struct_no0.{tid}.las")
        )

# %%
