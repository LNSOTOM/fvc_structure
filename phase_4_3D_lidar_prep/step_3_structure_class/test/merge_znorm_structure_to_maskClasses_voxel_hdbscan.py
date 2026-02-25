
#%%
## BEST + optional HDBSCAN refinement
import laspy
import numpy as np
from pathlib import Path
from collections import Counter

import hdbscan  # <- make sure this is installed

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


def refine_with_hdbscan(
    las_z: laspy.LasData,
    classification: np.ndarray,
    min_cluster_size: int = 50,
    min_samples: int = 10,
    classes_to_refine=(64, 65, 66, 67, 68),
) -> np.ndarray:
    """
    Refine voxel-assigned classes using HDBSCAN.

    For each class in classes_to_refine:
      - cluster its points in 3D (x, y, z)
      - points labelled as noise (-1) by HDBSCAN are set back to 0 (Unmatched)

    Returns a new classification array.
    """
    new_class = classification.copy()
    x = las_z.x
    y = las_z.y
    z = las_z.z

    total_before = (classification != 0).sum()

    for cls in classes_to_refine:
        idx = np.where(classification == cls)[0]
        n_cls = len(idx)
        print(f"\n🔍 HDBSCAN refinement for class {cls} ({CLASS_NAMES.get(cls,'?')}), points: {n_cls:,}")

        if n_cls < min_cluster_size:
            print("   ⚠️ Fewer points than min_cluster_size; setting them to 0.")
            new_class[idx] = 0
            continue

        coords = np.vstack((x[idx], y[idx], z[idx])).T

        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            metric="euclidean",
            cluster_selection_method="eom",
        )
        labels = clusterer.fit_predict(coords)  # -1 = noise

        n_noise = (labels == -1).sum()
        print(f"   HDBSCAN clusters: {len(np.unique(labels[labels >= 0]))}, noise points: {n_noise:,}")

        # set noise back to 0 (Unmatched)
        noise_idx = idx[labels == -1]
        new_class[noise_idx] = 0

    total_after = (new_class != 0).sum()
    print(f"\n🧹 HDBSCAN refinement summary:")
    print(f"   labelled before: {total_before:,}")
    print(f"   labelled after : {total_after:,}")
    print(f"   removed as noise: {total_before - total_after:,}")

    return new_class


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
    use_hdbscan: bool = False,
    hdbscan_min_cluster_size: int = 50,
    hdbscan_min_samples: int = 10,
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

    # === 3️⃣ Optional HDBSCAN refinement ===
    if use_hdbscan:
        print("\n🧠 Running HDBSCAN refinement on labelled points...")
        classification = refine_with_hdbscan(
            las_z,
            classification,
            min_cluster_size=hdbscan_min_cluster_size,
            min_samples=hdbscan_min_samples,
        )

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
    merge_znorm_structure_by_coords(
        z_norm_las="/data/calperumResearch/site1_1_DD0001/lidar/clipped/raster/z_norm/csf/z_norm_csf_tiles_multispectral.55.las",
        structure_las="/data/calperumResearch/site1_1_DD0001/lidar/clipped/mask/coreg/icp/mask_tiles_multispectral.55.las",
        output_las="/data/calperumResearch/site1_1_DD0001/lidar/clipped/3d_structure_hdbscan/z_norm_struct_tiles_multispectral.55.las",
        output_las_no0="/data/calperumResearch/site1_1_DD0001/lidar/clipped/3d_structure_mask_hdbscan/z_norm_mask_tiles_multispectral.55.las",
        voxel_size_xy=0.05,
        voxel_size_z=0.05,
        use_hdbscan=True,              # 🔥 turn refinement on/off here
        hdbscan_min_cluster_size=80,   # you can tune these
        hdbscan_min_samples=10,
    )
# %%
