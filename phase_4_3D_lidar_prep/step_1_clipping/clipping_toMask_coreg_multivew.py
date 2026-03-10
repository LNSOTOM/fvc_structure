# %%
"""
Multi-view style voxel labelling from a single top-view mask LAS.

Idea:
- Use top-view mask LAS (from your PDAL script) to get base labelled voxels.
- Build a voxel grid shared with the Z_norm LiDAR cloud.
- Propagate labels:
    1) Vertically (per (ix, iy) column)  -> "top-down" occlusion fill
    2) Along X (per (iy, iz) row)        -> front/back-like propagation
    3) Along Y (per (ix, iz) row)        -> left/right-like propagation
- Fuse these maps and assign final labels to each Z_norm point.

Inputs
------
- z_norm_las: LAS with Z_norm extra dim.
- structure_las_top: LAS from top-view mask (your PDAL output, 64–68 only).
"""

from __future__ import annotations

import laspy
import numpy as np
from pathlib import Path
from collections import Counter, defaultdict
from typing import Dict, Tuple, List


CLASS_NAMES: Dict[int, str] = {
    64: "Forb",
    65: "Hummock",
    66: "Shrub",
    67: "Woody Debris",
    68: "Mallee",
    0:  "Unmatched",
}


# -------------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------------
def ensure_las14_pf6(las_in: laspy.LasData) -> laspy.LasData:
    """
    Ensure LAS 1.4 / point format 6, copying coordinates, dims, and extra dims.
    """
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
        if dim in ("X", "Y", "Z"):
            continue
        out[dim] = las_in[dim]

    for eb in las_in.header.extra_dimensions:
        if eb.name not in out.point_format.dimension_names:
            out.add_extra_dim(eb)
        out[eb.name] = las_in[eb.name]

    return out


def compute_voxel_indices(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    origin: np.ndarray,
    vs_xy: float,
    vs_z: float,
) -> np.ndarray:
    """Return integer voxel indices (N, 3) from coordinates."""
    ix = np.floor((x - origin[0]) / vs_xy).astype(np.int32)
    iy = np.floor((y - origin[1]) / vs_xy).astype(np.int32)
    iz = np.floor((z - origin[2]) / vs_z).astype(np.int32)
    return np.stack([ix, iy, iz], axis=-1)


def estimate_voxel_sizes_from_density(
    las_z: laspy.LasData,
    target_points_per_voxel: float = 20.0,
    z_anisotropy: float = 0.5,
) -> Tuple[float, float]:
    """
    Estimate voxel sizes (XY, Z) from point density.
    """
    x_range = float(las_z.x.max() - las_z.x.min())
    y_range = float(las_z.y.max() - las_z.y.min())
    z_range = float(las_z.z.max() - las_z.z.min())

    eps = 1e-6
    x_range = max(x_range, eps)
    y_range = max(y_range, eps)
    z_range = max(z_range, eps)

    volume = x_range * y_range * z_range
    n_points = len(las_z.points)
    density = n_points / volume  # pts/m³

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


# -------------------------------------------------------------------------
# Build base voxel map from top-view mask LAS
# -------------------------------------------------------------------------
def build_voxel_structure_map(
    las_s: laspy.LasData,
    origin: np.ndarray,
    voxel_size_xy: float,
    voxel_size_z: float,
    structure_field: str = "classification",
    valid_min: int = 64,
    valid_max: int = 68,
) -> Dict[Tuple[int, int, int], int]:
    """
    Base voxel→class map from structure LAS (top view).

    Majority class per voxel, only for labels in [valid_min, valid_max].
    """
    print("📦 Building base voxel → structure-class map (top view)...")

    vox_s = compute_voxel_indices(
        las_s.x, las_s.y, las_s.z, origin, voxel_size_xy, voxel_size_z
    )
    raw_struct = las_s[structure_field].astype(np.uint16)

    raw_struct_masked = np.where(
        (raw_struct >= valid_min) & (raw_struct <= valid_max),
        raw_struct,
        0,
    )

    uniq_vox_s, inv_s = np.unique(vox_s, axis=0, return_inverse=True)
    voxel_class = np.zeros(len(uniq_vox_s), dtype=np.uint16)

    for k in range(len(uniq_vox_s)):
        vals = raw_struct_masked[inv_s == k]
        nonzero_vals = vals[vals != 0]
        if nonzero_vals.size == 0:
            continue
        cnt = Counter(nonzero_vals)
        voxel_class[k] = cnt.most_common(1)[0][0]

    voxel_struct_map = {
        (int(ix), int(iy), int(iz)): int(c)
        for (ix, iy, iz), c in zip(uniq_vox_s, voxel_class)
        if c != 0
    }

    print(f"✅ Base voxels with non-zero structure class: {len(voxel_struct_map):,}")
    return voxel_struct_map


# -------------------------------------------------------------------------
# Propagations: vertical + along X + along Y
# -------------------------------------------------------------------------
def propagate_vertical(
    voxel_struct_map: Dict[Tuple[int, int, int], int]
) -> Dict[Tuple[int, int, int], int]:
    """
    Vertical propagation (top-like occlusion fix):
      - for each (ix, iy) column, take majority class of all labelled voxels
      - fill from min(iz) to max(iz) with that class.
    """
    print("🧱 Vertical propagation along Z (columns)...")

    columns = defaultdict(list)  # (ix, iy) -> list[(iz, cls)]
    for (ix, iy, iz), cls in voxel_struct_map.items():
        columns[(ix, iy)].append((iz, cls))

    filled = {}

    for (ix, iy), entries in columns.items():
        z_vals = np.array([e[0] for e in entries], dtype=int)
        cls_vals = np.array([e[1] for e in entries], dtype=int)

        unique_cls, counts = np.unique(cls_vals, return_counts=True)
        col_class = int(unique_cls[np.argmax(counts)])

        z_min = int(z_vals.min())
        z_max = int(z_vals.max())

        for iz in range(z_min, z_max + 1):
            filled[(ix, iy, iz)] = col_class

    print(f"✅ Vertically filled voxels: {len(filled):,}")
    return filled


def propagate_along_x(
    voxel_struct_map: Dict[Tuple[int, int, int], int]
) -> Dict[Tuple[int, int, int], int]:
    """
    X-direction propagation (front/back-like):
      - group by (iy, iz) rows,
      - majority class per row,
      - fill from min(ix) to max(ix).
    """
    print("➡️  Propagation along X (rows at fixed Y,Z)...")

    rows = defaultdict(list)  # (iy, iz) -> list[(ix, cls)]
    for (ix, iy, iz), cls in voxel_struct_map.items():
        rows[(iy, iz)].append((ix, cls))

    filled = {}

    for (iy, iz), entries in rows.items():
        x_vals = np.array([e[0] for e in entries], dtype=int)
        cls_vals = np.array([e[1] for e in entries], dtype=int)

        unique_cls, counts = np.unique(cls_vals, return_counts=True)
        row_class = int(unique_cls[np.argmax(counts)])

        x_min = int(x_vals.min())
        x_max = int(x_vals.max())

        for ix in range(x_min, x_max + 1):
            filled[(ix, iy, iz)] = row_class

    print(f"✅ X-direction filled voxels: {len(filled):,}")
    return filled


def propagate_along_y(
    voxel_struct_map: Dict[Tuple[int, int, int], int]
) -> Dict[Tuple[int, int, int], int]:
    """
    Y-direction propagation (left/right-like):
      - group by (ix, iz) rows,
      - majority class per row,
      - fill from min(iy) to max(iy).
    """
    print("⬅️  Propagation along Y (rows at fixed X,Z)...")

    rows = defaultdict(list)  # (ix, iz) -> list[(iy, cls)]
    for (ix, iy, iz), cls in voxel_struct_map.items():
        rows[(ix, iz)].append((iy, cls))

    filled = {}

    for (ix, iz), entries in rows.items():
        y_vals = np.array([e[0] for e in entries], dtype=int)
        cls_vals = np.array([e[1] for e in entries], dtype=int)

        unique_cls, counts = np.unique(cls_vals, return_counts=True)
        row_class = int(unique_cls[np.argmax(counts)])

        y_min = int(y_vals.min())
        y_max = int(y_vals.max())

        for iy in range(y_min, y_max + 1):
            filled[(ix, iy, iz)] = row_class

    print(f"✅ Y-direction filled voxels: {len(filled):,}")
    return filled


def fuse_voxel_maps(maps: List[Dict[Tuple[int, int, int], int]]) -> Dict[Tuple[int, int, int], int]:
    """
    Fuse multiple voxel maps (base + vertical + X + Y) via majority voting.
    """
    print("🔗 Fusing voxel maps (base + vertical + X + Y)...")

    votes = defaultdict(list)  # idx -> [labels]

    for m in maps:
        for idx, cls in m.items():
            votes[idx].append(cls)

    fused = {}
    for idx, labels in votes.items():
        arr = np.array(labels, dtype=int)
        unique_cls, counts = np.unique(arr, return_counts=True)
        fused[idx] = int(unique_cls[np.argmax(counts)])

    print(f"✅ Fused voxel map size: {len(fused):,}")
    return fused


# -------------------------------------------------------------------------
# Assign to Z_norm points and write LAS
# -------------------------------------------------------------------------
def assign_to_points(
    las_z: laspy.LasData,
    origin: np.ndarray,
    voxel_size_xy: float,
    voxel_size_z: float,
    voxel_map: Dict[Tuple[int, int, int], int],
) -> np.ndarray:
    """
    Assign class labels to Z_norm LAS points based on fused voxel map.
    """
    print("🧩 Assigning fused voxel labels to Z_norm points...")

    vox_z = compute_voxel_indices(
        las_z.x, las_z.y, las_z.z, origin, voxel_size_xy, voxel_size_z
    )
    classification = np.zeros(len(las_z.points), dtype=np.uint8)

    matched = 0
    for i, (ix, iy, iz) in enumerate(vox_z):
        lbl = voxel_map.get((int(ix), int(iy), int(iz)), 0)
        classification[i] = lbl
        if lbl != 0:
            matched += 1

    coverage = 100.0 * matched / len(las_z.points)
    print(f"✅ Matched labels: {matched:,} / {len(las_z.points):,} ({coverage:.2f}% non-zero)")
    return classification


def merge_znorm_multi_view_from_top(
    z_norm_las: str | Path,
    structure_las_top: str | Path,
    output_las: str | Path,
    output_las_no0: str | Path | None = None,
    structure_field: str = "classification",
    voxel_size_xy: float | None = 0.05,
    voxel_size_z: float | None = 0.05,
    target_points_per_voxel: float = 5.0,
    z_anisotropy: float = 0.5,
    use_vertical: bool = True,
    use_x: bool = True,
    use_y: bool = True,
) -> None:
    """
    Multi-view-like voxel fusion using ONLY top-view mask LAS.
    """
    z_norm_las = Path(z_norm_las)
    structure_las_top = Path(structure_las_top)
    output_las = Path(output_las)
    if output_las_no0 is not None:
        output_las_no0 = Path(output_las_no0)

    print("────────────────────────────────────────────────────")
    print("🔗 Multi-view style merge (top mask → 3D voxels)")
    print(f"   Z_norm LAS:    {z_norm_las}")
    print(f"   Top mask LAS:  {structure_las_top}")
    print("────────────────────────────────────────────────────")

    las_z = laspy.read(z_norm_las)
    las_s_top = laspy.read(structure_las_top)

    print(f"🔎 Z_norm points: {len(las_z.points):,}")
    print(f"🔎 Top mask points: {len(las_s_top.points):,}")

    if "Z_norm" not in las_z.point_format.dimension_names:
        raise ValueError("Missing 'Z_norm' in z_norm LAS.")
    if structure_field not in las_s_top.point_format.dimension_names:
        raise ValueError(f"'{structure_field}' not found in top mask LAS.")

    # Voxel sizes
    if voxel_size_xy is None or voxel_size_z is None:
        voxel_size_xy, voxel_size_z = estimate_voxel_sizes_from_density(
            las_z,
            target_points_per_voxel=target_points_per_voxel,
            z_anisotropy=z_anisotropy,
        )
    else:
        print(f"📐 Using user-defined voxel sizes: XY={voxel_size_xy} m, Z={voxel_size_z} m")

    # Origin (cover both clouds)
    x_min = min(las_z.x.min(), las_s_top.x.min())
    y_min = min(las_z.y.min(), las_s_top.y.min())
    z_min = min(las_z.z.min(), las_s_top.z.min())
    origin = np.array([x_min, y_min, z_min], dtype=np.float64)
    print(f"🧱 Voxel grid origin (x_min, y_min, z_min): {origin}")

    # Base voxel map from top view
    base_map = build_voxel_structure_map(
        las_s=las_s_top,
        origin=origin,
        voxel_size_xy=voxel_size_xy,
        voxel_size_z=voxel_size_z,
        structure_field=structure_field,
    )

    maps = [base_map]

    if use_vertical:
        maps.append(propagate_vertical(base_map))
    if use_x:
        maps.append(propagate_along_x(base_map))
    if use_y:
        maps.append(propagate_along_y(base_map))

    fused_map = fuse_voxel_maps(maps)

    classification = assign_to_points(
        las_z=las_z,
        origin=origin,
        voxel_size_xy=voxel_size_xy,
        voxel_size_z=voxel_size_z,
        voxel_map=fused_map,
    )

    # Build output LAS
    las_out = laspy.LasData(las_z.header)
    las_out.points = las_z.points.copy()

    if "Z_norm" not in las_out.point_format.dimension_names:
        las_out.add_extra_dim(
            laspy.ExtraBytesParams(name="Z_norm", type=np.float32)
        )
    las_out["Z_norm"] = las_z["Z_norm"].astype(np.float32)

    las_out.classification = classification

    sw = "MultiViewFromTop_Znorm_Structure (Laura)"
    las_out.header.software_id = sw.encode("ascii", errors="ignore").decode("ascii")

    las_out = ensure_las14_pf6(las_out)

    output_las.parent.mkdir(parents=True, exist_ok=True)
    las_out.write(str(output_las))

    if output_las_no0 is not None:
        keep = classification != 0
        las_no0 = laspy.LasData(las_out.header)
        las_no0.points = las_out.points[keep].copy()

        sw_no0 = "MultiViewFromTop_Znorm_Structure_no0 (Laura)"
        las_no0.header.software_id = sw_no0.encode("ascii", errors="ignore").decode("ascii")

        output_las_no0.parent.mkdir(parents=True, exist_ok=True)
        las_no0.write(str(output_las_no0))

    counts = Counter(classification)
    print("\n📊 Histogram of extended classification of points:")
    for cid in sorted(counts):
        print(f"{counts[cid]:>12,}  {CLASS_NAMES.get(cid, 'Unknown'):<15} ({cid})")
    print(f"\n✅ Wrote: {output_las}")
    if output_las_no0 is not None:
        print(f"✅ Wrote (no class 0): {output_las_no0}")
    print("────────────────────────────────────────────────────")


# %%
if __name__ == "__main__":
    merge_znorm_multi_view_from_top(
        z_norm_las="/data/calperumResearch/site1_1_DD0001/lidar/clipped/raster/z_norm/csf/z_norm_csf_tiles_multispectral.71.las",
        structure_las_top="/data/calperumResearch/site1_1_DD0001/lidar/clipped/mask/coreg/icp_top/mask_tiles_multispectral.71.las",
        output_las="/data/calperumResearch/site1_1_DD0001/lidar/clipped/3d_structure_multiview/z_norm_struct_multiview_tiles_multispectral.71.las",
        output_las_no0="/data/calperumResearch/site1_1_DD0001/lidar/clipped/3d_structure_multiview/z_norm_mask_multiview_tiles_multispectral.71.las",
        voxel_size_xy=0.05,
        voxel_size_z=0.05,
        use_vertical=True,
        use_x=True,
        use_y=True,
    )
# %%
