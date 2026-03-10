# %%
"""
Merge Z-normalised LiDAR points with structure labels from mask-derived LAS
(top + sides + bottom, etc.) using a voxel grid.

Pipeline:
1. Build voxel->class maps from one or more structure LAS files.
2. Fuse them with a priority rule (Mallee > Shrub > Hummock > Forb > Debris).
3. Option A: vertical voxel propagation to reduce occlusion and make shrubs /
   hummocks volumetric objects.
4. Assign fused voxel classes back to the Z_norm LAS.

- Input 1 (z_norm_las): LAS with Z_norm extra dimension.
- Input 2 (structure_las): ONE path or a LIST of paths to structure LAS files
  with classes 64–68 from 2D masks (top, sides, bottom…).
- Output: LAS with Z_norm and classification (64–68 + 0 for unmatched).
"""

from __future__ import annotations

import laspy
import numpy as np
from pathlib import Path
from collections import Counter, defaultdict
from typing import Dict, Tuple, Iterable, Union, List


CLASS_NAMES: Dict[int, str] = {
    64: "Forb",
    65: "Hummock",
    66: "Shrub",
    67: "Woody Debris",
    68: "Mallee",
    0:  "Unmatched",
}

# smaller = higher priority when fusing
PRIORITY: Dict[int, int] = {
    68: 0,  # Mallee
    66: 1,  # Shrub
    65: 2,  # Hummock
    64: 3,  # Forb
    67: 4,  # Woody debris
    0:  99, # Unmatched / background
}


# -------------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------------
def ensure_las14_pf6(las_in: laspy.LasData) -> laspy.LasData:
    """
    Return a copy as LAS 1.4 / point format 6 so PDAL accepts classes 64–68.

    Works with laspy 2.x:
    - allocate points, copy raw integer coords X/Y/Z,
    - copy common standard dims,
    - copy ExtraBytes (e.g., Z_norm).
    """
    if las_in.header.version == "1.4" and las_in.header.point_format.id >= 6:
        return las_in

    new_hdr = laspy.LasHeader(version="1.4", point_format=6)
    new_hdr.scales = las_in.header.scales
    new_hdr.offsets = las_in.header.offsets

    out = laspy.LasData(new_hdr)
    out.points = laspy.ScaleAwarePointRecord.zeros(len(las_in.points), new_hdr.point_format)

    # raw integer coordinates
    out.X = las_in.X
    out.Y = las_in.Y
    out.Z = las_in.Z

    # overlapping standard dimensions (skip X/Y/Z which are already done)
    src_dims = set(las_in.point_format.dimension_names)
    dst_dims = set(out.point_format.dimension_names)
    for dim in (src_dims & dst_dims):
        if dim in ("X", "Y", "Z"):
            continue
        out[dim] = las_in[dim]

    # ExtraBytes
    for eb in las_in.header.extra_dimensions:
        if eb.name not in out.point_format.dimension_names:
            out.add_extra_dim(eb)
        out[eb.name] = las_in[eb.name]

    return out


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

    density = n_points / volume  # points per m³

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


# -------------------------------------------------------------------------
# Core logic: per-view voxel maps, fusion, propagation, assignment
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
    Build a base voxel -> structure-class map from ONE structure LAS.

    Majority vote per voxel, considering only classes in [valid_min, valid_max].
    """
    vox_s = compute_voxel_indices(
        las_s.x, las_s.y, las_s.z, origin, voxel_size_xy, voxel_size_z
    )
    raw_struct = las_s[structure_field].astype(np.uint16)

    # keep only extended classes, others become 0
    raw_struct_masked = np.where(
        (raw_struct >= valid_min) & (raw_struct <= valid_max),
        raw_struct,
        0,
    )

    uniq_vox_s, inv_s = np.unique(vox_s, axis=0, return_inverse=True)
    voxel_class = np.zeros(len(uniq_vox_s), dtype=np.uint16)

    for k in range(len(uniq_vox_s)):
        vals = raw_struct_masked[inv_s == k]
        if vals.size == 0:
            continue

        nonzero_vals = vals[vals != 0]
        if nonzero_vals.size == 0:
            voxel_class[k] = 0
            continue

        cnt = Counter(nonzero_vals)
        voxel_class[k] = cnt.most_common(1)[0][0]

    voxel_struct_map = {
        (int(ix), int(iy), int(iz)): int(c)
        for (ix, iy, iz), c in zip(uniq_vox_s, voxel_class)
        if c != 0
    }

    return voxel_struct_map


def fuse_voxel_maps(
    voxel_maps: Iterable[Dict[Tuple[int, int, int], int]],
    priority: Dict[int, int] = PRIORITY,
) -> Dict[Tuple[int, int, int], int]:
    """
    Fuse multiple voxel->class maps from different views using a priority scheme.

    If multiple views give different labels for the same voxel, the label with
    the smallest priority value wins.
    """
    fused: Dict[Tuple[int, int, int], int] = {}

    for vm in voxel_maps:
        for key, cls in vm.items():
            if cls == 0:
                continue

            if key not in fused:
                fused[key] = cls
            else:
                current = fused[key]
                if priority.get(cls, 99) < priority.get(current, 99):
                    fused[key] = cls

    print(f"✅ Fused voxel map: {len(fused):,} voxels with non-zero labels")
    return fused


def propagate_vertical_in_columns(
    voxel_struct_map: Dict[Tuple[int, int, int], int],
) -> Dict[Tuple[int, int, int], int]:
    """
    Option A: vertical label propagation to reduce occlusion.

    For each (ix, iy) column:
      - compute majority class of all labelled voxels in the column
      - fill continuously from min_iz to max_iz with that class

    This turns shrubs/hummocks into volumetric objects instead of hollow shells.
    """
    print("🧱 Applying vertical voxel label propagation (Option A)...")

    # group by (ix, iy)
    columns = defaultdict(list)
    for (ix, iy, iz), cls in voxel_struct_map.items():
        columns[(ix, iy)].append((iz, cls))

    voxel_struct_filled: Dict[Tuple[int, int, int], int] = {}

    for (ix, iy), entries in columns.items():
        z_vals = np.array([e[0] for e in entries], dtype=int)
        cls_vals = np.array([e[1] for e in entries], dtype=int)

        # majority class in this column
        unique_cls, counts = np.unique(cls_vals, return_counts=True)
        col_class = int(unique_cls[np.argmax(counts)])

        z_min = int(z_vals.min())
        z_max = int(z_vals.max())

        for iz in range(z_min, z_max + 1):
            voxel_struct_filled[(ix, iy, iz)] = col_class

    print(f"✅ Voxels after vertical propagation: {len(voxel_struct_filled):,}")
    return voxel_struct_filled


def assign_classes_to_z_norm_points(
    las_z: laspy.LasData,
    origin: np.ndarray,
    voxel_size_xy: float,
    voxel_size_z: float,
    voxel_struct_map: Dict[Tuple[int, int, int], int],
) -> np.ndarray:
    """
    Assign class labels to Z_norm LAS points based on voxel_struct_map.
    """
    print("🧩 Assigning voxel-based classes to Z_norm points (codes 64–68)...")

    vox_z = compute_voxel_indices(
        las_z.x, las_z.y, las_z.z, origin, voxel_size_xy, voxel_size_z
    )

    classification = np.zeros(len(las_z.points), dtype=np.uint8)

    matched = 0
    for i, (ix, iy, iz) in enumerate(vox_z):
        lbl = voxel_struct_map.get((int(ix), int(iy), int(iz)), 0)
        classification[i] = lbl
        if lbl != 0:
            matched += 1

    print(f"✅ Matched structure labels via voxels: {matched:,} / {len(las_z.points):,}")
    coverage = 100.0 * matched / len(las_z.points)
    print(f"   → Coverage: {coverage:.2f}% of points labelled (non-zero)")
    return classification


# -------------------------------------------------------------------------
# Main entry point
# -------------------------------------------------------------------------
def merge_znorm_structure_by_coords(
    z_norm_las: Union[str, Path],
    structure_las: Union[str, Path, Iterable[Union[str, Path]]],
    output_las: Union[str, Path],
    output_las_no0: Union[str, Path, None] = None,
    structure_field: str = "classification",
    voxel_size_xy: float | None = None,
    voxel_size_z: float | None = None,
    target_points_per_voxel: float = 5.0,
    z_anisotropy: float = 0.5,
    propagate_vertical: bool = True,
) -> None:
    """
    Merge Z_norm LAS with one or more structure LAS files using voxelised label transfer.

    structure_las can be:
      - a single path (top only), or
      - an iterable of paths [top, front, left, right, bottom, ...].
    """
    z_norm_las = Path(z_norm_las)

    # normalise structure_las to a list of Paths
    if isinstance(structure_las, (str, Path)):
        structure_paths: List[Path] = [Path(structure_las)]
    else:
        structure_paths = [Path(p) for p in structure_las]

    output_las = Path(output_las)
    if output_las_no0 is not None:
        output_las_no0 = Path(output_las_no0)

    print("────────────────────────────────────────────────────")
    print("🔗 Merge Z_norm + structure via voxel grid (multi-view + Option A)")
    print(f"   Z_norm LAS: {z_norm_las}")
    print("   Structure LAS files:")
    for p in structure_paths:
        print(f"     - {p}")
    print("────────────────────────────────────────────────────")

    # --- Load Z_norm and structure LAS
    las_z = laspy.read(z_norm_las)
    las_s_list = [laspy.read(p) for p in structure_paths]

    print(f"🔎 Z_norm points: {len(las_z.points):,}")
    for i, (p, las_s) in enumerate(zip(structure_paths, las_s_list), start=1):
        print(f"   Structure[{i}] {p.name}: {len(las_s.points):,} points")

    # --- Checks
    if "Z_norm" not in las_z.point_format.dimension_names:
        raise ValueError("Missing 'Z_norm' in z_norm LAS.")
    for p, las_s in zip(structure_paths, las_s_list):
        if structure_field not in las_s.point_format.dimension_names:
            raise ValueError(f"'{structure_field}' not found in structure LAS: {p}")

    # --- Voxel sizes: auto or manual
    if voxel_size_xy is None or voxel_size_z is None:
        voxel_size_xy, voxel_size_z = estimate_voxel_sizes_from_density(
            las_z,
            target_points_per_voxel=target_points_per_voxel,
            z_anisotropy=z_anisotropy,
        )
    else:
        print(f"📐 Using user-defined voxel sizes: XY={voxel_size_xy} m, "
              f"Z={voxel_size_z} m")

    # --- Define voxel grid origin (covers Z_norm + all structure clouds)
    x_min = min([las_z.x.min()] + [ls.x.min() for ls in las_s_list])
    y_min = min([las_z.y.min()] + [ls.y.min() for ls in las_s_list])
    z_min = min([las_z.z.min()] + [ls.z.min() for ls in las_s_list])
    origin = np.array([x_min, y_min, z_min], dtype=np.float64)
    print(f"🧱 Voxel grid origin (x_min, y_min, z_min): {origin}")

    # --- 1) Build voxel maps for each structure LAS
    voxel_maps = []
    for i, las_s in enumerate(las_s_list, start=1):
        vm = build_voxel_structure_map(
            las_s=las_s,
            origin=origin,
            voxel_size_xy=voxel_size_xy,
            voxel_size_z=voxel_size_z,
            structure_field=structure_field,
        )
        print(f"   ➕ Structure[{i}] voxel map: {len(vm):,} non-zero voxels")
        voxel_maps.append(vm)

    # --- 2) Fuse maps across views
    if len(voxel_maps) == 1:
        voxel_struct_map = voxel_maps[0]
        print("ℹ️  Only one structure LAS -> no multi-view fusion needed.")
    else:
        voxel_struct_map = fuse_voxel_maps(voxel_maps, priority=PRIORITY)

    # --- 3) Option A: vertical propagation (occlusion fix)
    if propagate_vertical:
        voxel_struct_map = propagate_vertical_in_columns(voxel_struct_map)

    # --- 4) Assign to Z_norm points
    classification = assign_classes_to_z_norm_points(
        las_z=las_z,
        origin=origin,
        voxel_size_xy=voxel_size_xy,
        voxel_size_z=voxel_size_z,
        voxel_struct_map=voxel_struct_map,
    )

    # --- 5) Build output LAS from z_norm; keep Z_norm and new classification
    las_out = laspy.LasData(las_z.header)
    las_out.points = las_z.points.copy()

    if "Z_norm" not in las_out.point_format.dimension_names:
        las_out.add_extra_dim(
            laspy.ExtraBytesParams(name="Z_norm", type=np.float32)
        )
    las_out["Z_norm"] = las_z["Z_norm"].astype(np.float32)

    las_out.classification = classification

    sw = "Merged Structure (64-68) + Z_norm (Laura, MultiView+OptionA)"
    las_out.header.software_id = sw.encode("ascii", errors="ignore").decode("ascii")

    las_out = ensure_las14_pf6(las_out)

    output_las.parent.mkdir(parents=True, exist_ok=True)
    las_out.write(str(output_las))

    # --- Optional: save file without class 0
    if output_las_no0 is not None:
        keep = classification != 0
        las_no0 = laspy.LasData(las_out.header)
        las_no0.points = las_out.points[keep].copy()

        sw_no0 = "Merged Structure (64-68) + Z_norm (Laura, MultiView+OptionA) no0"
        las_no0.header.software_id = sw_no0.encode("ascii", errors="ignore").decode("ascii")

        output_las_no0.parent.mkdir(parents=True, exist_ok=True)
        las_no0.write(str(output_las_no0))

    # --- Histogram summary
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
    # --- SINGLE VIEW (top only) example (what you had before) ---
    merge_znorm_structure_by_coords(
        z_norm_las="/data/calperumResearch/site1_1_DD0001/lidar/clipped/raster/z_norm/csf/z_norm_csf_tiles_multispectral.71.las",
        structure_las="/data/calperumResearch/site1_1_DD0001/lidar/clipped/mask/coreg/icp/mask_tiles_multispectral.71.las",
        output_las="/data/calperumResearch/site1_1_DD0001/lidar/clipped/3d_structure_density_topView/z_norm_struct_tiles_multispectral.71.las",
        output_las_no0="/data/calperumResearch/site1_1_DD0001/lidar/clipped/3d_structure_mask_density_topView/z_norm_mask_tiles_multispectral.71.las",
        voxel_size_xy=0.05,
        voxel_size_z=0.05,
    )

    # --- MULTI-VIEW example (top + sides + bottom) ---
    # Uncomment and adapt paths when you have the side/bottom classified LAS ready.
    #
    # merge_znorm_structure_by_coords(
    #     z_norm_las="/data/calperumResearch/site1_1_DD0001/lidar/clipped/raster/z_norm/csf/z_norm_csf_tiles_multispectral.71.las",
    #     structure_las=[
    #         "/data/.../classified_top_tiles_multispectral.71.las",
    #         "/data/.../classified_front_tiles_multispectral.71.las",
    #         "/data/.../classified_left_tiles_multispectral.71.las",
    #         "/data/.../classified_right_tiles_multispectral.71.las",
    #         "/data/.../classified_bottom_tiles_multispectral.71.las",
    #     ],
    #     output_las="/data/.../z_norm_struct_multiview_tiles_multispectral.71.las",
    #     output_las_no0="/data/.../z_norm_mask_multiview_tiles_multispectral.71.las",
    #     voxel_size_xy=0.05,
    #     voxel_size_z=0.05,
    # )
# %%
