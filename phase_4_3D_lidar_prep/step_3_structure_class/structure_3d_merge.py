#%% GOOD
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
    Make sure the LAS is 1.4 / point format 6 (needed for classes > 31).
    Works with laspy 2.x: allocate points, copy raw integer coords (X/Y/Z),
    copy overlapping standard dims, then copy ExtraBytes.
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

    # overlapping standard dims (skip X/Y/Z which are already copied)
    src_dims = set(las_in.point_format.dimension_names)
    dst_dims = set(out.point_format.dimension_names)
    for dim in (src_dims & dst_dims):
        if dim in ("X", "Y", "Z"):
            continue
        out[dim] = las_in[dim]

    # ExtraBytes (e.g., Z_norm)
    for eb in las_in.header.extra_dimensions:
        if eb.name not in out.point_format.dimension_names:
            out.add_extra_dim(eb)
        out[eb.name] = las_in[eb.name]

    return out


def merge_znorm_structure_by_coords(
    z_norm_las: str,
    structure_las: str,
    output_las: str,
    output_las_no0: str | None = None,   # optional: write file without class 0
    structure_field: str = "classification"
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

    # --- 1 cm coordinate match (XY/Z)
    coords_z = np.round(np.vstack((las_z.x, las_z.y, las_z.z)).T, 2)
    coords_s = np.round(np.vstack((las_s.x, las_s.y, las_s.z)).T, 2)

    # keep only extended classes 64..68
    raw_struct = las_s[structure_field].astype(np.uint16)
    struct_dict = {tuple(c): (lbl if 64 <= lbl <= 68 else 0) for c, lbl in zip(coords_s, raw_struct)}

    classification = np.zeros(len(coords_z), dtype=np.uint8)
    matched = 0
    for i, coord in enumerate(coords_z):
        lbl = struct_dict.get(tuple(coord), 0)
        classification[i] = lbl
        if lbl != 0:
            matched += 1

    print(f"✅ Matched structure labels: {matched:,} / {len(coords_z):,}")

    # --- Build output from z_norm; keep Z_norm and write classification
    las_out = laspy.LasData(las_z.header)
    las_out.points = las_z.points.copy()

    if "Z_norm" not in las_out.point_format.dimension_names:
        las_out.add_extra_dim(laspy.ExtraBytesParams(name="Z_norm", type=np.float32))
    las_out["Z_norm"] = las_z["Z_norm"].astype(np.float32)

    las_out.classification = classification

    # ASCII-only software_id to avoid odd VLR issues
    sw = "Merged Structure (64-68) + Z_norm (Laura)"
    las_out.header.software_id = sw.encode("ascii", errors="ignore").decode("ascii")

    # --- Force LAS 1.4 / PF6 for PDAL compatibility
    las_out = ensure_las14_pf6(las_out)

    out_path = Path(output_las)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    las_out.write(str(out_path))

    # --- Optional: save file without class 0
    if output_las_no0:
        keep = classification != 0
        las_no0 = ensure_las14_pf6(las_out)
        # IMPORTANT: assign filtered points in one go (copies all dims)
        las_no0.points = las_out.points[keep].copy()
        sw2 = "Merged Structure (64-68) + Z_norm (Laura) no0"
        las_no0.header.software_id = sw2.encode("ascii", errors="ignore").decode("ascii")

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
        z_norm_las="/media/laura/laura_usb/uas_data/DD0001/lidar/clipped/z_norm/z_norm_csf_tiles_multispectral.109.las",
        structure_las="/media/laura/laura_usb/uas_data/DD0001/lidar/clipped/mask/coreg/icp/mask_tiles_multispectral.109.las",
        output_las="/media/laura/laura_usb/uas_data/DD0001/lidar/clipped/3d_structure/z_norm_struct_tiles_multispectral.109.las",
        output_las_no0="/media/laura/laura_usb/uas_data/DD0001/lidar/clipped/3d_structure/z_norm_struct_tiles_multispectral_no0.109.las",
    )

# %%
# cleaning
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
    Make sure the LAS is 1.4 / point format 6 (needed for classes > 31).
    Works with laspy 2.x: allocate points, copy raw integer coords (X/Y/Z),
    copy overlapping standard dims, then copy ExtraBytes.
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

    # overlapping standard dims (skip X/Y/Z which are already copied)
    src_dims = set(las_in.point_format.dimension_names)
    dst_dims = set(out.point_format.dimension_names)
    for dim in (src_dims & dst_dims):
        if dim in ("X", "Y", "Z"):
            continue
        out[dim] = las_in[dim]

    # ExtraBytes (e.g., Z_norm)
    for eb in las_in.header.extra_dimensions:
        if eb.name not in out.point_format.dimension_names:
            out.add_extra_dim(eb)
        out[eb.name] = las_in[eb.name]

    return out


def merge_znorm_structure_by_coords(
    z_norm_las: str,
    structure_las: str,
    output_las: str,
    output_las_no0: str | None = None,  # optional: write file without class 0
    structure_field: str = "classification",
    shrub_height_threshold: float = 1.5 # NEW: Max height for a "Shrub" (class 66)
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

    # --- 1 cm coordinate match (XY/Z)
    coords_z = np.round(np.vstack((las_z.x, las_z.y, las_z.z)).T, 2)
    coords_s = np.round(np.vstack((las_s.x, las_s.y, las_s.z)).T, 2)

    # keep only extended classes 64..68
    raw_struct = las_s[structure_field].astype(np.uint16)
    struct_dict = {tuple(c): (lbl if 64 <= lbl <= 68 else 0) for c, lbl in zip(coords_s, raw_struct)}

    classification = np.zeros(len(coords_z), dtype=np.uint8)
    matched = 0
    for i, coord in enumerate(coords_z):
        lbl = struct_dict.get(tuple(coord), 0)
        classification[i] = lbl
        if lbl != 0:
            matched += 1

    print(f"✅ Matched structure labels: {matched:,} / {len(coords_z):,}")
    
    # --- NEW: Refine classification based on Z_norm ---
    # Get all Z_norm values
    z_norm_values = las_z["Z_norm"].copy()
    
    # Find points that are "Shrub" (66) AND are taller than the threshold
    is_shrub = (classification == 66)
    is_too_tall = (z_norm_values > shrub_height_threshold)
    
    # Create a combined boolean mask
    points_to_reclassify = is_shrub & is_too_tall
    
    num_reclassified = np.sum(points_to_reclassify)
    
    if num_reclassified > 0:
        # Re-classify these "tall shrubs" as "Mallee" (68)
        classification[points_to_reclassify] = 68
        print(f"♻️ Refined classification: Re-classified {num_reclassified:,} 'Shrub' points (class 66) with Z_norm > {shrub_height_threshold}m to 'Mallee' (class 68).")

    # --- Build output from z_norm; keep Z_norm and write classification
    las_out = laspy.LasData(las_z.header)
    las_out.points = las_z.points.copy()

    if "Z_norm" not in las_out.point_format.dimension_names:
        las_out.add_extra_dim(laspy.ExtraBytesParams(name="Z_norm", type=np.float32))
    las_out["Z_norm"] = z_norm_values # Use the z_norm values we already fetched

    las_out.classification = classification

    # ASCII-only software_id to avoid odd VLR issues
    sw = "Merged Structure (64-68) + Z_norm (Laura) - Refined"
    las_out.header.software_id = sw.encode("ascii", errors="ignore").decode("ascii")

    # --- Force LAS 1.4 / PF6 for PDAL compatibility
    las_out = ensure_las14_pf6(las_out)

    out_path = Path(output_las)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    las_out.write(str(out_path))

    # --- Optional: save file without class 0
    if output_las_no0:
        keep = classification != 0
        # Create a new, separate LasData object
        las_no0 = laspy.LasData(las_out.header)
        # IMPORTANT: assign filtered points in one go (copies all dims)
        las_no0.points = las_out.points[keep].copy()
        
        sw2 = "Merged Structure (64-68) + Z_norm (Laura) no0 - Refined"
        las_no0.header.software_id = sw2.encode("ascii", errors="ignore").decode("ascii")

        Path(output_las_no0).parent.mkdir(parents=True, exist_ok=True)
        las_no0.write(output_las_no0)

    # --- Histogram summary (will now reflect the refined counts)
    counts = Counter(classification)
    print("\n📊 Histogram of refined classification of points:")
    for cid in sorted(counts):
        print(f"{counts[cid]:>12,}  {CLASS_NAMES.get(cid,'Unknown'):<15} ({cid})")

    print(f"\n✅ Wrote: {output_las}")
    if output_las_no0:
        print(f"✅ Wrote (no class 0): {output_las_no0}")

#%%
if __name__ == "__main__":
    merge_znorm_structure_by_coords(
        z_norm_las="/media/laura/laura_usb/uas_data/DD0001/lidar/clipped/z_norm/z_norm_csf_tiles_multispectral.109.las",
        structure_las="/media/laura/laura_usb/uas_data/DD0001/lidar/clipped/mask/coreg/icp/mask_tiles_multispectral.109.las",
        output_las="/media/laura/laura_usb/uas_data/DD0001/lidar/clipped/3d_structure/z_norm_struct_tiles_multispectral.109.las",
        output_las_no0="/media/laura/laura_usb/uas_data/DD0001/lidar/clipped/3d_structure/z_norm_struct_tiles_multispectral_no0.109.las",
        shrub_height_threshold=1.5 # You can adjust this value (in meters)
    )

# %%