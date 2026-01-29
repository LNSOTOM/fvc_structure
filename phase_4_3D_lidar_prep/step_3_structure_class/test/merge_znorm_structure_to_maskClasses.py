#%%
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

    # --- Round XYZ for 1 cm matching
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
    
    # --- FIX: Replaced en dash (–) with hyphen (-) and added ASCII sanitization
    las_out.header.software_id = "Merged Structure (64-68) + Z_norm (Laura)"
    las_out.header.software_id = las_out.header.software_id.encode("ascii", errors="ignore").decode("ascii")

    # --- Force LAS 1.4 / PF6 for PDAL compatibility
    las_out = ensure_las14_pf6(las_out)

    out_path = Path(output_las)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    las_out.write(str(out_path))

    # --- Optional: save file without class 0
    if output_las_no0:
        keep = classification != 0 # This boolean mask has the *original* point count (e.g., 1594668)
        
        # --- FIX: Create a new, separate LasData object.
        # The line `las_no0 = ensure_las14_pf6(las_out)` was creating a *reference*
        # to las_out, not a new object. This meant `las_no0.points = ...`
        # was also modifying las_out.points, causing the IndexError.
        las_no0 = laspy.LasData(las_out.header)

        # Assign the filtered points. This operation copies all dimensions
        # (X, Y, Z, classification, Z_norm, etc.) at once.
        las_no0.points = las_out.points[keep].copy()
        
        # --- FIX: The lines below are now redundant and were the source of the error.
        # The .points copy above has already correctly filtered all data.
        #
        # (Removed) if "Z_norm" not in las_no0.point_format.dimension_names:
        # (Removed)     las_no0.add_extra_dim(laspy.ExtraBytesParams(name="Z_norm", type=np.float32))
        # (Removed) las_no0["Z_norm"] = las_out["Z_norm"][keep]  <-- ORIGINAL ERROR
        # (Removed) las_no0.classification = las_out.classification[keep]
        
        # --- This part is correct ---
        # Sanitize the header for the new no0 file
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
        output_las="/data/calperumResearch/site1_1_DD0001/lidar/clipped/3d_structure/z_norm_struct_tiles_multispectral.55.las",
        output_las_no0="/data/calperumResearch/site1_1_DD0001/lidar/clipped/3d_structure_mask/z_norm_mask_tiles_multispectral.55.las",
        
    )
# %%
