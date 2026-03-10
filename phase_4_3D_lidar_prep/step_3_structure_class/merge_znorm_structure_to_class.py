#%%
# GOOD
import laspy
import numpy as np
from pathlib import Path
from collections import Counter


def merge_znorm_structure_by_coords(
    z_norm_las: str,
    structure_las: str,
    output_las: str,
    structure_field: str = "classification"
):
    # Load LAS files
    las_z = laspy.read(z_norm_las)
    las_s = laspy.read(structure_las)

    print(f"🔎 Z_norm points: {len(las_z.points):,}, Structure points: {len(las_s.points):,}")

    # Validate presence of Z_norm
    if "Z_norm" not in las_z.point_format.dimension_names:
        raise ValueError("Missing 'Z_norm' in z_norm LAS.")

    # Validate presence of classification field
    if structure_field not in las_s.point_format.dimension_names:
        raise ValueError(f"'{structure_field}' not found in structure LAS.")

    # Round XYZ for coordinate-based matching (1 cm precision)
    coords_z = np.stack([
        np.round(las_z.x, 2),
        np.round(las_z.y, 2),
        np.round(las_z.z, 2)
    ], axis=-1)

    coords_s = np.stack([
        np.round(las_s.x, 2),
        np.round(las_s.y, 2),
        np.round(las_s.z, 2)
    ], axis=-1)

    # Structure labels from classification (use directly from mask)
    raw_struct = las_s[structure_field]

    # Build dictionary for coordinate → class label (no remapping)
    struct_dict = {tuple(c): lbl for c, lbl in zip(coords_s, raw_struct)}

    # Assign classes to Z_norm points
    classification = np.zeros(len(coords_z), dtype=np.uint8)
    matched = 0
    for i, coord in enumerate(coords_z):
        label = struct_dict.get(tuple(coord), 0)
        if label != 0:
            matched += 1
        classification[i] = label

    print(f"✅ Matched structure labels: {matched:,} / {len(coords_z):,}")

    # Prepare output LAS
    header = las_z.header
    las_out = laspy.LasData(header)
    las_out.points = las_z.points.copy()

    # ✅ Add Z_norm extra dimension correctly
    if "Z_norm" not in las_out.point_format.dimension_names:
        las_out.add_extra_dim(laspy.ExtraBytesParams(name="Z_norm", type=np.float32))
    las_out["Z_norm"] = las_z["Z_norm"].astype(np.float32)

    # ✅ Assign new classification
    las_out.classification = classification

    # Ensure software_id is ASCII-compliant
    las_out.header.software_id = las_out.header.software_id.encode("ascii", errors="ignore").decode("ascii")

    # Save
    out_path = Path(output_las)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    las_out.write(str(out_path))

    # Stats
    counts = Counter(classification)
    print(f"✅ Wrote merged LAS: {output_las}")
    print("✅ Class distribution:", dict(counts))
    print("💡 Classes: 0=forb, 1=Hummock, 2=Shrub, 3=WoodyDebris, 4=Mallee")



#%%
if __name__ == "__main__":
    merge_znorm_structure_by_coords(
        # SITE 1 + L1
        # z_norm_las="/media/laura/laura_usb/uas_data/DD0001/lidar/clipped/z_norm/z_norm_csf_tiles_multispectral.109.las",
        # structure_las="/media/laura/laura_usb/uas_data/DD0001/lidar/clipped/mask/coreg/icp/mask_tiles_multispectral.109.las",
        # output_las="/media/laura/laura_usb/uas_data/DD0001/lidar/clipped/3d_structure/z_norm_struct_tiles_multispectral.109.las"
        
        z_norm_las="/data/calperumResearch/site1_1_DD0001/lidar/clipped/raster/z_norm/csf/z_norm_csf_tiles_multispectral.55.las",
        structure_las="/data/calperumResearch/site1_1_DD0001/lidar/clipped/mask/coreg/icp/mask_tiles_multispectral.55.las",
        output_las="/data/calperumResearch/site1_1_DD0001/lidar/clipped/3d_structure/z_norm_struct_tiles_multispectral.55.las"
        
        
        #L2
        # z_norm_las="/media/laura/laura_usb/uas_data/DD0001/lidar_l2/clipped/z_norm/z_norm_tiles_multispectral.109.las",
        # structure_las="/media/laura/laura_usb/uas_data/DD0001/lidar_l2/clipped/mask/coreg/icp/mask_tiles_multispectral.109.las",
        # output_las="/media/laura/laura_usb/uas_data/DD0001/lidar_l2/clipped/3d_structure/z_norm_struct_tiles_multispectral.109.las"
    
    )

# %%
### DELTE CLASS 0