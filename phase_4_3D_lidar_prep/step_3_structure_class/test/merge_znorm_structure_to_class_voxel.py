#%%
# GOOD
import laspy
import numpy as np
from pathlib import Path
from collections import Counter


"""
Merge structure classes into a Z_norm LAS using a shared voxel grid.

Workflow:
    1. Build a voxel grid that covers BOTH point clouds.
    2. For the structure LAS, compute majority class per voxel (65–68 → 1–4).
    3. For the Z_norm LAS, assign each point the class of its voxel.

This avoids losing points due to imperfect XYZ coordinate matches and
is more robust to small misalignments and sampling differences.
"""



def merge_znorm_structure_by_coords(   # name unchanged for compatibility
    z_norm_las: str,
    structure_las: str,
    output_las: str,
    structure_field: str = "classification",
    voxel_size_xy: float = 0.03,       # 3 cm
    voxel_size_z: float = 0.05         # 5 cm
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

    # === Build a voxel grid covering both clouds ===
    x0 = min(las_z.x.min(), las_s.x.min())
    y0 = min(las_z.y.min(), las_s.y.min())
    z0 = min(las_z.z.min(), las_s.z.min())
    origin = np.array([x0, y0, z0], dtype=np.float64)

    def voxel_index(x, y, z):
        ix = np.floor((x - origin[0]) / voxel_size_xy).astype(np.int32)
        iy = np.floor((y - origin[1]) / voxel_size_xy).astype(np.int32)
        iz = np.floor((z - origin[2]) / voxel_size_z).astype(np.int32)
        return np.stack([ix, iy, iz], axis=-1)

    # === STRUCTURE LAS → majority class per voxel ===
    vox_s = voxel_index(las_s.x, las_s.y, las_s.z)
    raw_struct = np.asarray(las_s[structure_field], dtype=np.uint16)

    # Mapping to compact labels (1–5)
    structure_map = {
        64: 1,  # forb
        65: 2,  # hummock
        66: 3,  # shrub
        67: 4,  # woody debris
        68: 5   # mallee
    }
    labels_s = np.array([structure_map.get(lbl, 0) for lbl in raw_struct], dtype=np.uint8)

    uniq_vox_s, inv_s = np.unique(vox_s, axis=0, return_inverse=True)
    voxel_class = np.zeros(len(uniq_vox_s), dtype=np.uint8)

    for k in range(len(uniq_vox_s)):
        vals = labels_s[inv_s == k]
        if vals.size == 0:
            continue
        counts = np.bincount(vals, minlength=6)  # 0..5
        counts[0] = 0  # ignore unmatched
        voxel_class[k] = np.argmax(counts)

    voxel_struct_map = {
        (ix, iy, iz): c
        for (ix, iy, iz), c in zip(uniq_vox_s, voxel_class)
        if c != 0
    }

    print(f"✅ Voxels with structure class: {len(voxel_struct_map):,}")

    # === Z-NORM LAS → assign classification by voxel ===
    vox_z = voxel_index(las_z.x, las_z.y, las_z.z)

    classification = np.zeros(len(las_z.points), dtype=np.uint8)
    matched = 0
    for i, (ix, iy, iz) in enumerate(vox_z):
        c = voxel_struct_map.get((ix, iy, iz), 0)
        classification[i] = c
        if c != 0:
            matched += 1

    print(f"✅ Matched structure labels via voxels: {matched:,} / {len(las_z.points):,}")

    # === Write output LAS ===
    header = las_z.header
    las_out = laspy.LasData(header)
    las_out.points = las_z.points.copy()

    if "Z_norm" not in las_out.point_format.dimension_names:
        las_out.add_extra_dim(laspy.ExtraBytesParams(name="Z_norm", type=np.float32))
    las_out["Z_norm"] = las_z["Z_norm"].astype(np.float32)

    las_out.classification = classification

    # Fix ASCII
    las_out.header.software_id = (
        las_out.header.software_id.encode("ascii", errors="ignore").decode("ascii")
    )

    out_path = Path(output_las)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    las_out.write(str(out_path))

    # Stats
    counts = Counter(classification)
    print(f"✅ Wrote merged LAS: {output_las}")
    print("✅ Class distribution:", dict(counts))
    print("💡 Classes: 0=unmatched, 1=Forb, 2=Hummock, 3=Shrub, 4=WoodyDebris, 5=Mallee")
    print(f"💡 Voxel sizes: XY={voxel_size_xy}, Z={voxel_size_z}")


#%%
if __name__ == "__main__":
    merge_znorm_structure_by_coords(
        # SITE 1 + L1
        # z_norm_las="/media/laura/laura_usb/uas_data/DD0001/lidar/clipped/z_norm/z_norm_csf_tiles_multispectral.109.las",
        # structure_las="/media/laura/laura_usb/uas_data/DD0001/lidar/clipped/mask/coreg/icp/mask_tiles_multispectral.109.las",
        # output_las="/media/laura/laura_usb/uas_data/DD0001/lidar/clipped/3d_structure/z_norm_struct_tiles_multispectral.109.las"
        
        z_norm_las="/data/calperumResearch/site1_1_DD0001/lidar/clipped/raster/z_norm/csf/z_norm_csf_tiles_multispectral.55.las",
        structure_las="/data/calperumResearch/site1_1_DD0001/lidar/clipped/mask/coreg/icp/mask_tiles_multispectral.55.las",
        output_las="/data/calperumResearch/site1_1_DD0001/lidar/clipped/3d_structure_voxel/z_norm_struct_tiles_multispectral.55.las"
        
        
        #L2
        # z_norm_las="/media/laura/laura_usb/uas_data/DD0001/lidar_l2/clipped/z_norm/z_norm_tiles_multispectral.109.las",
        # structure_las="/media/laura/laura_usb/uas_data/DD0001/lidar_l2/clipped/mask/coreg/icp/mask_tiles_multispectral.109.las",
        # output_las="/media/laura/laura_usb/uas_data/DD0001/lidar_l2/clipped/3d_structure/z_norm_struct_tiles_multispectral.109.las"
    
    )
# %%
