# %%
"""
Unified preprocessing pipeline for Calperum LiDAR tiles.

Produces TWO outputs per LAS file:
-------------------------------------------------------
1) MINSU3D NPZ tensor
   coords, feats, semantic_label, instance_label, offset
   -> For voxel-based sparse 3D U-Net (MinkowskiEngine)

2) ForestFormer3D-compatible PLY
   x, y, z, semantic_seg, treeID, z_norm, intensity
   -> For point-based panoptic segmentation

This ensures both frameworks use the *same preprocessing*
so comparisons are valid and reproducible.
"""

import laspy
import numpy as np
import pandas as pd
from pathlib import Path
from pyntcloud import PyntCloud  # pip install pyntcloud

# ---------------- CONFIG ---------------- #

INPUT_DIR = Path("/data/calperumResearch/site1_1_DD0001/lidar/clipped/3d_structure_obj_polar_id")

OUT_NPZ_DIR = Path("/data/calperumResearch/site1_1_DD0001/lidar/clipped/preprocessed/minsu3d_npz/")
OUT_PLY_DIR = Path("/data/calperumResearch/site1_1_DD0001/lidar/clipped/preprocessed/forestformer3d_ply/")

OUT_NPZ_DIR.mkdir(parents=True, exist_ok=True)
OUT_PLY_DIR.mkdir(parents=True, exist_ok=True)

VOXEL_SIZE = 0.10
IGNORE_INDEX = -100

CLASS_REMAP = {
    64: 0,  # Forb
    65: 1,  # Woody debris
    66: 2,  # Hummock
    67: 3,  # Shrub
    68: 4,  # Mallee
    0: IGNORE_INDEX,  # Unmatched / background
}

# ------------------------------------------------ #


def normalise_intensity(intensity):
    intensity = intensity.astype(np.float32)
    if intensity.max() == 0:
        return np.zeros_like(intensity)
    max_val = float(intensity.max())
    if max_val <= 255:
        scale = 255.0
    elif max_val <= 65535:
        scale = 65535.0
    else:
        scale = max_val
    return np.clip(intensity / scale, 0.0, 1.0)


def get_z_norm(las, raw_xyz):
    dims = las.point_format.dimension_names
    if "Z_norm" in dims:
        return las["Z_norm"].astype(np.float32)
    z = raw_xyz[:, 2]
    return (z - z.min()).astype(np.float32)


def remap_classes(raw_labels):
    sem = np.full_like(raw_labels, IGNORE_INDEX, dtype=np.int32)
    for old, new in CLASS_REMAP.items():
        sem[raw_labels == old] = new
    return sem


def get_tree_ids(las, n, fname):
    dims = las.point_format.dimension_names
    if "treeID" in dims:
        return las["treeID"].astype(np.int32)
    print(f"⚠️ Warning: no treeID in {fname}, using zeros.")
    return np.zeros(n, dtype=np.int32)


# ------------------------------------------------ #
#   WRITE FORESTFORMER3D-READY PLY
# ------------------------------------------------ #
def write_forestformer_ply(out_path, xyz_abs, semantic, inst, z_norm, intensity):
    df = pd.DataFrame({
        "x": xyz_abs[:, 0],
        "y": xyz_abs[:, 1],
        "z": xyz_abs[:, 2],
        "semantic_seg": semantic,
        "treeID": inst,
        "z_norm": z_norm,
        "intensity": intensity,
    })
    cloud = PyntCloud(df)
    cloud.to_file(str(out_path))


# ------------------------------------------------ #
#   MAIN PROCESSING
# ------------------------------------------------ #
def process_las(path: Path):
    print(f"📦 Processing {path.name}")
    las = laspy.read(path)

    raw_xyz = np.vstack([las.x, las.y, las.z]).T
    offset = raw_xyz.min(axis=0)
    xyz_local = raw_xyz - offset

    # ---------- FEATURES ---------- #
    intensity = normalise_intensity(las.intensity)
    z_norm = get_z_norm(las, raw_xyz)

    feats = np.column_stack([xyz_local, z_norm, intensity]).astype(np.float32)

    # ---------- LABELS ---------- #
    raw_labels = las.classification.astype(np.int32)
    semantic = remap_classes(raw_labels)
    inst = get_tree_ids(las, len(raw_labels), path.name)

    # ---------- MINSU3D OUTPUT ---------- #
    coords = np.floor(xyz_local / VOXEL_SIZE).astype(np.int32)

    npz_path = OUT_NPZ_DIR / f"{path.stem}.npz"
    np.savez_compressed(
        npz_path,
        coords=coords,
        feats=feats,
        semantic_label=semantic,
        instance_label=inst,
        offset=offset.astype(np.float64),
    )

    # ---------- FORESTFORMER3D OUTPUT ---------- #
    ply_path = OUT_PLY_DIR / f"{path.stem}.ply"

    # FF3D can use absolute XYZ or local XYZ. 
    # Use ABSOLUTE xyz for consistency with ForAINetV2.
    write_forestformer_ply(
        ply_path,
        xyz_abs=raw_xyz,
        semantic=semantic,
        inst=inst,
        z_norm=z_norm,
        intensity=intensity,
    )

    return npz_path, ply_path


def main():
    las_files = sorted(INPUT_DIR.glob("*.las"))
    if not las_files:
        print("❌ No LAS files found.")
        return

    print(f"🚀 Found {len(las_files)} LAS files.")
    all_npz = []
    all_ply = []

    for f in las_files:
        try:
            npz, ply = process_las(f)
            all_npz.append(str(npz))
            all_ply.append(str(ply))
        except Exception as e:
            print(f"❌ Error in {f.name}: {e}")

    # Save lists for loaders
    with open(OUT_NPZ_DIR / "npz_list.txt", "w") as f:
        f.write("\n".join(all_npz))

    with open(OUT_PLY_DIR / "ply_list.txt", "w") as f:
        f.write("\n".join(all_ply))

    print("✅ All done!")
    print(f"📄 NPZ list: {OUT_NPZ_DIR / 'npz_list.txt'}")
    print(f"📄 PLY list: {OUT_PLY_DIR / 'ply_list.txt'}")


if __name__ == "__main__":
    main()
    
    
# %%: DROP IGNORE_INDEX POINTS

"""
Unified preprocessing pipeline for Calperum LiDAR tiles.

Produces TWO outputs per LAS file:
-------------------------------------------------------
1) MINSU3D NPZ tensor
   coords, feats, semantic_label, instance_label, offset
   -> For voxel-based sparse 3D U-Net (MinkowskiEngine)

2) ForestFormer3D-compatible PLY
   x, y, z, semantic_seg, treeID, z_norm, intensity
   -> For point-based panoptic segmentation

This ensures both frameworks use the *same preprocessing*
so comparisons are valid and reproducible.
"""

import laspy
import numpy as np
import pandas as pd
from pathlib import Path
from pyntcloud import PyntCloud  # pip install pyntcloud

# ---------------- CONFIG ---------------- #

INPUT_DIR = Path("/data/calperumResearch/site1_1_DD0001/lidar/clipped/3d_structure_obj_polar_id")

OUT_NPZ_DIR = Path("/data/calperumResearch/site1_1_DD0001/lidar/clipped/preprocessed/minsu3d_npz/")
OUT_PLY_DIR = Path("/data/calperumResearch/site1_1_DD0001/lidar/clipped/preprocessed/forestformer3d_ply/")

OUT_NPZ_DIR.mkdir(parents=True, exist_ok=True)
OUT_PLY_DIR.mkdir(parents=True, exist_ok=True)

VOXEL_SIZE = 0.10
IGNORE_INDEX = -100

CLASS_REMAP = {
    64: 0,        # Forb
    65: 1,        # Woody debris
    66: 2,        # Hummock
    67: 3,        # Shrub
    68: 4,        # Mallee
    0: IGNORE_INDEX,  # Unmatched / background
}

# ------------------------------------------------ #


def normalise_intensity(intensity):
    intensity = intensity.astype(np.float32)
    if intensity.size == 0:
        return intensity
    max_val = float(intensity.max())
    if max_val <= 0:
        return np.zeros_like(intensity, dtype=np.float32)
    if max_val <= 255:
        scale = 255.0
    elif max_val <= 65535:
        scale = 65535.0
    else:
        scale = max_val
    return np.clip(intensity / scale, 0.0, 1.0)


def get_z_norm(las, raw_xyz):
    dims = las.point_format.dimension_names
    if "Z_norm" in dims:
        return las["Z_norm"].astype(np.float32)
    z = raw_xyz[:, 2]
    return (z - z.min()).astype(np.float32)


def remap_classes(raw_labels):
    sem = np.full_like(raw_labels, IGNORE_INDEX, dtype=np.int32)
    for old, new in CLASS_REMAP.items():
        sem[raw_labels == old] = new
    return sem


def get_tree_ids(las, n, fname):
    dims = las.point_format.dimension_names
    if "treeID" in dims:
        return las["treeID"].astype(np.int32)
    print(f"⚠️ Warning: no treeID in {fname}, using zeros.")
    return np.zeros(n, dtype=np.int32)


# ------------------------------------------------ #
#   WRITE FORESTFORMER3D-READY PLY
# ------------------------------------------------ #
def write_forestformer_ply(out_path, xyz_abs, semantic, inst, z_norm, intensity):
    """
    Write a PLY for ForestFormer3D / ForAINetV2-style loaders.

    We drop points with semantic == IGNORE_INDEX so that
    all remaining points have valid class IDs (0..K-1).
    """
    valid_mask = semantic != IGNORE_INDEX
    n_valid = int(valid_mask.sum())

    if n_valid == 0:
        print(f"⚠️ No valid (non-ignore) points for {out_path.name}, skipping PLY.")
        return

    df = pd.DataFrame({
        "x": xyz_abs[valid_mask, 0],
        "y": xyz_abs[valid_mask, 1],
        "z": xyz_abs[valid_mask, 2],
        "semantic_seg": semantic[valid_mask],
        "treeID": inst[valid_mask],
        "z_norm": z_norm[valid_mask],
        "intensity": intensity[valid_mask],
    })
    cloud = PyntCloud(df)
    cloud.to_file(str(out_path))


# ------------------------------------------------ #
#   MAIN PROCESSING
# ------------------------------------------------ #
def process_las(path: Path):
    print(f"📦 Processing {path.name}")
    las = laspy.read(path)

    raw_xyz = np.vstack([las.x, las.y, las.z]).T
    if raw_xyz.size == 0:
        raise ValueError("Empty point cloud.")

    offset = raw_xyz.min(axis=0)
    xyz_local = raw_xyz - offset

    # ---------- FEATURES ---------- #
    intensity = normalise_intensity(las.intensity)
    z_norm = get_z_norm(las, raw_xyz)

    feats = np.column_stack([xyz_local, z_norm, intensity]).astype(np.float32)

    # ---------- LABELS ---------- #
    raw_labels = las.classification.astype(np.int32)
    semantic = remap_classes(raw_labels)
    inst = get_tree_ids(las, len(raw_labels), path.name)

    # ---------- MINSU3D OUTPUT ---------- #
    coords = np.floor(xyz_local / VOXEL_SIZE).astype(np.int32)

    npz_path = OUT_NPZ_DIR / f"{path.stem}.npz"
    np.savez_compressed(
        npz_path,
        coords=coords,
        feats=feats,
        semantic_label=semantic,
        instance_label=inst,
        offset=offset.astype(np.float64),
    )

    # ---------- FORESTFORMER3D OUTPUT ---------- #
    ply_path = OUT_PLY_DIR / f"{path.stem}.ply"

    # FF3D can use absolute XYZ or local XYZ.
    # Use ABSOLUTE xyz for consistency with ForAINetV2.
    write_forestformer_ply(
        ply_path,
        xyz_abs=raw_xyz,
        semantic=semantic,
        inst=inst,
        z_norm=z_norm,
        intensity=intensity,
    )

    return npz_path, ply_path


def main():
    las_files = sorted(INPUT_DIR.glob("*.las"))
    if not las_files:
        print("❌ No LAS files found.")
        return

    print(f"🚀 Found {len(las_files)} LAS files.")
    all_npz = []
    all_ply = []

    for f in las_files:
        try:
            npz, ply = process_las(f)
            all_npz.append(str(npz))
            all_ply.append(str(ply))
        except Exception as e:
            print(f"❌ Error in {f.name}: {e}")

    # Save lists for loaders
    with open(OUT_NPZ_DIR / "npz_list.txt", "w") as f:
        f.write("\n".join(all_npz))

    with open(OUT_PLY_DIR / "ply_list.txt", "w") as f:
        f.write("\n".join(all_ply))

    print("✅ All done!")
    print(f"📄 NPZ list: {OUT_NPZ_DIR / 'npz_list.txt'}")
    print(f"📄 PLY list: {OUT_PLY_DIR / 'ply_list.txt'}")


if __name__ == "__main__":
    main()
# %%
