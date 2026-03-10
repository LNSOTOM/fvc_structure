# %%
"""
Preprocess Calperum LAS files into voxel tensors suitable for:
- Minsu3D (MinkowskiEngine sparse voxels)
- ForestFormer3D-style point/label pipelines

Outputs one .npz per input LAS with:
  - coords: (N, 3) int32 voxel indices (local grid)
  - feats:  (N, C) float32 features [x_local, y_local, z_local, z_norm, intensity]
  - semantic_label: (N,) int32 remapped classes (0..K-1, or IGNORE_INDEX)
  - instance_label: (N,) int32 tree IDs (0 if missing)
  - offset: (3,) float64 XYZ offset used for local coordinates

You can:
  - Use coords/feats/labels directly for Minsu3D.
  - Or ignore coords and use xyz_local + labels as “points” for a ForestFormer3D-style
    conversion step (e.g. writing PLY with the same semantics / instances).
"""

import laspy
import numpy as np
from pathlib import Path

# ---------------- CONFIGURATION ---------------- #

# Root folder containing your LAS tiles with structure + treeID
INPUT_DIR = Path(
    "/data/calperumResearch/site1_1_DD0001/lidar/clipped/3d_structure_obj_polar_id"
)

# Output folder for NPZ tensors
OUT_DIR = Path(
    "/data/calperumResearch/site1_1_DD0001/lidar/clipped/voxel_tensors"
)
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Voxel size (must match what you use in Minsu3D / ME configs)
VOXEL_SIZE = 0.10  # 10 cm

# Ignore index (standard PyTorch convention)
IGNORE_INDEX = -100

# Mapping LAS classification codes -> model class IDs
# (adjust these once and reuse everywhere)
CLASS_REMAP = {
    64: 0,       # Forb
    65: 1,       # Woody debris
    66: 2,       # Hummock
    67: 3,       # Shrub
    68: 4,       # Mallee
    0: IGNORE_INDEX,  # background / unmatched
}

# ------------------------------------------------ #


def normalise_intensity(intensity: np.ndarray) -> np.ndarray:
    """
    Normalise raw intensity to [0, 1].
    Handles 8/16-bit or weird ranges robustly.
    """
    intensity = intensity.astype(np.float32)

    if intensity.size == 0:
        return intensity

    max_val = float(intensity.max())
    if max_val <= 0:
        return np.zeros_like(intensity, dtype=np.float32)

    # If it looks like 8-bit, scale by 255; if 16-bit, scale by 65535;
    # otherwise scale by its own max.
    if max_val <= 255.0:
        base = 255.0
    elif max_val <= 65535.0:
        base = 65535.0
    else:
        base = max_val

    return np.clip(intensity / base, 0.0, 1.0)


def get_z_norm(las, raw_xyz: np.ndarray) -> np.ndarray:
    """
    Return Z_norm as float32.
    Priority:
      1) Existing ExtraBytes dimension named 'Z_norm'
      2) Relative height within the tile (z - z_min)
    """
    dim_names = set(las.point_format.dimension_names)

    if "Z_norm" in dim_names:
        z_norm = las["Z_norm"].astype(np.float32)
    else:
        # Fallback: simple relative height within the tile
        z = raw_xyz[:, 2]
        z_norm = (z - z.min()).astype(np.float32)

    return z_norm


def remap_classes(raw_labels: np.ndarray) -> np.ndarray:
    """
    Remap LAS classification codes to a compact [0..K-1] label space,
    using CLASS_REMAP. Unmapped labels default to IGNORE_INDEX.
    """
    sem_labels = np.full_like(raw_labels, IGNORE_INDEX, dtype=np.int32)

    for las_cls, model_cls in CLASS_REMAP.items():
        mask = (raw_labels == las_cls)
        if np.any(mask):
            sem_labels[mask] = int(model_cls)

    return sem_labels


def get_instance_labels(las, n_points: int, path: Path) -> np.ndarray:
    """
    Retrieve tree instance IDs from 'treeID' ExtraBytes if present.
    Otherwise return zeros and warn.
    """
    dim_names = set(las.point_format.dimension_names)
    if "treeID" in dim_names:
        inst_labels = las["treeID"].astype(np.int32)
    else:
        print(f"⚠️  Warning: 'treeID' not found in {path.name}. Setting all instances to 0.")
        inst_labels = np.zeros(n_points, dtype=np.int32)

    return inst_labels


def process_las(path: Path) -> Path:
    """
    Process a single LAS tile into a compressed NPZ with:
      coords, feats, semantic_label, instance_label, offset
    """
    print(f"📦 Processing: {path.name}")
    las = laspy.read(path)

    # ---------- 1. Raw coordinates & local shift ---------- #
    raw_xyz = np.vstack([las.x, las.y, las.z]).T  # (N, 3)
    if raw_xyz.size == 0:
        raise ValueError("Empty point cloud.")

    offset = raw_xyz.min(axis=0)  # (3,)
    xyz_local = raw_xyz - offset  # local coords in metres

    # Integer voxel indices (N, 3)
    coords = np.floor(xyz_local / VOXEL_SIZE).astype(np.int32)

    # ---------- 2. Features ---------- #
    intensity = normalise_intensity(las.intensity)
    z_norm = get_z_norm(las, raw_xyz)

    # Feature vector: [x_local, y_local, z_local, z_norm, intensity]
    feats = np.column_stack([xyz_local, z_norm, intensity]).astype(np.float32)

    # ---------- 3. Semantic labels ---------- #
    raw_labels = las.classification.astype(np.int32)
    sem_labels = remap_classes(raw_labels)

    # ---------- 4. Instance labels ---------- #
    inst_labels = get_instance_labels(las, len(raw_labels), path)

    # ---------- 5. Save to NPZ ---------- #
    out_npz = OUT_DIR / f"{path.stem}.npz"
    np.savez_compressed(
        out_npz,
        coords=coords,                  # (N, 3) int32
        feats=feats,                    # (N, C) float32
        semantic_label=sem_labels,      # (N,) int32
        instance_label=inst_labels,     # (N,) int32
        offset=offset.astype(np.float64)  # (3,) float64
    )

    return out_npz


def main():
    if not INPUT_DIR.exists():
        print(f"❌ Input directory does not exist: {INPUT_DIR}")
        return

    las_files = sorted(INPUT_DIR.glob("*.las"))
    if not las_files:
        print(f"❌ No .las files found in {INPUT_DIR}")
        return

    print(f"🚀 Found {len(las_files)} LAS files.")
    print(f"📂 Output directory: {OUT_DIR}")

    dataset_paths = []

    for p in las_files:
        try:
            out_npz = process_las(p)
            dataset_paths.append(str(out_npz.resolve()))
        except Exception as e:
            print(f"❌ Error processing {p.name}: {e}")

    # Save a simple file list for downstream loaders / splitters
    list_file = OUT_DIR / "all_tiles_list.txt"
    with open(list_file, "w") as f:
        f.write("\n".join(dataset_paths))

    print(f"✅ Done. Processed {len(dataset_paths)} files.")
    print(f"📄 File list written to: {list_file}")


if __name__ == "__main__":
    main()
# %%
