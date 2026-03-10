import laspy
import numpy as np
from pathlib import Path

VOXEL_SIZE = 0.05   # 5 cm
OUT_DIR = Path("voxel_scenes")
OUT_DIR.mkdir(exist_ok=True)

def voxelize_points(xyz, voxel_size):
    """Convert continuous coordinates to integer voxel grid."""
    return np.floor(xyz / voxel_size).astype(np.int32)

def process_las(path):
    las = laspy.read(path)

    # --- Extract features ---
    xyz = np.vstack([las.x, las.y, las.z]).T.astype(np.float32)
    intensity = las.intensity.astype(np.float32)

    # Normalised height (already done in your file)
    if "Z_norm" in las.point_format.extra_dimension_names:
        z_norm = las["Z_norm"].astype(np.float32)
    else:
        z_norm = xyz[:, 2].copy()

    # Labels from ExtraBytes
    if "label" in las.point_format.extra_dimension_names:
        labels = las["label"].astype(np.int64)
    else:
        labels = np.zeros(len(xyz), dtype=np.int64)

    # Features concatenation
    feats = np.column_stack([xyz, z_norm, intensity])

    # Voxel coords
    coords = voxelize_points(xyz, VOXEL_SIZE)

    # Save NPZ
    out_npz = OUT_DIR / (path.stem + ".npz")
    np.savez_compressed(
        out_npz,
        coords=coords,
        feats=feats,
        labels=labels
    )

    print(f"Saved: {out_npz}")
    return out_npz


def main():
    las_path = Path("/data/calperumResearch/site1_1_DD0001/lidar/clipped/3d_structure_mask_density/z_norm_mask_tiles_multispectral.55.las")
    out_file = process_las(las_path)

    # Write metadata TXT
    with open("metadata.txt", "w") as f:
        f.write(str(out_file) + "\n")

    print("metadata.txt written.")


if __name__ == "__main__":
    main()
