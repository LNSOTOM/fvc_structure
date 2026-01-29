# normalize_las_z.py
import os
import numpy as np
import laspy
from pathlib import Path
from scipy.spatial import cKDTree


def normalize_las_z(
    input_las: str | Path,
    ground_las: str | Path,
    output_las: str | Path,
    *,
    k: int = 3,
    r_max: float = 1.5,
    extra_dim_name: str = "Z_norm",
) -> None:
    """
    Compute height above ground (HAG) using inverse-distance weighted kNN from ground LAS.

    Adds a new ExtraBytes dimension (default: Z_norm) to the LAS file.
    """
    input_las = Path(input_las)
    ground_las = Path(ground_las)
    output_las = Path(output_las)
    output_las.parent.mkdir(parents=True, exist_ok=True)

    # Load point clouds
    las_all = laspy.read(input_las)
    all_xyz = np.column_stack((las_all.x, las_all.y, las_all.z))

    las_ground = laspy.read(ground_las)
    ground_xyz = np.column_stack((las_ground.x, las_ground.y, las_ground.z))
    if len(ground_xyz) == 0:
        raise ValueError("❌ No ground points found in the CSF ground LAS.")

    # Check headers for consistency
    if not np.allclose(las_all.header.scales, las_ground.header.scales):
        print("⚠️ Warning: scale mismatch between input and ground LAS.")
    if not np.allclose(las_all.header.offsets, las_ground.header.offsets):
        print("⚠️ Warning: offset mismatch between input and ground LAS.")

    # Build KDTree
    tree = cKDTree(ground_xyz[:, :2])
    dist, idx = tree.query(all_xyz[:, :2], k=k, workers=-1)
    dist = np.maximum(dist, 1e-6)

    if dist.ndim == 1:  # ensure 2D shape
        dist = dist[:, None]
        idx = idx[:, None]

    # Inverse distance weights
    w = 1.0 / dist
    w /= w.sum(axis=1, keepdims=True)

    # Interpolated ground elevation
    gz_est = np.sum(ground_xyz[idx, 2] * w, axis=1)

    # Height Above Ground (HAG)
    hag = np.maximum(all_xyz[:, 2] - gz_est, 0.0).astype(np.float32)

    # Add or overwrite extra dimension
    if extra_dim_name not in las_all.point_format.dimension_names:
        las_all.add_extra_dim(laspy.ExtraBytesParams(name=extra_dim_name, type=np.float32))
    las_all[extra_dim_name] = hag

    las_all.header.software_id = "CSF_HAG_Normalization_by_Laura"
    las_all.write(output_las)

    print(f"✅ Wrote normalized LAS: {output_las}")
    print(f"   HAG stats (m): min={hag.min():.3f}, median={np.median(hag):.3f}, max={hag.max():.3f}")
    print(f"   Ground pts: {len(ground_xyz):,}, All pts: {len(all_xyz):,}, k={k}")