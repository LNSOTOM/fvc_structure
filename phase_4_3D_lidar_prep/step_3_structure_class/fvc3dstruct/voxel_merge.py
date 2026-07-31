"""Cartesian voxel merge: transfer structure labels via 3D voxel grid."""

import numpy as np


def cartesian_voxel_merge(las_z, las_s, voxel_xy=0.10, voxel_z=0.10, return_stats=False):
    """
    Transfer structure labels (64–68) from structure LAS to Z_norm LAS
    via a shared 3D voxel grid with majority voting within voxels.
    
    Args:
        las_z: Z_norm LAS (target points)
        las_s: Structure LAS (source labels)
        voxel_xy: Voxel size in XY (m)
        voxel_z: Voxel size in Z (m)
    
    Returns:
        classification: Array of transferred labels (0 or 64–68)
        stats: Optional dict with voxel-transfer summary when return_stats=True
    """
    x_min = min(las_z.x.min(), las_s.x.min())
    y_min = min(las_z.y.min(), las_s.y.min())
    z_min = min(las_z.z.min(), las_s.z.min())
    
    origin = np.array([x_min, y_min, z_min], dtype=np.float64)
    
    def vox(x, y, z):
        return np.stack(
            [
                np.floor((x - origin[0]) / voxel_xy).astype(int),
                np.floor((y - origin[1]) / voxel_xy).astype(int),
                np.floor((z - origin[2]) / voxel_z).astype(int),
            ],
            axis=-1,
        )
    
    # Build voxel map from structure LAS
    vox_s = vox(las_s.x, las_s.y, las_s.z)
    raw_struct = np.where(
        (las_s.classification >= 64) & (las_s.classification <= 68),
        las_s.classification,
        0,
    )
    
    valid_mask = raw_struct != 0
    voxel_struct_map = {}
    if np.any(valid_mask):
        vox_struct = np.column_stack((vox_s[valid_mask], raw_struct[valid_mask]))
        uniq_vox_struct, counts = np.unique(vox_struct, axis=0, return_counts=True)

        voxel_coords = uniq_vox_struct[:, :3]
        labels = uniq_vox_struct[:, 3]
        order = np.lexsort((labels, voxel_coords[:, 2], voxel_coords[:, 1], voxel_coords[:, 0]))
        voxel_coords = voxel_coords[order]
        labels = labels[order]
        counts = counts[order]

        unique_voxels, start_idx = np.unique(voxel_coords, axis=0, return_index=True)
        boundaries = np.append(start_idx[1:], len(counts))

        for voxel_idx, start, end in zip(unique_voxels, start_idx, boundaries):
            best = start + int(np.argmax(counts[start:end]))
            voxel_struct_map[tuple(voxel_idx)] = int(labels[best])

    print(f"  ✅ Voxels with non-zero structure class: {len(voxel_struct_map):,}")
    
    # Transfer to Z_norm points
    vox_z = vox(las_z.x, las_z.y, las_z.z)
    classification = np.zeros(len(las_z.points), dtype=np.uint8)
    
    for i, v in enumerate(vox_z):
        classification[i] = voxel_struct_map.get(tuple(v), 0)
    
    n_labeled = int(np.count_nonzero(classification))
    pct = 100.0 * n_labeled / len(classification) if len(classification) > 0 else 0.0
    print(f"  ✅ Transferred labels to {n_labeled:,}/{len(classification):,} Z_norm points ({pct:.1f}%)")

    if not return_stats:
        return classification

    stats = {
        "voxel_size_xy": float(voxel_xy),
        "voxel_size_z": float(voxel_z),
        "nonzero_structure_voxels": int(len(voxel_struct_map)),
        "labeled_points": n_labeled,
        "total_points": int(len(classification)),
        "labeled_pct": float(pct),
    }
    return classification, stats
