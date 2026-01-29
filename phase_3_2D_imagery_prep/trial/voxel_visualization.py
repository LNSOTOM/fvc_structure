#!/usr/bin/env python3
"""Voxel-based extraction of vegetation structure aligned to an FVC mask."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from voxel_pipeline import (
    LidarPoints,
    MaskSample,
    VoxelProducts,
    load_lidar_points,
    sample_mask,
    voxelise,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("las", type=Path, help="Height normalised LAS/LAZ file")
    parser.add_argument("mask", type=Path, help="FVC raster mask aligned to 5 cm grid")
    parser.add_argument("output", type=Path, help="Directory for saved voxel arrays")
    parser.add_argument(
        "--ignore-labels",
        nargs="*",
        type=float,
        default=(255, -9999, -1),
        help="Mask codes to drop before voxelisation",
    )
    parser.add_argument(
        "--exclude-classes",
        nargs="*",
        type=float,
        default=(0, 3),
        help="Vegetation mask classes to discard (e.g. bare earth, shadow)",
    )
    parser.add_argument(
        "--voxel-size",
        type=float,
        default=None,
        help="Optional voxel edge length in metres; defaults to mask pixel size",
    )
    parser.add_argument(
        "--z-override",
        type=str,
        default=None,
        help="Explicit LAS extra dimension to use for Z_norm (fallback otherwise)",
    )
    parser.add_argument(
        "--summary",
        action="store_true",
        help="Print voxel grid summary without saving arrays",
    )
    return parser.parse_args()


def save_outputs(
    products: VoxelProducts,
    sample: MaskSample,
    lidar: LidarPoints,
    output_dir: Path,
    mask_path: Path,
    las_path: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    np.save(output_dir / "voxel_occupancy.npy", products.occupancy)
    np.save(output_dir / "voxel_point_counts.npy", products.point_counts)
    np.save(output_dir / "voxel_label_dominant.npy", products.dominant_labels)
    np.save(output_dir / "xyz_norm_with_labels.npy", np.column_stack((sample.coords, sample.labels)))

    meta = {
        **products.metadata,
        "las_path": str(las_path),
        "mask_path": str(mask_path),
        "z_attribute": lidar.z_source,
        "total_las_points": int(lidar.total_points),
        "kept_fraction": sample.kept_fraction,
        "drop_stats": sample.drop_stats,
    }
    with open(output_dir / "voxel_meta.json", "w", encoding="utf8") as fh:
        json.dump(meta, fh, indent=2)


def run_pipeline(args: argparse.Namespace) -> None:
    lidar = load_lidar_points(args.las, z_override=args.z_override)
    sample = sample_mask(lidar, args.mask, args.ignore_labels, args.exclude_classes)
    products = voxelise(sample, voxel_size=args.voxel_size)

    if args.summary:
        print(json.dumps(products.metadata, indent=2))
        return

    save_outputs(products, sample, lidar, args.output, args.mask, args.las)
    print("Voxel volumes saved to", args.output)
    print(json.dumps(products.metadata, indent=2))


if __name__ == "__main__":
    run_pipeline(parse_args())
