"""Transfer 2D growth-form labels onto LiDAR with fvc3d.

Run from the repository root after `uv sync` (or `pip install -e .`):

    python examples/transfer_labels.py growth_forms.tif lidar.laz labelled_vegetation.laz
    python examples/transfer_labels.py --batch labels_dir/ lidar_dir/ out_dir/

Add --class-map paper for the published Zenodo rasters, which code classes 0–4.
"""

import argparse
import csv
import logging
import re
from pathlib import Path

from fvc3d import AlignmentError, transfer_labels


def one_tile(labels, lidar, output, class_map=None):
    result = transfer_labels(
        labels=labels,
        point_cloud=lidar,
        output=output,
        validate_alignment=True,
        class_map=class_map,  # e.g. "paper", or {1: "forb", 2: "woody_debris", 3: "hummock", 4: "shrub", 5: "mallee"}
        # config={"voxel_size_xy": 0.05, "voxel_size_z": 0.05},   # or "configs/transfer_labels.yaml"
    )
    print(result.summary())

    # Per-point arrays line up with the input cloud.
    print(result.classification[:10], result.tree_id[:10], result.z_norm[:10])


def batch(labels_dir, lidar_dir, out_dir, class_map=None):
    """Pair files by the last number in their name (e.g. mask.109.tif ↔ z_norm.109.las)."""
    tile_id = lambda p: (re.findall(r"\d+", p.stem) or [None])[-1]
    lidar = {tile_id(p): p for p in Path(lidar_dir).iterdir() if p.suffix.lower() in (".las", ".laz")}
    rows = []
    for labels in sorted(Path(labels_dir).glob("*.tif")):
        tid = tile_id(labels)
        if tid not in lidar:
            print(f"skip {labels.name}: no LiDAR tile {tid}")
            continue
        try:
            result = transfer_labels(labels, lidar[tid], Path(out_dir) / f"tile{tid}_labelled.laz",
                                     class_map=class_map, overwrite=True)
        except AlignmentError as err:
            print(f"skip tile {tid}: {err}")
            continue
        q = result.quality
        rows.append(result.metrics | {
            "canopy_shift_m": result.alignment.shift,
            "voxel_agreement_oa_pct": q.voxel_agreement["overall_accuracy_pct"],
            "crown_retention_pct": q.crown_retention["retention_pct"] if q.crown_retention else None,
        })
        print(f"tile {tid}: {result.metrics['final_veg_points']:,} points, {result.metrics['final_treeids']} trees")
    if rows:
        with open(Path(out_dir) / "metrics.csv", "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--batch", action="store_true", help="process folders of tiles")
    parser.add_argument("--class-map", choices=["paper"], help='"paper": the published 0–4 label rasters')
    parser.add_argument("labels")
    parser.add_argument("lidar")
    parser.add_argument("output")
    args = parser.parse_args()
    if args.batch:
        Path(args.output).mkdir(parents=True, exist_ok=True)
        batch(args.labels, args.lidar, args.output, args.class_map)
    else:
        one_tile(args.labels, args.lidar, args.output, args.class_map)
