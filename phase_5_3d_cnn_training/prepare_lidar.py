#!/usr/bin/env python3
"""Prepare one structure-classified LAS tile as a local-coordinate PLY."""

import argparse
from pathlib import Path

import laspy
import numpy as np
import yaml
from pyntcloud import PyntCloud

from lidar_preprocessing import PreprocessingConfig, process_las


CLASS_REMAP = {0: -100, 64: 1, 65: 2, 66: 3, 67: 4, 68: 5}
PROFILES = ("historical_five_class", "three_class_hsm")


def load_config(path):
    with path.open(encoding="utf-8") as file:
        job = yaml.safe_load(file)
    if not isinstance(job, dict):
        raise ValueError(f"Config must be a YAML mapping: {path}")
    for key in ("input_las", "output_dir", "class_profile"):
        if not isinstance(job.get(key), str) or not job[key].strip():
            raise ValueError(f"Missing config value: {key}")
    if job["class_profile"] not in PROFILES:
        raise ValueError(f"class_profile must be one of {PROFILES}")
    return job


def prepare(job):
    input_las = Path(job["input_las"]).expanduser()
    output_dir = Path(job["output_dir"]).expanduser()
    ply_path = output_dir / f"{input_las.stem}.ply"
    offset_path = output_dir / f"{input_las.stem}_offset.npy"
    for path in (ply_path, offset_path):
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite: {path}")

    las = laspy.read(input_las)
    if len(las.points) == 0:
        raise ValueError("Input LAS has no points")
    dimensions = set(las.point_format.dimension_names)
    missing = {"Z_norm", "treeID"} - dimensions
    if missing:
        raise ValueError(f"Input LAS is missing required dimensions: {sorted(missing)}")
    xyz = np.column_stack((las.x, las.y, las.z))
    heights = np.asarray(las["Z_norm"])
    tree_ids = np.asarray(las["treeID"])
    if not np.isfinite(xyz).all() or not np.isfinite(heights).all() or (heights < 0).any():
        raise ValueError("Coordinates and Z_norm must be finite; Z_norm must be non-negative")
    if (not np.isfinite(tree_ids).all() or (tree_ids < 0).any()
            or (tree_ids > np.iinfo(np.int32).max).any()
            or np.any(tree_ids != np.floor(tree_ids))):
        raise ValueError("treeID must contain non-negative int32-compatible integers")
    classes, counts = np.unique(np.asarray(las.classification), return_counts=True)
    class_counts = dict(zip(classes.tolist(), counts.tolist()))
    unsupported = set(class_counts) - set(CLASS_REMAP)
    if unsupported:
        raise ValueError(f"Unsupported LAS classifications: {sorted(unsupported)}")
    if job["class_profile"] == "three_class_hsm" and set(class_counts) & {64, 65}:
        raise ValueError(f"Three-class HSM rejects forb (64) and woody debris (65): {class_counts}")
    valid = np.asarray(las.classification) != 0
    if not valid.any():
        raise ValueError("Input LAS contains no supported vegetation points")
    print(f"Input points: {len(las.points)}; LAS classifications: {class_counts}")
    print(f"Excluded background (class 0): {int((~valid).sum())}")

    config = PreprocessingConfig(
        input_dir=input_las.parent,
        out_ply_dir=output_dir,
        out_npz_dir=None,
        save_npz=False,
        ply_use_local_coords=True,
        save_ply_offset=True,
        class_remap=dict(CLASS_REMAP),
    )
    process_las(input_las, config)
    if not ply_path.is_file() or not offset_path.is_file():
        raise RuntimeError("Preprocessing did not write the expected PLY and offset")
    points = PyntCloud.from_file(str(ply_path)).points
    expected_classes = np.asarray(las.classification, dtype=np.int32)[valid] - 63
    if (len(points) != int(valid.sum())
            or not np.array_equal(points["semantic_seg"].to_numpy(), expected_classes)
            or not np.array_equal(points["treeID"].to_numpy(), tree_ids[valid])
            or not np.array_equal(points["z_norm"].to_numpy(), heights[valid].astype(np.float32))):
        raise RuntimeError("PLY point count, labels or normalized heights do not match the input")
    offset = np.load(offset_path, allow_pickle=False)
    restored = points[["x", "y", "z"]].to_numpy(dtype=np.float64) + offset
    if not np.allclose(restored, xyz[valid], rtol=0, atol=1e-4):
        raise RuntimeError("PLY coordinates and offset do not reconstruct the input within 0.1 mm")
    print(f"PLY: {ply_path}\nOffset: {offset_path}")
    print(f"Validated {len(points)} points, class IDs, tree IDs and reconstructed coordinates.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    prepare(load_config(args.config))


if __name__ == "__main__":
    main()
