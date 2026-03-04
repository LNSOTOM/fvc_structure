#!/usr/bin/env python3
"""Sanity-check ForAINetV2 GT conversions.

Verifies that per-point GT fields in raw PLYs (`semantic_seg`, `treeID`) match:
- `forainetv2_instance_data/*_sem_label.npy` / `*_ins_label.npy`
- `semantic_mask/*.bin` / `instance_mask/*.bin`

Assumptions (based on current preprocessing):
- PLY `semantic_seg` is 1-indexed: {1,2,3} -> {0,1,2} via `-1`.
- Background semantic class is 0 (ground), and its instance id is forced to 0.

Usage examples:
  python data/ForAINetV2/sanity_check_ply_vs_masks.py --split test --max-scenes 3
  python data/ForAINetV2/sanity_check_ply_vs_masks.py --ids NIBIO2_NIBIO2_plot10_annotated_test
"""

from __future__ import annotations

import argparse
import os
from collections import Counter
from pathlib import Path

import numpy as np

# Local import (same folder)
from plyutils import read_ply  # type: ignore


def _find_ply(data_root: Path, scene_id: str) -> Path:
    candidates = [
        data_root / "train_val_data" / f"{scene_id}.ply",
        data_root / "test_data" / f"{scene_id}.ply",
        data_root / "test_data_calp" / f"{scene_id}.ply",
    ]
    for p in candidates:
        if p.is_file():
            return p
    raise FileNotFoundError(
        f"Could not find PLY for '{scene_id}'. Tried: " + ", ".join(str(p) for p in candidates)
    )


def _load_bin(path: Path, dtype: np.dtype) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(str(path))
    return np.fromfile(path, dtype=dtype)


def _fmt_counts(arr: np.ndarray, max_keys: int = 8) -> str:
    c = Counter(arr.tolist())
    items = sorted(c.items(), key=lambda kv: (-kv[1], kv[0]))
    if len(items) > max_keys:
        shown = items[:max_keys]
        rest = sum(v for _, v in items[max_keys:])
        return ", ".join(f"{k}:{v}" for k, v in shown) + f", …(+{rest} pts)"
    return ", ".join(f"{k}:{v}" for k, v in items)


def _compare(name: str, a: np.ndarray, b: np.ndarray) -> tuple[bool, str]:
    if a.shape != b.shape:
        return False, f"shape {a.shape} vs {b.shape}"
    if a.dtype != b.dtype:
        # dtype mismatch is usually OK if values match; compare after casting
        eq = np.array_equal(a.astype(np.int64), b.astype(np.int64))
        if not eq:
            mism = int(np.sum(a.astype(np.int64) != b.astype(np.int64)))
            return False, f"dtype {a.dtype} vs {b.dtype}, mismatches {mism}/{a.size}"
        return True, f"dtype {a.dtype} vs {b.dtype} (values identical)"

    eq = np.array_equal(a, b)
    if eq:
        return True, "identical"
    mism = int(np.sum(a != b))
    return False, f"mismatches {mism}/{a.size} ({mism / max(1, a.size):.6%})"


def check_scene(data_root: Path, scene_id: str, strict: bool) -> bool:
    ply_path = _find_ply(data_root, scene_id)
    pcd = read_ply(str(ply_path))

    required = {"x", "y", "z", "semantic_seg", "treeID"}
    missing = required - set(pcd.dtype.names)
    if missing:
        raise KeyError(f"PLY missing fields {sorted(missing)}: {ply_path}")

    semantic_seg = pcd["semantic_seg"].astype(np.int64)
    tree_id = pcd["treeID"].astype(np.int64)

    sem_expected = semantic_seg - 1
    ins_expected = tree_id.copy()
    ins_expected[sem_expected == 0] = 0

    sem_bin = _load_bin(data_root / "semantic_mask" / f"{scene_id}.bin", dtype=np.int64)
    ins_bin = _load_bin(data_root / "instance_mask" / f"{scene_id}.bin", dtype=np.int64)

    # Intermediate NPYS
    sem_npy_path = data_root / "forainetv2_instance_data" / f"{scene_id}_sem_label.npy"
    ins_npy_path = data_root / "forainetv2_instance_data" / f"{scene_id}_ins_label.npy"
    sem_npy = np.load(sem_npy_path).astype(np.int64)
    ins_npy = np.load(ins_npy_path).astype(np.int64)

    ok = True

    n_pts = int(semantic_seg.shape[0])
    print("=" * 80)
    print(f"Scene: {scene_id}")
    print(f"PLY:   {ply_path}")
    print(f"n_pts: {n_pts}")
    print(f"PLY semantic_seg raw: min={semantic_seg.min()}, max={semantic_seg.max()} counts: {_fmt_counts(semantic_seg)}")
    print(f"Expected sem (semantic_seg-1): min={sem_expected.min()}, max={sem_expected.max()} counts: {_fmt_counts(sem_expected)}")

    # Length checks
    if sem_bin.shape[0] != n_pts:
        ok = False
        print(f"[FAIL] semantic .bin length {sem_bin.shape[0]} != PLY n_pts {n_pts}")
    if ins_bin.shape[0] != n_pts:
        ok = False
        print(f"[FAIL] instance .bin length {ins_bin.shape[0]} != PLY n_pts {n_pts}")
    if sem_npy.shape[0] != n_pts:
        ok = False
        print(f"[FAIL] semantic .npy length {sem_npy.shape[0]} != PLY n_pts {n_pts}")
    if ins_npy.shape[0] != n_pts:
        ok = False
        print(f"[FAIL] instance .npy length {ins_npy.shape[0]} != PLY n_pts {n_pts}")

    # Value comparisons
    s_ok, s_msg = _compare("sem_bin", sem_bin.astype(np.int64), sem_expected.astype(np.int64))
    i_ok, i_msg = _compare("ins_bin", ins_bin.astype(np.int64), ins_expected.astype(np.int64))
    sn_ok, sn_msg = _compare("sem_npy", sem_npy.astype(np.int64), sem_expected.astype(np.int64))
    in_ok, in_msg = _compare("ins_npy", ins_npy.astype(np.int64), ins_expected.astype(np.int64))

    print(f"semantic_mask .bin vs PLY-derived expected: {s_msg}")
    print(f"instance_mask .bin vs PLY-derived expected: {i_msg}")
    print(f"*_sem_label.npy vs PLY-derived expected:     {sn_msg}")
    print(f"*_ins_label.npy vs PLY-derived expected:     {in_msg}")

    ok = ok and s_ok and i_ok and sn_ok and in_ok

    # Quick extra stats
    n_bg = int(np.sum(sem_expected == 0))
    n_ins_nonzero = int(np.sum(ins_expected != 0))
    uniq_ins = np.unique(ins_expected)
    uniq_ins_nonzero = uniq_ins[uniq_ins != 0]
    print(f"BG semantic (0) points: {n_bg}")
    print(f"Nonzero instance points: {n_ins_nonzero}")
    print(f"Unique instances (excluding 0): {len(uniq_ins_nonzero)}")

    if not ok and strict:
        raise SystemExit(2)
    return ok


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-root",
        default=None,
        help="Path to ForestFormer3D/data/ForAINetV2 (defaults to this script's folder)",
    )
    parser.add_argument("--split", choices=["train", "val", "test", "all"], default="all")
    parser.add_argument("--max-scenes", type=int, default=1)
    parser.add_argument("--ids", nargs="*", default=None, help="Explicit scene ids to check")
    parser.add_argument("--strict", action="store_true", help="Exit nonzero on first failure")
    args = parser.parse_args()

    data_root = Path(args.data_root) if args.data_root else Path(__file__).resolve().parent

    if args.ids:
        scene_ids = args.ids
    else:
        split_files = []
        if args.split in ("train", "all"):
            split_files.append(data_root / "meta_data" / "train_list.txt")
        if args.split in ("val", "all"):
            split_files.append(data_root / "meta_data" / "val_list.txt")
        if args.split in ("test", "all"):
            split_files.append(data_root / "meta_data" / "test_list.txt")

        scene_ids = []
        for sf in split_files:
            if not sf.is_file():
                raise FileNotFoundError(str(sf))
            with sf.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        scene_ids.append(line)
                        break  # first entry per split

    scene_ids = scene_ids[: max(1, int(args.max_scenes))]

    all_ok = True
    for sid in scene_ids:
        ok = check_scene(data_root, sid, strict=args.strict)
        all_ok = all_ok and ok

    print("=" * 80)
    print("RESULT:", "OK" if all_ok else "FAIL")


if __name__ == "__main__":
    main()
