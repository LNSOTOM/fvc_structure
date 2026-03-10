#!/usr/bin/env python3
"""Create a MED test_data folder compatible with the 3-class ForAINetV2 pipeline.

Background:
    The ForAINetV2 exporter in this repo does `label_ids = semantic_seg - 1`.
    The downstream dataset helper is hardcoded to 3 classes (label_ids in {0,1,2}).

Therefore, the output PLYs written by this script must have `semantic_seg` in {1,2,3}.

This script supports two modes:
    1) Constant semantic label (legacy): force all points to the same `semantic_seg`.
    2) Remap: map source semantic labels (e.g. 3/4/5) into 1/2/3.

`treeID` is always preserved (for instance bboxes). All other vertex fields are copied as-is.

It writes files named <scene_id>.ply into the destination directory, where
scene_id lines come from the provided list (e.g. znorm_3Dlabels_m_106).

Example:
  python tools/make_test_data_med_forestformer3.py \
    --src-dir /data/calperumResearch/site1_1_DD0001/lidar/clipped/preprocessed/forestformer3d_ply \
    --dst-dir data/ForAINetV2/test_data_med_forest3 \
    --scene-list data/ForAINetV2/meta_data/test_list_med.txt
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Iterable, List

import numpy as np


def _read_list(path: Path) -> List[str]:
    return [ln.strip() for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]


def _infer_src_path(src_dir: Path, scene_id: str) -> Path:
    """Map scene_id like 'znorm_3Dlabels_m_106' to a source filename.

    Site preprocessing output usually uses 'znorm_3Dlabels_106.ply' (no '_m_').
    """
    sid = scene_id
    if sid.endswith(".ply"):
        sid = sid[:-4]

    base = sid
    if base.startswith("znorm_3Dlabels_m_"):
        base = base.replace("znorm_3Dlabels_m_", "znorm_3Dlabels_", 1)

    # Prefer the no-_m_ source, but fall back to exact name.
    candidate1 = src_dir / f"{base}.ply"
    candidate2 = src_dir / f"{sid}.ply"
    if candidate1.is_file():
        return candidate1
    if candidate2.is_file():
        return candidate2
    raise FileNotFoundError(f"Missing source PLY for '{scene_id}'. Tried: {candidate1} and {candidate2}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--src-dir", required=True, type=Path)
    p.add_argument("--dst-dir", required=True, type=Path)
    p.add_argument("--scene-list", required=True, type=Path)
    p.add_argument(
        "--semantic-seg",
        type=int,
        default=3,
        help="Constant semantic_seg to write into output PLY (default: 3 -> label_id 2 after -1). Ignored if --remap is provided.",
    )
    p.add_argument(
        "--remap",
        type=str,
        default=None,
        help=(
            "Optional semantic remap in the form 'src:dst,src:dst'. "
            "Example for rangeland 3/4/5 -> 1/2/3: --remap '3:1,4:2,5:3'."
        ),
    )
    p.add_argument(
        "--unknown-semantic",
        type=int,
        default=1,
        help="If --remap is used, semantic_seg values not present in the mapping are set to this value (must be >=1).",
    )
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def _parse_remap(spec: str) -> dict[int, int]:
    mapping: dict[int, int] = {}
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if ":" not in part:
            raise ValueError(f"Bad --remap entry '{part}'. Expected 'src:dst'.")
        src_s, dst_s = [x.strip() for x in part.split(":", 1)]
        mapping[int(src_s)] = int(dst_s)
    if not mapping:
        raise ValueError("--remap provided but parsed empty mapping")
    return mapping


def main() -> None:
    args = parse_args()

    if args.remap is not None:
        remap = _parse_remap(args.remap)
        if args.unknown_semantic < 1:
            raise ValueError("--unknown-semantic must be >= 1 (semantic_seg=0 would break exporter)")
    else:
        remap = None

    repo_root = Path(__file__).resolve().parents[1]
    plyutils_dir = repo_root / "data" / "ForAINetV2"
    if str(plyutils_dir) not in sys.path:
        sys.path.insert(0, str(plyutils_dir))

    from plyutils import read_ply, write_ply  # type: ignore

    scene_ids = _read_list(args.scene_list)
    args.dst_dir.mkdir(parents=True, exist_ok=True)

    num_done = 0
    for scene_id in scene_ids:
        src_path = _infer_src_path(args.src_dir, scene_id)
        dst_path = args.dst_dir / f"{scene_id}.ply"

        if dst_path.exists() and not args.overwrite:
            continue

        pcd = read_ply(str(src_path))
        if "x" not in pcd.dtype.names or "y" not in pcd.dtype.names or "z" not in pcd.dtype.names:
            raise KeyError(f"{src_path} missing x/y/z fields")

        if "treeID" not in pcd.dtype.names:
            raise KeyError(f"{src_path} missing treeID field (required for instance export)")

        # Build output fields: copy everything but force semantic_seg
        field_names: List[str] = list(pcd.dtype.names)
        fields: List[np.ndarray] = []

        for name in field_names:
            arr = pcd[name]
            if name == "semantic_seg":
                if remap is None:
                    arr = np.full(arr.shape, args.semantic_seg, dtype=arr.dtype)
                else:
                    src = arr.astype(np.int64, copy=False)
                    out = np.full(src.shape, int(args.unknown_semantic), dtype=np.int64)
                    for src_id, dst_id in remap.items():
                        out[src == int(src_id)] = int(dst_id)
                    if out.min(initial=1) < 1:
                        raise ValueError("Remap produced semantic_seg < 1, which is incompatible with exporter")
                    arr = out.astype(arr.dtype, copy=False)
            fields.append(arr)

        ok = write_ply(str(dst_path), fields, field_names)
        if not ok:
            raise RuntimeError(f"Failed writing {dst_path}")

        num_done += 1
        if num_done % 5 == 0:
            print(f"Wrote {num_done}/{len(scene_ids)} PLYs...")

    print(f"Done. Wrote {num_done} PLYs into {args.dst_dir}")


if __name__ == "__main__":
    main()
