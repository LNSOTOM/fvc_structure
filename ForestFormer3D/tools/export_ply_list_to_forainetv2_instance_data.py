#!/usr/bin/env python3
"""Export a list of PLY scenes into ForAINetV2-style instance-data npys.

This is the missing step before you can build a test-only PKL using
`tools/create_forainetv2_test_only.py`.

It writes, per scene id, into:
  <out-root>/forainetv2_instance_data/<scene_id>_vert.npy
  <out-root>/forainetv2_instance_data/<scene_id>_sem_label.npy
  <out-root>/forainetv2_instance_data/<scene_id>_ins_label.npy
  <out-root>/forainetv2_instance_data/<scene_id>_aligned_bbox.npy
  <out-root>/forainetv2_instance_data/<scene_id>_unaligned_bbox.npy
  <out-root>/forainetv2_instance_data/<scene_id>_axis_align_matrix.npy
  <out-root>/forainetv2_instance_data/<scene_id>_offsets.npy (extra; optional)

This reuses `data/ForAINetV2/load_forainetv2_data.export()`.

Example:
  python tools/export_ply_list_to_forainetv2_instance_data.py \
    --ply-dir data/ForAINetV2/test_data_calp \
    --scene-list data/ForAINetV2/meta_data/test_list_calp.txt \
    --out-root data/ForAINetV2_calp
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import List

import numpy as np


def _read_list(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        return [ln.strip() for ln in f.readlines() if ln.strip()]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ply-dir", required=True, help="Directory containing <scene_id>.ply")
    p.add_argument("--scene-list", required=True, help="Text file with scene ids (one per line)")
    p.add_argument("--out-root", required=True, help="Output dataset root")
    p.add_argument("--skip-existing", action="store_true", help="Skip scenes that already have *_vert.npy")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # Ensure we can import the ForAINetV2 exporter (it relies on local plyutils).
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    forainetv2_data_dir = os.path.join(repo_root, "data", "ForAINetV2")
    if forainetv2_data_dir not in sys.path:
        sys.path.insert(0, forainetv2_data_dir)

    from load_forainetv2_data import export  # type: ignore

    scene_ids = _read_list(args.scene_list)
    out_dir = os.path.join(args.out_root, "forainetv2_instance_data")
    os.makedirs(out_dir, exist_ok=True)

    num_done = 0
    num_skip = 0
    for scene_id in scene_ids:
        ply_path = os.path.join(args.ply_dir, f"{scene_id}.ply")
        if not os.path.isfile(ply_path):
            raise FileNotFoundError(f"Missing PLY: {ply_path}")

        out_prefix = os.path.join(out_dir, scene_id)
        vert_path = out_prefix + "_vert.npy"
        if args.skip_existing and os.path.isfile(vert_path):
            num_skip += 1
            continue

        pts, sem, ins, unaligned_bb, aligned_bb, axis_mat, offsets = export(
            ply_path,
            output_file=out_prefix,
            test_mode=False,
        )

        # Save offsets too (not required by the dataloader, but useful for debugging)
        np.save(out_prefix + "_offsets.npy", offsets)

        num_done += 1
        if num_done % 5 == 0:
            print(f"Exported {num_done}/{len(scene_ids)} scenes...")

    print(f"Done. Exported={num_done}, skipped={num_skip}, total={len(scene_ids)}")


if __name__ == "__main__":
    main()
