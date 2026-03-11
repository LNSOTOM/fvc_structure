#!/usr/bin/env python3
"""Create ForAINetV2 train/val info PKLs from explicit scene-id lists.

This complements `tools/create_forainetv2_test_only.py` for Experiment 2 (few-shot
fine-tuning), where train/val are small custom subsets.

It assumes you already exported per-scene npys under:
  <root-path>/<FORAINETV2_INSTANCE_DATA_DIR>/

and you want to generate:
  - forainetv2_oneformer3d_infos_train_L1.pkl
  - forainetv2_oneformer3d_infos_val_L1.pkl

IMPORTANT:
- Set these env vars consistently BEFORE running this script and BEFORE training:
  - FORAINETV2_INSTANCE_DATA_DIR
  - FORAINETV2_BIN_BASE_DIR

Those control where the script writes the generated `*.bin` files and what paths
end up inside the PKL.

Example (inside container):
  FORAINETV2_PROTOCOL=forest \
  FORAINETV2_INSTANCE_DATA_DIR=forainetv2_instance_data_exp2A \
  FORAINETV2_BIN_BASE_DIR=forainetv2_instance_data_exp2A \
  python /workspace/tools/create_forainetv2_train_val_from_lists.py \
    --root-path /workspace/data/ForAINetV2 \
    --out-dir /workspace/data/ForAINetV2 \
        --train-list /workspace/data/ForAINetV2/meta_data/train_list_exp2_mediumL1_lowL1_fewshot.txt \
        --val-list /workspace/data/ForAINetV2/meta_data/val_list_exp2_mediumL1_lowL1_fewshot.txt \
    --overwrite
"""

from __future__ import annotations

import argparse
from os import path as osp
from pathlib import Path
from typing import List

import mmengine

from forainetv2_data_utils import ForAINetV2Data
from update_infos_to_v2 import update_pkl_infos


def _read_ids(path: Path) -> List[str]:
    if not path.is_file():
        raise FileNotFoundError(f"List not found: {path}")
    ids = [ln.strip() for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    if not ids:
        raise ValueError(f"List is empty: {path}")
    return ids


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root-path", type=Path, required=True, help="Dataset root containing meta_data/ and instance-data dir")
    p.add_argument("--out-dir", type=Path, required=True, help="Where to write the generated PKLs")
    p.add_argument("--train-list", type=Path, required=True, help="Train scene list (one id per line)")
    p.add_argument("--val-list", type=Path, required=True, help="Val scene list (one id per line)")
    p.add_argument("--train-out", type=str, default="forainetv2_oneformer3d_infos_train_L1.pkl")
    p.add_argument("--val-out", type=str, default="forainetv2_oneformer3d_infos_val_L1.pkl")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def _write_split(root_path: str, out_dir: str, split: str, sample_ids: List[str], out_name: str, *, workers: int, overwrite: bool) -> Path:
    # Build dataset helper (it requires meta_data/<split>_list.txt to exist, but we override sample_id_list).
    ds = ForAINetV2Data(root_path=root_path, split=split)
    infos = ds.get_infos(num_workers=workers, has_label=True, sample_id_list=sample_ids)

    out_path = Path(out_dir) / out_name
    if out_path.exists() and not overwrite:
        raise FileExistsError(
            f"Output already exists: {out_path}\n"
            "Refusing to overwrite. Pass --overwrite to replace it, or choose a different --train-out/--val-out."
        )

    mmengine.mkdir_or_exist(out_dir)
    mmengine.dump(infos, osp.join(out_dir, out_name), "pkl")

    # Convert to v2 format (dataset type remains forainetv2)
    update_pkl_infos("forainetv2", out_dir=out_dir, pkl_path=str(out_path))
    return out_path


def main() -> None:
    args = parse_args()

    root_path = str(args.root_path)
    out_dir = str(args.out_dir)

    train_ids = _read_ids(args.train_list)
    val_ids = _read_ids(args.val_list)

    train_pkl = _write_split(
        root_path,
        out_dir,
        split="train",
        sample_ids=train_ids,
        out_name=args.train_out,
        workers=int(args.workers),
        overwrite=bool(args.overwrite),
    )
    val_pkl = _write_split(
        root_path,
        out_dir,
        split="val",
        sample_ids=val_ids,
        out_name=args.val_out,
        workers=int(args.workers),
        overwrite=bool(args.overwrite),
    )

    print("Wrote:")
    print(f"- {train_pkl}  (n={len(train_ids)})")
    print(f"- {val_pkl}    (n={len(val_ids)})")


if __name__ == "__main__":
    # Ensure registries are loaded similarly to other tools.
    from mmdet3d.utils import register_all_modules

    register_all_modules()
    main()
