#!/usr/bin/env python3
"""Stage a subset of ForAINetV2 PLYs into a single directory (copy or symlink).

This is useful for Experiment 2 (few-shot fine-tuning), where you want a small
train/val pool assembled from multiple source PLY directories (e.g.,
`test_data_medium_L1`, `test_data_low_L1`, ...).

Given one or more `--scene-list` txt files (one scene id per line), this script
finds `<scene_id>.ply` in the provided `--src-dirs` and stages it into `--out-dir`.

Example (inside container):
  python /workspace/tools/make_ply_subset_dir.py \
        --src-dirs /workspace/data/ForAINetV2/test_data_medium_L1 /workspace/data/ForAINetV2/test_data_low_L1 \
        --scene-list /workspace/data/ForAINetV2/meta_data/train_list_exp2_mediumL1_lowL1_fewshot.txt \
        --scene-list /workspace/data/ForAINetV2/meta_data/val_list_exp2_mediumL1_lowL1_fewshot.txt \
    --out-dir /workspace/data/ForAINetV2/train_val_data_exp2A \
    --mode symlink \
    --overwrite
"""

from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path
from typing import Iterable, List


def _read_list(path: Path) -> List[str]:
    if not path.is_file():
        raise FileNotFoundError(f"List not found: {path}")
    out: List[str] = []
    for ln in path.read_text(encoding="utf-8").splitlines():
        ln = ln.strip()
        if ln:
            out.append(ln)
    if not out:
        raise ValueError(f"List is empty: {path}")
    return out


def _iter_scene_ids(list_paths: Iterable[Path]) -> List[str]:
    ids: List[str] = []
    for p in list_paths:
        ids.extend(_read_list(p))
    # keep order, drop duplicates
    seen = set()
    uniq: List[str] = []
    for sid in ids:
        if sid in seen:
            continue
        seen.add(sid)
        uniq.append(sid)
    return uniq


def _find_ply(scene_id: str, src_dirs: List[Path]) -> Path:
    name = f"{scene_id}.ply"
    for d in src_dirs:
        cand = d / name
        if cand.is_file():
            return cand
    raise FileNotFoundError(
        f"Could not find {name} in any --src-dirs. Searched: {', '.join(map(str, src_dirs))}"
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--src-dirs", nargs="+", type=Path, required=True, help="Source directories to search for <scene_id>.ply")
    p.add_argument("--scene-list", action="append", type=Path, required=True, help="Scene list txt (repeatable)")
    p.add_argument("--out-dir", type=Path, required=True, help="Output directory to populate with PLYs")
    p.add_argument("--mode", choices=["symlink", "copy"], default="symlink")
    p.add_argument("--overwrite", action="store_true", help="Overwrite existing files in out-dir")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    src_dirs = [d.resolve() for d in args.src_dirs]
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    scene_ids = _iter_scene_ids(args.scene_list)

    n_done = 0
    for sid in scene_ids:
        src = _find_ply(sid, src_dirs)
        dst = out_dir / src.name

        if dst.exists() or dst.is_symlink():
            if not args.overwrite:
                continue
            if dst.is_dir():
                shutil.rmtree(dst)
            else:
                dst.unlink()

        if args.mode == "copy":
            shutil.copy2(src, dst)
        else:
            # Use a relative symlink when possible (more portable inside /workspace).
            try:
                rel = os.path.relpath(src, start=dst.parent)
                dst.symlink_to(rel)
            except Exception:
                dst.symlink_to(src)

        n_done += 1

    print(f"Staged {n_done}/{len(scene_ids)} scenes into: {out_dir}")


if __name__ == "__main__":
    main()
