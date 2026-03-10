#!/usr/bin/env python3
"""Zero-shot inference helper for ForAINetV2-style splits.

This script is meant for the "forest-trained -> rangeland inference" experiment:
    - forest test (existing)
    - medium rangeland (primary)
    - low rangeland

It performs the minimal reproducible pipeline:
  1) Export per-scene instance/semantic arrays from PLYs into
     data/ForAINetV2/forainetv2_instance_data/
  2) Build a *test-only* infos pkl for the requested scene list
  3) Run `tools/test.py` with cfg overrides so the test dataloader uses that pkl

Example (medium rangeland):
  python tools/zero_shot_infer_split.py \
    --split medium \
    --config configs/oneformer3d_qs_radius16_qp300_2many.py \
    --checkpoint work_dirs/clean_forestformer/epoch_3000_fix.pth \
    --work-dir work_dirs/zero_shot_medium

Notes:
  - Assumes the PLYs contain `x,y,z,semantic_seg,treeID` fields.
  - Does not retrain; only runs evaluation/inference.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import Iterable


def _run(cmd: list[str], cwd: Path | None = None) -> None:
    print("\n[cmd] " + " ".join(cmd))
    subprocess.run(cmd, cwd=str(cwd) if cwd else None, check=True)


def _get_protocol() -> str:
    protocol = os.environ.get("FORAINETV2_PROTOCOL", "forest").strip().lower()
    if protocol not in {"forest", "sparse"}:
        raise ValueError(
            f"Invalid FORAINETV2_PROTOCOL='{protocol}'. Use one of: forest, sparse."
        )
    return protocol


def _instance_cache_dirname() -> str:
    protocol = _get_protocol()
    return f"forainetv2_instance_data_{protocol}"


def _rel_or_abs(path: Path, base: Path) -> str:
    """Return path relative to base if possible, else absolute."""
    try:
        return os.fspath(path.relative_to(base))
    except Exception:
        return os.fspath(path)


def _read_scene_list(path: Path) -> list[str]:
    if not path.is_file():
        raise FileNotFoundError(f"Scene list not found: {path}")
    scene_ids: list[str] = []
    for ln in path.read_text(encoding="utf-8").splitlines():
        ln = ln.strip()
        if ln:
            scene_ids.append(ln)
    if not scene_ids:
        raise ValueError(f"Scene list is empty: {path}")
    return scene_ids


def _validate_plys(scene_ids: Iterable[str], ply_dir: Path) -> None:
    missing: list[str] = []
    for sid in scene_ids:
        ply = ply_dir / f"{sid}.ply"
        if not ply.is_file():
            missing.append(str(ply))
    if missing:
        msg = "\n".join(missing[:20])
        more = "" if len(missing) <= 20 else f"\n... and {len(missing) - 20} more"
        raise FileNotFoundError(
            f"{len(missing)} PLY files referenced by the scene list were not found under {ply_dir}:\n{msg}{more}"
        )


def _default_paths(split: str, data_root: Path) -> tuple[Path, Path, str]:
    # Returns (scene_list, ply_dir, extra_tag)
    meta = data_root / "meta_data"
    if split == "forest":
        return meta / "test_list.txt", data_root / "test_data", "forainetv2_forest"
    if split == "medium":
        return meta / "test_list_medium.txt", data_root / "test_data_medium", "forainetv2_medium"
    if split == "low":
        return meta / "test_list_low.txt", data_root / "test_data_low", "forainetv2_low"
    raise ValueError(f"Unknown split: {split}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--data-root",
        type=Path,
        default=Path("data/ForAINetV2"),
        help="ForAINetV2 data root (default: data/ForAINetV2)",
    )
    p.add_argument(
        "--split",
        choices=["forest", "medium", "low"],
        default="medium",
        help="Which predefined split to run (default: medium)",
    )
    p.add_argument(
        "--scene-list",
        type=Path,
        default=None,
        help="Override path to a scene list txt (one scene id per line)",
    )
    p.add_argument(
        "--ply-dir",
        type=Path,
        default=None,
        help="Override PLY directory containing <scene_id>.ply files",
    )
    p.add_argument(
        "--extra-tag",
        type=str,
        default=None,
        help="Override extra-tag used for the generated test-only pkl (default derived from split)",
    )
    p.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Model config to use for inference",
    )
    p.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Forest-trained checkpoint to evaluate",
    )
    p.add_argument(
        "--work-dir",
        type=Path,
        required=True,
        help="Work dir for this run (metrics + prediction PLYs)",
    )
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--overwrite-pkl", action="store_true")
    p.add_argument("--skip-export", action="store_true", help="Skip PLY->npy export step")
    p.add_argument(
        "--dist",
        action="store_true",
        help="Run distributed inference via tools/dist_test.sh (torchrun/launch).",
    )
    p.add_argument(
        "--gpus",
        type=int,
        default=1,
        help="Number of GPUs for distributed test (default: 1).",
    )
    p.add_argument(
        "--cuda-visible-devices",
        type=str,
        default=None,
        help=(
            "Optional CUDA_VISIBLE_DEVICES string, e.g. '0,1,2,3,4,6,7'. "
            "If set, it will be applied to the inference subprocess."
        ),
    )
    p.add_argument(
        "--cfg-options",
        nargs="+",
        default=None,
        help=(
            "Extra config overrides forwarded to tools/test.py in key=value format. "
            "Example: --cfg-options test_evaluator.stuff_class_inds=[0,1] test_evaluator.thing_class_inds=[2]"
        ),
    )
    return p.parse_args()


def _resolve_under(base: Path, path: Path) -> Path:
    """Resolve a potentially-relative path under `base` if it exists there."""
    if path.is_absolute():
        return path
    candidate = base / path
    if candidate.exists():
        return candidate
    return path


def main() -> None:
    args = parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    data_root = (repo_root / args.data_root).resolve() if not args.data_root.is_absolute() else args.data_root

    default_scene_list, default_ply_dir, default_tag = _default_paths(args.split, data_root)
    scene_list = args.scene_list or default_scene_list
    ply_dir = args.ply_dir or default_ply_dir
    # Normalize user-provided data paths to avoid nested relative-path issues.
    scene_list = _resolve_under(data_root, scene_list).resolve() if not scene_list.is_absolute() else scene_list
    ply_dir = _resolve_under(data_root, ply_dir).resolve() if not ply_dir.is_absolute() else ply_dir
    extra_tag = args.extra_tag or default_tag

    config_path = _resolve_under(repo_root, args.config)
    checkpoint_path = _resolve_under(repo_root, args.checkpoint)
    if not config_path.is_file():
        raise FileNotFoundError(f"Config not found: {config_path}")
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    scene_ids = _read_scene_list(scene_list)
    _validate_plys(scene_ids, ply_dir)

    # Use explicit env overrides when provided; otherwise default to protocol-specific cache.
    cache_dir_name = os.environ.get("FORAINETV2_INSTANCE_DATA_DIR", "").strip()
    if not cache_dir_name:
        cache_dir_name = _instance_cache_dirname()
    bin_base_dir = os.environ.get("FORAINETV2_BIN_BASE_DIR", "").strip()
    if not bin_base_dir:
        bin_base_dir = cache_dir_name

    os.environ["FORAINETV2_INSTANCE_DATA_DIR"] = cache_dir_name
    os.environ["FORAINETV2_BIN_BASE_DIR"] = bin_base_dir

    # 1) Export per-scene npys into protocol-specific instance_data cache.
    if not args.skip_export:
        exporter = (data_root / "batch_load_ForAINetV2_data.py")
        if not exporter.is_file():
            raise FileNotFoundError(f"Exporter script not found: {exporter}")

        _run(
            [
                sys.executable,
                str(exporter),
                "--only-test",
                "--test_forainetv2_dir",
                _rel_or_abs(ply_dir, data_root),
                "--test_scan_names_file",
                _rel_or_abs(scene_list, data_root),
                "--output_folder",
                f"./{cache_dir_name}",
            ],
            cwd=data_root,
        )

    # 2) Create a test-only pkl for this list
    create_test_only = repo_root / "tools" / "create_forainetv2_test_only.py"
    if not create_test_only.is_file():
        raise FileNotFoundError(f"create_forainetv2_test_only.py not found: {create_test_only}")

    cmd = [
        sys.executable,
        str(create_test_only),
        "--root-path",
        os.fspath(data_root),
        "--out-dir",
        os.fspath(data_root),
        "--test-list",
        os.fspath(scene_list),
        "--extra-tag",
        extra_tag,
        "--workers",
        str(args.workers),
    ]
    if args.overwrite_pkl:
        cmd.append("--overwrite")
    _run(cmd, cwd=repo_root)

    # 3) Run inference/eval using that pkl
    ann_file = f"{extra_tag}_oneformer3d_infos_test.pkl"
    env = os.environ.copy()
    if args.cuda_visible_devices is not None:
        env["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices

    if args.dist:
        dist_sh = repo_root / "tools" / "dist_test.sh"
        if not dist_sh.is_file():
            raise FileNotFoundError(f"tools/dist_test.sh not found: {dist_sh}")
        if args.gpus < 1:
            raise ValueError("--gpus must be >= 1")

        cfg_opts = [
            f"test_dataloader.dataset.ann_file={ann_file}",
            f"test_dataloader.dataset.data_prefix.pts={bin_base_dir}/points",
            f"test_dataloader.dataset.data_prefix.pts_instance_mask={bin_base_dir}/instance_mask",
            f"test_dataloader.dataset.data_prefix.pts_semantic_mask={bin_base_dir}/semantic_mask",
        ]
        if args.cfg_options:
            cfg_opts.extend(list(args.cfg_options))

        cmd = [
            "bash",
            os.fspath(dist_sh),
            os.fspath(config_path),
            os.fspath(checkpoint_path),
            str(args.gpus),
            "--work-dir",
            os.fspath(args.work_dir),
            "--cfg-options",
            *cfg_opts,
        ]
        print("\n[info] Running distributed test")
        if args.cuda_visible_devices is not None:
            print(f"[info] CUDA_VISIBLE_DEVICES={args.cuda_visible_devices}")
        subprocess.run(cmd, cwd=str(repo_root), env=env, check=True)
    else:
        test_py = repo_root / "tools" / "test.py"
        if not test_py.is_file():
            raise FileNotFoundError(f"tools/test.py not found: {test_py}")
        cfg_opts = [
            f"test_dataloader.dataset.ann_file={ann_file}",
            f"test_dataloader.dataset.data_prefix.pts={bin_base_dir}/points",
            f"test_dataloader.dataset.data_prefix.pts_instance_mask={bin_base_dir}/instance_mask",
            f"test_dataloader.dataset.data_prefix.pts_semantic_mask={bin_base_dir}/semantic_mask",
        ]
        if args.cfg_options:
            cfg_opts.extend(list(args.cfg_options))

        cmd = [
            sys.executable,
            os.fspath(test_py),
            os.fspath(config_path),
            os.fspath(checkpoint_path),
            "--work-dir",
            os.fspath(args.work_dir),
            "--cfg-options",
            *cfg_opts,
        ]
        subprocess.run(cmd, cwd=str(repo_root), env=env, check=True)

    print("\nDone.")
    print(f"- Protocol: {_get_protocol()}")
    print(f"- Instance cache dir: {data_root / cache_dir_name}")
    print(f"- Bin base dir: {data_root / bin_base_dir}")
    print(f"- Split: {args.split}")
    print(f"- Scene list: {scene_list}")
    print(f"- PLY dir: {ply_dir}")
    print(f"- Test PKL: {data_root / ann_file}")
    print(f"- Work dir: {args.work_dir}")


if __name__ == "__main__":
    main()
