"""
Batch processing with parallel execution for ground classification and normalization
(+ optional: write FULL clouds with Classification=2 for ground).
"""

from pathlib import Path
from typing import Tuple
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import laspy
from scipy.spatial import cKDTree

from .core import run_csf_ground_filter, run_smrf_ground_filter, normalize_las_z


def _process_one_csf(
    input_las: Path,
    ground_dir: Path,
    norm_dir: Path,
    normalize: bool,
    resolution: float,
    iterations: int,
    threshold: float,
    rigidness: int,
) -> tuple[str, bool]:
    """Process a single LAS file with CSF ground filter."""
    tile_name = input_las.stem
    ground_out = ground_dir / f"csf_{tile_name}.las"
    norm_out = norm_dir / f"z_norm_csf_{tile_name}.las"

    # If everything needed already exists, skip
    if ground_out.exists() and (not normalize or norm_out.exists()):
        return tile_name, True

    try:
        # 1) CSF ground filter (ground-only)
        if not ground_out.exists():
            run_csf_ground_filter(
                input_las=input_las,
                output_las=ground_out,
                resolution=resolution,
                iterations=iterations,
                threshold=threshold,
                rigidness=rigidness,
                save_pipeline=False,
            )

        # 2) Z normalization (optional)
        if normalize and not norm_out.exists():
            normalize_las_z(
                input_las=input_las,
                ground_las=ground_out,
                output_las=norm_out,
                k=3,
            )

        return tile_name, True

    except Exception as e:
        print(f"❌ Error processing {tile_name}: {e}")
        return tile_name, False


def _process_one_smrf(
    input_las: Path,
    ground_dir: Path,
    norm_dir: Path,
    normalize: bool,
    slope: float,
    window: float,
    threshold: float,
    scalar: float,
) -> tuple[str, bool]:
    """Process a single LAS file with SMRF ground filter."""
    tile_name = input_las.stem
    ground_out = ground_dir / f"smrf_{tile_name}.las"
    norm_out = norm_dir / f"z_norm_smrf_{tile_name}.las"

    if ground_out.exists() and (not normalize or norm_out.exists()):
        return tile_name, True

    try:
        # 1) SMRF ground filter (ground-only)
        if not ground_out.exists():
            run_smrf_ground_filter(
                input_las=input_las,
                output_las=ground_out,
                slope=slope,
                window=window,
                threshold=threshold,
                scalar=scalar,
                save_pipeline=False,
            )

        # 2) Z normalization (optional)
        if normalize and not norm_out.exists():
            normalize_las_z(
                input_las=input_las,
                ground_las=ground_out,
                output_las=norm_out,
                k=3,
            )

        return tile_name, True

    except Exception as e:
        print(f"❌ Error processing {tile_name}: {e}")
        return tile_name, False


def _label_ground_in_full_cloud(
    full_las_path: Path,
    ground_las_path: Path,
    out_las_path: Path,
    *,
    ground_class: int = 2,
    tol_xy: float = 0.02,
    tol_z: float = 0.05,
) -> Tuple[int, int]:
    """
    Write a new LAS where points in `full_las` that match `ground_las`
    are assigned Classification=ground_class.

    Returns (n_ground_labeled, n_full_points).
    """
    full = laspy.read(full_las_path)
    ground = laspy.read(ground_las_path)

    full_xy = np.c_[full.x, full.y]
    g_xy = np.c_[ground.x, ground.y]

    tree = cKDTree(full_xy)
    d, idx = tree.query(g_xy, k=1, workers=-1)

    ok_xy = d <= tol_xy
    idx_ok = idx[ok_xy]

    g_z_ok = ground.z[ok_xy]
    full_z_at = full.z[idx_ok]
    ok_z = np.abs(full_z_at - g_z_ok) <= tol_z

    matched_full_idx = idx_ok[ok_z]

    cls = np.array(full.classification, copy=True)
    cls[matched_full_idx] = ground_class
    full.classification = cls

    out_las_path.parent.mkdir(parents=True, exist_ok=True)
    full.write(out_las_path)

    return int(len(matched_full_idx)), int(len(full.points))


def process_batch_csf(
    input_dir: Path,
    output_base: Path,
    *,
    pattern: str = "*.las",
    normalize: bool = True,
    n_workers: int = 4,
    resolution: float = 0.5,
    iterations: int = 500,
    threshold: float = 0.01,
    rigidness: int = 2,
) -> None:
    """
    Process multiple LAS files with CSF ground filter in parallel.

    Outputs:
      - ground_only: output_base/ground_classification/csf/csf_<tile>.las
      - z_norm:      output_base/z_norm/csf/z_norm_csf_<tile>.las (optional)
    """
    input_dir = Path(input_dir)
    output_base = Path(output_base)

    ground_dir = output_base / "ground_classification" / "csf"
    norm_dir = output_base / "z_norm" / "csf"

    ground_dir.mkdir(parents=True, exist_ok=True)
    if normalize:
        norm_dir.mkdir(parents=True, exist_ok=True)

    las_files = sorted(input_dir.glob(pattern))
    if not las_files:
        print(f"❌ No LAS files found in {input_dir} with pattern {pattern}")
        return

    print(f"Found {len(las_files)} LAS files to process with CSF")
    print(f"Using {n_workers} workers")

    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = {
            executor.submit(
                _process_one_csf,
                las,
                ground_dir,
                norm_dir,
                normalize,
                resolution,
                iterations,
                threshold,
                rigidness,
            ): las
            for las in las_files
        }
        for future in tqdm(as_completed(futures), total=len(las_files), desc="CSF processing"):
            tile_name, success = future.result()
            if success:
                print(f"✅ {tile_name}")

    print("\n🎉 CSF processing complete!")


def process_batch_smrf(
    input_dir: Path,
    output_base: Path,
    *,
    pattern: str = "*.las",
    normalize: bool = True,
    n_workers: int = 4,
    slope: float = 0.15,
    window: float = 3.5,
    threshold: float = 0.25,
    scalar: float = 1.35,
) -> None:
    """
    Process multiple LAS files with SMRF ground filter in parallel.

    Outputs:
      - ground_only: output_base/ground_classification/smrf/smrf_<tile>.las
      - z_norm:      output_base/z_norm/smrf/z_norm_smrf_<tile>.las (optional)
    """
    input_dir = Path(input_dir)
    output_base = Path(output_base)

    ground_dir = output_base / "ground_classification" / "smrf"
    norm_dir = output_base / "z_norm" / "smrf"

    ground_dir.mkdir(parents=True, exist_ok=True)
    if normalize:
        norm_dir.mkdir(parents=True, exist_ok=True)

    las_files = sorted(input_dir.glob(pattern))
    if not las_files:
        print(f"❌ No LAS files found in {input_dir} with pattern {pattern}")
        return

    print(f"Found {len(las_files)} LAS files to process with SMRF")
    print(f"Using {n_workers} workers")

    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = {
            executor.submit(
                _process_one_smrf,
                las,
                ground_dir,
                norm_dir,
                normalize,
                slope,
                window,
                threshold,
                scalar,
            ): las
            for las in las_files
        }
        for future in tqdm(as_completed(futures), total=len(las_files), desc="SMRF processing"):
            tile_name, success = future.result()
            if success:
                print(f"✅ {tile_name}")

    print("\n🎉 SMRF processing complete!")
