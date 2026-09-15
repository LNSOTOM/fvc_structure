#!/usr/bin/env python3
"""Run the focused sensitivity analysis requested by the reviewers.

The script changes one parameter at a time, reruns the existing 3D fusion
pipeline, and writes:

* parameter_sensitivity_per_tile.csv
* parameter_sensitivity_summary.csv
* pipeline_stage_effects.csv

The 2D-derived structure cloud is the available reference. Agreement therefore
measures preservation of the input labels, not accuracy against independent
3D ground truth.

Example
-------
python sensitivity.py --workers 8
"""

from __future__ import annotations

import argparse
import contextlib
import io
import math
import re
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import laspy
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree


STEP_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(STEP_DIR))

from fvc3dstruct.config import DEFAULT_CONFIG  # noqa: E402
from fvc3dstruct.pipeline import process_tile  # noqa: E402


DATA_ROOT = Path("/mnt/d/paper2/data/calperumResearch")
OUTPUT_DIR = Path(__file__).resolve().parent / "results"

SITES = {
    "site1_1_DD0001": "Medium",
    "site2_1_DD0011": "Low",
}
Z_NORM_DIR = "lidar/clipped/raster/z_norm/csf"
STRUCTURE_DIR = {
    "site1_1_DD0001": "lidar/clipped/mask/coreg/icp_top",
    "site2_1_DD0011": "lidar/clipped/mask/coreg/icp",
}

# The medium site used automatically derived crown parameters in the delivered
# workflow. The low site used the package defaults.
SITE_CONFIG: Dict[str, Dict[str, object]] = {
    "site1_1_DD0001": {"min_tree_dist": "auto", "smoothing_sigma": "auto"},
    "site2_1_DD0011": {},
}

CLASS_CODES = (64, 65, 66, 67, 68)
MATCH_TOLERANCE_M = 0.005

# One-at-a-time tests. Each range contains the delivered value.
SWEEPS: Dict[str, Tuple[float, ...]] = {
    "voxel_size": (0.05, 0.10, 0.15, 0.20, 0.25),
    "height_match_radius": (0.05, 0.10, 0.20, 0.30, 0.50),
    "height_prior_out_of_range_weight": (1.0, 0.5, 0.1, 0.0),
    "min_conf": (0.2, 0.3, 0.4, 0.5),
    "min_flip_ratio": (1.0, 1.5, 2.0),
}
DEFAULTS = {
    "voxel_size": float(DEFAULT_CONFIG["voxel_size_xy"]),
    "height_match_radius": float(DEFAULT_CONFIG["height_match_radius"]),
    "height_prior_out_of_range_weight": float(
        DEFAULT_CONFIG["height_prior_out_of_range_weight"]
    ),
    "min_conf": float(DEFAULT_CONFIG["min_conf"]),
    "min_flip_ratio": float(DEFAULT_CONFIG["min_flip_ratio"]),
}


def tile_id(path: Path) -> str | None:
    """Return the numeric tile identifier at the end of a LAS stem."""
    match = re.search(r"(\d+)$", path.stem)
    return match.group(1) if match else None


def find_tiles(data_root: Path, sites: Iterable[str], wanted: set[str]) -> List[Tuple]:
    """Find Z_norm/structure-cloud pairs for the selected sites."""
    pairs: List[Tuple] = []
    for site in sites:
        base = data_root / site
        z_files = {tile_id(p): p for p in (base / Z_NORM_DIR).glob("*.las")}
        s_files = {tile_id(p): p for p in (base / STRUCTURE_DIR[site]).glob("*.las")}
        for tile in sorted((set(z_files) & set(s_files)) - {None}, key=int):
            if not wanted or tile in wanted:
                pairs.append((site, tile, z_files[tile], s_files[tile]))
    return pairs


def parameter_config(parameter: str | None = None, value: float | None = None) -> Dict:
    """Return an analysis-only pipeline configuration for one test value."""
    config = dict(DEFAULT_CONFIG)
    config.update(write_outputs=False, return_classification_arrays=True)
    if parameter == "voxel_size":
        config["voxel_size_xy"] = value
        config["voxel_size_z"] = value
    elif parameter is not None:
        config[parameter] = value
    return config


def run_pipeline(
    site: str,
    z_path: Path,
    s_path: Path,
    parameter: str | None = None,
    value: float | None = None,
) -> Dict:
    """Run one silent, analysis-only pipeline configuration."""
    config = parameter_config(parameter, value)
    config.update(SITE_CONFIG[site])
    with contextlib.redirect_stdout(io.StringIO()):
        return process_tile(z_path, s_path, Path("/dev/null"), config=config)


def reference_lookup(las_z: laspy.LasData, las_s: laspy.LasData):
    """Match labelled structure points to their points in the Z_norm cloud."""
    xyz_z = np.column_stack((las_z.x, las_z.y, las_z.z))
    xyz_s = np.column_stack((las_s.x, las_s.y, las_s.z))
    distance, index = cKDTree(xyz_z).query(xyz_s, k=1)
    reference = np.asarray(las_s.classification)
    keep = (distance <= MATCH_TOLERANCE_M) & np.isin(reference, CLASS_CODES)
    if not np.any(keep):
        raise ValueError("No structure points matched the Z_norm cloud within 5 mm.")
    return index[keep], reference[keep]


def agreement(
    predicted: np.ndarray,
    reference_index: np.ndarray,
    reference: np.ndarray,
) -> Tuple[float, float]:
    """Point agreement and macro-IoU (%) at the original labelled points."""
    predicted = predicted[reference_index]
    accuracy = 100.0 * np.mean(predicted == reference)
    ious = []
    for code in CLASS_CODES:
        truth = reference == code
        if not np.any(truth):
            continue
        estimate = predicted == code
        union = np.count_nonzero(truth | estimate)
        ious.append(np.count_nonzero(truth & estimate) / union if union else math.nan)
    return float(accuracy), 100.0 * float(np.nanmean(ious))


def changed(a: np.ndarray, b: np.ndarray) -> float:
    """Percentage of points whose class differs between two stages or runs."""
    return 100.0 * float(np.mean(a != b))


def analyse_tile(site: str, tile: str, z_path: Path, s_path: Path):
    """Run all tests for one tile and return parameter and stage rows."""
    las_z = laspy.read(str(z_path))
    las_s = laspy.read(str(s_path))
    reference_index, reference = reference_lookup(las_z, las_s)

    xy_z = np.column_stack((las_z.x, las_z.y))
    xy_s = np.column_stack((las_s.x, las_s.y))
    xy_distance, _ = cKDTree(xy_z).query(xy_s, k=1)
    valid_structure = np.isin(np.asarray(las_s.classification), CLASS_CODES)
    xy_distance = xy_distance[valid_structure]
    del las_z, las_s, xy_z, xy_s

    default = run_pipeline(site, z_path, s_path)
    default_stage2 = np.asarray(default["classification_baseline"])
    default_stage4 = np.asarray(default["classification_refined"])
    default_final = np.asarray(default["classification_final"])

    stage2_accuracy, _ = agreement(default_stage2, reference_index, reference)
    n_points = len(default_final)
    default_radius = DEFAULTS["height_match_radius"]

    stage_rows = [
        {
            "Site": site,
            "Site condition": SITES[site],
            "Tile": tile,
            "Stage": "1 Height matching",
            "Measure": "Structure points beyond the 0.30 m radius",
            "Measure type": "error proxy",
            "Affected points (%)": 100.0 * np.mean(xy_distance > default_radius),
        },
        {
            "Site": site,
            "Site condition": SITES[site],
            "Tile": tile,
            "Stage": "2 Voxel transfer",
            "Measure": "Mismatch at the original labelled points",
            "Measure type": "error proxy",
            "Affected points (%)": 100.0 - stage2_accuracy,
        },
        {
            "Site": site,
            "Site condition": SITES[site],
            "Tile": tile,
            "Stage": "3 Crown detection",
            "Measure": "Points inside detected crowns",
            "Measure type": "exposure, not error",
            "Affected points (%)": 100.0 * default["stage3_points_in_crowns"] / n_points,
        },
        {
            "Site": site,
            "Site condition": SITES[site],
            "Tile": tile,
            "Stage": "4 Polar refinement",
            "Measure": "Labels changed relative to Stage 2",
            "Measure type": "change",
            "Affected points (%)": changed(default_stage4, default_stage2),
        },
        {
            "Site": site,
            "Site condition": SITES[site],
            "Tile": tile,
            "Stage": "5 Consolidation",
            "Measure": "Labels changed relative to Stage 4",
            "Measure type": "change",
            "Affected points (%)": changed(default_final, default_stage4),
        },
    ]

    parameter_rows = []
    for parameter, values in SWEEPS.items():
        for value in values:
            if np.isclose(value, DEFAULTS[parameter]):
                result = default
            else:
                result = run_pipeline(site, z_path, s_path, parameter, value)

            stage2 = np.asarray(result["classification_baseline"])
            stage4 = np.asarray(result["classification_refined"])
            final = np.asarray(result["classification_final"])
            accuracy, miou = agreement(final, reference_index, reference)

            parameter_rows.append(
                {
                    "Site": site,
                    "Site condition": SITES[site],
                    "Tile": tile,
                    "Parameter": parameter,
                    "Value": value,
                    "Default": bool(np.isclose(value, DEFAULTS[parameter])),
                    "Reference points": len(reference),
                    "Reference agreement (%)": accuracy,
                    "Reference mIoU (%)": miou,
                    "Stage 2 labelled points (%)": result["stage2_labeled_pct"],
                    "Stage 4 changed points (%)": changed(stage4, stage2),
                    "Stage 5 changed points (%)": changed(final, stage4),
                    "Final labelled points (%)": result["final_veg_pct"],
                    "Stage 2 changed from default (%)": changed(stage2, default_stage2),
                    "Final changed from default (%)": changed(final, default_final),
                    "Detected crowns": result["stage3_watershed_crowns"],
                    "Match coverage (%)": (
                        100.0 * np.mean(xy_distance <= value)
                        if parameter == "height_match_radius"
                        else math.nan
                    ),
                }
            )

    return parameter_rows, stage_rows


def summarise_parameters(per_tile: pd.DataFrame) -> pd.DataFrame:
    """Mean and standard deviation across tiles for each site and test value."""
    ids = ["Site", "Site condition", "Parameter", "Value", "Default"]
    metrics = [
        "Reference agreement (%)",
        "Reference mIoU (%)",
        "Stage 2 labelled points (%)",
        "Stage 4 changed points (%)",
        "Stage 5 changed points (%)",
        "Final labelled points (%)",
        "Stage 2 changed from default (%)",
        "Final changed from default (%)",
        "Detected crowns",
        "Match coverage (%)",
    ]
    summary = per_tile.groupby(ids)[metrics].agg(["mean", "std"]).reset_index()
    summary.columns = [
        " ".join(str(part) for part in column if part).strip()
        for column in summary.columns.to_flat_index()
    ]
    return summary


def save_results(parameter_rows: List[Dict], stage_rows: List[Dict], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    per_tile = pd.DataFrame(parameter_rows).sort_values(
        ["Site condition", "Parameter", "Value", "Tile"]
    )
    per_tile.to_csv(output_dir / "parameter_sensitivity_per_tile.csv", index=False)
    summarise_parameters(per_tile).to_csv(
        output_dir / "parameter_sensitivity_summary.csv", index=False
    )

    stages = pd.DataFrame(stage_rows).sort_values(["Site condition", "Stage", "Tile"])
    stage_summary = (
        stages.groupby(
            ["Site", "Site condition", "Stage", "Measure", "Measure type"]
        )["Affected points (%)"]
        .agg(["mean", "std", "max"])
        .reset_index()
    )
    stage_summary.to_csv(output_dir / "pipeline_stage_effects.csv", index=False)
    print(f"Wrote results to {output_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument(
        "--sites", default=",".join(SITES), help="Comma-separated site folder names"
    )
    parser.add_argument("--tiles", default="", help="Optional comma-separated tile IDs")
    parser.add_argument("--workers", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sites = [site.strip() for site in args.sites.split(",") if site.strip()]
    unknown = set(sites) - set(SITES)
    if unknown:
        raise SystemExit(f"Unknown sites: {sorted(unknown)}")
    wanted = {tile.strip() for tile in args.tiles.split(",") if tile.strip()}
    jobs = find_tiles(args.data_root, sites, wanted)
    if not jobs:
        raise SystemExit("No matching Z_norm and structure LAS tiles were found.")

    parameter_rows: List[Dict] = []
    stage_rows: List[Dict] = []
    if args.workers <= 1:
        for number, job in enumerate(jobs, 1):
            rows, stages = analyse_tile(*job)
            parameter_rows.extend(rows)
            stage_rows.extend(stages)
            print(f"[{number}/{len(jobs)}] {job[0]} tile {job[1]}")
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(analyse_tile, *job): job for job in jobs}
            for number, future in enumerate(as_completed(futures), 1):
                rows, stages = future.result()
                parameter_rows.extend(rows)
                stage_rows.extend(stages)
                site, tile = futures[future][:2]
                print(f"[{number}/{len(jobs)}] {site} tile {tile}")

    save_results(parameter_rows, stage_rows, args.output_dir)


if __name__ == "__main__":
    main()
