#!/usr/bin/env python3
"""One-at-a-time sensitivity of the fusion pipeline to its tuned parameters.

Answers the reviewer's requests to justify the voxel size (10 cm), the Stage-1
height matching radius (0.30 m), the Stage-4 height-prior weights (1 in range,
0.1 out of range), the confidence threshold (`min_conf` 0.3) and the flip ratio
(`min_flip_ratio` 1.5). Each parameter is varied across a range while the others
stay at their defaults, on every matched tile of both sites.

What the pipeline is given decides how to score it. The structure cloud is an
exact subset of the Z_norm cloud: the lidar points that fall on labelled cells of
the delivered 5 cm label raster, each carrying that cell's class. The `input` rows
verify this per tile (`Structure points with an exact twin (%)`, `Points on
labelled 5 cm cells`). Every run is then scored per lidar point, against two
references.

*Input.* The 5 cm raster the structure cloud was cut from. Against it the labels
the pipeline was handed are perfect by construction, so every departure is the
pipeline's own doing. `Outside footprint (%)` is the share of labelled points on
cells the reference leaves unlabelled, i.e. how far labels have been spread past
their delineation; `Footprint recall (%)` the share of points on labelled cells
that carry the cell's class; `Point precision (%)` the share of labelled points
whose class matches the cell beneath them.

*1 cm.* The annotator's 1 cm raster, the same three measures. The `input` rows
score the structure labels themselves against it: what the 1 cm -> 5 cm
downsample had already cost before the pipeline started (phase 3 step 3).

Both references are also compared in plan view. The 3D labels are rasterised
top-down onto the reference's 5 cm grid, each cell taking the majority class of
its labelled points, and compared cell by cell: with the 5 cm input raster as
delivered, and with the 1 cm raster put on a 5 cm grid by majority of its
sub-pixels (the rule `phase_3.../label_qa.py` uses). Class IoU penalises both
failure modes a parameter can cause: labels spread onto cells the annotator left
unlabelled or gave another class, and labelled cells left empty.

Neither reference is independent of the labels being scored, so these measure how
faithfully the 3D labels preserve the 2D delineation, not their accuracy against
3D truth, which does not exist for these sites. `changed vs default (%)` needs no
reference at all: the share of points whose label differs from the default run's.

Writes into --output-dir:

    parameter_sensitivity_per_tile.csv   one row per (site, tile, parameter, value)
    parameter_sensitivity_summary.csv    mean / median / std across tiles, per site

Usage
-----
python parameter_sensitivity.py --output-dir assets/parameter_sensitivity --workers 12
"""

from __future__ import annotations

import argparse
import contextlib
import io
import math
import re
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import laspy
import numpy as np
import pandas as pd
import rasterio
from scipy.spatial import cKDTree

# This copy lives in `fvc3dstruct/voxel_test/`, two levels below the step directory, so it
# puts its own imports on the path rather than relying on the working directory. The
# superseded original sat directly in the step directory and needed neither line.
STEP_DIR = Path(__file__).resolve().parents[2]
REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(STEP_DIR))  # the fvc3dstruct package, and compare_voxel_sizes
sys.path.insert(0, str(REPO_ROOT / "phase_3_2D_imagery_prep" / "step_3_label_quality"))

from fvc3dstruct.config import DEFAULT_CONFIG  # noqa: E402
from fvc3dstruct.height_stats import (  # noqa: E402
    auto_mallee_min_height, class_height_ranges, compute_class_height_thresholds,
    match_structure_heights)
from fvc3dstruct.pipeline import process_tile  # noqa: E402
from fvc3dstruct.watershed import auto_watershed_params  # noqa: E402
from label_qa import majority_on_grid, remap_class_codes  # noqa: E402

DATA_ROOT = Path("/mnt/d/paper2/data/calperumResearch")
SITES = {
    "site1_1_DD0001": "Medium",
    "site2_1_DD0011": "Low",
}
# Structure-label point clouds per site. DD0011's `coreg/icp_top` is a byte-for-byte
# copy of DD0001's tiles (IDs 16-205), so the low site must be read from `coreg/icp`.
STRUCTURE_DIR = {
    "site1_1_DD0001": "lidar/clipped/mask/coreg/icp_top",
    "site2_1_DD0011": "lidar/clipped/mask/coreg/icp",
}
# Crown detection as the delivered labels were produced (cell 30 of the step's
# `notebooks/01_run_pipeline_site*_workflow.ipynb`): the medium site with auto-derived
# watershed parameters, the low site with the defaults. On these tiles `auto`
# resolves to min_tree_dist 0.8 m / sigma 1.33 px wherever there is canopy: it
# estimates spacing from neighbouring lidar returns (~3 cm), which the 0.8 m floor clamps.
SITE_CONFIG: Dict[str, Dict[str, object]] = {
    "site1_1_DD0001": {"min_tree_dist": "auto", "smoothing_sigma": "auto"},
    "site2_1_DD0011": {},
}
Z_NORM_DIR = "lidar/clipped/raster/z_norm/csf"
REFERENCE_DIR = "imagery/inputs/masks/tiles_3072/raw/structure_class/annotation_raster"
# The 5 cm labels the structure clouds were cut from (`notebooks/maskCoreg*.py`).
INPUT_DIR = f"{REFERENCE_DIR}/downsample_5cm_nearest"

# LAS classification codes 64-68 are struct_id 0-4 in the 2D rasters.
LAS_CODES = (64, 65, 66, 67, 68)
CLASS_NAMES = {64: "Forb", 65: "Woody debris", 66: "Hummock grass", 67: "Shrub", 68: "Mallee"}
NODATA = 255

# Evaluation grid: the 5 cm resolution the labels are consumed at downstream.
EVAL_CELL_M = 0.05

# Each parameter's tested values. The default is included in every range, so
# the default configuration is run once per tile and shared by all sweeps.
SWEEPS: Dict[str, Tuple[float, ...]] = {
    "voxel_size": (0.05, 0.10, 0.15, 0.20, 0.25),
    "height_match_radius": (0.01, 0.05, 0.10, 0.30, 1.00),
    "height_prior_out_of_range_weight": (1.0, 0.5, 0.1, 0.01, 0.0),
    "min_conf": (0.2, 0.3, 0.4, 0.5, 0.6, 0.8),
    "min_flip_ratio": (1.0, 1.25, 1.5, 2.0, 3.0),
}
DEFAULTS = {
    "voxel_size": DEFAULT_CONFIG["voxel_size_xy"],
    "height_match_radius": DEFAULT_CONFIG["height_match_radius"],
    "height_prior_out_of_range_weight": DEFAULT_CONFIG["height_prior_out_of_range_weight"],
    "min_conf": DEFAULT_CONFIG["min_conf"],
    "min_flip_ratio": DEFAULT_CONFIG["min_flip_ratio"],
}

ID_COLUMNS = ("Site", "Site condition", "Tile", "Parameter", "Value")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--output-dir", type=Path,
                        default=STEP_DIR / "assets" / "parameter_sensitivity")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--sites", default=",".join(SITES),
                        help="Comma-separated site folders to include.")
    parser.add_argument("--tiles", default="", help="Comma-separated tile IDs to restrict to.")
    parser.add_argument("--parameters", default=",".join(SWEEPS),
                        help="Comma-separated sweeps to run. Default: all.")
    return parser.parse_args()


def tile_id(path: Path) -> str | None:
    match = re.search(r"(\d+)$", path.stem)
    return match.group(1) if match else None


def site_tiles(data_root: Path, site: str) -> List[Tuple[str, Path, Path, Path, Path]]:
    """(tile, Z_norm LAS, structure LAS, 5 cm input raster, 1 cm reference raster)
    for every tile with all four."""
    base = data_root / site
    maps = [
        {tile_id(p): p for p in (base / Z_NORM_DIR).glob("*.las")},
        {tile_id(p): p for p in (base / STRUCTURE_DIR[site]).glob("*.las")},
        {tile_id(p): p for p in (base / INPUT_DIR).glob("*.tif")},
        {tile_id(p): p for p in (base / REFERENCE_DIR).glob("*.tif")},
    ]
    common = set.intersection(*(set(m) for m in maps)) - {None}
    return [(t, *(m[t] for m in maps)) for t in sorted(common, key=int)]


def configurations(parameters: Sequence[str]) -> List[Tuple[str, float, Dict[str, float]]]:
    """(parameter, value, config overrides), the shared default listed first as 'default'."""
    base = {p: DEFAULTS[p] for p in DEFAULTS if p != "voxel_size"}
    base["voxel_size_xy"] = base["voxel_size_z"] = DEFAULTS["voxel_size"]
    runs = [("default", math.nan, dict(base))]
    for parameter in parameters:
        for value in SWEEPS[parameter]:
            if math.isclose(value, DEFAULTS[parameter]):
                continue
            config = dict(base)
            if parameter == "voxel_size":
                config["voxel_size_xy"] = config["voxel_size_z"] = value
            else:
                config[parameter] = value
            runs.append((parameter, value, config))
    return runs


def read_labels(raster_path: Path) -> Tuple[np.ndarray, rasterio.Affine]:
    """A label raster as struct_id 0-4 / NODATA, whichever coding it was written in."""
    with rasterio.open(raster_path) as src:
        return remap_class_codes(src.read(1)), src.transform


def as_codes(struct_ids: np.ndarray) -> np.ndarray:
    """struct_id 0-4 / NODATA as LAS codes 64-68 / NODATA."""
    codes = np.full(struct_ids.shape, NODATA, dtype=np.uint8)
    for struct_id, code in enumerate(LAS_CODES):
        codes[struct_ids == struct_id] = code
    return codes


def majority_grid(fine: np.ndarray, transform: rasterio.Affine) -> Tuple[np.ndarray, rasterio.Affine]:
    """The 1 cm reference put on the 5 cm evaluation grid, coded 64-68 / NODATA."""
    factor = int(round(EVAL_CELL_M / abs(transform.a)))
    shape = (fine.shape[0] // factor, fine.shape[1] // factor)
    grid_transform = transform * rasterio.Affine.scale(factor)
    return as_codes(majority_on_grid(fine, transform, grid_transform, shape)), grid_transform


def classes_beneath(labels: np.ndarray, transform: rasterio.Affine,
                    x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """The class of the raster cell beneath each point, coded 64-68, 0 where unlabelled."""
    cols = np.floor((x - transform.c) / transform.a).astype(np.int64)
    rows = np.floor((y - transform.f) / transform.e).astype(np.int64)
    inside = (rows >= 0) & (rows < labels.shape[0]) & (cols >= 0) & (cols < labels.shape[1])
    values = np.full(len(x), NODATA, dtype=np.uint8)
    values[inside] = labels[rows[inside], cols[inside]]
    codes = np.zeros(len(x), dtype=np.uint8)
    labelled = values < len(LAS_CODES)
    codes[labelled] = values[labelled] + LAS_CODES[0]
    return codes


def labels_on_grid(x: np.ndarray, y: np.ndarray, classes: np.ndarray,
                   transform: rasterio.Affine, shape: Tuple[int, int]) -> np.ndarray:
    """Majority class of the labelled points in each 5 cm cell; empty cells NODATA."""
    labelled = np.isin(classes, LAS_CODES)
    cols = np.floor((x[labelled] - transform.c) / transform.a).astype(np.int64)
    rows = np.floor((y[labelled] - transform.f) / transform.e).astype(np.int64)
    inside = (rows >= 0) & (rows < shape[0]) & (cols >= 0) & (cols < shape[1])
    cell = rows[inside] * shape[1] + cols[inside]
    index = classes[labelled][inside].astype(np.int64) - LAS_CODES[0]
    counts = np.bincount(cell * len(LAS_CODES) + index,
                         minlength=shape[0] * shape[1] * len(LAS_CODES)
                         ).reshape(-1, len(LAS_CODES))
    grid = np.full(shape[0] * shape[1], NODATA, dtype=np.uint8)
    occupied = counts.sum(axis=1) > 0
    grid[occupied] = np.asarray(LAS_CODES, dtype=np.uint8)[np.argmax(counts[occupied], axis=1)]
    return grid.reshape(shape)


def agreement(predicted: np.ndarray, reference: np.ndarray, prefix: str) -> Dict[str, float]:
    """Class IoU of the predicted grid against the reference, over classes the reference holds."""
    row: Dict[str, float] = {}
    scores = []
    for code in LAS_CODES:
        truth, pred = reference == code, predicted == code
        union = int(np.count_nonzero(truth | pred))
        score = np.count_nonzero(truth & pred) / union if union else math.nan
        row[f"{prefix}IoU {CLASS_NAMES[code]}"] = score
        if truth.any():
            scores.append(score)
    row[f"{prefix}mIoU"] = float(np.mean(scores)) if scores else math.nan
    labelled_ref = reference != NODATA
    labelled_pred = predicted != NODATA
    # Cells labelled in the 3D product where the annotator labelled nothing:
    # labels spread past the delineation.
    row[f"{prefix}Commission (%)"] = (
        100.0 * np.count_nonzero(labelled_pred & ~labelled_ref) / np.count_nonzero(labelled_pred)
        if labelled_pred.any() else math.nan)
    # Reference cells that received no label at all.
    row[f"{prefix}Omission (%)"] = (
        100.0 * np.count_nonzero(labelled_ref & ~labelled_pred) / np.count_nonzero(labelled_ref)
        if labelled_ref.any() else math.nan)
    return row


def point_agreement(predicted: np.ndarray, reference: np.ndarray, prefix: str) -> Dict[str, float]:
    labelled = np.isin(predicted, LAS_CODES)
    in_footprint = reference > 0
    n_labelled = int(np.count_nonzero(labelled))
    n_footprint = int(np.count_nonzero(in_footprint))
    return {
        f"{prefix}Outside footprint (%)":
            100.0 * np.count_nonzero(labelled & ~in_footprint) / n_labelled if n_labelled else math.nan,
        f"{prefix}Point precision (%)":
            100.0 * np.count_nonzero(labelled & (predicted == reference)) / n_labelled
            if n_labelled else math.nan,
        f"{prefix}Footprint recall (%)":
            100.0 * np.count_nonzero(in_footprint & (predicted == reference)) / n_footprint
            if n_footprint else math.nan,
    }


def score(classes: np.ndarray, x: np.ndarray, y: np.ndarray,
          references: Dict[str, Tuple[np.ndarray, np.ndarray, rasterio.Affine]],
          stage: str) -> Dict[str, float]:
    """Per-point and plan-view agreement of one label array with every reference."""
    row: Dict[str, float] = {}
    for name, (points, grid, transform) in references.items():
        prefix = f"{stage} vs {name} "
        row.update(point_agreement(classes, points, prefix))
        row.update(agreement(labels_on_grid(x, y, classes, transform, grid.shape), grid, prefix))
    return row


def exact_twins(las_s: laspy.LasData, las_z: laspy.LasData) -> Tuple[np.ndarray, np.ndarray]:
    """Index of each structure point's coordinate twin in the Z_norm cloud (to 1 mm),
    and whether it has one."""
    coords = [(np.asarray(getattr(las_s, axis)), np.asarray(getattr(las_z, axis)))
              for axis in ("x", "y", "z")]
    keys = []
    for cloud in (0, 1):
        key = np.zeros(len(coords[0][cloud]), dtype=np.int64)
        for values in coords:
            origin = min(values[0].min(), values[1].min())
            mm = np.round((values[cloud] - origin) * 1000).astype(np.int64)
            if mm.max() >= 1 << 20:
                raise ValueError("Tile spans more than 1 km on one axis; widen the key.")
            key = (key << 20) | mm
        keys.append(key)
    order = np.argsort(keys[1], kind="stable")
    position = np.clip(np.searchsorted(keys[1][order], keys[0]), 0, len(order) - 1)
    twin = order[position]
    return twin, keys[1][twin] == keys[0]


def stage1_matching(las_s: laspy.LasData, las_z: laspy.LasData) -> Dict[str, float]:
    """How the Stage-1 height lookup behaves on this tile.

    `Match distance XY max (m)` is the quantity the 0.30 m radius caps. The
    lookup queries XY only, so where several returns share a 1 cm XY it can take
    the height of another return in the same column: the class ranges are
    computed both from the heights it takes and from each point's own.
    """
    radius = DEFAULTS["height_match_radius"]
    distance, _ = cKDTree(np.column_stack((las_z.x, las_z.y))).query(
        np.column_stack((las_s.x, las_s.y)), k=1)
    twin, found = exact_twins(las_s, las_z)
    classes = np.asarray(las_s.classification)
    zn = np.asarray(las_z["Z_norm"], dtype=float)
    own = np.where(found, zn[twin], np.nan)
    with contextlib.redirect_stdout(io.StringIO()):
        taken = match_structure_heights(las_s, las_z, radius)
        ranges_taken = class_height_ranges(taken, classes)
        ranges_own = class_height_ranges(own, classes)
    n = len(classes)
    return {
        "Structure points with an exact twin (%)": 100.0 * np.count_nonzero(found) / n,
        "Match distance XY max (m)": float(distance.max()),
        "Structure points beyond match radius (%)": 100.0 * np.count_nonzero(distance > radius) / n,
        "Stage-1 heights from another return (%)": 100.0 * np.count_nonzero(np.abs(taken - own) > 0.01) / n,
        "Stage-1 range shift, max (m)": max(
            abs(a - b) for code in LAS_CODES for a, b in zip(ranges_taken[code], ranges_own[code])),
    }


def auto_crown_parameters(las_s: laspy.LasData, las_z: laspy.LasData) -> Dict[str, float]:
    """What `auto` crown detection resolves to on this tile, and the spacing it starts from.

    `watershed.auto_watershed_params` takes the 60th percentile of nearest-neighbour
    distances between canopy points (mallee-labelled, or failing that returns above the
    minimum crown height) and clamps it to 0.8-3.0 m. The unclamped value is recomputed
    the same way, on a seeded sample of at most 50,000 points, to show whether the clamp
    decides the result.
    """
    with contextlib.redirect_stdout(io.StringIO()):
        mallee_min = auto_mallee_min_height(compute_class_height_thresholds(las_s, las_z))
    min_tree_dist, sigma = auto_watershed_params(las_s, las_z, grid_res=0.10, fallback_min_height=mallee_min)
    mallee = np.asarray(las_s.classification) == 68
    if mallee.any():
        canopy = np.column_stack((np.asarray(las_s.x)[mallee], np.asarray(las_s.y)[mallee]))
    else:
        tall = np.asarray(las_z["Z_norm"], dtype=float) >= mallee_min
        canopy = np.column_stack((np.asarray(las_z.x)[tall], np.asarray(las_z.y)[tall]))
    spacing = math.nan
    if len(canopy) >= 100:  # auto's own minimum; below it auto returns its defaults
        if len(canopy) > 50_000:
            canopy = canopy[np.random.default_rng(0).choice(len(canopy), 50_000, replace=False)]
        distance, _ = cKDTree(canopy).query(canopy, k=2)
        spacing = float(np.percentile(distance[:, 1], 60))
    return {
        "Auto min_tree_dist (m)": float(min_tree_dist),
        "Auto smoothing_sigma (px)": float(sigma),
        "Canopy point spacing, 60th percentile (m)": spacing,
    }


def voxel_occupancy(x: np.ndarray, y: np.ndarray, z: np.ndarray,
                    input_points: np.ndarray, size: float) -> Dict[str, float]:
    """How the structure points fill the Stage-2 voxel grid at `size`.

    The grid origin is the minimum over both clouds, as in
    `voxel_merge.cartesian_voxel_merge`: the Z_norm cloud's, since the structure
    cloud is its labelled subset.
    """
    labelled = input_points > 0
    key = np.zeros(int(np.count_nonzero(labelled)), dtype=np.int64)
    for values in (x, y, z):
        key = (key << 20) | np.floor((values[labelled] - values.min()) / size).astype(np.int64)
    voxels, voxel, counts = np.unique(key, return_inverse=True, return_counts=True)
    pairs = np.unique(voxel.ravel() * len(LAS_CODES) + (input_points[labelled] - LAS_CODES[0]))
    mixed = np.bincount(pairs // len(LAS_CODES), minlength=len(voxels)) > 1
    return {
        "Structure voxels": len(voxels),
        "Structure points per voxel, median": float(np.median(counts)),
        "Single-point voxels (%)": 100.0 * np.count_nonzero(counts == 1) / len(voxels),
        "Mixed-class voxels (%)": 100.0 * np.count_nonzero(mixed) / len(voxels),
        "Structure points in mixed-class voxels (%)": 100.0 * counts[mixed].sum() / counts.sum(),
    }


def run_tile(site: str, tile: str, z_path: Path, s_path: Path, input_path: Path,
             reference_path: Path, parameters: Sequence[str]) -> List[Dict]:
    started = time.time()
    las = laspy.read(str(z_path))
    las_s = laspy.read(str(s_path))
    x, y, z = np.asarray(las.x), np.asarray(las.y), np.asarray(las.z)

    input_labels, input_transform = read_labels(input_path)
    fine, fine_transform = read_labels(reference_path)
    input_points = classes_beneath(input_labels, input_transform, x, y)
    references = {
        "input": (input_points, as_codes(input_labels), input_transform),
        "1 cm": (classes_beneath(fine, fine_transform, x, y), *majority_grid(fine, fine_transform)),
    }

    header = {"Site": site, "Site condition": SITES[site], "Tile": tile}
    rows = [{
        **header, "Parameter": "input", "Value": math.nan,
        "Z_norm points": len(x),
        "Structure points": len(las_s.points),
        "Points on labelled 5 cm cells": int(np.count_nonzero(input_points)),
        **stage1_matching(las_s, las),
        **auto_crown_parameters(las_s, las),
        **score(input_points, x, y, references, "Final"),
    }]
    del las, las_s

    default = None
    for parameter, value, overrides in configurations(parameters):
        config = dict(overrides, **SITE_CONFIG[site], write_outputs=False, return_classification_arrays=True)
        with contextlib.redirect_stdout(io.StringIO()):
            result = process_tile(z_path, s_path, out_path=Path("/dev/null"), config=config)
        baseline = np.asarray(result["classification_baseline"])
        refined = np.asarray(result["classification_refined"])
        final = np.asarray(result["classification_final"])
        if default is None:  # configurations() lists the default first
            default = baseline, final
        n = len(final)
        refined_change = refined != baseline
        row = {
            **header, "Parameter": parameter, "Value": value,
            "Stage-2 labelled points (%)": result["stage2_labeled_pct"],
            "Stage-3 crowns": result["stage3_watershed_crowns"],
            "Stage-3 points in crowns (%)": 100.0 * result["stage3_points_in_crowns"] / n,
            "Stage-4 changed points (%)": result["stage4_changed_pct"],
            "Stage-4 filled points (%)": 100.0 * np.count_nonzero(refined_change & (baseline == 0)) / n,
            "Stage-4 flipped points (%)": 100.0 * np.count_nonzero(refined_change & (baseline != 0)) / n,
            "Stage-4 mean confidence": result["stage4_mean_confidence"],
            "Stage-4 below min_conf (%)": result["stage4_low_confidence_pct"],
            "Stage-5 promoted points (%)": 100.0 * result["stage5_promoted_points"] / n,
            "Stage-5 crowns voted": result["stage5_crowns_voted"],
            "Stage-5 crowns with mallee majority": result["stage5_crowns_mallee_majority"],
            "Stage-5 least-mallee crown (%)": result["stage5_min_crown_mallee_pct"],
            "Stage-5 crown-majority points (%)": 100.0 * result["stage5_majority_reclassified_points"] / n,
            "Final labelled points (%)": result["final_veg_pct"],
            "Final treeIDs": result["final_treeids"],
            "Stage 2 changed vs default (%)": 100.0 * np.count_nonzero(baseline != default[0]) / n,
            "Final changed vs default (%)": 100.0 * np.count_nonzero(final != default[1]) / n,
        }
        if parameter in ("default", "voxel_size"):
            row.update(voxel_occupancy(x, y, z, input_points, config["voxel_size_xy"]))
        row.update(score(baseline, x, y, references, "Stage 2"))
        row.update(score(final, x, y, references, "Final"))
        rows.append(row)
    rows[0]["Seconds"] = time.time() - started
    return rows


def expand_default(per_tile: pd.DataFrame, parameters: Sequence[str]) -> pd.DataFrame:
    """Copy the shared default run into each sweep at its default value, so every
    parameter's curve is complete on its own."""
    default = per_tile[per_tile["Parameter"] == "default"]
    copies = [default.assign(Parameter=p, Value=DEFAULTS[p]) for p in parameters]
    swept = per_tile[per_tile["Parameter"] != "default"]
    return pd.concat([swept, *copies], ignore_index=True).sort_values(
        ["Site", "Parameter", "Value", "Tile"],
        key=lambda s: s.map(int) if s.name == "Tile" else s)


def summarise(per_tile: pd.DataFrame) -> pd.DataFrame:
    metrics = [c for c in per_tile.columns if c not in ID_COLUMNS]
    grouped = per_tile.groupby(["Site", "Site condition", "Parameter", "Value"], dropna=False)[metrics]
    summary = grouped.agg(["mean", "median", "std"])
    summary.columns = [f"{metric} {stat}" for metric, stat in summary.columns]
    summary.insert(0, "Tiles", grouped.size())
    return summary.reset_index()


def main() -> None:
    args = parse_args()
    parameters = [p.strip() for p in args.parameters.split(",") if p.strip()]
    unknown = set(parameters) - set(SWEEPS)
    if unknown:
        raise SystemExit(f"Unknown parameters: {sorted(unknown)}; choose from {list(SWEEPS)}")
    wanted = {t.strip() for t in args.tiles.split(",") if t.strip()}
    jobs = []
    for site in [s.strip() for s in args.sites.split(",") if s.strip()]:
        for tile, *paths in site_tiles(args.data_root, site):
            if not wanted or tile in wanted:
                jobs.append((site, tile, *paths, parameters))
    if not jobs:
        raise SystemExit("No tiles with Z_norm LAS, structure LAS and both label rasters found.")
    n_configs = len(configurations(parameters))
    print(f"{len(jobs)} tiles x {n_configs} configurations = {len(jobs) * n_configs} runs", flush=True)

    records: List[Dict] = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(run_tile, *job): job for job in jobs}
        for done, future in enumerate(as_completed(futures), 1):
            site, tile = futures[future][:2]
            rows = future.result()
            records.extend(rows)
            print(f"[{done}/{len(jobs)}] {site} tile {tile} ({rows[0]['Seconds']:.0f} s)", flush=True)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    per_tile = expand_default(pd.DataFrame.from_records(records), parameters)
    per_tile.to_csv(args.output_dir / "parameter_sensitivity_per_tile.csv", index=False)
    summarise(per_tile).to_csv(args.output_dir / "parameter_sensitivity_summary.csv", index=False)
    print(f"Wrote: {args.output_dir}")


if __name__ == "__main__":
    main()
