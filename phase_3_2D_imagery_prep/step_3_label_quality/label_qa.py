#!/usr/bin/env python3
"""Quality assessment of the 2D reference labels.

The 2D structure labels produced in `step_2_structure_2D_labels` are the
reference used by every downstream experiment. This script quantifies their
quality from the delivered artefacts alone, and writes two tables:

    <site>_qa_summary.csv   one row per tile, plus an Average and a Median row
    <site>_qa_issues.csv    one row per individual problem found

The assessment is four checks, following the label from the drawn polygon to
the array a model reads:

    Shapefiles
        |
        +-- 1. Annotation QA
        |       invalid geometry, cross-class overlap, below MMU
        |
        +-- 2. Polygon -> 1 cm raster
        |       class IoU
        |
        +-- 3. 1 cm -> 5 cm label grid
        |       class IoU, polygon IoU, centroid RMSE
        |
        +-- 4. Label grid -> predictor grid
        |       class IoU, polygon IoU, centroid RMSE, grid offset
        |
        +-- 5. +/-5 cm boundary perturbation
                delineation IoU

*Annotation QA*  - digitising artefacts: invalid rings, polygons below the
  minimum mapping unit, and polygons of different classes that overlap and so
  leave pixels with no single truth.

*Class IoU*  - one number per class for each lossy step:

    IoU 1 cm        polygons rasterised exactly (centroid rule) against the
                    delivered 1 cm raster. `2_2_structure_2D_rasterLabels.py`
                    uses all_touched=True and resolves overlaps by row order;
                    this IoU is how much of the class survives both.
    IoU 5 cm        the majority class of each 5 cm cell's 1 cm sub-pixels
                    against the delivered nearest-neighbour 5 cm raster.
    IoU predictor   the 1 cm labels put on the predictor grid by majority,
                    against the delivered 5 cm raster read there by nearest
                    neighbour, the way a dataloader reads it.

*Per feature*  - class IoU says how much of a class survives but not where it
  went. After Haynes et al. (2025), each polygon's footprint on one grid is
  compared with its footprint on the previous one: the centroid RMSE says how
  far it moved, the polygon IoU how much of it survived.

*Delineation sensitivity*  - with no second annotator, erode and dilate each
  class by `--boundary` metres; the IoU between the two is the agreement two
  delineations disagreeing by that much everywhere would reach. A worst case,
  not an estimate.

Only `--shp-dir`, `--site` and `--output-dir` are required. Checks 2 and 5 need
`--raster-dir`, check 3 additionally needs `--downsampled-dir`, and check 4
additionally needs `--image-dir`; any measure whose inputs are absent is left
blank rather than failing. Tiles are matched on the trailing numeric ID, as in
phase 4.

Example
-------
python label_qa.py \
    --site site1_1_DD0001 --site-condition Medium \
    --shp-dir         'D:\\paper2\\...\\structure_class\\annotations_shp' \
    --raster-dir      'D:\\paper2\\...\\structure_class\\annotation_raster' \
    --downsampled-dir 'D:\\paper2\\...\\annotation_raster\\downsample_5cm_nearest' \
    --image-dir       'D:\\paper2\\...\\predictors\\tiles_3072\\multispec\\res_05\\stacked' \
    --output-dir      'D:\\paper2\\...\\validation\\label_quality'
"""

from __future__ import annotations

import argparse
import csv
import math
import re
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.features import rasterize
from scipy.ndimage import binary_dilation, binary_erosion
from shapely.validation import explain_validity


# --------------------------------------------------------------------------
# Class scheme (mirrors 2_1_structure_2D_labels.py / 2_3_downsampling_labels.py)
# --------------------------------------------------------------------------

CLASS_LABELS = {0: "Forb", 1: "Woody debris", 2: "Grass", 3: "Shrub", 4: "Mallee"}
STRUCT_IDS = tuple(CLASS_LABELS)
NODATA = 255

# `2_2_structure_2D_rasterLabels.py` writes the 1 cm rasters as struct_id, then
# `2_3_downsampling_labels.py` remaps the 5 cm rasters to these LAS codes. Every
# label raster is read through remap_class_codes(), so either coding works.
VALUE_REMAP = {0: 64, 1: 65, 2: 66, 3: 67, 4: 68}

# A polygon smaller than this is below the minimum mapping unit: 5 cm is the
# resolution the labels are consumed at downstream, so 4 cells is the practical MMU.
DEFAULT_MMU_M2 = 4 * 0.05 * 0.05

# How far a hand-drawn boundary is assumed to be uncertain by, in metres. One
# 5 cm cell, i.e. the resolution the labels are consumed at.
DEFAULT_BOUNDARY_M = 0.05

# Below this, two delineations differing by DEFAULT_BOUNDARY_M would agree on
# less than half the class, so the class is boundary-dominated and is flagged.
DELINEATION_IOU_ALARM = 0.50

# An IoU this low against the delivered raster is not an all_touched effect; it
# means the shapefile and the raster were built from different material.
RASTER_IOU_ALARM = 0.50

# A label and an image pixel at the same row and column have to be the same
# ground. Below this the two grids do not agree well enough for that to hold.
REGISTRATION_IOU_ALARM = 0.95

# Origins written by different GDAL calls differ in the last decimal; below this
# an offset is float noise in the GeoTIFF header rather than a real shift.
GRID_OFFSET_TOLERANCE_CM = 0.1

# The grids a feature is carried onto after the 1 cm raster, in pipeline order.
# Each is compared with the one before it.
FEATURE_STAGES = ("5 cm", "predictor")


SUMMARY_COLUMNS = [
    "Site", "Site condition", "Tile", "Polygons",
    # 1. Annotation QA
    "Invalid geometries", "Cross-class overlap (%)", "Below MMU", "Below MMU (%)",
] + [f"IoU 1 cm {name}" for name in CLASS_LABELS.values()] \
  + [f"IoU 5 cm {name}" for name in CLASS_LABELS.values()] \
  + ["Polygon IoU mean 5 cm", "Polygon IoU median 5 cm", "Centroid RMSE 5 cm (m)"] \
  + [f"IoU predictor {name}" for name in CLASS_LABELS.values()] \
  + ["Polygon IoU mean predictor", "Polygon IoU median predictor",
     "Centroid RMSE predictor (m)", "Predictor grid offset (cm)", "Predictor GSD (cm)"] \
  + [f"Delineation IoU {name}" for name in CLASS_LABELS.values()]

ISSUE_COLUMNS = ["Site", "Tile", "Polygon", "Issue", "Details"]


# --------------------------------------------------------------------------
# CLI and path handling
# --------------------------------------------------------------------------

def wsl_path(path: str) -> Path:
    """Accept Windows drive paths, preferring a /tmp mirror when the drive is unmounted."""
    match = re.match(r"^([A-Za-z]):\\(.*)$", path)
    if match:
        drive, rest = match.groups()
        rest_path = rest.replace("\\", "/")
        mounted_path = Path("/mnt") / drive.lower() / rest_path
        mirror_path = Path("/tmp") / f"wsl_drive_{drive.lower()}" / rest_path
        return mirror_path if mirror_path.parent.exists() else mounted_path
    return Path(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Quality assessment of the 2D structure reference labels.")
    parser.add_argument("--site", required=True,
                        help="Site name, used as the output filename prefix.")
    parser.add_argument("--site-condition", default="",
                        help="Value written to the 'Site condition' column.")
    parser.add_argument("--shp-dir", type=wsl_path, required=True,
                        help="Directory of structure-classified annotation shapefiles.")
    parser.add_argument("--raster-dir", type=wsl_path,
                        help="Directory of 1 cm label rasters (enables checks 2 and 5).")
    parser.add_argument("--downsampled-dir", type=wsl_path,
                        help="Directory of 5 cm label rasters (enables check 3).")
    parser.add_argument("--image-dir", type=wsl_path,
                        help="Directory of stacked predictor tiles, i.e. the imagery the "
                             "models read (enables check 4).")
    parser.add_argument("--output-dir", type=wsl_path, required=True,
                        help="Directory for the two CSVs.")
    parser.add_argument("--shp-glob", default="*.shp")
    parser.add_argument("--raster-glob", default="*.tif")
    parser.add_argument("--image-glob", default="*.tif")
    parser.add_argument("--mmu", type=float, default=DEFAULT_MMU_M2,
                        help=f"Minimum mapping unit in m2. Default {DEFAULT_MMU_M2:.4f}.")
    parser.add_argument("--boundary", type=float, default=DEFAULT_BOUNDARY_M,
                        help="Boundary perturbation in metres for the delineation IoU. "
                             f"Default {DEFAULT_BOUNDARY_M}.")
    parser.add_argument("--tiles", default="",
                        help="Comma-separated tile IDs to restrict the run to.")
    return parser.parse_args()


def detect_tile_id(path: Path) -> str | None:
    match = re.search(r"(\d+)$", path.stem)
    return match.group(1) if match else None


def scan_tile_map(directory: Path | None, pattern: str) -> Dict[str, Path]:
    if directory is None or not directory.exists():
        return {}
    tile_map: Dict[str, Path] = {}
    for path in sorted(directory.glob(pattern)):
        if not path.is_file():
            continue
        tile_id = detect_tile_id(path)
        if tile_id is None or tile_id in tile_map:
            continue
        tile_map[tile_id] = path
    return tile_map


def read_annotations(shp_path: Path) -> gpd.GeoDataFrame:
    gdf = gpd.read_file(shp_path)
    if "struct_id" not in gdf.columns:
        raise KeyError(
            f"{shp_path.name} has no 'struct_id' column "
            f"(found {list(gdf.columns)}). Run 2_1_structure_2D_labels.py first.")
    gdf = gdf[gdf.geometry.notna()].reset_index(drop=True)
    gdf["struct_id"] = gdf["struct_id"].astype(int)
    return gdf


def iou(a: np.ndarray, b: np.ndarray) -> float:
    """Intersection over union of two boolean masks."""
    union = int(np.count_nonzero(a | b))
    return float(np.count_nonzero(a & b) / union) if union else math.nan


def remap_class_codes(values: np.ndarray) -> np.ndarray:
    """Bring the LAS codes 64-68 that `2_3_downsampling_labels.py` writes into
    the 5 cm rasters back to struct_id, leaving a raster already coded as
    struct_id untouched.

    Testing `max() > 4` would not do: the 255 nodata makes that true of every
    raster, so a correctly coded one would be blanked. Presence of the codes
    themselves is the only safe test.
    """
    if not np.isin(values, list(VALUE_REMAP.values())).any():
        return values
    remapped = np.full(values.shape, NODATA, dtype=np.uint8)
    for struct_id, code in VALUE_REMAP.items():
        remapped[values == code] = struct_id
    return remapped


def flag_low(row: Dict[str, object], prefix: str, alarm: float, issue: "IssueLog",
             kind: str, message: str) -> None:
    """Log every class whose `<prefix> <class>` score falls below `alarm`."""
    for name in CLASS_LABELS.values():
        score = row.get(f"{prefix} {name}")
        if isinstance(score, float) and not math.isnan(score) and score < alarm:
            issue.add("", kind, f"{name}: IoU {score:.3f} {message}")


# --------------------------------------------------------------------------
# 1. Annotation QA: invalid geometry, cross-class overlap, below MMU
# --------------------------------------------------------------------------

def assess_annotation(gdf: gpd.GeoDataFrame, mmu: float,
                      issue: "IssueLog") -> Dict[str, object]:
    geoms = gdf.geometry
    areas = geoms.area.to_numpy(dtype=np.float64)
    n = len(gdf)

    invalid_mask = (~geoms.is_valid).to_numpy()
    for index in np.flatnonzero(invalid_mask):
        # explain_validity returns "Self-intersection[x y]"; keep only the kind.
        reason = explain_validity(geoms.iloc[int(index)]).split("[")[0].strip()
        issue.add(int(index), "invalid_geometry", reason)

    below_mmu = areas < mmu
    for index in np.flatnonzero(below_mmu):
        issue.add(int(index), "below_mmu", f"{areas[index]:.6f} m2 (MMU {mmu:.4f})")

    return {
        "Polygons": n,
        "Invalid geometries": int(invalid_mask.sum()),
        "Cross-class overlap (%)": cross_class_overlap(gdf, float(areas.sum()), issue),
        "Below MMU": int(below_mmu.sum()),
        "Below MMU (%)": (int(below_mmu.sum()) / n * 100.0) if n else math.nan,
    }


def cross_class_overlap(gdf: gpd.GeoDataFrame, labelled: float,
                        issue: "IssueLog") -> float:
    """Share of labelled area under two polygons of different classes.

    Such a pixel has no single reference value, and which one it receives is
    decided by shapefile row order in 2_2_structure_2D_rasterLabels.py.
    """
    geoms = gdf.geometry.to_numpy()
    class_ids = gdf["struct_id"].to_numpy()

    left, right = gdf.sindex.query(gdf.geometry, predicate="intersects")
    keep = (left < right) & (class_ids[left] != class_ids[right])
    cross_area = 0.0
    for i, j in zip(left[keep], right[keep]):
        a, b = geoms[i], geoms[j]
        try:
            overlap = (a if a.is_valid else a.buffer(0)).intersection(
                b if b.is_valid else b.buffer(0)).area
        except Exception:
            continue
        if overlap <= 0:
            continue  # shared edge only, not an overlap
        cross_area += overlap
        issue.add(int(i), "cross_class_overlap",
                  f"with polygon {int(j)} ({CLASS_LABELS.get(int(class_ids[i]), class_ids[i])} / "
                  f"{CLASS_LABELS.get(int(class_ids[j]), class_ids[j])}); {overlap:.6f} m2")
    return (cross_area / labelled * 100.0) if labelled else math.nan


# --------------------------------------------------------------------------
# Class IoU at each step
# --------------------------------------------------------------------------

def iou_polygon_to_1cm(gdf: gpd.GeoDataFrame, raster_path: Path) -> Dict[str, object]:
    """2. Polygons rasterised exactly against the delivered 1 cm raster.

    The reference uses the centroid rule (all_touched=False), so the shortfall
    from 1.0 is the combined cost of the all_touched dilation and of resolving
    overlaps by draw order - the two things step 2 does to make the raster.
    """
    with rasterio.open(raster_path) as src:
        delivered = remap_class_codes(src.read(1))
        transform = src.transform
        shape = (src.height, src.width)

    reference = rasterize(
        list(zip(gdf.geometry, gdf["struct_id"])),
        out_shape=shape, transform=transform,
        fill=NODATA, all_touched=False, dtype=rasterio.uint8)

    return {f"IoU 1 cm {name}": iou(reference == struct_id, delivered == struct_id)
            for struct_id, name in CLASS_LABELS.items()}


def _fine_to_coarse_index(fine_transform: rasterio.Affine, n: int,
                          coarse_transform: rasterio.Affine, axis: str) -> np.ndarray:
    """Map fine-grid indices along one axis onto coarse-grid indices, using fine
    pixel centres. Assumes both grids are north-up, which holds for every
    product written by phase 3."""
    idx = np.arange(n, dtype=np.float64) + 0.5
    if axis == "x":
        coords = fine_transform.c + idx * fine_transform.a
        return np.floor((coords - coarse_transform.c) / coarse_transform.a).astype(np.int64)
    coords = fine_transform.f + idx * fine_transform.e
    return np.floor((coords - coarse_transform.f) / coarse_transform.e).astype(np.int64)


def majority_on_grid(fine: np.ndarray, fine_transform: rasterio.Affine,
                     target_transform: rasterio.Affine,
                     target_shape: tuple[int, int]) -> np.ndarray:
    """Class of each target cell by majority of the 1 cm label pixels inside it.

    Unlabelled sub-pixels are counted as their own category, so a cell that is
    mostly nodata resolves to nodata. Taking the majority over labelled
    sub-pixels only would call a cell "woody debris" on the strength of a single
    1 cm pixel in 25, which no resampling rule would ever deliver and which
    would depress every sparse class.
    """
    height, width = target_shape
    rows_idx = _fine_to_coarse_index(fine_transform, fine.shape[0], target_transform, "y")
    cols_idx = _fine_to_coarse_index(fine_transform, fine.shape[1], target_transform, "x")

    in_grid = ((rows_idx[:, None] >= 0) & (rows_idx[:, None] < height)
               & (cols_idx[None, :] >= 0) & (cols_idx[None, :] < width))
    cell = rows_idx[:, None] * width + cols_idx[None, :]

    unlabelled = len(STRUCT_IDS)
    n_categories = unlabelled + 1
    flat_cell = cell[in_grid].astype(np.int64).ravel()
    flat_class = fine[in_grid].astype(np.int64)
    flat_class[flat_class > unlabelled] = unlabelled

    counts = np.bincount(flat_cell * n_categories + flat_class,
                         minlength=height * width * n_categories
                         ).reshape(height * width, n_categories)
    majority = np.argmax(counts, axis=1).astype(np.int64)
    majority[majority == unlabelled] = NODATA
    return majority.reshape(height, width)


def iou_1cm_to_5cm(fine_path: Path, coarse_path: Path) -> Dict[str, object]:
    """3. Majority class of each 5 cm cell's 1 cm sub-pixels against the
    delivered nearest-neighbour 5 cm raster. 1.0 means the downsample lost nothing.

    A 3072 px tile at 1 cm is 30.72 m, and 30.72 / 0.05 is not an integer, so
    `gdalwarp -tr 0.05` in `2_3_downsampling_labels.py` cannot always land on
    the 1 cm grid; where it shifts the origin, nearest resampling with
    `-srcnodata 255` dilates every sparse class, and this IoU drops.
    """
    with rasterio.open(fine_path) as src:
        fine = remap_class_codes(src.read(1))
        fine_transform = src.transform
    with rasterio.open(coarse_path) as src:
        coarse = remap_class_codes(src.read(1))
        coarse_transform = src.transform
        coarse_shape = src.shape

    majority = majority_on_grid(fine, fine_transform, coarse_transform, coarse_shape)
    return {f"IoU 5 cm {name}": iou(majority == struct_id, coarse == struct_id)
            for struct_id, name in CLASS_LABELS.items()}


def assess_registration(fine_path: Path, coarse_path: Path, image_path: Path,
                        issue: "IssueLog") -> Dict[str, object]:
    """4. Whether a label and a predictor pixel at the same row and column are
    the same ground.

    A model indexes the label tensor and the predictor tensor together, so the
    two grids have to interlock. They are built by different steps and nothing
    between here and training checks that they agree.

    `Predictor grid offset` is how far the predictor origin sits from the label
    grid, reduced modulo one label cell, so 0 means the grids interlock and half
    a cell is the worst case. `Predictor GSD` is carried alongside it because an
    offset only means "interlock" when the two cell sizes match.

    `IoU predictor <class>` is what the mismatch costs each class: the 1 cm
    labels placed on the predictor grid correctly, by majority of the sub-pixels
    each pixel covers, against the delivered 5 cm labels read there by nearest
    neighbour, the way a dataloader that trusts the grids to match reads them.
    """
    with rasterio.open(fine_path) as src:
        fine = remap_class_codes(src.read(1))
        fine_transform = src.transform
    with rasterio.open(coarse_path) as src:
        delivered = remap_class_codes(src.read(1))
        label_transform = src.transform
        label_gsd = abs(src.transform.a)
    with rasterio.open(image_path) as src:
        image_transform = src.transform
        image_shape = src.shape
        image_gsd = abs(src.transform.a)

    def grid_offset(a: float, b: float) -> float:
        gap = abs(a - b) % label_gsd
        return min(gap, label_gsd - gap)

    offset = max(grid_offset(image_transform.c, label_transform.c),
                 grid_offset(image_transform.f, label_transform.f))

    reference = majority_on_grid(fine, fine_transform, image_transform, image_shape)

    rows_idx = _fine_to_coarse_index(image_transform, image_shape[0], label_transform, "y")
    cols_idx = _fine_to_coarse_index(image_transform, image_shape[1], label_transform, "x")
    inside = ((rows_idx[:, None] >= 0) & (rows_idx[:, None] < delivered.shape[0])
              & (cols_idx[None, :] >= 0) & (cols_idx[None, :] < delivered.shape[1]))
    nearest = np.full(image_shape, NODATA, dtype=np.uint8)
    sampled = delivered[np.ix_(np.clip(rows_idx, 0, delivered.shape[0] - 1),
                               np.clip(cols_idx, 0, delivered.shape[1] - 1))]
    nearest[inside] = sampled[inside]

    row: Dict[str, object] = {
        "Predictor GSD (cm)": image_gsd * 100.0,
        "Predictor grid offset (cm)": offset * 100.0,
    }
    for struct_id, name in CLASS_LABELS.items():
        row[f"IoU predictor {name}"] = iou(reference == struct_id, nearest == struct_id)

    if abs(image_gsd - label_gsd) > 1e-6:
        issue.add("", "predictor_gsd_differs_from_labels",
                  f"predictor {image_gsd * 100:.2f} cm against labels {label_gsd * 100:.2f} cm; "
                  f"the two tensors cannot be indexed together without resampling one again")
    if offset * 100.0 > GRID_OFFSET_TOLERANCE_CM:
        issue.add("", "predictor_grid_offset",
                  f"{offset * 100:.2f} cm between the predictor and label grids, "
                  f"{offset / label_gsd:.2f} of a label cell")
    flag_low(row, "IoU predictor", REGISTRATION_IOU_ALARM, issue,
             "predictor_grid_disagreement",
             "between the labels resampled onto the predictor grid correctly and "
             "the same labels read by nearest neighbour")
    return row


# --------------------------------------------------------------------------
# Per feature: polygon IoU and centroid RMSE, after Haynes et al. (2025)
# --------------------------------------------------------------------------

def _grid_index(coords: np.ndarray, origin: float, step: float) -> np.ndarray:
    """Cell index containing each map coordinate along one axis."""
    return np.floor((coords - origin) / step).astype(np.int64)


def _feature_on_grid(geom, transform: rasterio.Affine,
                     shape: Tuple[int, int]) -> Tuple[np.ndarray, int, int]:
    """One polygon rasterised (centroid rule) inside its own window on `transform`.

    Returns the boolean footprint and the window's row/column offset. Working
    per polygon in a local window keeps each feature's identity, which a
    whole-tile rasterisation loses wherever two polygons overlap.
    """
    minx, miny, maxx, maxy = geom.bounds
    col_lo = _grid_index(np.array([minx]), transform.c, transform.a)[0] - 1
    col_hi = _grid_index(np.array([maxx]), transform.c, transform.a)[0] + 2
    row_lo = _grid_index(np.array([maxy]), transform.f, transform.e)[0] - 1
    row_hi = _grid_index(np.array([miny]), transform.f, transform.e)[0] + 2

    col_lo, row_lo = max(col_lo, 0), max(row_lo, 0)
    col_hi, row_hi = min(col_hi, shape[1]), min(row_hi, shape[0])
    if col_hi <= col_lo or row_hi <= row_lo:
        return np.zeros((0, 0), dtype=bool), row_lo, col_lo

    window_transform = rasterio.Affine(
        transform.a, transform.b, transform.c + col_lo * transform.a,
        transform.d, transform.e, transform.f + row_lo * transform.e)
    footprint = rasterize(
        [(geom, 1)], out_shape=(row_hi - row_lo, col_hi - col_lo),
        transform=window_transform, fill=0, all_touched=False,
        dtype=rasterio.uint8).astype(bool)
    return footprint, row_lo, col_lo


def _footprint_centroid(footprint: np.ndarray, row_off: int, col_off: int,
                        transform: rasterio.Affine) -> Tuple[float, float] | None:
    """Map coordinates of a footprint's centre of mass, cell centres equally weighted."""
    rows, cols = np.nonzero(footprint)
    if rows.size == 0:
        return None
    x = transform.c + (col_off + cols.mean() + 0.5) * transform.a
    y = transform.f + (row_off + rows.mean() + 0.5) * transform.e
    return float(x), float(y)


def _footprint_on_grid(footprint: np.ndarray, row_off: int, col_off: int,
                       source_transform: rasterio.Affine,
                       target_rows: np.ndarray, target_cols: np.ndarray,
                       target_transform: rasterio.Affine) -> np.ndarray:
    """A footprint read onto another grid's cells by nearest neighbour at their
    centres - the way a dataloader reads a label on a grid it was not written to."""
    x = target_transform.c + (target_cols + 0.5) * target_transform.a
    y = target_transform.f + (target_rows + 0.5) * target_transform.e
    cols = _grid_index(x, source_transform.c, source_transform.a) - col_off
    rows = _grid_index(y, source_transform.f, source_transform.e) - row_off

    inside = ((rows >= 0) & (rows < footprint.shape[0])
              & (cols >= 0) & (cols < footprint.shape[1]))
    out = np.zeros(rows.shape, dtype=bool)
    if footprint.size:
        out[inside] = footprint[rows[inside], cols[inside]]
    return out


def assess_features(gdf: gpd.GeoDataFrame, fine_path: Path, coarse_path: Path,
                    image_path: Path | None, issue: "IssueLog") -> Dict[str, object]:
    """Per feature, how far it moves and how much of it survives at each step.

    Each polygon is rasterised onto each grid and compared with its footprint on
    the grid before it:

        5 cm        5 cm label footprint against 1 cm footprint, IoU on the 1 cm grid
        predictor   predictor footprint against 5 cm label footprint, IoU on the
                    5 cm grid, the resolution the models consume

    Centroid RMSE follows Haynes et al.'s equation (2) over the Euclidean
    distances between the two footprints' centroids, in metres. Haynes et al.
    compare two independent delineations; here there is one, so these columns
    bound the positional cost of the grids, not of the annotator.
    """
    grids: List[Tuple[str, rasterio.Affine, Tuple[int, int]]] = []
    for name, path in (("1 cm", fine_path), ("5 cm", coarse_path), ("predictor", image_path)):
        if path is None:
            break
        with rasterio.open(path) as src:
            grids.append((name, src.transform, src.shape))

    distances: Dict[str, List[float]] = {stage: [] for stage, *_ in grids[1:]}
    ious: Dict[str, List[float]] = {stage: [] for stage, *_ in grids[1:]}

    for geom in gdf.geometry:
        if geom is None or geom.is_empty or not geom.is_valid:
            continue
        previous = None
        for stage, transform, shape in grids:
            footprint, row_off, col_off = _feature_on_grid(geom, transform, shape)
            centre = _footprint_centroid(footprint, row_off, col_off, transform)
            if centre is None:
                break
            current = (footprint, row_off, col_off, transform, centre)
            if previous is not None:
                prev_fp, prev_row, prev_col, prev_transform, prev_centre = previous
                distances[stage].append(math.hypot(centre[0] - prev_centre[0],
                                                   centre[1] - prev_centre[1]))
                rows, cols = np.meshgrid(np.arange(prev_fp.shape[0]) + prev_row,
                                         np.arange(prev_fp.shape[1]) + prev_col,
                                         indexing="ij")
                test = _footprint_on_grid(footprint, row_off, col_off, transform,
                                          rows, cols, prev_transform)
                union = np.count_nonzero(prev_fp | test)
                if union:
                    ious[stage].append(np.count_nonzero(prev_fp & test) / union)
            previous = current

    result: Dict[str, object] = {}
    for stage, transform, _ in grids[1:]:
        if distances[stage]:
            rmse = float(np.sqrt(np.mean(np.square(distances[stage]))))
            result[f"Centroid RMSE {stage} (m)"] = rmse
            # A feature displaced by more than one cell of the grid it is read on
            # is no longer on the pixel a model will pair it with.
            if rmse > abs(transform.a):
                issue.add("", "feature_displacement",
                          f"centroid RMSE {rmse:.3f} m on the {stage} grid exceeds its "
                          f"{abs(transform.a):.3f} m cell")
        if ious[stage]:
            result[f"Polygon IoU mean {stage}"] = float(np.mean(ious[stage]))
            result[f"Polygon IoU median {stage}"] = float(np.median(ious[stage]))
    return result


# --------------------------------------------------------------------------
# 5. +/- boundary perturbation: delineation IoU
# --------------------------------------------------------------------------

SQUARE = np.ones((3, 3), dtype=bool)


def assess_delineation(raster_path: Path, radius_m: float,
                       issue: "IssueLog") -> Dict[str, object]:
    """How exposed each class is to where the annotator put the line.

    Erode and dilate each class by `radius_m` and take the IoU of the two. That
    is the agreement two delineations that disagreed by `radius_m` *everywhere*
    would reach, so it is a worst case rather than an estimate - a class scoring
    high cannot be much affected by boundary placement no matter who drew it.
    """
    with rasterio.open(raster_path) as src:
        labels = remap_class_codes(src.read(1))
        pixel = abs(src.transform.a)

    radius = max(int(round(radius_m / pixel)), 1)
    row: Dict[str, object] = {}
    for struct_id, name in CLASS_LABELS.items():
        mask = labels == struct_id
        if not mask.any():
            row[f"Delineation IoU {name}"] = math.nan
            continue
        # border_value=1 stops the tile edge from counting as an eroded boundary.
        core = binary_erosion(mask, SQUARE, iterations=radius, border_value=1)
        grown = binary_dilation(mask, SQUARE, iterations=radius)
        row[f"Delineation IoU {name}"] = iou(core, grown)

    flag_low(row, "Delineation IoU", DELINEATION_IOU_ALARM, issue,
             "boundary_dominated_class",
             f"between the class eroded and dilated by {radius_m * 100:.0f} cm")
    return row


# --------------------------------------------------------------------------
# Output
# --------------------------------------------------------------------------

class IssueLog:
    """Flat log of individual problems, so a flagged polygon can be opened in QGIS."""

    def __init__(self, site: str) -> None:
        self.site = site
        self.rows: List[Dict[str, object]] = []
        self._tile = ""

    def tile(self, tile_id: str) -> None:
        self._tile = tile_id

    def add(self, polygon: int | str, issue: str, details: str) -> None:
        self.rows.append({"Site": self.site, "Tile": self._tile,
                          "Polygon": polygon, "Issue": issue, "Details": details})

    def counts(self) -> Dict[str, int]:
        tally: Dict[str, int] = {}
        for row in self.rows:
            key = str(row["Issue"])
            tally[key] = tally.get(key, 0) + 1
        return dict(sorted(tally.items(), key=lambda kv: -kv[1]))


def reduce_ignore_nan(values: Iterable[object], reducer) -> float:
    numeric = np.asarray(
        [v for v in values if isinstance(v, (int, float, np.integer, np.floating))],
        dtype=np.float64)
    if numeric.size == 0 or np.all(np.isnan(numeric)):
        return math.nan
    return float(reducer(numeric))


def aggregate_row(rows: Sequence[Dict[str, object]], site: str, condition: str,
                  label: str, reducer) -> Dict[str, object]:
    """One across-tile summary row. Non-numeric columns are blanked.

    Two are written, `Average` and `Median`. A handful of tiles decide the mean
    of an IoU column whenever one tile fails differently from the rest, so the
    median says what a typical tile scores and the gap between the two says how
    concentrated the damage is.
    """
    aggregate: Dict[str, object] = {"Site": site, "Site condition": condition, "Tile": label}
    for column in SUMMARY_COLUMNS:
        if column in aggregate:
            continue
        values = [r.get(column) for r in rows]
        aggregate[column] = reduce_ignore_nan(values, reducer) if any(
            isinstance(v, (int, float, np.integer, np.floating)) for v in values) else ""
    return aggregate


def format_value(value: object) -> str:
    if isinstance(value, float):
        if math.isnan(value):
            return ""
        return str(int(value)) if value.is_integer() else f"{value:.3f}"
    return "" if value is None else str(value)


def write_csv(path: Path, fieldnames: Sequence[str], rows: Sequence[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({f: format_value(row.get(f)) for f in fieldnames})


# --------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    shp_dir = args.shp_dir.resolve()
    if not shp_dir.exists():
        raise SystemExit(f"Shapefile directory does not exist: {shp_dir}")
    output_dir = args.output_dir.resolve()
    condition = args.site_condition.strip()

    shp_map = scan_tile_map(shp_dir, args.shp_glob)
    raster_map = scan_tile_map(args.raster_dir, args.raster_glob)
    coarse_map = scan_tile_map(args.downsampled_dir, args.raster_glob)
    image_map = scan_tile_map(args.image_dir, args.image_glob)

    tile_ids = sorted(shp_map, key=int)
    if args.tiles:
        wanted = {t.strip() for t in args.tiles.split(",") if t.strip()}
        tile_ids = [t for t in tile_ids if t in wanted]
    if not tile_ids:
        raise SystemExit(f"No shapefiles matched {args.shp_glob} in {shp_dir}")

    for enabled, name, hint in (
        (bool(raster_map), "checks 2 and 5", "--raster-dir"),
        (bool(raster_map) and bool(coarse_map), "check 3", "--raster-dir and --downsampled-dir"),
        (bool(raster_map) and bool(coarse_map) and bool(image_map), "check 4",
         "--raster-dir, --downsampled-dir and --image-dir"),
    ):
        if not enabled:
            print(f"Leaving {name} blank: needs {hint}")

    issues = IssueLog(args.site)
    summary: List[Dict[str, object]] = []

    for index, tile_id in enumerate(tile_ids, 1):
        print(f"[{index}/{len(tile_ids)}] Tile {tile_id}", flush=True)
        issues.tile(tile_id)
        try:
            gdf = read_annotations(shp_map[tile_id])
        except KeyError as exc:
            print(f"  skipped: {exc}", file=sys.stderr)
            continue
        if gdf.empty:
            print("  skipped: no polygons")
            continue

        fine_path = raster_map.get(tile_id)
        coarse_path = coarse_map.get(tile_id)
        image_path = image_map.get(tile_id)

        row: Dict[str, object] = {"Site": args.site, "Site condition": condition,
                                  "Tile": tile_id}
        row.update(assess_annotation(gdf, args.mmu, issues))

        if fine_path is None:
            issues.add("", "missing_raster", "no 1 cm label raster for this tile")
            summary.append(row)
            continue

        row.update(iou_polygon_to_1cm(gdf, fine_path))
        flag_low(row, "IoU 1 cm", RASTER_IOU_ALARM, issues, "raster_disagrees_with_shapefile",
                 "between the polygons on disk and the delivered 1 cm raster")
        row.update(assess_delineation(fine_path, args.boundary, issues))

        if coarse_path is None:
            issues.add("", "missing_downsampled", "no 5 cm label raster for this tile")
        else:
            row.update(iou_1cm_to_5cm(fine_path, coarse_path))
            if image_path is None:
                if image_map:
                    issues.add("", "missing_imagery", "no predictor stack for this tile")
            else:
                row.update(assess_registration(fine_path, coarse_path, image_path, issues))
            row.update(assess_features(gdf, fine_path, coarse_path, image_path, issues))

        summary.append(row)

    if not summary:
        raise SystemExit("No tiles produced a summary row.")

    tile_rows = list(summary)
    summary.append(aggregate_row(tile_rows, args.site, condition, "Average", np.nanmean))
    summary.append(aggregate_row(tile_rows, args.site, condition, "Median", np.nanmedian))
    summary_path = output_dir / f"{args.site}_qa_summary.csv"
    issues_path = output_dir / f"{args.site}_qa_issues.csv"

    write_csv(summary_path, SUMMARY_COLUMNS, summary)
    write_csv(issues_path, ISSUE_COLUMNS, issues.rows)

    print(f"\nTiles assessed: {len(tile_rows)}")
    print(f"Issues logged:  {len(issues.rows)}")
    for name, count in issues.counts().items():
        print(f"  {name:32s} {count}")
    for path in (summary_path, issues_path):
        print(f"  Wrote: {path}")


if __name__ == "__main__":
    main()
