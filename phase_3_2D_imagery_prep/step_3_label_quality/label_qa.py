#!/usr/bin/env python3
"""Quality assessment of the 2D reference labels.

The 2D structure labels produced in `step_2_structure_2D_labels` are the
reference used by every downstream experiment. This script quantifies their
quality from the delivered artefacts alone, and writes two tables:

    <site>_qa_summary.csv   one row per tile, plus an Average row
    <site>_qa_issues.csv    one row per individual problem found

and a generated prose report, `<site>_qa_report.md`, in the form it would
appear in a response to a reviewer.

What the summary measures, in three groups:

*Annotation*  - how much of the tile carries a reference value, the class
  balance, and the digitising artefacts (invalid rings, duplicates, slivers,
  and polygons of different classes that overlap and so leave pixels with no
  single truth).

*Preservation, as IoU*  - the labels pass through two lossy transformations
  before a model sees them, and each gets one number per class:

    IoU raster  polygons rasterised exactly (centroid rule) against the
                delivered 1 cm raster. `2_2_structure_2D_rasterLabels.py` uses
                all_touched=True, which dilates every polygon by up to one 1 cm
                ring and resolves overlaps by shapefile row order; this IoU is
                how much of the class survives both effects.
    IoU 5 cm    the majority class of each 5 cm cell's 1 cm sub-pixels against
                the delivered nearest-neighbour 5 cm raster. 1.0 means the
                downsample lost nothing.

*Plausibility*  - per polygon, the 95th percentile of `Z_norm` among the LiDAR
  points inside it, i.e. its top-of-canopy height, against the height range the
  class scheme implies. LiDAR height was never used to make the 2D labels, so
  this is a semi-independent check that the classes mean what they say.

  Note this is deliberately a *per-polygon top-of-canopy* statistic. Taking the
  median of every point in the column under a crown footprint instead measures
  the ground beneath the plant, not the plant, and reports tall classes as
  implausible by construction.

Only `--shp-dir`, `--site` and `--output-dir` are required. IoU raster needs
`--raster-dir`, IoU 5 cm additionally needs `--downsampled-dir`, and the height
columns need `--las-dir`; any measure whose inputs are absent is left blank
rather than failing. Tiles are matched on the trailing numeric ID, as in phase 4.

Example
-------
python label_qa.py \
    --site site1_1_DD0001 --site-condition Medium \
    --shp-dir         'D:\\paper2\\...\\structure_class\\annotations_shp' \
    --raster-dir      'D:\\paper2\\...\\structure_class\\annotation_raster' \
    --downsampled-dir 'D:\\paper2\\...\\annotation_raster\\downsample_5cm_nearest' \
    --las-dir         'D:\\paper2\\...\\lidar\\clipped\\raster\\z_norm\\csf' \
    --output-dir      'D:\\paper2\\...\\lidar\\clipped\\validation\\label_quality'
"""

from __future__ import annotations

import argparse
import csv
import math
import re
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.features import rasterize
from shapely.validation import explain_validity


# --------------------------------------------------------------------------
# Class scheme (mirrors 2_1_structure_2D_labels.py / 2_3_downsampling_labels.py)
# --------------------------------------------------------------------------

CLASS_LABELS = {0: "Forb", 1: "Woody debris", 2: "Grass", 3: "Shrub", 4: "Mallee"}
STRUCT_IDS = tuple(CLASS_LABELS)
NODATA = 255

# struct_id -> LAS classification code applied by 2_3_downsampling_labels.py
VALUE_REMAP = {0: 64, 1: 65, 2: 66, 3: 67, 4: 68}

# Z_norm ranges expected per class, from fvc3dstruct.height_stats.DEFAULT_HEIGHT_RANGES.
HEIGHT_RANGES = {0: (0.00, 0.40), 1: (0.05, 5.00), 2: (0.00, 0.80), 3: (0.40, 3.00), 4: (1.50, 8.00)}

# A polygon smaller than this is reported as a sliver: 5 cm is the resolution the
# labels are actually consumed at downstream, so 4 cells is the practical MMU.
DEFAULT_MMU_M2 = 4 * 0.05 * 0.05

# Top-of-canopy height per polygon. P95 rather than the maximum, which a single
# stray return above the crown would set.
CANOPY_PERCENTILE = 95

# An IoU this low against the delivered raster is not an all_touched effect; it
# means the shapefile and the raster were built from different material.
RASTER_IOU_ALARM = 0.50

# Below this many returns a polygon's P95 is the maximum of a handful of points,
# so the polygon is excluded from the height columns rather than judged on noise.
DEFAULT_MIN_POINTS = 10

# Sentinel for "no polygon here" in the polygon-index raster.
NO_POLYGON = -1


SUMMARY_COLUMNS = [
    "Site", "Site condition", "Tile", "Polygons", "Classes present",
    "Tile area (m2)", "Labelled area (m2)", "Labelled area (%)",
    "Median polygon area (m2)",
    "Invalid geometries", "Repairable by buffer(0)", "Duplicate geometries",
    "Empty or zero-area", "Slivers below MMU", "Slivers below MMU (%)",
    "Overlapping pairs", "Cross-class overlap (m2)", "Cross-class overlap (%)",
    "Polygons with too few returns (%)",
    "5 cm grid offset (cm)", "Labelled area change at 5 cm (%)",
] + [f"{name} area (%)" for name in CLASS_LABELS.values()] \
  + [f"{name} polygons" for name in CLASS_LABELS.values()] \
  + [f"IoU raster {name}" for name in CLASS_LABELS.values()] \
  + [f"IoU 5 cm {name}" for name in CLASS_LABELS.values()] \
  + [f"P95 Znorm {name} (m)" for name in CLASS_LABELS.values()] \
  + [f"Height outliers {name} (%)" for name in CLASS_LABELS.values()]

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
                        help="Directory of 1 cm label rasters (enables IoU raster).")
    parser.add_argument("--downsampled-dir", type=wsl_path,
                        help="Directory of 5 cm label rasters (enables IoU 5 cm).")
    parser.add_argument("--las-dir", type=wsl_path,
                        help="Directory of Z-normalised LAS tiles (enables the height columns).")
    parser.add_argument("--output-dir", type=wsl_path, required=True,
                        help="Directory for the two CSVs and the report.")
    parser.add_argument("--shp-glob", default="*.shp")
    parser.add_argument("--raster-glob", default="*.tif")
    parser.add_argument("--las-glob", default="*.las")
    parser.add_argument("--mmu", type=float, default=DEFAULT_MMU_M2,
                        help=f"Minimum mapping unit in m2. Default {DEFAULT_MMU_M2:.4f}.")
    parser.add_argument("--min-points", type=int, default=DEFAULT_MIN_POINTS,
                        help="LiDAR returns a polygon needs before its top-of-canopy "
                             f"height is used. Default {DEFAULT_MIN_POINTS}.")
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


# --------------------------------------------------------------------------
# Annotation: coverage, class balance, geometry, overlap
# --------------------------------------------------------------------------

def assess_annotation(gdf: gpd.GeoDataFrame, raster_path: Path | None,
                      mmu: float, issue: "IssueLog") -> Dict[str, object]:
    areas = gdf.geometry.area.to_numpy(dtype=np.float64)
    labelled = float(areas.sum())

    # Tile footprint from the label raster when available: the convex hull of
    # sparse annotations under-states an incompletely labelled tile.
    if raster_path is not None and raster_path.exists():
        with rasterio.open(raster_path) as src:
            footprint = abs(src.bounds.right - src.bounds.left) * \
                        abs(src.bounds.top - src.bounds.bottom)
    else:
        minx, miny, maxx, maxy = gdf.total_bounds
        footprint = float((maxx - minx) * (maxy - miny))

    present = sorted(set(gdf["struct_id"]) & set(STRUCT_IDS))
    row: Dict[str, object] = {
        "Polygons": int(len(gdf)),
        "Classes present": ", ".join(CLASS_LABELS[c] for c in present),
        "Tile area (m2)": footprint,
        "Labelled area (m2)": labelled,
        "Labelled area (%)": (labelled / footprint * 100.0) if footprint else math.nan,
        "Median polygon area (m2)": float(np.median(areas)) if areas.size else math.nan,
    }
    class_ids = gdf["struct_id"].to_numpy()
    for struct_id, name in CLASS_LABELS.items():
        selection = class_ids == struct_id
        row[f"{name} area (%)"] = (float(areas[selection].sum()) / labelled * 100.0) \
            if labelled else math.nan
        row[f"{name} polygons"] = int(selection.sum())

    unmapped = sorted(set(gdf["struct_id"]) - set(STRUCT_IDS))
    for value in unmapped:
        for index in np.flatnonzero(gdf["struct_id"].to_numpy() == value):
            issue.add(int(index), "unmapped_class", f"struct_id {value}")

    geoms = gdf.geometry
    invalid_mask = (~geoms.is_valid).to_numpy()
    repairable = 0
    for index in np.flatnonzero(invalid_mask):
        geom = geoms.iloc[int(index)]
        # explain_validity returns "Self-intersection[x y]"; keep only the kind.
        reason = explain_validity(geom).split("[")[0].strip()
        issue.add(int(index), "invalid_geometry", reason)
        repaired = geom.buffer(0)
        if (not repaired.is_empty) and repaired.is_valid:
            repairable += 1

    empty_mask = (geoms.is_empty.to_numpy() | (areas <= 0))
    for index in np.flatnonzero(empty_mask):
        issue.add(int(index), "empty_geometry", "")

    sliver_mask = (areas > 0) & (areas < mmu)
    for index in np.flatnonzero(sliver_mask):
        issue.add(int(index), "small_area", f"{areas[index]:.6f} m2 (MMU {mmu:.4f})")

    # normalize() makes geometrically identical rings compare equal as WKB.
    seen: Dict[bytes, int] = {}
    duplicates = 0
    for index, geom in enumerate(geoms):
        if geom.is_empty:
            continue
        key = geom.normalize().wkb
        if key in seen:
            duplicates += 1
            issue.add(index, "duplicate_geometry", f"same rings as polygon {seen[key]}")
        else:
            seen[key] = index

    row.update({
        "Invalid geometries": int(invalid_mask.sum()),
        "Repairable by buffer(0)": repairable,
        "Duplicate geometries": duplicates,
        "Empty or zero-area": int(empty_mask.sum()),
        "Slivers below MMU": int(sliver_mask.sum()),
        "Slivers below MMU (%)": (int(sliver_mask.sum()) / len(gdf) * 100.0) if len(gdf) else math.nan,
    })
    row.update(assess_overlap(gdf, labelled, issue))
    return row


def assess_overlap(gdf: gpd.GeoDataFrame, labelled: float,
                   issue: "IssueLog") -> Dict[str, object]:
    """Overlapping polygons are label ambiguity: a pixel under two polygons of
    different classes has no single reference value, and which one it receives
    is decided by shapefile row order in 2_2_structure_2D_rasterLabels.py."""
    geoms = gdf.geometry.to_numpy()
    class_ids = gdf["struct_id"].to_numpy()

    left, right = gdf.sindex.query(gdf.geometry, predicate="intersects")
    keep = left < right  # each unordered pair once, and drops self-matches
    left, right = left[keep], right[keep]

    pairs = 0
    cross_area = 0.0
    for i, j in zip(left, right):
        a, b = geoms[i], geoms[j]
        if not a.is_valid:
            a = a.buffer(0)
        if not b.is_valid:
            b = b.buffer(0)
        try:
            overlap = a.intersection(b).area
        except Exception:
            continue
        if overlap <= 0:
            continue  # shared edge only, not an overlap
        pairs += 1
        same_class = class_ids[i] == class_ids[j]
        if not same_class:
            cross_area += overlap
        issue.add(int(i),
                  "overlap_same_class" if same_class else "overlap_different_class",
                  f"with polygon {int(j)} "
                  f"({CLASS_LABELS.get(int(class_ids[i]), class_ids[i])} / "
                  f"{CLASS_LABELS.get(int(class_ids[j]), class_ids[j])}); "
                  f"{overlap:.6f} m2")

    return {
        "Overlapping pairs": pairs,
        "Cross-class overlap (m2)": cross_area,
        "Cross-class overlap (%)": (cross_area / labelled * 100.0) if labelled else math.nan,
    }


# --------------------------------------------------------------------------
# Preservation: IoU through rasterisation and through downsampling
# --------------------------------------------------------------------------

def iou_vector_to_raster(gdf: gpd.GeoDataFrame, raster_path: Path) -> Dict[str, object]:
    """Polygons rasterised exactly against the delivered 1 cm raster.

    The reference uses the centroid rule (all_touched=False), so the shortfall
    from 1.0 is the combined cost of the all_touched dilation and of resolving
    overlaps by draw order - the two things step 2 does to make the raster.
    """
    with rasterio.open(raster_path) as src:
        delivered = src.read(1)
        transform = src.transform
        shape = (src.height, src.width)

    reference = rasterize(
        list(zip(gdf.geometry, gdf["struct_id"])),
        out_shape=shape, transform=transform,
        fill=NODATA, all_touched=False, dtype=rasterio.uint8)

    return {f"IoU raster {name}": iou(reference == struct_id, delivered == struct_id)
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


def iou_fine_to_coarse(fine_path: Path, coarse_path: Path) -> Dict[str, object]:
    """Majority class of each 5 cm cell's 1 cm sub-pixels against the delivered
    nearest-neighbour 5 cm raster. 1.0 means the downsample lost nothing.

    Also reports the offset between the two grids and the change in total
    labelled area, which together diagnose a low IoU. A 3072 px tile at 1 cm is
    30.72 m, and 30.72 / 0.05 is not an integer, so `gdalwarp -tr 0.05` in
    `2_3_downsampling_labels.py` cannot land on the 1 cm grid: it snaps to 614
    cells and shifts the origin. Because that call also passes `-srcnodata 255`,
    nearest resampling onto the shifted grid takes the nearest *valid* source
    pixel in preference to a nodata one, which dilates every sparse class.
    """
    with rasterio.open(fine_path) as src:
        fine = src.read(1)
        fine_transform = src.transform
    with rasterio.open(coarse_path) as src:
        coarse = src.read(1)
        coarse_transform = src.transform
        coarse_h, coarse_w = src.shape

    if coarse.max() > max(STRUCT_IDS):  # delivered rasters carry the 64-68 LAS codes
        inverse = {v: k for k, v in VALUE_REMAP.items()}
        remapped = np.full(coarse.shape, NODATA, dtype=np.uint8)
        for code, struct_id in inverse.items():
            remapped[coarse == code] = struct_id
        coarse = remapped

    rows_idx = _fine_to_coarse_index(fine_transform, fine.shape[0], coarse_transform, "y")
    cols_idx = _fine_to_coarse_index(fine_transform, fine.shape[1], coarse_transform, "x")

    n_classes = len(STRUCT_IDS)
    n_cells = coarse_h * coarse_w
    cell = rows_idx[:, None] * coarse_w + cols_idx[None, :]
    in_grid = ((rows_idx[:, None] >= 0) & (rows_idx[:, None] < coarse_h)
               & (cols_idx[None, :] >= 0) & (cols_idx[None, :] < coarse_w))

    labelled = (fine != NODATA) & in_grid

    # Unlabelled sub-pixels are counted as their own category, so a cell that is
    # mostly nodata resolves to nodata. Taking the majority over labelled
    # sub-pixels only would call a cell "woody debris" on the strength of a
    # single 1 cm return in 25, which no downsampling rule would ever deliver
    # and which would depress the IoU of every sparse class.
    n_categories = n_classes + 1
    unlabelled = n_classes
    flat_cell_all = cell[in_grid].astype(np.int64).ravel()
    flat_class_all = np.where(labelled[in_grid], fine[in_grid], unlabelled).astype(np.int64)
    flat_class_all[flat_class_all > unlabelled] = unlabelled

    counts = np.bincount(flat_cell_all * n_categories + flat_class_all,
                         minlength=n_cells * n_categories).reshape(n_cells, n_categories)
    majority = np.argmax(counts, axis=1)

    delivered = coarse.reshape(-1).astype(np.int64)
    result: Dict[str, object] = {
        f"IoU 5 cm {name}": iou(majority == struct_id, delivered == struct_id)
        for struct_id, name in CLASS_LABELS.items()
    }

    expected_cells = int(np.count_nonzero(labelled)) * abs(
        (fine_transform.a * fine_transform.e) / (coarse_transform.a * coarse_transform.e))
    delivered_cells = int(np.count_nonzero(delivered <= max(STRUCT_IDS)))
    result["5 cm grid offset (cm)"] = 100.0 * max(
        abs(coarse_transform.c - fine_transform.c), abs(coarse_transform.f - fine_transform.f))
    result["Labelled area change at 5 cm (%)"] = (
        (delivered_cells - expected_cells) / expected_cells * 100.0) if expected_cells else math.nan
    return result


# --------------------------------------------------------------------------
# Plausibility: per-polygon top-of-canopy height
# --------------------------------------------------------------------------

def read_z_norm(las) -> np.ndarray:
    dims = set(las.point_format.dimension_names)
    if "Z_norm" in dims:
        return np.asarray(las["Z_norm"], dtype=np.float32)
    if "z_norm" in dims:
        return np.asarray(las["z_norm"], dtype=np.float32)
    z_raw = np.asarray(las.z, dtype=np.float32)
    return z_raw - float(np.nanpercentile(z_raw, 1))


def assess_height(gdf: gpd.GeoDataFrame, raster_path: Path, las_path: Path,
                  min_points: int, issue: "IssueLog") -> Dict[str, object]:
    """Per polygon, the P95 of Z_norm among the points inside it, against the
    height range its class implies.

    Points are assigned to polygons through a polygon-index raster on the 1 cm
    grid, which is O(points) and avoids a spatial join over millions of returns.

    A polygon carrying fewer than `min_points` returns is left out of the class
    statistics: a P95 over three points is the maximum of three points, and the
    median polygon here is 0.1 m2, so this is the difference between measuring
    canopy height and measuring where the returns happened to land.
    """
    import laspy  # optional dependency, only needed for this stage

    with rasterio.open(raster_path) as src:
        transform = src.transform
        shape = (src.height, src.width)

    index_raster = rasterize(
        ((geom, index) for index, geom in enumerate(gdf.geometry)),
        out_shape=shape, transform=transform,
        fill=NO_POLYGON, all_touched=False, dtype=rasterio.int32)

    las = laspy.read(str(las_path))
    z = read_z_norm(las)
    rows_idx, cols_idx = rasterio.transform.rowcol(
        transform, np.asarray(las.x), np.asarray(las.y))
    rows_idx, cols_idx = np.asarray(rows_idx), np.asarray(cols_idx)
    inside = ((rows_idx >= 0) & (rows_idx < shape[0])
              & (cols_idx >= 0) & (cols_idx < shape[1]))

    polygon_of_point = index_raster[rows_idx[inside], cols_idx[inside]]
    z_inside = z[inside]
    hit = (polygon_of_point != NO_POLYGON) & np.isfinite(z_inside)
    polygon_of_point, z_inside = polygon_of_point[hit], z_inside[hit]

    canopy = np.full(len(gdf), np.nan)
    n_points = np.zeros(len(gdf), dtype=np.int64)
    if polygon_of_point.size:
        order = np.argsort(polygon_of_point, kind="stable")
        sorted_polygons = polygon_of_point[order]
        sorted_z = z_inside[order]
        boundaries = np.flatnonzero(np.diff(sorted_polygons)) + 1
        for group, values in zip(np.split(sorted_polygons, boundaries),
                                 np.split(sorted_z, boundaries)):
            if values.size:
                n_points[group[0]] = values.size
                canopy[group[0]] = float(np.percentile(values, CANOPY_PERCENTILE))

    thin = n_points < min_points
    for index in np.flatnonzero(thin):
        issue.add(int(index), "too_few_lidar_points",
                  f"{n_points[index]} returns, need {min_points} for a "
                  f"P{CANOPY_PERCENTILE}; excluded from the height columns")

    class_ids = gdf["struct_id"].to_numpy()
    row: Dict[str, object] = {
        "Polygons with too few returns (%)": float(np.mean(thin) * 100.0) if len(gdf) else math.nan,
    }
    for struct_id, name in CLASS_LABELS.items():
        selection = (class_ids == struct_id) & np.isfinite(canopy) & ~thin
        values = canopy[selection]
        low, high = HEIGHT_RANGES[struct_id]
        outside = (values < low) | (values > high)
        row[f"P95 Znorm {name} (m)"] = float(np.median(values)) if values.size else math.nan
        row[f"Height outliers {name} (%)"] = \
            float(np.mean(outside) * 100.0) if values.size else math.nan
        for index in np.flatnonzero(selection)[outside]:
            row_index = int(index)
            issue.add(row_index, "height_implausible",
                      f"{name}: P{CANOPY_PERCENTILE} Z_norm {canopy[row_index]:.3f} m "
                      f"outside {low:.2f}-{high:.2f} m")
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


def mean_ignore_nan(values: Iterable[object]) -> float:
    numeric = np.asarray(
        [v for v in values if isinstance(v, (int, float, np.integer, np.floating))],
        dtype=np.float64)
    if numeric.size == 0 or np.all(np.isnan(numeric)):
        return math.nan
    return float(np.nanmean(numeric))


def average_row(rows: Sequence[Dict[str, object]], site: str,
                condition: str) -> Dict[str, object]:
    """Across-tile means. Non-numeric columns are blanked."""
    average: Dict[str, object] = {"Site": site, "Site condition": condition, "Tile": "Average"}
    for column in SUMMARY_COLUMNS:
        if column in average:
            continue
        values = [r.get(column) for r in rows]
        average[column] = mean_ignore_nan(values) if any(
            isinstance(v, (int, float, np.integer, np.floating)) for v in values) else ""
    return average


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


def write_report(path: Path, site: str, condition: str, n_tiles: int,
                 average: Dict[str, object], issues: IssueLog, mmu: float,
                 min_points: int) -> None:
    """The paragraph as it would appear in a response to the reviewer, generated
    from the numbers so it stays correct when the script is re-run elsewhere."""
    def value(column: str) -> float:
        raw = average.get(column)
        return float(raw) if isinstance(raw, (int, float, np.integer, np.floating)) else math.nan

    def per_class(prefix: str) -> Dict[str, float]:
        return {name: value(f"{prefix} {name}") for name in CLASS_LABELS.values()}

    present = [name for name in CLASS_LABELS.values()
               if not math.isnan(value(f"IoU raster {name}"))
               or not math.isnan(value(f"P95 Znorm {name} (m)"))]
    if not present:
        present = list(CLASS_LABELS.values())
    absent = [name for name in CLASS_LABELS.values() if name not in present]

    lines = [
        f"# 2D reference label quality - {site}",
        "",
        f"{condition} condition, {n_tiles} tiles, means across tiles. "
        f"Generated by `label_qa.py`.",
        "",
        "## Annotation",
        "",
        f"- **{value('Labelled area (%)'):.1f}% of tile area carries a reference value** "
        f"({value('Labelled area (m2)'):.1f} m2 of {value('Tile area (m2)'):.0f} m2), "
        f"from {value('Polygons'):.0f} polygons per tile of median area "
        f"{value('Median polygon area (m2)'):.3f} m2. The annotation is sparse by design, "
        f"not wall-to-wall.",
        f"- {value('Invalid geometries'):.1f} invalid geometries per tile, "
        f"{value('Repairable by buffer(0)'):.1f} of them repairable by `buffer(0)`; "
        f"{value('Duplicate geometries'):.1f} duplicates; "
        f"{value('Empty or zero-area'):.1f} zero-area; "
        f"{value('Slivers below MMU (%)'):.2f}% below the {mmu:.4f} m2 minimum mapping unit.",
        f"- {value('Overlapping pairs'):.1f} overlapping polygon pairs per tile. Cross-class "
        f"overlap - pixels with no single reference value - is "
        f"{value('Cross-class overlap (m2)'):.3f} m2, "
        f"**{value('Cross-class overlap (%)'):.3f}% of labelled area**.",
        "",
        "## Preservation (IoU)",
        "",
        "How much of each class survives the two lossy steps between the polygons and "
        "the tensor a model sees. 1.000 is lossless.",
        "",
        "| Class | IoU vector to 1 cm raster | IoU 1 cm to 5 cm |",
        "|---|---|---|",
    ]
    raster_iou, coarse_iou = per_class("IoU raster"), per_class("IoU 5 cm")
    for name in present:
        lines.append(f"| {name} | {format_value(raster_iou[name]) or 'n/a'} | "
                     f"{format_value(coarse_iou[name]) or 'n/a'} |")

    broken = [name for name in present
              if not math.isnan(raster_iou[name]) and raster_iou[name] < RASTER_IOU_ALARM]
    if broken:
        lines += [
            "",
            f"**{', '.join(broken)}: IoU below {RASTER_IOU_ALARM:.2f} against the delivered 1 cm "
            f"raster.** `all_touched` costs a class a few percent, never this. An IoU near zero "
            f"means the shapefiles on disk are not the version the rasters were built from, so "
            f"re-running `2_2_structure_2D_rasterLabels.py` over them would not reproduce the "
            f"reference the experiments used. Check the polygons of that class before reporting "
            f"anything from it.",
        ]

    offset = value("5 cm grid offset (cm)")
    area_change = value("Labelled area change at 5 cm (%)")
    if not math.isnan(offset) and offset > 0:
        lines += [
            "",
            f"The 5 cm grid sits {offset:.2f} cm from the 1 cm grid on average, and the 5 cm "
            f"raster carries {area_change:+.1f}% of the labelled area the 1 cm raster implies. "
            f"A 3072 px tile at 1 cm is 30.72 m and 30.72 / 0.05 is not an integer, so "
            f"`gdalwarp -tr 0.05` in `2_3_downsampling_labels.py` cannot land on the 1 cm grid; "
            f"combined with `-srcnodata 255`, nearest resampling onto the shifted grid takes the "
            f"nearest *valid* pixel over a nodata one and dilates every sparse class. Per-tile "
            f"values are in the summary CSV: tiles whose grids happen to align show an offset of "
            f"0 and an IoU near 1.",
        ]

    with_height = [name for name in present
                   if not math.isnan(value(f"P95 Znorm {name} (m)"))]
    if with_height:
        lines += [
            "",
            "## Plausibility (top-of-canopy height)",
            "",
            f"Per polygon, the P{CANOPY_PERCENTILE} of `Z_norm` among the LiDAR returns inside "
            f"it, against the range its class implies. LiDAR height was never used to make the "
            f"2D labels, so this is a semi-independent check on class meaning.",
            "",
            f"{value('Polygons with too few returns (%)'):.1f}% of polygons carry fewer than "
            f"{min_points} returns and are excluded here, since a P{CANOPY_PERCENTILE} over a "
            f"handful of points measures where the returns landed rather than canopy height.",
            "",
            f"| Class | Expected range (m) | Median P{CANOPY_PERCENTILE} Z_norm (m) "
            f"| Polygons outside (%) |",
            "|---|---|---|---|",
        ]
        for name in with_height:
            struct_id = next(k for k, v in CLASS_LABELS.items() if v == name)
            low, high = HEIGHT_RANGES[struct_id]
            lines.append(f"| {name} | {low:.2f}-{high:.2f} | "
                         f"{format_value(value(f'P95 Znorm {name} (m)'))} | "
                         f"{format_value(value(f'Height outliers {name} (%)'))} |")
    else:
        lines += [
            "",
            "## Plausibility (top-of-canopy height)",
            "",
            "Not computed: no Z-normalised LAS was matched to these tiles (`--las-dir`).",
        ]

    tally = issues.counts()
    lines += ["", "## Issues logged", "",
              f"{len(issues.rows)} individual problems across {n_tiles} tiles, "
              f"listed per polygon in `{site}_qa_issues.csv`.", ""]
    if tally:
        lines += ["| Issue | Count |", "|---|---|"]
        lines += [f"| {name} | {count} |" for name, count in tally.items()]
    else:
        lines.append("None.")

    if absent:
        lines += ["", f"**Not represented in this site:** {', '.join(absent)} "
                      f"(0 polygons, so every per-class column above is blank for them)."]
    lines += [
        "",
        "## Limitation",
        "",
        "These measures characterise internal consistency and preservation. They cannot "
        "measure annotation *reproducibility* - whether a second annotator would draw the "
        "same thing. That needs a repeat delineation of a tile subset, compared with "
        "`compare_reannotation.py`.",
        "",
    ]

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


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
    las_map = scan_tile_map(args.las_dir, args.las_glob)

    tile_ids = sorted(shp_map, key=int)
    if args.tiles:
        wanted = {t.strip() for t in args.tiles.split(",") if t.strip()}
        tile_ids = [t for t in tile_ids if t in wanted]
    if not tile_ids:
        raise SystemExit(f"No shapefiles matched {args.shp_glob} in {shp_dir}")

    for enabled, name, hint in (
        (bool(raster_map), "IoU raster", "--raster-dir"),
        (bool(raster_map) and bool(coarse_map), "IoU 5 cm", "--raster-dir and --downsampled-dir"),
        (bool(raster_map) and bool(las_map), "height columns", "--raster-dir and --las-dir"),
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
        las_path = las_map.get(tile_id)

        row: Dict[str, object] = {"Site": args.site, "Site condition": condition,
                                  "Tile": tile_id}
        row.update(assess_annotation(gdf, fine_path, args.mmu, issues))

        if fine_path is None:
            issues.add("", "missing_raster", "no 1 cm label raster for this tile")
        else:
            row.update(iou_vector_to_raster(gdf, fine_path))
            for struct_id, name in CLASS_LABELS.items():
                score = row[f"IoU raster {name}"]
                if isinstance(score, float) and not math.isnan(score) \
                        and score < RASTER_IOU_ALARM:
                    n_class = int(np.count_nonzero(gdf["struct_id"].to_numpy() == struct_id))
                    issues.add("", "raster_disagrees_with_shapefile",
                               f"{name}: IoU {score:.3f} between the {n_class} polygons on "
                               f"disk and the delivered 1 cm raster")
            if coarse_path is None:
                issues.add("", "missing_downsampled", "no 5 cm label raster for this tile")
            else:
                row.update(iou_fine_to_coarse(fine_path, coarse_path))
                offset = float(row["5 cm grid offset (cm)"])
                if offset > 0:
                    issues.add("", "grid_misaligned_5cm",
                               f"5 cm grid offset {offset:.2f} cm from the 1 cm grid; "
                               f"labelled area changes "
                               f"{float(row['Labelled area change at 5 cm (%)']):+.1f}%")
            if las_path is None:
                if las_map:
                    issues.add("", "missing_las", "no Z-normalised LAS for this tile")
            else:
                row.update(assess_height(gdf, fine_path, las_path,
                                         args.min_points, issues))

        summary.append(row)

    if not summary:
        raise SystemExit("No tiles produced a summary row.")

    summary.append(average_row(summary, args.site, condition))
    summary_path = output_dir / f"{args.site}_qa_summary.csv"
    issues_path = output_dir / f"{args.site}_qa_issues.csv"
    report_path = output_dir / f"{args.site}_qa_report.md"

    write_csv(summary_path, SUMMARY_COLUMNS, summary)
    write_csv(issues_path, ISSUE_COLUMNS, issues.rows)
    write_report(report_path, args.site, condition, len(summary) - 1,
                 summary[-1], issues, args.mmu, args.min_points)

    print(f"\nTiles assessed: {len(summary) - 1}")
    print(f"Issues logged:  {len(issues.rows)}")
    for name, count in issues.counts().items():
        print(f"  {name:28s} {count}")
    for path in (summary_path, issues_path, report_path):
        print(f"  Wrote: {path}")


if __name__ == "__main__":
    main()
