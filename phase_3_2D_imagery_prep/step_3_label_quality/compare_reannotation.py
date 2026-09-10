#!/usr/bin/env python3
"""Agreement between two independent delineations of the same tiles.

This is the direct measure of 2D reference-label quality: a subset of tiles is
digitised a second time (a second annotator, or the same annotator re-digitising
blind after a delay), and this script reports how much the two versions agree.

Reported per tile and as across-tile means:

- OA, Cohen's kappa, per-class IoU and F1 over all labelled cells
- the same metrics with an N-cell buffer around class boundaries excluded, which
  separates *classification* disagreement (what is this plant) from *delineation*
  disagreement (exactly where its edge falls)
- per-class area agreement and the full confusion matrix

Both inputs are annotation shapefiles carrying `struct_id`, rasterised onto the
grid of the corresponding reference raster so the comparison is pixel-exact.
Rasterisation deliberately uses all_touched=True to match
2_2_structure_2D_rasterLabels.py, so the number describes the labels as the
experiments actually consume them.

Example
-------
python compare_reannotation.py \
    --shp-dir-a  /path/annotation_shp \
    --shp-dir-b  /path/annotation_shp_reannotation \
    --raster-dir /path/annotation_raster \
    --site-condition Medium \
    --output-dir /path/validation/label_quality
"""

from __future__ import annotations

import argparse
import csv
import math
import re
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.features import rasterize
from scipy.ndimage import binary_dilation


CLASS_LABELS = {0: "Forb", 1: "Woody debris", 2: "Grass", 3: "Shrub", 4: "Mallee"}
STRUCT_IDS = tuple(CLASS_LABELS)
NODATA = 255

AGREEMENT_COLUMNS = [
    "Tile", "Site condition", "Scope", "Compared cells",
    "OA (%)", "Kappa",
] + [f"IoU {name}" for name in CLASS_LABELS.values()] \
  + [f"F1 {name}" for name in CLASS_LABELS.values()]

AREA_COLUMNS = [
    "Tile", "Site condition", "Class",
    "Polygons A", "Polygons B", "Area A (m2)", "Area B (m2)",
    "Area difference (m2)", "Area difference (%)",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Agreement between two independent 2D label delineations.")
    parser.add_argument("--shp-dir-a", type=Path, required=True,
                        help="Directory of the original annotation shapefiles.")
    parser.add_argument("--shp-dir-b", type=Path, required=True,
                        help="Directory of the repeat annotation shapefiles.")
    parser.add_argument("--raster-dir", type=Path, required=True,
                        help="Directory of reference rasters defining the comparison grid.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--site-condition", default="")
    parser.add_argument("--shp-glob", default="*.shp")
    parser.add_argument("--raster-glob", default="*.tif")
    parser.add_argument("--boundary-buffer", type=int, default=5,
                        help="Cells excluded either side of a class boundary for the "
                             "interior-only scope. Default 5 (5 cm at 1 cm pixels).")
    parser.add_argument("--all-touched", action="store_true", default=True,
                        help="Rasterise with all_touched, matching step 2. Default on.")
    parser.add_argument("--no-all-touched", dest="all_touched", action="store_false")
    return parser.parse_args()


def detect_tile_id(path: Path) -> str | None:
    match = re.search(r"(\d+)$", path.stem)
    return match.group(1) if match else None


def scan_tile_map(directory: Path, pattern: str) -> Dict[str, Path]:
    tile_map: Dict[str, Path] = {}
    for path in sorted(directory.glob(pattern)):
        if not path.is_file():
            continue
        tile_id = detect_tile_id(path)
        if tile_id is None or tile_id in tile_map:
            continue
        tile_map[tile_id] = path
    return tile_map


def rasterise_annotations(shp_path: Path, transform: rasterio.Affine,
                          shape: tuple[int, int], all_touched: bool) -> tuple[np.ndarray, gpd.GeoDataFrame]:
    gdf = gpd.read_file(shp_path)
    if "struct_id" not in gdf.columns:
        raise KeyError(f"{shp_path.name} has no 'struct_id' column")
    gdf = gdf[gdf.geometry.notna()].copy()
    gdf["struct_id"] = gdf["struct_id"].astype(int)
    if gdf.empty:
        return np.full(shape, NODATA, dtype=np.uint8), gdf
    array = rasterize(
        list(zip(gdf.geometry, gdf["struct_id"])),
        out_shape=shape, transform=transform,
        fill=NODATA, all_touched=all_touched, dtype=rasterio.uint8,
    )
    return array, gdf


def boundary_mask(array: np.ndarray, buffer_cells: int) -> np.ndarray:
    """Cells within `buffer_cells` of any change in class value, including the
    edge against nodata."""
    if buffer_cells <= 0:
        return np.zeros(array.shape, dtype=bool)
    edges = np.zeros(array.shape, dtype=bool)
    edges[:-1, :] |= array[:-1, :] != array[1:, :]
    edges[1:, :] |= array[:-1, :] != array[1:, :]
    edges[:, :-1] |= array[:, :-1] != array[:, 1:]
    edges[:, 1:] |= array[:, :-1] != array[:, 1:]
    structure = np.ones((3, 3), dtype=bool)
    return binary_dilation(edges, structure=structure, iterations=buffer_cells)


def agreement_metrics(a: np.ndarray, b: np.ndarray) -> Dict[str, object]:
    """OA, kappa and per-class IoU/F1 over cells labelled in both versions.

    Cells labelled in only one version are omission/commission of the annotation
    itself; they are counted in the area table rather than folded into OA, where
    they would be indistinguishable from a class confusion.
    """
    metrics: Dict[str, object] = {"Compared cells": int(a.size), "OA (%)": math.nan, "Kappa": math.nan}
    for name in CLASS_LABELS.values():
        metrics[f"IoU {name}"] = math.nan
        metrics[f"F1 {name}"] = math.nan
    if a.size == 0:
        return metrics

    metrics["OA (%)"] = float(np.mean(a == b) * 100.0)

    n = a.size
    p_observed = float(np.mean(a == b))
    p_expected = 0.0
    for struct_id in STRUCT_IDS:
        p_expected += (np.count_nonzero(a == struct_id) / n) * (np.count_nonzero(b == struct_id) / n)
    metrics["Kappa"] = float((p_observed - p_expected) / (1.0 - p_expected)) if p_expected < 1.0 else math.nan

    for struct_id, name in CLASS_LABELS.items():
        in_a = a == struct_id
        in_b = b == struct_id
        tp = int(np.count_nonzero(in_a & in_b))
        fp = int(np.count_nonzero(~in_a & in_b))
        fn = int(np.count_nonzero(in_a & ~in_b))
        union = tp + fp + fn
        metrics[f"IoU {name}"] = float(tp / union) if union else math.nan
        metrics[f"F1 {name}"] = float(2 * tp / (2 * tp + fp + fn)) if union else math.nan
    return metrics


def confusion_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    n = len(STRUCT_IDS)
    if a.size == 0:
        return np.zeros((n, n), dtype=np.int64)
    return np.bincount(a.astype(np.int64) * n + b.astype(np.int64),
                       minlength=n * n).reshape(n, n)


def mean_ignore_nan(values: Iterable[object]) -> float:
    numeric = np.asarray(
        [v for v in values if isinstance(v, (int, float, np.integer, np.floating))],
        dtype=np.float64)
    if numeric.size == 0 or np.all(np.isnan(numeric)):
        return math.nan
    return float(np.nanmean(numeric))


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


def write_markdown(path: Path, fieldnames: Sequence[str], rows: Sequence[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        handle.write("| " + " | ".join(fieldnames) + " |\n")
        handle.write("| " + " | ".join(["---"] * len(fieldnames)) + " |\n")
        for row in rows:
            handle.write("| " + " | ".join(format_value(row.get(f)) for f in fieldnames) + " |\n")


def write_confusion(path: Path, matrix: np.ndarray, site: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    names = list(CLASS_LABELS.values())
    totals = matrix.sum(axis=1)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow([f"Site condition: {site}"])
        writer.writerow(["Rows = annotation A, columns = annotation B, cells"])
        writer.writerow(["A \\ B"] + names + ["Total A", "Producer agreement (%)"])
        for i, name in enumerate(names):
            correct = matrix[i, i]
            pct = correct / totals[i] * 100.0 if totals[i] else float("nan")
            writer.writerow([name] + [int(v) for v in matrix[i]] +
                            [int(totals[i]), format_value(float(pct))])
        writer.writerow(["Total B"] + [int(v) for v in matrix.sum(axis=0)] +
                        [int(matrix.sum()), ""])


def main() -> None:
    args = parse_args()
    site = args.site_condition.strip()

    map_a = scan_tile_map(args.shp_dir_a.resolve(), args.shp_glob)
    map_b = scan_tile_map(args.shp_dir_b.resolve(), args.shp_glob)
    map_r = scan_tile_map(args.raster_dir.resolve(), args.raster_glob)

    tile_ids = sorted(set(map_a) & set(map_b) & set(map_r), key=int)
    if not tile_ids:
        raise SystemExit(
            "No tiles present in all three of --shp-dir-a, --shp-dir-b and --raster-dir.")

    agreement_rows: List[Dict[str, object]] = []
    area_rows: List[Dict[str, object]] = []
    total_confusion = np.zeros((len(STRUCT_IDS), len(STRUCT_IDS)), dtype=np.int64)

    for index, tile_id in enumerate(tile_ids, 1):
        print(f"[{index}/{len(tile_ids)}] Tile {tile_id}", flush=True)
        with rasterio.open(map_r[tile_id]) as src:
            transform = src.transform
            shape = src.shape
            pixel_area = abs(transform.a * transform.e)

        array_a, gdf_a = rasterise_annotations(map_a[tile_id], transform, shape, args.all_touched)
        array_b, gdf_b = rasterise_annotations(map_b[tile_id], transform, shape, args.all_touched)

        both = (array_a != NODATA) & (array_b != NODATA)
        near_edge = boundary_mask(array_a, args.boundary_buffer) | \
                    boundary_mask(array_b, args.boundary_buffer)

        for scope, selection in (("All labelled cells", both),
                                 (f"Interior only ({args.boundary_buffer}-cell buffer)",
                                  both & ~near_edge)):
            row: Dict[str, object] = {"Tile": tile_id, "Site condition": site, "Scope": scope}
            row.update(agreement_metrics(array_a[selection], array_b[selection]))
            agreement_rows.append(row)

        total_confusion += confusion_matrix(array_a[both], array_b[both])

        for struct_id, name in CLASS_LABELS.items():
            area_a = float(gdf_a.geometry[gdf_a["struct_id"] == struct_id].area.sum()) if not gdf_a.empty else 0.0
            area_b = float(gdf_b.geometry[gdf_b["struct_id"] == struct_id].area.sum()) if not gdf_b.empty else 0.0
            denom = (area_a + area_b) / 2.0
            area_rows.append({
                "Tile": tile_id,
                "Site condition": site,
                "Class": name,
                "Polygons A": int((gdf_a["struct_id"] == struct_id).sum()) if not gdf_a.empty else 0,
                "Polygons B": int((gdf_b["struct_id"] == struct_id).sum()) if not gdf_b.empty else 0,
                "Area A (m2)": area_a,
                "Area B (m2)": area_b,
                "Area difference (m2)": area_b - area_a,
                # Symmetric percentage difference: neither version is "truth".
                "Area difference (%)": ((area_b - area_a) / denom * 100.0) if denom > 0 else math.nan,
            })
        _ = pixel_area  # grid metadata kept for provenance; areas come from the vectors

    scopes: List[str] = []
    for row in agreement_rows:
        if row["Scope"] not in scopes:
            scopes.append(str(row["Scope"]))
    agreement_avg: List[Dict[str, object]] = []
    for scope in scopes:
        group = [r for r in agreement_rows if r["Scope"] == scope]
        avg: Dict[str, object] = {"Tile": "Average", "Site condition": site, "Scope": scope}
        for column in AGREEMENT_COLUMNS:
            if column in avg:
                continue
            avg[column] = mean_ignore_nan(r.get(column) for r in group)
        agreement_avg.append(avg)

    area_avg: List[Dict[str, object]] = []
    for struct_id, name in CLASS_LABELS.items():
        group = [r for r in area_rows if r["Class"] == name]
        if not group:
            continue
        avg = {"Tile": "Average", "Site condition": site, "Class": name}
        for column in AREA_COLUMNS:
            if column in avg:
                continue
            avg[column] = mean_ignore_nan(r.get(column) for r in group)
        area_avg.append(avg)

    output_dir = args.output_dir.resolve()
    outputs = [
        (output_dir / "r1_reannotation_agreement_per_tile.csv", AGREEMENT_COLUMNS, agreement_rows),
        (output_dir / "r1_reannotation_agreement_per_tile.md", AGREEMENT_COLUMNS, agreement_rows),
        (output_dir / "r1_reannotation_agreement_average.csv", AGREEMENT_COLUMNS, agreement_avg),
        (output_dir / "r1_reannotation_agreement_average.md", AGREEMENT_COLUMNS, agreement_avg),
        (output_dir / "r2_reannotation_area_per_tile.csv", AREA_COLUMNS, area_rows),
        (output_dir / "r2_reannotation_area_average.csv", AREA_COLUMNS, area_avg),
        (output_dir / "r2_reannotation_area_average.md", AREA_COLUMNS, area_avg),
    ]
    for path, fieldnames, rows in outputs:
        if path.suffix == ".csv":
            write_csv(path, fieldnames, rows)
        else:
            write_markdown(path, fieldnames, rows)

    confusion_path = output_dir / "r3_reannotation_confusion_matrix.csv"
    write_confusion(confusion_path, total_confusion, site)

    print(f"\nTiles compared: {len(tile_ids)}")
    print(f"Output directory: {output_dir}")
    for path, _, _ in outputs:
        print(f"  Wrote: {path}")
    print(f"  Wrote: {confusion_path}")


if __name__ == "__main__":
    main()
