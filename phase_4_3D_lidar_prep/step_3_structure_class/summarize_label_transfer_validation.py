#!/usr/bin/env python3
"""Batch summaries for the 2D-to-3D label-transfer validation workflow.

This extracts the tile-level validation logic from the Tile 109 notebook and
writes manuscript-ready summaries across all matched tiles:

- Table 5: label-agreement metrics between projected and refined labels
- Table 6: height-plausibility metrics for refined labels
- TreeID: Step 4 watershed crown counts and Step 9 treeID instance statistics
         (skipped when --skip-treeid is passed)

For each table the script writes both per-tile outputs and across-tile means.
"""

from __future__ import annotations

import argparse
import csv
import math
import re
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import laspy
import numpy as np
import rasterio

from fvc3dstruct.height_stats import DEFAULT_HEIGHT_RANGES
from fvc3dstruct.watershed import get_mallee_centroids_watershed
from make_label_transfer_validation_figure import filter_points_by_class
from make_label_transfer_validation_figure import get_las_z_norm
from make_label_transfer_validation_figure import read_las_points
from make_label_transfer_validation_figure import read_mask
from make_label_transfer_validation_figure import validation_z


CLASS_LABELS = {
    64: "Forb",
    65: "Woody debris",
    66: "Hummock",
    67: "Shrub",
    68: "Mallee",
}

HEIGHT_RANGES = DEFAULT_HEIGHT_RANGES

VALIDATION_CLASS_CODES = (64, 65, 66, 67, 68)

DEFAULT_MASK_DIR = Path(
    "inputs/masks/tiles_3072/raw/structure_class/annotation_raster/downsample_5cm_nearest"
)
DEFAULT_PROJECTED_DIR = Path("lidar/clipped/mask/coreg/icp")
DEFAULT_REFINED_DIR = Path("lidar/clipped/3d_structure_mask_malleid")
DEFAULT_OUTPUT_DIR = Path("lidar/clipped/validation")

TABLE5_COLUMNS = [
    "Tile",
    "Site condition",
    "OA (%)",
    "IoU Forb",
    "IoU Woody debris",
    "IoU Hummock",
    "IoU Shrub",
    "IoU Mallee",
    "Common occupied cells",
    "Projected occupied cells",
    "Refined occupied cells",
]

TABLE6_COLUMNS = [
    "Tile",
    "Site condition",
    "Class",
    "Points",
    "Configured min Znorm (m)",
    "Configured max Znorm (m)",
    "Median Znorm (m)",
    "IQR (m)",
    "Outside configured range (%)",
]

TREEID_COLUMNS = [
    "Tile",
    "Site condition",
    "Watershed crowns (Step 4)",
    "Points in crowns (Step 4)",
    "Final non-zero treeIDs (Step 9)",
    "Mallee points with treeID",
    "Crown count delta (Step 9 - Step 4)",
    "Crown retention (%)",
    "Mean points per watershed crown",
    "Mean points per treeID",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize label-transfer validation across all matched tiles."
    )
    parser.add_argument(
        "--site-root",
        type=Path,
        help=(
            "Site root used to infer default mask/projected/refined/output "
            "directories."
        ),
    )
    parser.add_argument("--mask-dir", type=Path, help="Directory containing mask rasters.")
    parser.add_argument(
        "--projected-dir",
        type=Path,
        help="Directory containing projected label LAS tiles.",
    )
    parser.add_argument(
        "--refined-dir",
        type=Path,
        help="Directory containing refined label LAS tiles.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Directory for CSV/Markdown summaries. Defaults under --site-root.",
    )
    parser.add_argument(
        "--site-condition",
        default="",
        help=(
            "Value written to the 'Site condition' column. If omitted, the "
            "site-root directory name is used when available."
        ),
    )
    parser.add_argument(
        "--mask-glob",
        default="*.tif",
        help="Glob pattern used to scan mask rasters. Default: *.tif",
    )
    parser.add_argument(
        "--znorm-dir",
        type=Path,
        help=(
            "Directory containing Z-normalized LAS tiles used for watershed computation. "
            "When omitted, the script infers the path from the projected LAS location "
            "(4-level parent / raster / z_norm / csf)."
        ),
    )
    parser.add_argument(
        "--skip-treeid",
        action="store_true",
        help="Skip the watershed / treeID summary table (faster when that step is not needed).",
    )
    return parser.parse_args()


def resolve_dirs(args: argparse.Namespace) -> tuple[Path, Path, Path, Path, str]:
    site_root = args.site_root.resolve() if args.site_root else None

    def _resolve(path_arg: Path | None, default_rel: Path | None) -> Path | None:
        if path_arg is not None:
            return path_arg.resolve()
        if site_root is not None and default_rel is not None:
            return (site_root / default_rel).resolve()
        return None

    mask_dir = _resolve(args.mask_dir, DEFAULT_MASK_DIR)
    projected_dir = _resolve(args.projected_dir, DEFAULT_PROJECTED_DIR)
    refined_dir = _resolve(args.refined_dir, DEFAULT_REFINED_DIR)
    output_dir = _resolve(args.output_dir, DEFAULT_OUTPUT_DIR)

    missing = []
    if mask_dir is None:
        missing.append("--mask-dir")
    if projected_dir is None:
        missing.append("--projected-dir")
    if refined_dir is None:
        missing.append("--refined-dir")
    if output_dir is None:
        missing.append("--output-dir")
    if missing:
        joined = ", ".join(missing)
        raise SystemExit(
            f"Missing required paths: {joined}. Provide them directly or pass --site-root."
        )

    site_condition = args.site_condition.strip()
    if not site_condition and site_root is not None:
        site_condition = site_root.name

    return mask_dir, projected_dir, refined_dir, output_dir, site_condition


def detect_tile_id(path: Path) -> str | None:
    match = re.search(r"(\d+)$", path.stem)
    if match:
        return match.group(1)
    return None


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


def aggregate_majority_by_mask_cell(
    points: Dict[str, np.ndarray | None],
    transform: rasterio.Affine,
    shape: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    x_values = np.asarray(points["x"])
    y_values = np.asarray(points["y"])
    class_values = np.asarray(points["class"])
    keep = np.isin(class_values, VALIDATION_CLASS_CODES)
    x_values = x_values[keep]
    y_values = y_values[keep]
    class_values = class_values[keep]

    rows, cols = rasterio.transform.rowcol(transform, x_values, y_values)
    rows = np.asarray(rows)
    cols = np.asarray(cols)
    valid = (rows >= 0) & (rows < shape[0]) & (cols >= 0) & (cols < shape[1])
    rows = rows[valid]
    cols = cols[valid]
    class_values = class_values[valid]

    if len(class_values) == 0:
        return np.array([], dtype=np.int64), np.array([], dtype=np.int16)

    cell_ids = rows.astype(np.int64) * shape[1] + cols.astype(np.int64)
    order = np.argsort(cell_ids)
    cell_ids = cell_ids[order]
    class_values = class_values[order]

    unique_cells: List[int] = []
    majority_classes: List[int] = []
    start = 0
    for stop in range(1, len(cell_ids) + 1):
        if stop < len(cell_ids) and cell_ids[stop] == cell_ids[start]:
            continue
        unique_cells.append(int(cell_ids[start]))
        values = class_values[start:stop]
        codes, counts = np.unique(values, return_counts=True)
        majority_classes.append(int(codes[np.argmax(counts)]))
        start = stop

    return np.asarray(unique_cells, dtype=np.int64), np.asarray(majority_classes, dtype=np.int16)


def compute_table5_row(
    tile_id: str,
    site_condition: str,
    mask_path: Path,
    projected_las_path: Path,
    refined_las_path: Path,
) -> Dict[str, object]:
    validation_mask, validation_transform = read_mask(str(mask_path))

    projected_points = filter_points_by_class(
        read_las_points(str(projected_las_path)),
        VALIDATION_CLASS_CODES,
    )
    refined_points = filter_points_by_class(
        read_las_points(str(refined_las_path)),
        VALIDATION_CLASS_CODES,
    )

    projected_cells, projected_classes = aggregate_majority_by_mask_cell(
        projected_points,
        validation_transform,
        validation_mask.shape,
    )
    refined_cells, refined_classes = aggregate_majority_by_mask_cell(
        refined_points,
        validation_transform,
        validation_mask.shape,
    )

    row: Dict[str, object] = {
        "Tile": tile_id,
        "Site condition": site_condition,
        "OA (%)": math.nan,
        "Common occupied cells": 0,
        "Projected occupied cells": int(len(projected_cells)),
        "Refined occupied cells": int(len(refined_cells)),
    }
    for code in VALIDATION_CLASS_CODES:
        row[f"IoU {CLASS_LABELS[code]}"] = math.nan

    common_cells = np.intersect1d(projected_cells, refined_cells)
    row["Common occupied cells"] = int(len(common_cells))
    if len(common_cells) == 0:
        return row

    projected_lookup = dict(zip(projected_cells.tolist(), projected_classes.tolist()))
    refined_lookup = dict(zip(refined_cells.tolist(), refined_classes.tolist()))
    projected_common = np.asarray(
        [projected_lookup[int(cell)] for cell in common_cells],
        dtype=np.int16,
    )
    refined_common = np.asarray(
        [refined_lookup[int(cell)] for cell in common_cells],
        dtype=np.int16,
    )

    row["OA (%)"] = float(np.mean(projected_common == refined_common) * 100.0)

    for code in VALIDATION_CLASS_CODES:
        true_mask = projected_common == code
        pred_mask = refined_common == code
        tp = int(np.sum(true_mask & pred_mask))
        fp = int(np.sum(~true_mask & pred_mask))
        fn = int(np.sum(true_mask & ~pred_mask))
        denom = tp + fp + fn
        row[f"IoU {CLASS_LABELS[code]}"] = float(tp / denom) if denom else math.nan

    return row


def class_height_values(points: Dict[str, np.ndarray | None]) -> Dict[int, np.ndarray]:
    z_values = validation_z(points).astype(np.float32)
    class_values = np.asarray(points["class"])
    return {
        code: z_values[class_values == code]
        for code in VALIDATION_CLASS_CODES
    }


def compute_table6_rows(
    tile_id: str,
    site_condition: str,
    refined_las_path: Path,
) -> List[Dict[str, object]]:
    refined_points = filter_points_by_class(
        read_las_points(str(refined_las_path)),
        VALIDATION_CLASS_CODES,
    )
    height_data = class_height_values(refined_points)

    rows: List[Dict[str, object]] = []
    for code in VALIDATION_CLASS_CODES:
        values = np.asarray(height_data[code], dtype=np.float32)
        zmin, zmax = HEIGHT_RANGES[code]
        row: Dict[str, object] = {
            "Tile": tile_id,
            "Site condition": site_condition,
            "Class": CLASS_LABELS[code],
            "Points": int(len(values)),
            "Configured min Znorm (m)": float(zmin),
            "Configured max Znorm (m)": float(zmax),
            "Median Znorm (m)": math.nan,
            "IQR (m)": math.nan,
            "Outside configured range (%)": math.nan,
        }
        if len(values) > 0:
            q1, q3 = np.percentile(values, [25, 75])
            out_of_range = ((values < zmin) | (values > zmax)).mean() * 100.0
            row["Median Znorm (m)"] = float(np.median(values))
            row["IQR (m)"] = float(q3 - q1)
            row["Outside configured range (%)"] = float(out_of_range)
        rows.append(row)
    return rows


def _resolve_znorm_las(projected_las_path: Path, tile_id: str, znorm_dir: Path | None) -> Path | None:
    if znorm_dir is not None:
        candidate = znorm_dir / f"z_norm_csf_tiles_multispectral.{tile_id}.las"
    else:
        # Infer from projected LAS: site_root/lidar/clipped/mask/coreg/icp/<file>
        # -> 4 levels up lands at site_root/lidar/clipped/
        clipped_root = projected_las_path.parent.parent.parent.parent
        candidate = clipped_root / "raster" / "z_norm" / "csf" / f"z_norm_csf_tiles_multispectral.{tile_id}.las"
    return candidate if candidate.exists() else None


def load_watershed_summary(
    projected_las_path: Path,
    znorm_dir: Path | None = None,
) -> Dict[str, object]:
    structure_las = laspy.read(str(projected_las_path))
    tile_id_match = re.search(r"(\d+)$", projected_las_path.stem)
    tile_id = tile_id_match.group(1) if tile_id_match else None

    fallback_points = None
    if tile_id is not None:
        znorm_las_path = _resolve_znorm_las(projected_las_path, tile_id, znorm_dir)
        if znorm_las_path is not None:
            znorm_las = laspy.read(str(znorm_las_path))
            znorm_values = get_las_z_norm(znorm_las)
            if znorm_values is None:
                z_values = np.asarray(znorm_las.z, dtype=np.float32)
                znorm_values = z_values - float(np.percentile(z_values, 1))
            fallback_points = (znorm_las.x, znorm_las.y, znorm_values)

    common_kwargs: dict = dict(
        grid_res=0.10,
        min_tree_dist=0.80,
        smoothing_sigma=1.33,
    )
    if fallback_points is not None:
        labels, _, min_x, min_y = get_mallee_centroids_watershed(
            structure_las,
            fallback_points=fallback_points,
            fallback_min_height=1.0,
            **common_kwargs,
        )
    else:
        labels, _, min_x, min_y = get_mallee_centroids_watershed(
            structure_las,
            **common_kwargs,
        )

    crown_count = int(labels.max()) if labels.size else 0
    points_in_crowns = 0
    if crown_count > 0:
        grid_res = 0.10
        cols = np.floor((np.asarray(structure_las.x) - float(min_x)) / grid_res).astype(np.int64)
        rows = np.floor((np.asarray(structure_las.y) - float(min_y)) / grid_res).astype(np.int64)
        valid = (
            (rows >= 0)
            & (rows < labels.shape[0])
            & (cols >= 0)
            & (cols < labels.shape[1])
        )
        if np.any(valid):
            crown_ids = labels[rows[valid], cols[valid]]
            points_in_crowns = int(np.count_nonzero(crown_ids > 0))

    return {
        "Watershed crowns (Step 4)": crown_count,
        "Points in crowns (Step 4)": points_in_crowns,
    }


def compute_treeid_summary_row(
    tile_id: str,
    site_condition: str,
    projected_las_path: Path,
    refined_las_path: Path,
    znorm_dir: Path | None = None,
) -> Dict[str, object]:
    row: Dict[str, object] = {
        "Tile": tile_id,
        "Site condition": site_condition,
        "Watershed crowns (Step 4)": 0,
        "Points in crowns (Step 4)": 0,
        "Final non-zero treeIDs (Step 9)": 0,
        "Mallee points with treeID": 0,
        "Crown count delta (Step 9 - Step 4)": math.nan,
        "Crown retention (%)": math.nan,
        "Mean points per watershed crown": math.nan,
        "Mean points per treeID": math.nan,
    }

    watershed_summary = load_watershed_summary(projected_las_path, znorm_dir=znorm_dir)
    row.update(watershed_summary)

    las = laspy.read(str(refined_las_path))
    dims = set(las.point_format.dimension_names)
    if "treeID" not in dims:
        return row

    tree_ids = np.asarray(las["treeID"], dtype=np.int32)
    class_values = np.asarray(las.classification, dtype=np.int16)
    keep = (tree_ids > 0) & (class_values == 68)
    if not np.any(keep):
        return row

    kept_tree_ids = tree_ids[keep]
    unique_ids = np.unique(kept_tree_ids)
    row["Final non-zero treeIDs (Step 9)"] = int(len(unique_ids))
    row["Mallee points with treeID"] = int(len(kept_tree_ids))
    row["Mean points per treeID"] = float(len(kept_tree_ids) / len(unique_ids)) if len(unique_ids) else math.nan

    crown_count = row["Watershed crowns (Step 4)"]
    crown_points = row["Points in crowns (Step 4)"]
    final_treeids = row["Final non-zero treeIDs (Step 9)"]
    if isinstance(crown_count, (int, float)) and crown_count:
        row["Crown count delta (Step 9 - Step 4)"] = float(final_treeids) - float(crown_count)
        row["Crown retention (%)"] = float(final_treeids) / float(crown_count) * 100.0
        row["Mean points per watershed crown"] = float(crown_points) / float(crown_count)
    return row


def mean_ignore_nan(values: Iterable[object]) -> float:
    numeric = np.asarray(
        [v for v in values if isinstance(v, (int, float, np.integer, np.floating))],
        dtype=np.float64,
    )
    if numeric.size == 0:
        return math.nan
    if np.all(np.isnan(numeric)):
        return math.nan
    return float(np.nanmean(numeric))


def build_table5_average(rows: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    if not rows:
        return []
    avg: Dict[str, object] = {
        "Tile": "Average",
        "Site condition": rows[0]["Site condition"],
        "Common occupied cells": mean_ignore_nan(row["Common occupied cells"] for row in rows),
        "Projected occupied cells": mean_ignore_nan(row["Projected occupied cells"] for row in rows),
        "Refined occupied cells": mean_ignore_nan(row["Refined occupied cells"] for row in rows),
    }
    for column in ("OA (%)", "IoU Forb", "IoU Woody debris", "IoU Hummock", "IoU Shrub", "IoU Mallee"):
        avg[column] = mean_ignore_nan(row[column] for row in rows)
    return [avg]


def build_table6_average(rows: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    averages: List[Dict[str, object]] = []
    for code in VALIDATION_CLASS_CODES:
        class_name = CLASS_LABELS[code]
        class_rows = [row for row in rows if row["Class"] == class_name]
        if not class_rows:
            continue
        averages.append(
            {
                "Tile": "Average",
                "Site condition": class_rows[0]["Site condition"],
                "Class": class_name,
                "Points": mean_ignore_nan(row["Points"] for row in class_rows),
                "Configured min Znorm (m)": mean_ignore_nan(
                    row["Configured min Znorm (m)"] for row in class_rows
                ),
                "Configured max Znorm (m)": mean_ignore_nan(
                    row["Configured max Znorm (m)"] for row in class_rows
                ),
                "Median Znorm (m)": mean_ignore_nan(row["Median Znorm (m)"] for row in class_rows),
                "IQR (m)": mean_ignore_nan(row["IQR (m)"] for row in class_rows),
                "Outside configured range (%)": mean_ignore_nan(
                    row["Outside configured range (%)"] for row in class_rows
                ),
            }
        )
    return averages


def build_treeid_average(rows: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    if not rows:
        return []
    return [
        {
            "Tile": "Average",
            "Site condition": rows[0]["Site condition"],
            "Watershed crowns (Step 4)": mean_ignore_nan(row["Watershed crowns (Step 4)"] for row in rows),
            "Points in crowns (Step 4)": mean_ignore_nan(row["Points in crowns (Step 4)"] for row in rows),
            "Final non-zero treeIDs (Step 9)": mean_ignore_nan(row["Final non-zero treeIDs (Step 9)"] for row in rows),
            "Mallee points with treeID": mean_ignore_nan(row["Mallee points with treeID"] for row in rows),
            "Crown count delta (Step 9 - Step 4)": mean_ignore_nan(row["Crown count delta (Step 9 - Step 4)"] for row in rows),
            "Crown retention (%)": mean_ignore_nan(row["Crown retention (%)"] for row in rows),
            "Mean points per watershed crown": mean_ignore_nan(row["Mean points per watershed crown"] for row in rows),
            "Mean points per treeID": mean_ignore_nan(row["Mean points per treeID"] for row in rows),
        }
    ]


def format_value(value: object) -> str:
    if isinstance(value, float):
        if math.isnan(value):
            return ""
        if value.is_integer():
            return str(int(value))
        return f"{value:.3f}"
    if value is None:
        return ""
    return str(value)


def write_csv(path: Path, fieldnames: Sequence[str], rows: Sequence[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: format_value(row.get(field)) for field in fieldnames})


def write_markdown(path: Path, fieldnames: Sequence[str], rows: Sequence[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        header = "| " + " | ".join(fieldnames) + " |\n"
        divider = "| " + " | ".join(["---"] * len(fieldnames)) + " |\n"
        handle.write(header)
        handle.write(divider)
        for row in rows:
            values = [format_value(row.get(field)) for field in fieldnames]
            handle.write("| " + " | ".join(values) + " |\n")


def main() -> None:
    args = parse_args()
    mask_dir, projected_dir, refined_dir, output_dir, site_condition = resolve_dirs(args)

    if not mask_dir.exists():
        raise SystemExit(f"Mask directory does not exist: {mask_dir}")
    if not projected_dir.exists():
        raise SystemExit(f"Projected LAS directory does not exist: {projected_dir}")
    if not refined_dir.exists():
        raise SystemExit(f"Refined LAS directory does not exist: {refined_dir}")

    mask_map = scan_tile_map(mask_dir, args.mask_glob)
    projected_map = scan_tile_map(projected_dir, "*.las")
    refined_map = scan_tile_map(refined_dir, "*.las")

    tile_ids = sorted(set(mask_map) & set(projected_map) & set(refined_map), key=int)
    if not tile_ids:
        raise SystemExit(
            "No matched tiles were found across mask/projected/refined inputs."
        )

    znorm_dir = args.znorm_dir.resolve() if args.znorm_dir else None
    skip_treeid = args.skip_treeid

    table5_rows: List[Dict[str, object]] = []
    table6_rows: List[Dict[str, object]] = []
    treeid_rows: List[Dict[str, object]] = []

    for idx, tile_id in enumerate(tile_ids, 1):
        print(f"[{idx}/{len(tile_ids)}] Tile {tile_id}", flush=True)
        mask_path = mask_map[tile_id]
        projected_path = projected_map[tile_id]
        refined_path = refined_map[tile_id]

        table5_rows.append(
            compute_table5_row(
                tile_id=tile_id,
                site_condition=site_condition,
                mask_path=mask_path,
                projected_las_path=projected_path,
                refined_las_path=refined_path,
            )
        )
        table6_rows.extend(
            compute_table6_rows(
                tile_id=tile_id,
                site_condition=site_condition,
                refined_las_path=refined_path,
            )
        )
        if not skip_treeid:
            treeid_rows.append(
                compute_treeid_summary_row(
                    tile_id=tile_id,
                    site_condition=site_condition,
                    projected_las_path=projected_path,
                    refined_las_path=refined_path,
                    znorm_dir=znorm_dir,
                )
            )

    table5_average = build_table5_average(table5_rows)
    table6_average = build_table6_average(table6_rows)

    outputs = [
        (output_dir / "table5_label_agreement_per_tile.csv", TABLE5_COLUMNS, table5_rows),
        (output_dir / "table5_label_agreement_per_tile.md", TABLE5_COLUMNS, table5_rows),
        (output_dir / "table5_label_agreement_average.csv", TABLE5_COLUMNS, table5_average),
        (output_dir / "table5_label_agreement_average.md", TABLE5_COLUMNS, table5_average),
        (output_dir / "table6_height_plausibility_per_tile.csv", TABLE6_COLUMNS, table6_rows),
        (output_dir / "table6_height_plausibility_per_tile.md", TABLE6_COLUMNS, table6_rows),
        (output_dir / "table6_height_plausibility_average.csv", TABLE6_COLUMNS, table6_average),
        (output_dir / "table6_height_plausibility_average.md", TABLE6_COLUMNS, table6_average),
    ]

    if not skip_treeid:
        treeid_average = build_treeid_average(treeid_rows)
        outputs += [
            (output_dir / "treeid_instance_summary_per_tile.csv", TREEID_COLUMNS, treeid_rows),
            (output_dir / "treeid_instance_summary_per_tile.md", TREEID_COLUMNS, treeid_rows),
            (output_dir / "treeid_instance_summary_average.csv", TREEID_COLUMNS, treeid_average),
            (output_dir / "treeid_instance_summary_average.md", TREEID_COLUMNS, treeid_average),
        ]

    for path, fieldnames, rows in outputs:
        if path.suffix == ".csv":
            write_csv(path, fieldnames, rows)
        else:
            write_markdown(path, fieldnames, rows)

    print(f"\nMatched tiles: {len(tile_ids)}")
    print(f"Output directory: {output_dir}")
    for path, _, _ in outputs:
        print(f"  Wrote: {path}")


if __name__ == "__main__":
    main()
