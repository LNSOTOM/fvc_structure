#!/usr/bin/env python3
"""Vertical distribution of lidar returns, as absolute counts per height bin.

The voxel-size analysis in `parameter_sensitivity.py` treats the grid in plan: it
reports labelled returns per voxel and how far labels travel across the ground. A
voxel is a cube, though, so whether it holds a second labelled return also depends
on how returns stack in height. Canopy returns sit metres above the ground returns
directly beneath them and never share a 10 cm cube with them, however dense the
survey is in plan.

This counts returns in fixed height bins, per site, separating:

    all         every return in the Z_norm cloud
    labelled    the structure cloud, i.e. returns carrying a 2D class
    per class   those labelled returns split by growth form

Counts are absolute, so sites and heights are compared on the number of returns
actually available to the transfer rather than on a normalised shape.

Writes into --output-dir:

    vertical_profile.csv    one row per (site, tile, bin, class) with the count

Usage
-----
python vertical_profile.py --output-dir assets/parameter_sensitivity --workers 8
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List

import laspy
import numpy as np
import pandas as pd

from parameter_sensitivity import (CLASS_NAMES, DATA_ROOT, LAS_CODES, SITES,
                                   STRUCTURE_DIR, Z_NORM_DIR, exact_twins, tile_id)

# Ground returns dominate every bin near zero and would set the x-axis for the
# whole profile; the vegetation the labels describe sits above this.
DEFAULT_BIN_M = 0.10
DEFAULT_MAX_M = 8.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--output-dir", type=Path,
                        default=Path(__file__).resolve().parent / "assets" / "parameter_sensitivity")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--bin", type=float, default=DEFAULT_BIN_M,
                        help=f"Height bin width in metres. Default {DEFAULT_BIN_M}.")
    parser.add_argument("--max-height", type=float, default=DEFAULT_MAX_M,
                        help=f"Returns above this go in a final overflow bin. Default {DEFAULT_MAX_M}.")
    parser.add_argument("--tiles", default="", help="Comma-separated tile IDs to restrict to.")
    return parser.parse_args()


def z_norm(las: laspy.LasData) -> np.ndarray:
    """Height above ground, from the Z_norm dimension the CSF step writes."""
    if "Z_norm" in las.point_format.dimension_names:
        return np.asarray(las["Z_norm"], dtype=np.float64)
    raise KeyError("No Z_norm dimension; run the CSF height normalisation first.")


def bin_counts(heights: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """Returns per bin, with everything above the last edge folded into the top bin."""
    clipped = np.clip(heights, edges[0], edges[-1] - 1e-9)
    return np.histogram(clipped, bins=edges)[0]


def run_tile(site: str, tile: str, z_path: Path, s_path: Path,
             edges: np.ndarray) -> List[Dict]:
    las_z = laspy.read(str(z_path))
    las_s = laspy.read(str(s_path))
    height_all = z_norm(las_z)
    # The structure cloud carries the class but not the height: it is an exact
    # coordinate subset of the Z_norm cloud, so each point reads its twin's height.
    twin, found = exact_twins(las_s, las_z)
    height_lab = height_all[twin[found]]
    classes = np.asarray(las_s.classification)[found]

    header = {"Site": site, "Site condition": SITES[site], "Tile": tile}
    centres = (edges[:-1] + edges[1:]) / 2

    rows: List[Dict] = []
    for name, counts in (("all", bin_counts(height_all, edges)),
                         ("labelled", bin_counts(height_lab, edges))):
        for centre, count in zip(centres, counts):
            rows.append({**header, "Layer": name, "Height (m)": centre, "Points": int(count)})
    for code in LAS_CODES:
        counts = bin_counts(height_lab[classes == code], edges)
        for centre, count in zip(centres, counts):
            rows.append({**header, "Layer": CLASS_NAMES[code],
                         "Height (m)": centre, "Points": int(count)})
    return rows


def main() -> None:
    args = parse_args()
    edges = np.arange(0.0, args.max_height + args.bin, args.bin)
    wanted = {t.strip() for t in args.tiles.split(",") if t.strip()}

    jobs = []
    for site in SITES:
        base = args.data_root / site
        z_map = {tile_id(p): p for p in (base / Z_NORM_DIR).glob("*.las")}
        s_map = {tile_id(p): p for p in (base / STRUCTURE_DIR[site]).glob("*.las")}
        for tile in sorted(set(z_map) & set(s_map) - {None}, key=int):
            if not wanted or tile in wanted:
                jobs.append((site, tile, z_map[tile], s_map[tile], edges))
    if not jobs:
        raise SystemExit("No tiles with both a Z_norm and a structure cloud were found.")
    print(f"{len(jobs)} tiles, bins of {args.bin * 100:.0f} cm to {args.max_height:.0f} m", flush=True)

    records: List[Dict] = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(run_tile, *job): job for job in jobs}
        for done, future in enumerate(as_completed(futures), 1):
            site, tile = futures[future][:2]
            records.extend(future.result())
            print(f"[{done}/{len(jobs)}] {site} tile {tile}", flush=True)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    out = args.output_dir / "vertical_profile.csv"
    pd.DataFrame.from_records(records).to_csv(out, index=False)
    print(f"Wrote: {out}")


if __name__ == "__main__":
    main()
