#!/usr/bin/env python3
"""What voxel size does to the transferred labels, drawn on one tile.

The companion to `supplementary_voxel_size_sensitivity.png`: where that figure counts
labels spread past their 2D footprint, this one shows them. In a small plan-view window
every lidar return is drawn from above in one of three states:

    inside    labelled, on a cell the 2D labels also label
    spread    labelled by the transfer, on a cell the 2D labels leave unlabelled
    unlabelled

Panel (a) is the 2D labels read onto the returns, the transfer's input; the others are
the Stage-2 result at each voxel size, with the voxel grid drawn and the 2D footprint
outlined. Every spread label sits in a voxel that also holds a labelled return. Grid
squares are voxel *columns*, though: a return takes a label only from returns in the same
voxel, at the same height as well, so an unlabelled return can share a square with
labelled ones.

The window is chosen, not typical: the one holding the most spread labels at the largest
voxel size among windows that are between a quarter and three quarters labelled.

Usage
-----
python plot_voxel_transfer.py --site site1_1_DD0001 --tile 106 \\
    --output assets/parameter_sensitivity/voxel_transfer_tile106
"""

from __future__ import annotations

import argparse
import contextlib
import io
from pathlib import Path
from typing import Dict, Sequence, Tuple

import laspy
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.collections import LineCollection  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

from fvc3dstruct.voxel_merge import cartesian_voxel_merge  # noqa: E402
from parameter_sensitivity import (  # noqa: E402
    DATA_ROOT, INPUT_DIR, STRUCTURE_DIR, SWEEPS, Z_NORM_DIR, classes_beneath, read_labels)

# Emphasis: the spread labels are the story (slot-2 accent), labels inside the
# footprint the context (slot 1), unlabelled returns recede. Slots 1-2 validated on white.
INSIDE = "#2a78d6"
SPREAD = "#eb6834"
UNLABELLED = "#c3c2b7"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
GRID = "#d4d3cc"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--site", default="site1_1_DD0001", choices=sorted(STRUCTURE_DIR))
    parser.add_argument("--tile", default="106")
    parser.add_argument("--window", type=float, default=2.0, help="Window side (m).")
    parser.add_argument("--output", type=Path, required=True, help="Output path without suffix.")
    return parser.parse_args()


def tile_paths(data_root: Path, site: str, tile: str) -> Tuple[Path, Path, Path]:
    base = data_root / site
    return (base / Z_NORM_DIR / f"z_norm_csf_tiles_multispectral.{tile}.las",
            base / STRUCTURE_DIR[site] / f"mask_tiles_multispectral.{tile}.las",
            base / INPUT_DIR / f"mask_structure_3072.{tile}.tif")


def transfer_by_size(las_z: laspy.LasData, las_s: laspy.LasData,
                     sizes: Sequence[float]) -> Dict[float, np.ndarray]:
    """Stage-2 labels for every Z_norm point at each voxel size."""
    transfers = {}
    for size in sizes:
        with contextlib.redirect_stdout(io.StringIO()):
            transfers[size] = cartesian_voxel_merge(las_z, las_s, voxel_xy=size, voxel_z=size)
    return transfers


def choose_window(x: np.ndarray, y: np.ndarray, input_points: np.ndarray, spread: np.ndarray,
                  side: float, step: float = 0.25) -> Tuple[float, float]:
    """Lower-left corner of the `side` window holding the most spread labels, among
    windows a quarter to three quarters labelled, searched on a `step` grid."""
    x0, y0 = x.min() + side, y.min() + side  # keep a window's width clear of the tile edge
    nx, ny = int((x.max() - side - x0) // step), int((y.max() - side - y0) // step)
    ix = np.floor((x - x0) / step).astype(np.int64)
    iy = np.floor((y - y0) / step).astype(np.int64)
    ok = (ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny)

    def summed(weights: np.ndarray) -> np.ndarray:
        grid = np.zeros((ny + 1, nx + 1))
        np.add.at(grid, (iy[ok] + 1, ix[ok] + 1), weights[ok])
        return grid.cumsum(0).cumsum(1)

    k = int(round(side / step))

    def window_sum(table: np.ndarray) -> np.ndarray:
        return table[k:, k:] - table[:-k, k:] - table[k:, :-k] + table[:-k, :-k]

    points = window_sum(summed(np.ones(len(x))))
    labelled = window_sum(summed((input_points > 0).astype(float)))
    spread_count = window_sum(summed(spread.astype(float)))
    share = np.divide(labelled, points, out=np.zeros_like(points), where=points > 0)
    spread_count[(share < 0.25) | (share > 0.75)] = -1
    row, col = np.unravel_index(np.argmax(spread_count), spread_count.shape)
    return x0 + col * step, y0 + row * step


def footprint_outline(labels: np.ndarray, transform, bounds: Tuple[float, float, float, float]) -> LineCollection:
    """Cell edges between labelled and unlabelled 5 cm cells inside `bounds`."""
    xmin, ymin, xmax, ymax = bounds
    c0 = max(int(np.floor((xmin - transform.c) / transform.a)) - 1, 0)
    c1 = min(int(np.ceil((xmax - transform.c) / transform.a)) + 1, labels.shape[1])
    r0 = max(int(np.floor((ymax - transform.f) / transform.e)) - 1, 0)
    r1 = min(int(np.ceil((ymin - transform.f) / transform.e)) + 1, labels.shape[0])
    lab = labels[r0:r1, c0:c1] < 5
    segments = []
    rows, cols = np.nonzero(lab[:, 1:] != lab[:, :-1])  # vertical edges
    for r, c in zip(rows + r0, cols + c0 + 1):
        xe = transform.c + c * transform.a
        segments.append([(xe, transform.f + r * transform.e), (xe, transform.f + (r + 1) * transform.e)])
    rows, cols = np.nonzero(lab[1:, :] != lab[:-1, :])  # horizontal edges
    for r, c in zip(rows + r0 + 1, cols + c0):
        ye = transform.f + r * transform.e
        segments.append([(transform.c + c * transform.a, ye), (transform.c + (c + 1) * transform.a, ye)])
    return LineCollection(segments, colors=INK, linewidths=0.7, zorder=4)


def draw_panel(ax: plt.Axes, x: np.ndarray, y: np.ndarray, classes: np.ndarray,
               input_points: np.ndarray, corner: Tuple[float, float], side: float,
               outline: LineCollection, grid: Tuple[float, float, float] | None) -> float:
    """One plan-view panel; returns the share of labelled points spread past the footprint."""
    x0, y0 = corner
    if grid is not None:
        size, gx, gy = grid
        for value in np.arange(gx + np.ceil((x0 - gx) / size) * size, x0 + side, size):
            ax.axvline(value, color=GRID, linewidth=0.4, zorder=1)
        for value in np.arange(gy + np.ceil((y0 - gy) / size) * size, y0 + side, size):
            ax.axhline(value, color=GRID, linewidth=0.4, zorder=1)
    labelled = classes > 0
    spread = labelled & (input_points == 0)
    for mask, colour, order in ((~labelled, UNLABELLED, 2), (labelled & ~spread, INSIDE, 3),
                                (spread, SPREAD, 3)):
        ax.scatter(x[mask], y[mask], s=1.1, c=colour, linewidths=0, zorder=order, rasterized=True)
    ax.add_collection(outline)
    ax.set_xlim(x0, x0 + side)
    ax.set_ylim(y0, y0 + side)
    ax.set_aspect("equal")
    ticks = np.arange(0, side + 1e-9, 0.5)
    ax.set_xticks(x0 + ticks)
    ax.set_yticks(y0 + ticks)
    ax.set_xticklabels([f"{t:g}" for t in ticks])
    ax.set_yticklabels([f"{t:g}" for t in ticks])
    ax.tick_params(colors=INK_SECONDARY, labelsize=7, length=2.5, width=0.6)
    for spine in ax.spines.values():
        spine.set_color("#c3c2b7")
        spine.set_linewidth(0.6)
    return 100.0 * np.count_nonzero(spread) / max(int(np.count_nonzero(labelled)), 1)


def plot_tile(data_root: Path, site: str, tile: str, output: Path, side: float = 2.0,
              sizes: Sequence[float] = SWEEPS["voxel_size"]) -> Path:
    z_path, s_path, input_path = tile_paths(data_root, site, tile)
    las_z, las_s = laspy.read(str(z_path)), laspy.read(str(s_path))
    x, y = np.asarray(las_z.x), np.asarray(las_z.y)
    labels, transform = read_labels(input_path)
    input_points = classes_beneath(labels, transform, x, y)
    transfers = transfer_by_size(las_z, las_s, sizes)
    # voxel_merge's grid origin: the minimum over both clouds
    gx = min(float(las_z.x.min()), float(las_s.x.min()))
    gy = min(float(las_z.y.min()), float(las_s.y.min()))

    largest = transfers[max(sizes)]
    corner = choose_window(x, y, input_points, (largest > 0) & (input_points == 0), side)
    pad = 0.05
    inside = ((x >= corner[0] - pad) & (x <= corner[0] + side + pad)
              & (y >= corner[1] - pad) & (y <= corner[1] + side + pad))
    bounds = (corner[0], corner[1], corner[0] + side, corner[1] + side)

    fig, axes = plt.subplots(2, 3, figsize=(7.2, 5.6), constrained_layout=True)
    panels = [("2D labels on the lidar returns", input_points, None)]
    panels += [(f"{size * 100:g} cm voxels", transfers[size], (size, gx, gy)) for size in sizes]
    for index, (ax, (title, classes, grid)) in enumerate(zip(axes.ravel(), panels)):
        share = draw_panel(ax, x[inside], y[inside], classes[inside], input_points[inside],
                           corner, side, footprint_outline(labels, transform, bounds), grid)
        subtitle = "" if grid is None else f"\n{share:.0f}% of labels past the footprint"
        ax.set_title(f"({chr(97 + index)}) {title}{subtitle}", fontsize=8, color=INK, loc="left")
        if index % 3 == 0:
            ax.set_ylabel("m", fontsize=7, color=INK_SECONDARY)
        if index >= 3:
            ax.set_xlabel("m", fontsize=7, color=INK_SECONDARY)

    handles = [
        Line2D([], [], linestyle="none", marker="o", markersize=4, markerfacecolor=INSIDE,
               markeredgewidth=0, label="Label inside its 2D footprint"),
        Line2D([], [], linestyle="none", marker="o", markersize=4, markerfacecolor=SPREAD,
               markeredgewidth=0, label="Label spread past the footprint"),
        Line2D([], [], linestyle="none", marker="o", markersize=4, markerfacecolor=UNLABELLED,
               markeredgewidth=0, label="Unlabelled return"),
        Line2D([], [], color=INK, linewidth=0.7, label="2D label footprint"),
        Line2D([], [], color=GRID, linewidth=0.8, label="Voxel columns"),
    ]
    fig.legend(handles=handles, loc="outside upper center", ncol=3, frameon=False, fontsize=7.5,
               labelcolor=INK_SECONDARY)
    output.parent.mkdir(parents=True, exist_ok=True)
    for suffix in (".png", ".pdf"):
        fig.savefig(output.with_suffix(suffix), dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Wrote: {output.with_suffix('.png')} (+ .pdf); window lower-left "
          f"({corner[0]:.2f}, {corner[1]:.2f}), {side:g} m")
    return output.with_suffix(".png")


def main() -> None:
    args = parse_args()
    plot_tile(args.data_root, args.site, args.tile, args.output, args.window)


if __name__ == "__main__":
    main()
