#!/usr/bin/env python3
"""Every figure and table behind the four reviewer replies, from `sensitivity.py`'s CSVs.

    supplementary_error_budget.png / .pdf             where the label error comes from (6.1)
    supplementary_voxel_size_sensitivity.png / .pdf   the voxel sweep, 5-25 cm (3.2.2)
    supplementary_parameter_sensitivity.png / .pdf    radius, height-prior weight,
                                                      min_conf, min_flip_ratio
    reviewer_response_tables.md                       every table, as markdown

Curves are the median across a site's tiles with the inter-quartile range shaded, and
tables are `mean / median` across tiles unless the row says otherwise. Nothing here
computes a number the CSVs do not already contain, except by pooling across tiles, so
rerun this after any rerun of `sensitivity.py`.

Usage
-----
python figures.py
"""

from __future__ import annotations

import argparse
import contextlib
import io
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import laspy
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib import colors  # noqa: E402
from matplotlib.collections import LineCollection  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from matplotlib.ticker import MaxNLocator  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))

from sensitivity import (  # noqa: E402
    DATA_ROOT, DEFAULTS, INPUT_DIR, OUTPUT_DIR, STRUCTURE_DIR, SWEEPS,
    Z_NORM_DIR, classes_beneath, read_labels)

from fvc3dstruct.voxel_merge import cartesian_voxel_merge  # noqa: E402

# Categorical slots 1-2 of the reference palette, validated all-pairs on white.
SITE_COLOURS = {"Medium": "#2a78d6", "Low": "#eb6834"}
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"

# The three parts of the pipeline the error budget separates, in order.
STAGE_COLOURS = {"2D annotation": "#8c6d31", "Onto the returns": "#d9b44a",
                 "Voxel transfer": "#2a78d6", "Crown stages": "#2e8b57"}

PARAMETER_LABELS = {
    "voxel_size": "Voxel size (cm)",
    "height_match_radius": "Matching radius (m)",
    "height_prior_out_of_range_weight": "Out-of-range weight",
    "min_conf": "min_conf",
    "min_flip_ratio": "min_flip_ratio",
}

# Growth forms in height order, with the palette used for the vertical profile.
CLASS_ORDER = ("Forb", "Woody debris", "Hummock grass", "Shrub", "Mallee")
CLASS_COLOURS = ("#8c6d31", "#694714", "#d9b44a", "#2a78d6", "#2e8b57")

# Tiles are 3072 px at 1 cm, so 30.72 m square; used to express counts per m2.
TILE_SIDE_M = 30.72

# (column, panel title, y-axis label) for the voxel figure.
VOXEL_PANELS = [
    ("Stage-2 labelled points (%)", "Labelled by voxel transfer", "% of lidar points"),
    ("Stage 2 vs input Outside footprint (%)", "Labels outside 2D footprint", "% of labelled points"),
    ("Stage 2 vs input Footprint recall (%)", "Footprint points keeping class", "% of footprint points"),
    ("Structure points per voxel, median", "Labelled points per voxel", "median count"),
    # Stage 2, not the final labels: Stage 5 removes woody debris below 0.5 m and
    # merges crowns into mallee, which would set the final mIoU's level, not voxel size.
    ("Stage 2 vs input mIoU", "Plan-view agreement", "mIoU vs 2D labels"),
    ("Final changed vs default (%)", "Final labels changed vs 10 cm", "% of lidar points"),
]
# Columns that cannot go below zero, drawn from a zero baseline.
ZERO_BASED = ("Final changed vs default (%)", "Stage-4 changed points (%)")

# (column, row label, worst tile instead of median) for the other-parameter figure,
# one column per parameter. The final-label row shows the worst tile, because the
# claim it carries is that no tile changes, which a median cannot show.
PARAMETER_ROWS = [
    ("Stage-4 changed points (%)", "Stage-4 labels changed\n(% of points)", False),
    ("Final changed vs default (%)", "Final labels changed from\ndefault, worst tile (%)", True),
]
OTHER_PARAMETERS = ("height_match_radius", "height_prior_out_of_range_weight",
                    "min_conf", "min_flip_ratio")

# (column, table heading, decimals) for the two voxel-size tables.
TRANSFER_COLUMNS = [
    ("Stage-2 labelled points (%)", "Labelled by transfer (%)", 1),
    ("Stage 2 vs input Outside footprint (%)", "Outside footprint (%)", 1),
    ("Stage 2 vs input Footprint recall (%)", "Footprint recall (%)", 2),
    ("Stage 2 vs input mIoU", "mIoU vs 5 cm input", 3),
    ("Structure points per voxel, median", "Points per voxel", 1),
    ("Single-point voxels (%)", "Single-point voxels (%)", 1),
    ("Mixed-class voxels (%)", "Mixed-class voxels (%)", 2),
]
FINAL_COLUMNS = [
    ("Final labelled points (%)", "Labelled (%)", 1),
    ("Final vs input Outside footprint (%)", "Outside footprint (%)", 1),
    ("Final vs input mIoU", "mIoU vs 5 cm input", 3),
    ("Final vs 1 cm mIoU", "mIoU vs 1 cm", 3),
    ("Final changed vs default (%)", "Changed from 10 cm (%)", 2),
]

# Height bands for the vertical profile table, chosen to separate ground, understorey
# and the canopy where the sparse classes sit.
HEIGHT_BANDS = [
    (0.0, 0.1, "0–10 cm (ground)"),
    (0.1, 0.5, "10–50 cm (understorey)"),
    (0.5, 1.5, "0.5–1.5 m"),
    (1.5, 3.0, "1.5–3 m"),
    (3.0, 8.0, ">3 m (upper canopy)"),
]

# Densities to scale the voxel rule to, as multiples of this survey.
SENSOR_FACTORS = ((0.5, "half as dense"), (1.0, "this survey"), (2.0, "twice as dense"))


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def fmt(value: float, digits: int) -> str:
    return "–" if pd.isna(value) else f"{value:.{digits}f}"


def mean_median(frame: pd.DataFrame, column: str, digits: int) -> str:
    return f"{fmt(frame[column].mean(), digits)} / {fmt(frame[column].median(), digits)}"


def pooled(frame: pd.DataFrame, prefix: str) -> np.ndarray:
    """Every value of a per-class family of columns (`IoU 5 cm <class>`), tiles and
    classes together, NaNs dropped. label_qa.py leaves a class blank where a site has none."""
    columns = [c for c in frame.columns if c.startswith(prefix)]
    values = frame[columns].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float).ravel()
    return values[~np.isnan(values)]


def pooled_mean_median(frame: pd.DataFrame, prefix: str, digits: int = 3) -> str:
    values = pooled(frame, prefix)
    if not len(values):
        return "–"
    return f"{values.mean():.{digits}f} / {np.median(values):.{digits}f}"


def defaults_only(per_tile: pd.DataFrame) -> pd.DataFrame:
    """The runs at every default value. The voxel sweep carries the shared default run,
    so its 10 cm rows are the delivered configuration."""
    return per_tile[(per_tile["Parameter"] == "voxel_size")
                    & np.isclose(per_tile["Value"], DEFAULTS["voxel_size"])]


def style_axes(ax: plt.Axes) -> None:
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
        ax.spines[side].set_linewidth(0.8)
    ax.tick_params(colors=INK_SECONDARY, labelsize=7.5, length=3, width=0.8)
    ax.grid(axis="y", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def save(fig: plt.Figure, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    for suffix in (".png", ".pdf"):
        fig.savefig(path.with_suffix(suffix), dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Wrote: {path.with_suffix('.png')} (+ .pdf)")
    return path.with_suffix(".png")


# ---------------------------------------------------------------------------
# 1. The error budget (discussion 6.1)
# ---------------------------------------------------------------------------
#
# One reference runs through the whole pipeline: the annotator's 1 cm raster, which
# predates every step. Putting each intermediate product on the 5 cm grid by majority
# and scoring it against that reference gives every step a value on one scale, so the
# steps can be compared with each other rather than only described.
#
# The four steps the ladder separates, and where each level is measured:
#
#   1 cm annotation          1.000 by definition, the reference
#   -> 1 cm to 5 cm grid     label_qa.py `IoU 5 cm`, phase 3 step 3
#   -> onto the lidar returns  the sweep's `input` rows: the 5 cm labels carried by the
#                            structure cloud, which is where cells holding no return are lost
#   -> Stage 2 voxel transfer  `Stage 2 vs 1 cm mIoU` at the delivered 10 cm
#   -> Stages 3-5            `Final vs 1 cm mIoU`
#
# The two steps before the raster exist but cannot join this column: delineation has no
# reference at all, and polygon -> 1 cm compares a vector against the raster that *is* the
# reference here. Both are quoted in their own units in `error_sources_table`.

LADDER = [
    ("The annotator's 1 cm raster", "2D annotation", None, None),
    ("1 cm → 5 cm label grid", "2D annotation", "label_qa", "IoU 5 cm "),
    ("5 cm labels → labelled lidar returns", "Onto the returns", "input",
     "Final vs 1 cm mIoU"),
    ("Stage 2: voxel transfer (10 cm)", "Voxel transfer", "default", "Stage 2 vs 1 cm mIoU"),
    ("Stages 3–5: crown detection, refinement, vote", "Crown stages", "default",
     "Final vs 1 cm mIoU"),
]


def ladder_levels(per_tile: pd.DataFrame, label_quality: pd.DataFrame,
                  condition: str) -> List[Tuple[str, str, float]]:
    """(step, part of the pipeline, plan-view mIoU against the 1 cm annotation) for one site."""
    qa = label_quality[label_quality["Site condition"] == condition]
    inputs = per_tile[(per_tile["Parameter"] == "input")
                      & (per_tile["Site condition"] == condition)]
    default = defaults_only(per_tile)
    default = default[default["Site condition"] == condition]
    levels = []
    for label, part, source, column in LADDER:
        if source is None:
            value = 1.0
        elif source == "label_qa":
            value = float(pooled(qa, column).mean())
        else:
            frame = inputs if source == "input" else default
            value = float(frame[column].mean())
        levels.append((label, part, value))
    return levels


def error_budget_table(per_tile: pd.DataFrame, label_quality: pd.DataFrame) -> str:
    """The ladder: each step's level on the one common scale, and what it cost."""
    lines = ["| Step | Part of the pipeline | "
             + " | ".join(f"{c}: mIoU | {c}: lost" for c in SITE_COLOURS) + " |",
             "|---|---|" + "---:|" * (2 * len(SITE_COLOURS))]
    ladders = {c: ladder_levels(per_tile, label_quality, c) for c in SITE_COLOURS}
    for index in range(len(LADDER)):
        label, part, _ = ladders["Medium"][index]
        cells = []
        for condition in SITE_COLOURS:
            level = ladders[condition][index][2]
            previous = ladders[condition][index - 1][2] if index else None
            cells.append(fmt(level, 3))
            cells.append("–" if previous is None else f"{level - previous:+.3f}")
        lines.append(f"| {label} | {part} | " + " | ".join(cells) + " |")
    shares = []
    for condition in SITE_COLOURS:
        levels = [v for _, _, v in ladders[condition]]
        total = levels[0] - levels[-1]
        steps = ", ".join(f"{100 * (levels[i] - levels[i + 1]) / total:.0f}%"
                          for i in range(len(levels) - 1))
        shares.append(f"**{condition}** {steps} (total {total:.3f})")
    lines.append("")
    lines.append("Share of the total loss, step by step in the order above — "
                 + "; ".join(shares) + ".")
    return "\n".join(lines)


def error_sources_table(per_tile: pd.DataFrame, label_quality: pd.DataFrame) -> str:
    """Every step of the pipeline in its own units, including the two the ladder cannot carry.

    The ladder says how much each step costs on one scale; this says what the error at
    each step actually is, which is what a reader needs to judge whether it matters.
    """
    inputs = per_tile[per_tile["Parameter"] == "input"]
    default = defaults_only(per_tile)
    voxel = per_tile[per_tile["Parameter"] == "voxel_size"]

    def qa(condition: str) -> pd.DataFrame:
        return label_quality[label_quality["Site condition"] == condition]

    def sweep(frame: pd.DataFrame, condition: str) -> pd.DataFrame:
        return frame[frame["Site condition"] == condition]

    rows = [
        ("Delineation", "Where the annotator put each boundary",
         "IoU of the class eroded against the class dilated by ±5 cm (a floor, not an estimate)",
         lambda c: pooled_mean_median(qa(c), "Delineation IoU ")),
        ("Delineation", "Pixels claimed by two classes at once",
         "Cross-class overlap, share of labelled area",
         lambda c: f"{pd.to_numeric(qa(c)['Cross-class overlap (%)'], errors='coerce').mean():.3f}%"),
        ("Polygon → 1 cm raster", "Rasterisation and draw-order resolution of overlaps",
         "Class IoU, polygons rasterised exactly against the delivered 1 cm raster",
         lambda c: pooled_mean_median(qa(c), "IoU 1 cm ")),
        ("1 cm → 5 cm grid", "Downsampling onto a grid that does not divide the tile",
         "Class IoU against the majority of each cell's 1 cm sub-pixels",
         lambda c: pooled_mean_median(qa(c), "IoU 5 cm ")),
        ("1 cm → 5 cm grid", "How far a feature's centre moves",
         "Centroid RMSE per polygon",
         lambda c: f"{pd.to_numeric(qa(c)['Centroid RMSE 5 cm (m)'], errors='coerce').mean():.3f} m"),
        ("5 cm labels → returns", "Labelled cells holding no lidar return cannot be labelled",
         "Plan-view mIoU of the labelled returns against the 5 cm raster they came from",
         lambda c: mean_median(sweep(inputs, c), "Final vs input mIoU", 3)),
        ("5 cm labels → returns", "Returns on labelled cells missing from the structure cloud",
         "Share of the lidar points on labelled cells, all tiles",
         lambda c: (lambda s: f"{int((s['Points on labelled 5 cm cells'] - s['Structure points']).sum()):,} "
                              f"({100 * (s['Points on labelled 5 cm cells'] - s['Structure points']).sum() / s['Points on labelled 5 cm cells'].sum():.3f}%)")(sweep(inputs, c))),
        ("Stage 1: height lookup", "The nearest-neighbour search is in XY only",
         "Largest shift in any class height range it causes, any tile",
         lambda c: f"{sweep(inputs, c)['Stage-1 range shift, max (m)'].max():.3f} m"),
        ("Stage 1: height lookup", "Points beyond the 0.30 m matching radius",
         "Share of structure points, any tile",
         lambda c: f"{sweep(inputs, c)['Structure points beyond match radius (%)'].max():.3f}%"),
        ("Stage 2: voxel transfer", "Labels extended past their 2D footprint",
         "Share of the labels the transfer assigns",
         lambda c: mean_median(sweep(default, c), "Stage 2 vs input Outside footprint (%)", 1) + "%"),
        ("Stage 2: voxel transfer", "Labels lost inside their 2D footprint",
         "100 − footprint recall, worst tile at any voxel size",
         lambda c: f"{100 - sweep(voxel, c)['Stage 2 vs input Footprint recall (%)'].min():.2f}%"),
        ("Stage 2: voxel transfer", "Voxels holding more than one class, settled by majority",
         "Share of occupied voxels",
         lambda c: mean_median(sweep(default, c), "Mixed-class voxels (%)", 2) + "%"),
        ("Stage 4: crown refinement", "Labels the polar refinement changes inside crowns",
         "Share of all lidar points",
         lambda c: mean_median(sweep(default, c), "Stage-4 changed points (%)", 3) + "%"),
        ("Stage 5: crown vote", "Crowns turned wholly mallee, overwriting Stage 4",
         "Crowns voted, of which mallee-majority",
         lambda c: (lambda s: f"{int(s['Stage-5 crowns voted'].sum())}, "
                              f"{int(s['Stage-5 crowns with mallee majority'].sum())} mallee")(sweep(default, c))),
        ("Stage 5: crown vote", "Unlabelled returns promoted to mallee on height alone",
         "Share of all lidar points",
         lambda c: mean_median(sweep(default, c), "Stage-5 promoted points (%)", 3) + "%"),
        ("Stage 5: crown vote", "Woody debris removed: it is not a growth form",
         "Plan-view IoU, Stage 2 → final",
         lambda c: f"{sweep(default, c)['Stage 2 vs input IoU Woody debris'].mean():.3f} → "
                   f"{sweep(default, c)['Final vs input IoU Woody debris'].mean():.3f}"),
    ]
    lines = ["| Step | What it introduces | How it is measured | " + " | ".join(SITE_COLOURS) + " |",
             "|---|---|---|" + "---:|" * len(SITE_COLOURS)]
    for step, introduces, measure, cell in rows:
        values = [cell(condition) for condition in SITE_COLOURS]
        lines.append(f"| {step} | {introduces} | {measure} | " + " | ".join(values) + " |")
    return "\n".join(lines)


def class_attribution_table(per_tile: pd.DataFrame) -> str:
    """Which growth forms the crown stages cost, and how much of the drop each explains.

    A tile's mIoU is the mean over the classes its reference holds, so a class's
    contribution to the change is its own change divided by that count. Summed over the
    classes, the column reproduces the tile's mIoU change exactly.
    """
    default = defaults_only(per_tile)
    lines = ["| Site | Class | Stage 2 | Final | Change | Share of the mIoU drop |",
             "|---|---|---:|---:|---:|---:|"]
    for condition in SITE_COLOURS:
        site = default[default["Site condition"] == condition]
        present = [f"Stage 2 vs input IoU {c}" for c in CLASS_ORDER]
        n_classes = site[present].notna().sum(axis=1)
        total = (site["Final vs input mIoU"] - site["Stage 2 vs input mIoU"]).mean()
        for name in CLASS_ORDER:
            start, end = site[f"Stage 2 vs input IoU {name}"], site[f"Final vs input IoU {name}"]
            if start.isna().all() and end.isna().all():
                continue
            contribution = ((end - start) / n_classes).fillna(0.0).mean()
            share = f"{100 * contribution / total:.0f}%" if abs(total) > 1e-9 else "–"
            lines.append(f"| {condition} | {name} | {fmt(start.mean(), 3)} | {fmt(end.mean(), 3)} "
                         f"| {contribution:+.3f} | {share} |")
        lines.append(f"| {condition} | **All classes (mIoU)** | "
                     f"{fmt(site['Stage 2 vs input mIoU'].mean(), 3)} | "
                     f"{fmt(site['Final vs input mIoU'].mean(), 3)} | {total:+.3f} | 100% |")
    return "\n".join(lines)


def plot_error_budget(per_tile: pd.DataFrame, label_quality: pd.DataFrame, path: Path) -> Path:
    """The ladder as a cascade: one bar per step, falling from the annotation to the labels."""
    fig, axes = plt.subplots(1, len(SITE_COLOURS), figsize=(7.2, 3.6), constrained_layout=True,
                             sharey=True)
    for ax, condition in zip(np.atleast_1d(axes), SITE_COLOURS):
        style_axes(ax)
        levels = ladder_levels(per_tile, label_quality, condition)
        values = [v for _, _, v in levels]
        total = values[0] - values[-1]
        for index in range(1, len(values)):
            top, bottom = values[index - 1], values[index]
            drop, part = top - bottom, levels[index][1]
            ax.bar(index, drop, bottom=bottom, width=0.62, color=STAGE_COLOURS[part],
                   linewidth=0, zorder=3)
            # Inside the bar where it is tall enough to hold two lines, above it otherwise.
            roomy = drop > 0.09
            ax.annotate(f"−{drop:.3f}\n{100 * drop / total:.0f}%",
                        (index, (top + bottom) / 2 if roomy else top),
                        ha="center", va="center" if roomy else "bottom", fontsize=6.8,
                        color="white" if roomy else INK_SECONDARY,
                        xytext=(0, 0) if roomy else (0, 3), textcoords="offset points", zorder=4)
        # The running level, so the cascade can be read as a single descent. Each level is
        # labelled *below* its marker, which is the foot of the bar that lands on it.
        # Behind the bars: its vertical segments are the drops the bars already draw, so
        # hiding them there keeps the drop labels clear.
        ax.step(range(len(values)), values, where="post", color=INK_SECONDARY, linewidth=0.9,
                zorder=2)
        ax.plot(range(len(values)), values, "o", color=INK_SECONDARY, markersize=3.2,
                markeredgecolor="white", markeredgewidth=0.8, zorder=6)
        for index, value in enumerate(values):
            ax.annotate(f"{value:.3f}", (index, value), ha="center", va="top",
                        fontsize=6.8, color=INK, xytext=(0, -5), textcoords="offset points",
                        zorder=6)
        ax.set_title(f"{condition} site", fontsize=8.5, color=INK, loc="left")
        ax.set_xticks(range(len(values)))
        ax.set_xticklabels(["1 cm\nannotation", "5 cm\ngrid", "onto the\nreturns",
                            "Stage 2\ntransfer", "Stages 3–5\ncrowns"], fontsize=7)
        ax.set_xlim(-0.55, len(values) - 0.45)
        ax.set_ylim(0, 1.12)
    np.atleast_1d(axes)[0].set_ylabel("plan-view mIoU against the 1 cm annotation",
                                      fontsize=7.5, color=INK_SECONDARY)
    handles = [Patch(facecolor=colour, label=part) for part, colour in STAGE_COLOURS.items()]
    fig.legend(handles=handles, loc="outside upper center", ncol=len(handles), frameon=False,
               fontsize=7.5, labelcolor=INK_SECONDARY)
    return save(fig, path)


# ---------------------------------------------------------------------------
# 2. Voxel size (3.2.2)
# ---------------------------------------------------------------------------

def quartiles(frame: pd.DataFrame, column: str, worst: bool = False) -> pd.DataFrame:
    """Median and IQR across tiles per value, or with `worst` the largest tile value (no band)."""
    grouped = frame.groupby("Value")[column]
    if worst:
        largest = grouped.max()
        return pd.DataFrame({"q1": largest, "median": largest, "q3": largest}).sort_index()
    return pd.DataFrame({"q1": grouped.quantile(0.25), "median": grouped.median(),
                         "q3": grouped.quantile(0.75)}).sort_index()


def draw_sweep(ax: plt.Axes, per_tile: pd.DataFrame, parameter: str, column: str,
               positions: Dict[float, float] | None = None,
               worst: bool = False) -> Dict[str, pd.DataFrame]:
    """Median and IQR per site against the parameter value (or the worst tile); returns the curves."""
    rows = per_tile[per_tile["Parameter"] == parameter]
    curves = {}
    for condition in SITE_COLOURS:
        site = rows[rows["Site condition"] == condition]
        if not site.empty and not site[column].isna().all():
            curves[condition] = quartiles(site, column, worst)
    # Where the sites coincide the later line would hide the earlier one: draw
    # the first wider underneath so both stay visible.
    medians = [curve["median"].to_numpy() for curve in curves.values()]
    coincide = len(medians) == 2 and medians[0].shape == medians[1].shape and np.allclose(*medians)
    for order, (condition, curve) in enumerate(curves.items()):
        colour = SITE_COLOURS[condition]
        x = np.array([positions[v] if positions else v for v in curve.index], dtype=float)
        if parameter == "voxel_size" and not positions:
            x = x * 100.0
        under = coincide and order == 0
        ax.fill_between(x, curve["q1"], curve["q3"], color=colour, alpha=0.12, linewidth=0)
        ax.plot(x, curve["median"], color=colour, linewidth=3.2 if under else 1.6,
                solid_capstyle="round", solid_joinstyle="round", marker="o",
                markersize=8 if under else 5.5, markeredgecolor="white",
                markeredgewidth=1.2, label=condition, zorder=3 + order)
    default = DEFAULTS[parameter]
    x_default = positions[default] if positions else (
        default * 100.0 if parameter == "voxel_size" else default)
    ax.axvline(x_default, color=INK_MUTED, linewidth=0.8, zorder=1)
    if column in ZERO_BASED:
        zero_based_ylim([ax], [curves])
    return curves


def zero_based_ylim(axes: Sequence[plt.Axes], curves: Sequence[Dict[str, pd.DataFrame]]) -> None:
    """A zero baseline with room above the highest band across `axes`, which may share y."""
    tops = [float(c["q3"].max()) for panel in curves for c in panel.values()]
    top = max([t for t in tops if np.isfinite(t)] or [0.0])
    for ax in axes:
        ax.set_ylim(0, top * 1.15 if top > 0 else 0.05)


def figure_legend(fig: plt.Figure, default_label: str) -> None:
    handles = [
        Line2D([], [], color=colour, linewidth=1.6, marker="o", markersize=5.5,
               markeredgecolor="white", markeredgewidth=1.2, label=f"{condition} site")
        for condition, colour in SITE_COLOURS.items()
    ]
    handles.append(Line2D([], [], color=INK_MUTED, linewidth=0.8, label=default_label))
    fig.legend(handles=handles, loc="outside upper center", ncol=len(handles), frameon=False,
               fontsize=8, labelcolor=INK_SECONDARY)


def plot_voxel_figure(per_tile: pd.DataFrame, path: Path) -> Path:
    fig, axes = plt.subplots(2, 3, figsize=(7.2, 4.9), constrained_layout=True)
    for index, (ax, (column, title, ylabel)) in enumerate(zip(axes.ravel(), VOXEL_PANELS)):
        style_axes(ax)
        draw_sweep(ax, per_tile, "voxel_size", column)
        ax.set_title(f"({chr(97 + index)}) {title}", fontsize=8.5, color=INK, loc="left")
        ax.set_ylabel(ylabel, fontsize=7.5, color=INK_SECONDARY)
        ax.set_xticks([5, 10, 15, 20, 25])
        ax.set_xlim(3.5, 26.5)
        if index >= 3:
            ax.set_xlabel(PARAMETER_LABELS["voxel_size"], fontsize=7.5, color=INK_SECONDARY)
    figure_legend(fig, "Default (10 cm)")
    return save(fig, path)


def voxel_volume_profile(per_tile: pd.DataFrame) -> pd.DataFrame:
    """Tile-paired occupied-voxel volume and mIoU changes relative to 10 cm.

    Occupied volume is a resolution-dependent support proxy, not vegetation volume:
    each occupied voxel contributes its full cubic volume regardless of point count.
    """
    voxel = per_tile[per_tile["Parameter"] == "voxel_size"].copy()
    voxel["Occupied support volume (m3)"] = voxel["Structure voxels"] * voxel["Value"] ** 3
    keys = ["Site", "Tile"]
    default = voxel[np.isclose(voxel["Value"], DEFAULTS["voxel_size"])].set_index(keys)
    baseline_volume = default["Occupied support volume (m3)"]
    baseline_miou = default["Stage 2 vs input mIoU"]
    index = pd.MultiIndex.from_frame(voxel[keys])
    voxel["Occupied volume change vs 10 cm (%)"] = (
        100.0 * (voxel["Occupied support volume (m3)"].to_numpy()
                 / baseline_volume.reindex(index).to_numpy() - 1.0))
    voxel["mIoU change vs 10 cm"] = (
        voxel["Stage 2 vs input mIoU"].to_numpy()
        - baseline_miou.reindex(index).to_numpy())
    return voxel


def plot_voxel_volume_profile(per_tile: pd.DataFrame, path: Path) -> Path:
    """Resolution sensitivity of occupied support volume beside label fidelity."""
    profile = voxel_volume_profile(per_tile)
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.1), constrained_layout=True)
    panels = [
        ("Occupied volume change vs 10 cm (%)", "(a) Occupied support volume", "% change from 10 cm"),
        ("mIoU change vs 10 cm", "(b) Plan-view label fidelity", "mIoU change from 10 cm"),
    ]
    for ax, (column, title, ylabel) in zip(axes, panels):
        style_axes(ax)
        draw_sweep(ax, profile, "voxel_size", column)
        ax.axhline(0, color=AXIS, linewidth=0.8, zorder=1)
        ax.set_title(title, fontsize=8.5, color=INK, loc="left")
        ax.set_ylabel(ylabel, fontsize=7.5, color=INK_SECONDARY)
        ax.set_xlabel(PARAMETER_LABELS["voxel_size"], fontsize=7.5, color=INK_SECONDARY)
        ax.set_xticks([5, 10, 15, 20, 25])
        ax.set_xlim(3.5, 26.5)
    figure_legend(fig, "Reference (10 cm)")
    return save(fig, path)


def table(per_tile: pd.DataFrame, parameter: str,
          columns: Sequence[Tuple[str, str, int]]) -> str:
    """Markdown table: one row per (site, value), `mean / median` across tiles."""
    header = ("| Site | " + PARAMETER_LABELS[parameter] + " | "
              + " | ".join(label for _, label, _ in columns) + " |")
    lines = [header, "|---|---:|" + "---:|" * len(columns)]
    rows = per_tile[per_tile["Parameter"] == parameter]
    for condition in SITE_COLOURS:
        site = rows[rows["Site condition"] == condition]
        for value, group in site.groupby("Value"):
            shown = f"{value * 100:g}" if parameter == "voxel_size" else f"{value:g}"
            if np.isclose(value, DEFAULTS[parameter]):
                shown = f"**{shown}**"
            cells = [mean_median(group, column, digits) for column, _, digits in columns]
            lines.append(f"| {condition} | {shown} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def coverage_cost_table(per_tile: pd.DataFrame) -> str:
    """What a larger voxel buys in coverage and what it costs in fidelity.

    `Outside footprint` is a share of labelled points, so it is put back on the
    all-points base before differencing; otherwise the two cannot be compared.
    The last column is the share of the *additional* labels that land on cells the
    2D annotation leaves unlabelled.
    """
    voxel = per_tile[per_tile["Parameter"] == "voxel_size"]
    sizes = sorted(voxel["Value"].unique())
    base = min(sizes)
    lines = [f"| Site | Voxel (cm) | Labelled (% of points) | Outside footprint (% of points) "
             f"| Extra labelled vs {base * 100:g} cm | Of which outside footprint |",
             "|---|---:|---:|---:|---:|---:|"]
    for condition in SITE_COLOURS:
        site = voxel[voxel["Site condition"] == condition]
        means = site.groupby("Value")[["Stage-2 labelled points (%)",
                                       "Stage 2 vs input Outside footprint (%)"]].mean()
        labelled = means["Stage-2 labelled points (%)"]
        outside = labelled * means["Stage 2 vs input Outside footprint (%)"] / 100.0
        for size in sizes:
            gained = labelled[size] - labelled[base]
            spread = outside[size] - outside[base]
            share = f"{100 * spread / gained:.0f}%" if gained > 1e-9 else "–"
            shown = (f"**{size * 100:g}**" if np.isclose(size, DEFAULTS["voxel_size"])
                     else f"{size * 100:g}")
            lines.append(f"| {condition} | {shown} | {labelled[size]:.1f} | {outside[size]:.2f} "
                         f"| {gained:+.2f} | {share} |")
    return "\n".join(lines)


def spacing_table(per_tile: pd.DataFrame) -> str:
    """Each voxel size as a multiple of the mean spacing between labelled returns.

    Spacing is planimetric, 1/sqrt(labelled returns per m2). The transfer needs two
    labelled returns inside one cube, so it is *their* spacing that sets the useful
    voxel size, not the spacing of all returns.
    """
    inputs = per_tile[per_tile["Parameter"] == "input"]
    sizes = sorted(per_tile[per_tile["Parameter"] == "voxel_size"]["Value"].unique())
    area = TILE_SIDE_M ** 2
    header = ("| Site | Labelled returns per m² | Mean spacing (cm) | "
              + " | ".join(f"{s * 100:g} cm" for s in sizes) + " |")
    lines = [header, "|---|---:|---:|" + "---:|" * len(sizes)]
    for condition in SITE_COLOURS:
        site = inputs[inputs["Site condition"] == condition]
        density = (site["Structure points"] / area).mean()
        spacing = 100.0 / np.sqrt(density)
        ratios = [f"{s * 100 / spacing:.2f}×" for s in sizes]
        lines.append(f"| {condition} | {density:.0f} | {spacing:.2f} | " + " | ".join(ratios) + " |")
    return "\n".join(lines)


def sensor_scaling_table(per_tile: pd.DataFrame,
                         factors: Sequence[Tuple[float, str]] = SENSOR_FACTORS) -> str:
    """How the voxel would scale with survey density, at two spacings per voxel.

    Spacing goes as 1/sqrt(density), so a denser survey reaches the same occupancy
    with a proportionally smaller voxel.
    """
    inputs = per_tile[per_tile["Parameter"] == "input"]
    area = TILE_SIDE_M ** 2
    spacings = [100.0 / np.sqrt((inputs[inputs["Site condition"] == c]["Structure points"] / area).mean())
                for c in SITE_COLOURS]
    lines = ["| Survey density | Labelled spacing (cm) | Voxel at 2× spacing (cm) |",
             "|---|---:|---:|"]
    for factor, label in factors:
        lo, hi = sorted(s / np.sqrt(factor) for s in spacings)
        lines.append(f"| {label} ({factor:g}×) | {lo:.1f}–{hi:.1f} | {2 * lo:.0f}–{2 * hi:.0f} |")
    return "\n".join(lines)


def voxel_extremes(per_tile: pd.DataFrame) -> str:
    """Per-tile extremes behind the voxel section's every-tile statements."""
    voxel = per_tile[per_tile["Parameter"] == "voxel_size"]
    lines = []
    for condition in SITE_COLOURS:
        site = voxel[voxel["Site condition"] == condition]
        fine = site[np.isclose(site["Value"], min(SWEEPS["voxel_size"]))]
        per_voxel = fine["Structure points per voxel, median"]
        lines.append(
            f"- {condition} site: footprint recall on the lowest tile at any voxel size "
            f"**{site['Stage 2 vs input Footprint recall (%)'].min():.2f}%**; at 5 cm the per-tile "
            f"median of labelled points per voxel ranges {per_voxel.min():g}–{per_voxel.max():g}.")
    return "\n".join(lines)


def class_iou_table(per_tile: pd.DataFrame) -> str:
    """Stage-2 plan-view IoU against the 2D labels, per class and voxel size (mean across
    tiles), with the relative loss from 10 to 20 cm: which growth forms a coarser grid costs most."""
    voxel = per_tile[per_tile["Parameter"] == "voxel_size"]
    sizes = sorted(voxel["Value"].unique())
    header = "| Site | Class | " + " | ".join(f"{s * 100:g} cm" for s in sizes) + " | 10 → 20 cm |"
    lines = [header, "|---|---|" + "---:|" * (len(sizes) + 1)]
    for condition in SITE_COLOURS:
        site = voxel[voxel["Site condition"] == condition]
        means = site.groupby("Value")[[f"Stage 2 vs input IoU {c}" for c in CLASS_ORDER]].mean()
        for name in CLASS_ORDER:
            column = means[f"Stage 2 vs input IoU {name}"]
            if column.isna().all():
                continue
            ten = column.loc[np.isclose(column.index, 0.10)].iloc[0]
            twenty = column.loc[np.isclose(column.index, 0.20)].iloc[0]
            cells = [f"**{fmt(v, 3)}**" if np.isclose(s, DEFAULTS["voxel_size"]) else fmt(v, 3)
                     for s, v in column.items()]
            lines.append(f"| {condition} | {name} | " + " | ".join(cells)
                         + f" | {100 * (twenty - ten) / ten:+.0f}% |")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# The vertical profile: why 10 cm, and what a plan-view lookup costs
# ---------------------------------------------------------------------------

def plot_vertical_profile(profile: pd.DataFrame, path: Path, max_height: float = 6.0) -> Path:
    """Absolute returns per height bin: all returns, those carrying a 2D label, and by class.

    Summed over tiles, so the bars are the number of returns the transfer actually has
    to work with at each height, not a normalised shape.
    """
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 3.4), constrained_layout=True)
    shown = profile[profile["Height (m)"] <= max_height]

    for ax, (title, layers) in zip(axes, [
            ("(a) All returns", ["all"]),
            ("(b) Returns carrying a 2D label", ["labelled"]),
            ("(c) Labelled returns by class", list(CLASS_ORDER))]):
        style_axes(ax)
        ax.grid(axis="y", color=GRID, linewidth=0.6)
        if len(layers) == 1:
            for condition, colour in SITE_COLOURS.items():
                block = shown[(shown["Site condition"] == condition)
                              & (shown["Layer"] == layers[0])]
                totals = block.groupby("Height (m)")["Points"].sum().sort_index()
                ax.plot(totals.to_numpy(), totals.index.to_numpy(), color=colour,
                        linewidth=1.6, label=f"{condition} site")
        else:
            # One site only; two would double the lines and the classes differ between them.
            block = shown[shown["Site condition"] == "Medium"]
            for layer, colour in zip(layers, CLASS_COLOURS):
                totals = block[block["Layer"] == layer].groupby("Height (m)")["Points"].sum().sort_index()
                if totals.sum() == 0:
                    continue
                ax.plot(totals.to_numpy(), totals.index.to_numpy(), color=colour,
                        linewidth=1.4, label=layer)
            ax.legend(frameon=False, fontsize=6.5, labelcolor=INK_SECONDARY, loc="upper right")
        ax.set_title(title, fontsize=8.5, color=INK, loc="left")
        ax.set_xlabel("returns per 10 cm bin", fontsize=7.5, color=INK_SECONDARY)
        ax.set_xscale("log")
        ax.set_ylim(0, max_height)
    axes[0].set_ylabel("height above ground (m)", fontsize=7.5, color=INK_SECONDARY)
    axes[0].legend(frameon=False, fontsize=7, labelcolor=INK_SECONDARY, loc="upper right")
    return save(fig, path)


def vertical_table(profile: pd.DataFrame,
                   bands: Sequence[Tuple[float, float, str]] = HEIGHT_BANDS) -> str:
    """Returns in each height band, summed over tiles: absolute counts and the labelled share."""
    lines = ["| Site | Band | All returns | Labelled returns | Labelled (%) | Labelled per m² |",
             "|---|---|---:|---:|---:|---:|"]
    area = TILE_SIDE_M ** 2
    for condition in SITE_COLOURS:
        site = profile[profile["Site condition"] == condition]
        tiles = site["Tile"].nunique()
        for lo, hi, label in bands:
            band = site[(site["Height (m)"] >= lo) & (site["Height (m)"] < hi)]
            total = band[band["Layer"] == "all"]["Points"].sum()
            labelled = band[band["Layer"] == "labelled"]["Points"].sum()
            share = 100.0 * labelled / total if total else float("nan")
            lines.append(f"| {condition} | {label} | {total:,} | {labelled:,} | "
                         f"{share:.1f} | {labelled / (tiles * area):.1f} |")
    return "\n".join(lines)


def _height_quantile(block: pd.DataFrame, q: float) -> float:
    """Height below which share `q` of a class's returns fall, from the binned counts."""
    counts = block.groupby("Height (m)")["Points"].sum().sort_index()
    if counts.sum() == 0:
        return float("nan")
    cumulative = counts.cumsum() / counts.sum()
    return float(cumulative.index[np.searchsorted(cumulative.to_numpy(), q)])


def class_height_table(profile: pd.DataFrame, quantiles: Sequence[float] = (0.5, 0.85, 0.95, 0.99),
                       tail_m: float = 1.0) -> str:
    """Height percentiles of each class's labelled returns.

    `filters.colorization` is a plan-view raster lookup with no height gate, so every
    return in the column above a labelled cell inherits that cell's class. Where a
    ground class sits under canopy it therefore grows an upper tail that is not the
    growth form. A flat p95-to-p99 step means the class is clean; a large one is that
    contamination, and it is the one error source in the budget that no IoU reveals.
    """
    header = ("| Site | Class | Returns | " + " | ".join(f"p{q * 100:g}" for q in quantiles)
              + f" | Above {tail_m:g} m |")
    lines = [header, "|---|---|---:|" + "---:|" * (len(quantiles) + 1)]
    for condition in SITE_COLOURS:
        site = profile[profile["Site condition"] == condition]
        for name in CLASS_ORDER:
            block = site[site["Layer"] == name]
            total = block["Points"].sum()
            if total == 0:
                continue
            cells = [f"{_height_quantile(block, q):.2f} m" for q in quantiles]
            tail = block[block["Height (m)"] > tail_m]["Points"].sum()
            lines.append(f"| {condition} | {name} | {total:,} | " + " | ".join(cells)
                         + f" | {100 * tail / total:.1f}% |")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 3-4. The matching radius and the Stage-4 parameters
# ---------------------------------------------------------------------------

def plot_parameter_figure(per_tile: pd.DataFrame, path: Path) -> Path:
    fig, axes = plt.subplots(len(PARAMETER_ROWS), len(OTHER_PARAMETERS), figsize=(7.2, 3.9),
                             constrained_layout=True, sharey="row")
    panel = 0
    for row, (column, row_label, worst) in enumerate(PARAMETER_ROWS):
        row_curves = []
        for col, parameter in enumerate(OTHER_PARAMETERS):
            ax = axes[row, col]
            style_axes(ax)
            values = SWEEPS[parameter]
            positions = {v: float(i) for i, v in enumerate(sorted(values))}
            row_curves.append(draw_sweep(ax, per_tile, parameter, column, positions, worst))
            ax.set_xticks(list(positions.values()))
            ax.set_xticklabels([f"{v:g}" for v in sorted(values)])
            ax.set_xlim(-0.5, len(values) - 0.5)
            if row == 0:
                ax.set_title(f"({chr(97 + panel)}) {PARAMETER_LABELS[parameter]}", fontsize=8.5,
                             color=INK, loc="left")
            else:
                ax.set_title(f"({chr(97 + panel)})", fontsize=8.5, color=INK, loc="left")
                ax.set_xlabel(PARAMETER_LABELS[parameter], fontsize=7.5, color=INK_SECONDARY)
            if col == 0:
                ax.set_ylabel(row_label, fontsize=7.5, color=INK_SECONDARY)
            panel += 1
        # The row shares y, so each panel's own limit would be overwritten by the last.
        if column in ZERO_BASED:
            zero_based_ylim(axes[row], row_curves)
    figure_legend(fig, "Default value")
    return save(fig, path)


# What each parameter decides, for the one-line answer to the parameter comments.
WHAT_IT_DECIDES = {
    "voxel_size": "how far a 2D label extends to neighbouring returns",
    "height_match_radius": "how far a point may look for its height above ground",
    "height_prior_out_of_range_weight": "how strongly a class outside its height range is penalised",
    "min_conf": "whether an unlabelled point inside a crown is filled",
    "min_flip_ratio": "whether an existing label is overturned",
}
# Values carrying a unit in the answer table; the rest are dimensionless. Voxel size is
# quoted in centimetres there, as the manuscript and the review comments do.
UNITS = {"voxel_size": " cm", "height_match_radius": " m"}
SCALE = {"voxel_size": 100.0}


def answer_table(per_tile: pd.DataFrame) -> str:
    """One row per parameter: the largest change any tested value made to the final
    labels on any tile. The whole answer to the parameter comments in one table."""
    lines = ["| Parameter | Default | What it decides | Values tested | "
             + " | ".join(f"{c}: largest change (%)" for c in SITE_COLOURS) + " |",
             "|---|---:|---|---|" + "---:|" * len(SITE_COLOURS)]
    for parameter, decides in WHAT_IT_DECIDES.items():
        rows = per_tile[per_tile["Parameter"] == parameter]
        scale, unit = SCALE.get(parameter, 1.0), UNITS.get(parameter, "")
        default = f"{DEFAULTS[parameter] * scale:g}{unit}"
        tested = ", ".join(f"{v * scale:g}" for v in SWEEPS[parameter])
        cells = [fmt(rows[rows["Site condition"] == c]["Final changed vs default (%)"].max(), 2)
                 for c in SITE_COLOURS]
        name = PARAMETER_LABELS[parameter].split(" (")[0]
        lines.append(f"| {name} | {default} | {decides} | {tested} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def input_table(per_tile: pd.DataFrame) -> str:
    """Per-site checks of what the pipeline is given, from the `input` rows. Each is the
    extreme across tiles the claim needs, not an average."""
    inputs = per_tile[per_tile["Parameter"] == "input"]
    lines = ["| Check | " + " | ".join(SITE_COLOURS) + " |", "|---|" + "---:|" * len(SITE_COLOURS)]
    checks = [
        ("Tiles", lambda s: str(s["Tile"].nunique())),
        ("Structure points with an exact Z_norm twin, lowest tile (%)",
         lambda s: fmt(s["Structure points with an exact twin (%)"].min(), 3)),
        ("Largest XY match distance, any tile (m)",
         lambda s: fmt(s["Match distance XY max (m)"].max(), 3)),
        ("Structure points beyond the 0.30 m radius, any tile (%)",
         lambda s: fmt(s["Structure points beyond match radius (%)"].max(), 3)),
        ("Largest class-range shift from XY-only matching, any tile (m)",
         lambda s: fmt(s["Stage-1 range shift, max (m)"].max(), 3)),
    ]
    for label, cell in checks:
        lines.append(f"| {label} | " + " | ".join(cell(inputs[inputs["Site condition"] == c])
                                                  for c in SITE_COLOURS) + " |")
    return "\n".join(lines)


def crown_table(per_tile: pd.DataFrame) -> str:
    """Stages 3-5 at the default configuration, as counts over a site's tiles."""
    defaults = defaults_only(per_tile)
    lines = ["| At the defaults | " + " | ".join(SITE_COLOURS) + " |",
             "|---|" + "---:|" * len(SITE_COLOURS)]
    rows = [
        ("Tiles with a detected crown",
         lambda s: f"{int(s['Stage-3 crowns'].gt(0).sum())} of {s['Tile'].nunique()}"),
        # Not `Stage-3 crowns`: watershed labels can skip IDs, so its max overcounts.
        ("Crowns holding lidar points, put to the Stage-5 vote, all tiles",
         lambda s: f"{int(s['Stage-5 crowns voted'].sum())}"),
        ("… of which more than half mallee, so turned wholly mallee",
         lambda s: f"{int(s['Stage-5 crowns with mallee majority'].sum())}"),
        ("Mallee share of the least-mallee crown, any tile (%)",
         lambda s: fmt(s["Stage-5 least-mallee crown (%)"].min(), 1)),
        ("Lidar points inside crowns (mean / median, %)",
         lambda s: mean_median(s, "Stage-3 points in crowns (%)", 1)),
        ("Stage-4 labels changed (mean / median, % of points)",
         lambda s: mean_median(s, "Stage-4 changed points (%)", 3)),
        ("Stage-4 mean confidence, tiles with crowns (mean)",
         lambda s: fmt(s["Stage-4 mean confidence"].mean(), 4)),
        ("Points turned mallee by the vote (mean / median, %)",
         lambda s: mean_median(s, "Stage-5 crown-majority points (%)", 2)),
    ]
    for label, cell in rows:
        lines.append(f"| {label} | " + " | ".join(cell(defaults[defaults["Site condition"] == c])
                                                  for c in SITE_COLOURS) + " |")
    return "\n".join(lines)


def worst_table(per_tile: pd.DataFrame) -> str:
    """Largest final-label change on any tile, per parameter and site."""
    lines = ["| Parameter | Values tested | "
             + " | ".join(f"{c}: worst tile (%)" for c in SITE_COLOURS) + " |",
             "|---|---|" + "---:|" * len(SITE_COLOURS)]
    for parameter in OTHER_PARAMETERS:
        rows = per_tile[per_tile["Parameter"] == parameter]
        cells = [fmt(rows[rows["Site condition"] == c]["Final changed vs default (%)"].max(), 4)
                 for c in SITE_COLOURS]
        values = ", ".join(f"{v:g}" for v in SWEEPS[parameter])
        lines.append(f"| {PARAMETER_LABELS[parameter]} | {values} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def stage4_table(per_tile: pd.DataFrame, condition: str = "Medium") -> str:
    """The four Stage-1/4 parameters in one table for one site, `mean / median`."""
    columns = [
        ("Stage-4 filled points (%)", "Stage-4 filled (%)", 3),
        ("Stage-4 flipped points (%)", "Stage-4 flipped (%)", 3),
        ("Stage-4 below min_conf (%)", "Evaluated points below min_conf (%)", 3),
        ("Final changed vs default (%)", "Final labels changed (%)", 3),
    ]
    lines = ["| Parameter | Value | " + " | ".join(label for _, label, _ in columns) + " |",
             "|---|---:|" + "---:|" * len(columns)]
    site = per_tile[per_tile["Site condition"] == condition]
    for parameter in OTHER_PARAMETERS:
        for value, group in site[site["Parameter"] == parameter].groupby("Value"):
            shown = f"**{value:g}**" if np.isclose(value, DEFAULTS[parameter]) else f"{value:g}"
            cells = [mean_median(group, column, digits) for column, _, digits in columns]
            lines.append(f"| {PARAMETER_LABELS[parameter]} | {shown} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def min_conf_extremes(per_tile: pd.DataFrame) -> str:
    """Evaluated points below `min_conf` on the worst tile, per value: whether the gate binds."""
    rows = per_tile[per_tile["Parameter"] == "min_conf"]
    lines = []
    for condition in SITE_COLOURS:
        worst = rows[rows["Site condition"] == condition].groupby("Value")["Stage-4 below min_conf (%)"].max()
        cells = ", ".join(f"{value:g}: {share:.4f}%" for value, share in worst.items())
        lines.append(f"- {condition} site, evaluated points below `min_conf` on the worst tile: {cells}.")
    return "\n".join(lines)


def invariance_sentences(per_tile: pd.DataFrame) -> str:
    """What the tables' means cannot show: the extremes across every tile and value."""
    swept = per_tile[per_tile["Parameter"].isin(OTHER_PARAMETERS)]
    lines = []
    for condition in SITE_COLOURS:
        site = swept[swept["Site condition"] == condition]
        tiles = site["Tile"].nunique()
        no_crowns = site.groupby("Tile")["Stage-3 crowns"].max().eq(0).sum()
        lines.append(
            f"- {condition} site: largest final-label change across {tiles} tiles and every value of the "
            f"four parameters **{site['Final changed vs default (%)'].max():.4f}%** of points; largest "
            f"Stage-4 change {site['Stage-4 changed points (%)'].max():.3f}%; tiles without a detected "
            f"crown {no_crowns} of {tiles}.")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# The two figures that re-read the lidar tiles
# ---------------------------------------------------------------------------

# Emphasis: the spread labels are the story (slot-2 accent), labels inside the
# footprint the context (slot 1), unlabelled returns recede.
INSIDE = "#2a78d6"
SPREAD = "#eb6834"
UNLABELLED = "#c3c2b7"
WINDOW_GRID = "#d4d3cc"


def tile_paths(data_root: Path, site: str, tile: str) -> Tuple[Path, Path, Path]:
    base = data_root / site
    return (base / Z_NORM_DIR / f"z_norm_csf_tiles_multispectral.{tile}.las",
            base / STRUCTURE_DIR[site] / f"mask_tiles_multispectral.{tile}.las",
            base / INPUT_DIR / f"mask_structure_3072.{tile}.tif")


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

    def window_sum(table_: np.ndarray) -> np.ndarray:
        return table_[k:, k:] - table_[:-k, k:] - table_[k:, :-k] + table_[:-k, :-k]

    points = window_sum(summed(np.ones(len(x))))
    labelled = window_sum(summed((input_points > 0).astype(float)))
    spread_count = window_sum(summed(spread.astype(float)))
    share = np.divide(labelled, points, out=np.zeros_like(points), where=points > 0)
    spread_count[(share < 0.25) | (share > 0.75)] = -1
    row, col = np.unravel_index(np.argmax(spread_count), spread_count.shape)
    return x0 + col * step, y0 + row * step


def footprint_outline(labels: np.ndarray, transform,
                      bounds: Tuple[float, float, float, float]) -> LineCollection:
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
        segments.append([(xe, transform.f + r * transform.e),
                         (xe, transform.f + (r + 1) * transform.e)])
    rows, cols = np.nonzero(lab[1:, :] != lab[:-1, :])  # horizontal edges
    for r, c in zip(rows + r0 + 1, cols + c0):
        ye = transform.f + r * transform.e
        segments.append([(transform.c + c * transform.a, ye),
                         (transform.c + (c + 1) * transform.a, ye)])
    return LineCollection(segments, colors=INK, linewidths=0.7, zorder=4)


def draw_window_panel(ax: plt.Axes, x: np.ndarray, y: np.ndarray, classes: np.ndarray,
                      input_points: np.ndarray, corner: Tuple[float, float], side: float,
                      outline: LineCollection,
                      grid: Tuple[float, float, float] | None) -> float:
    """One plan-view panel; returns the share of labelled points spread past the footprint."""
    x0, y0 = corner
    if grid is not None:
        size, gx, gy = grid
        for value in np.arange(gx + np.ceil((x0 - gx) / size) * size, x0 + side, size):
            ax.axvline(value, color=WINDOW_GRID, linewidth=0.4, zorder=1)
        for value in np.arange(gy + np.ceil((y0 - gy) / size) * size, y0 + side, size):
            ax.axhline(value, color=WINDOW_GRID, linewidth=0.4, zorder=1)
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
        spine.set_color(AXIS)
        spine.set_linewidth(0.6)
    return 100.0 * np.count_nonzero(spread) / max(int(np.count_nonzero(labelled)), 1)


def plot_transfer_window(data_root: Path, site: str, tile: str, output: Path, side: float = 2.0,
                         sizes: Sequence[float] = SWEEPS["voxel_size"]) -> Path:
    """What voxel size does to the transferred labels, drawn on one tile.

    The companion to the voxel figure: where that counts labels spread past their 2D
    footprint, this shows them. Every lidar return in a small plan-view window is drawn
    from above, blue where its label lies inside its 2D footprint, orange where the
    transfer spread a label past it, grey where unlabelled.

    The window is chosen, not typical: the one holding the most spread labels at the
    largest voxel size, among windows a quarter to three quarters labelled.
    """
    z_path, s_path, input_path = tile_paths(data_root, site, tile)
    las_z, las_s = laspy.read(str(z_path)), laspy.read(str(s_path))
    x, y = np.asarray(las_z.x), np.asarray(las_z.y)
    labels, transform = read_labels(input_path)
    input_points = classes_beneath(labels, transform, x, y)
    transfers = {}
    for size in sizes:
        with contextlib.redirect_stdout(io.StringIO()):
            transfers[size] = cartesian_voxel_merge(las_z, las_s, voxel_xy=size, voxel_z=size)
    # voxel_merge's grid origin: the minimum over both clouds.
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
        share = draw_window_panel(ax, x[inside], y[inside], classes[inside], input_points[inside],
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
        Line2D([], [], color=WINDOW_GRID, linewidth=0.8, label="Voxel columns"),
    ]
    fig.legend(handles=handles, loc="outside upper center", ncol=3, frameon=False, fontsize=7.5,
               labelcolor=INK_SECONDARY)
    print(f"Window lower-left ({corner[0]:.2f}, {corner[1]:.2f}), {side:g} m")
    return save(fig, output)


def plot_voxel_volumes(las_path: Path, output: Path, tile: str,
                       sizes: Sequence[float] = SWEEPS["voxel_size"],
                       max_points: int = 180_000, seed: int = 42) -> Path:
    """One tile divided into voxels at each size, in 3D, coloured by returns per voxel.

    Geometry only: how coarse each grid is, not what happens to the labels. Drawn from a
    seeded random sample of the tile's returns, so the voxel counts describe the sample.
    Both rows use Z_norm, with ground at zero. Lower panels count voxel centres in
    shared 0.5 m height-above-ground bands.
    """
    las = laspy.read(str(las_path))
    if "Z_norm" not in las.point_format.dimension_names:
        raise ValueError(f"{las_path} requires Z_norm for ground-normalized voxel profiles.")
    points = np.column_stack([np.asarray(las[axis], dtype=np.float64)
                              for axis in ("x", "y", "Z_norm")])
    if not len(points) or not np.isfinite(points).all():
        raise ValueError("Voxel profiles require nonempty, finite X, Y and Z_norm coordinates.")
    points[:, :2] -= points[:, :2].min(axis=0)
    if len(points) > max_points:
        points = points[np.random.default_rng(seed).choice(len(points), max_points, replace=False)]
    limits = [(0.0, max(float(points[:, axis].max()) + max(sizes), max(sizes)))
              for axis in range(3)]

    voxel_sets = []
    for size in sizes:
        index = np.floor(points / size).astype(np.int32)
        unique, counts = np.unique(index, axis=0, return_counts=True)
        centres = (unique.astype(np.float64) + 0.5) * size
        order = np.argsort(counts)
        voxel_sets.append((centres[order], counts[order].astype(np.float32)))

    density = np.concatenate([c for _, c in voxel_sets if len(c)])
    vmin = max(1.0, float(np.percentile(density, 5)))
    norm = colors.LogNorm(vmin=vmin, vmax=max(vmin + 1.0, float(np.percentile(density, 99.5))))

    lowest_centre = min(float(centres[:, 2].min()) for centres, _ in voxel_sets)
    height_min = np.floor(min(0.0, lowest_centre) / 0.5) * 0.5
    height_edges = np.arange(height_min, np.ceil(limits[2][1] / 0.5) * 0.5 + 0.25, 0.5)
    limits[2] = (float(height_edges[0]), float(height_edges[-1]))
    profiles = [np.histogram(centres[:, 2], bins=height_edges)[0]
                for centres, _ in voxel_sets]
    profile_max = max(int(profile.max()) for profile in profiles)
    font_size = 16
    fig = plt.figure(figsize=(5.0 * len(sizes), 11.0), layout="none")
    grid = fig.add_gridspec(2, len(sizes), height_ratios=(1.25, 1),
                           left=0.045, right=0.92, bottom=0.09, top=0.90,
                           wspace=0.27, hspace=0.16)
    axes, scatter = [], None
    for index, (size, (centres, counts)) in enumerate(zip(sizes, voxel_sets)):
        ax = fig.add_subplot(grid[0, index], projection="3d")
        axes.append(ax)
        scatter = ax.scatter(centres[:, 0], centres[:, 1], centres[:, 2], c=counts, s=4,
                             cmap="plasma", norm=norm, alpha=0.88, linewidths=0)
        ax.set_title(f"{size * 100:g} cm\n{len(counts):,} occupied voxels", fontsize=font_size,
                     color=INK)
        for axis, (label, limit) in zip("xyz", zip(("Local X (m)", "Local Y (m)", "Z_norm (m)"), limits)):
            getattr(ax, f"set_{axis}label")(label, fontsize=font_size, color=INK_SECONDARY,
                                           labelpad=24)
            getattr(ax, f"set_{axis}lim")(*limit)
        ax.tick_params(colors=INK_SECONDARY, labelsize=font_size)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=3, prune="both"))
        ax.yaxis.set_major_locator(MaxNLocator(nbins=3, prune="both"))
        ax.zaxis.set_major_locator(MaxNLocator(nbins=2))
        ax.view_init(elev=24, azim=-62)
        ax.set_box_aspect([max(hi - lo, 1.0) for lo, hi in limits], zoom=0.8)
        profile_ax = fig.add_subplot(grid[1, index])
        style_axes(profile_ax)
        profile_ax.tick_params(labelsize=font_size)
        profile_ax.stairs(profiles[index], height_edges, orientation="horizontal",
                          color=SITE_COLOURS["Medium"], linewidth=1.4)
        profile_ax.set_xlim(0, max(profile_max * 1.05, 1))
        profile_ax.xaxis.set_major_locator(MaxNLocator(nbins=3, integer=True))
        profile_ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
        profile_ax.set_ylim(height_edges[0], height_edges[-1])
        profile_ax.set_xlabel("Occupied voxels\nper 0.5 m band", fontsize=font_size, color=INK_SECONDARY)
        if index == 0:
            profile_ax.set_ylabel("Height above ground (m)", fontsize=font_size,
                                  color=INK_SECONDARY)
    bar = fig.colorbar(scatter, cax=fig.add_axes([0.97, 0.58, 0.008, 0.28]))
    bar.set_label("returns per voxel", fontsize=font_size, color=INK_SECONDARY)
    bar.ax.tick_params(colors=INK_SECONDARY, labelsize=font_size, which="both")
    fig.suptitle(f"Tile {tile}, random sample of {len(points):,} returns", fontsize=font_size,
                 color=INK_SECONDARY)
    return save(fig, output)


# ---------------------------------------------------------------------------

def write_tables(per_tile: pd.DataFrame, label_quality: pd.DataFrame, path: Path) -> None:
    sections: List[str] = [
        "# Reviewer response tables", "",
        "Generated by `figures.py` from the CSVs `sensitivity.py` writes. Cells are "
        "`mean / median` across a site's tiles unless the row says otherwise. Defaults in bold.",
        "",
        "## 1. Error budget (discussion 6.1)", "",
        "### Where the error comes from, on one scale", "", error_budget_table(per_tile, label_quality), "",
        "### Every step in its own units", "", error_sources_table(per_tile, label_quality), "",
        "## 2. Voxel size (3.2.2)", "",
        "### Selected transfer metrics", "", table(per_tile, "voxel_size", TRANSFER_COLUMNS), "",
        "### Voxel size against the spacing of labelled returns", "", spacing_table(per_tile), "",
        "## 3. What the pipeline is given (the matching radius)", "", input_table(per_tile), "",
        "## 4. The parameter values", "",
        "### The answer in brief", "", answer_table(per_tile), "",
        "### Final labels: the worst tile", "", worst_table(per_tile), "",
        invariance_sentences(per_tile), "",
    ]
    path.write_text("\n".join(sections))
    print(f"Wrote: {path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--assets", type=Path, default=OUTPUT_DIR,
                        help="Directory holding sensitivity.py's CSVs; figures are written here.")
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--site", default="site1_1_DD0001", help="Site for the tile figures.")
    parser.add_argument("--tile", default="106", help="Tile for the tile figures.")
    parser.add_argument("--no-tile-figures", action="store_true",
                        help="Skip the two figures that re-read the lidar tiles.")
    parser.add_argument("--voxel-volumes", action="store_true",
                        help="Also draw the 3D voxel-geometry figure (appendix).")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    per_tile = pd.read_csv(args.assets / "parameter_sensitivity_per_tile.csv")
    label_quality = pd.read_csv(args.assets / "label_quality_summary.csv")

    plot_error_budget(per_tile, label_quality, args.assets / "supplementary_error_budget")
    plot_voxel_figure(per_tile, args.assets / "supplementary_voxel_size_sensitivity")
    plot_parameter_figure(per_tile, args.assets / "supplementary_parameter_sensitivity")
    write_tables(per_tile, label_quality, args.assets / "reviewer_response_tables.md")

    if not args.no_tile_figures:
        plot_transfer_window(args.data_root, args.site, args.tile,
                             args.assets / f"voxel_transfer_tile{args.tile}")
        if args.voxel_volumes:
            z_path = (args.data_root / args.site / Z_NORM_DIR
                      / f"z_norm_csf_tiles_multispectral.{args.tile}.las")
            plot_voxel_volumes(z_path, args.assets / f"voxel_volumes_tile{args.tile}", args.tile)


if __name__ == "__main__":
    main()
