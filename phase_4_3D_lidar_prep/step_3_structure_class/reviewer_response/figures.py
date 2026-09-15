#!/usr/bin/env python3
"""The three supplementary figures and the tables behind the four reviewer replies.

Reads the two CSVs `sensitivity.py` writes and puts these in `--assets`:

    supplementary_error_budget.png / .pdf             where the label error comes from (6.1)
    supplementary_voxel_size_sensitivity.png / .pdf   the voxel sweep, 5-25 cm (3.2.2)
    supplementary_parameter_sensitivity.png / .pdf    radius, height-prior weight,
                                                      min_conf, min_flip_ratio
    reviewer_response_tables.md                       the six tables, as markdown

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
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))

from sensitivity import DEFAULTS, OUTPUT_DIR, SWEEPS  # noqa: E402

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


def invariance_sentences(per_tile: pd.DataFrame) -> str:
    """What the tables' means cannot show: the extremes across every tile and value."""
    swept = per_tile[per_tile["Parameter"].isin(OTHER_PARAMETERS)]
    defaults = defaults_only(per_tile)
    lines = []
    for condition in SITE_COLOURS:
        site = swept[swept["Site condition"] == condition]
        tiles = site["Tile"].nunique()
        no_crowns = site.groupby("Tile")["Stage-3 crowns"].max().eq(0).sum()
        # The confidence the `min_conf` answer rests on: the gate only binds below it.
        confidence = defaults[defaults["Site condition"] == condition]["Stage-4 mean confidence"]
        lines.append(
            f"- {condition} site: largest final-label change across {tiles} tiles and every value of the "
            f"four parameters **{site['Final changed vs default (%)'].max():.4f}%** of points; largest "
            f"Stage-4 change {site['Stage-4 changed points (%)'].max():.3f}%; mean Stage-4 confidence at "
            f"the defaults {fmt(confidence.mean(), 3)}; tiles without a detected crown "
            f"{no_crowns} of {tiles}.")
    return "\n".join(lines)


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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    per_tile = pd.read_csv(args.assets / "parameter_sensitivity_per_tile.csv")
    label_quality = pd.read_csv(args.assets / "label_quality_summary.csv")

    plot_error_budget(per_tile, label_quality, args.assets / "supplementary_error_budget")
    plot_voxel_figure(per_tile, args.assets / "supplementary_voxel_size_sensitivity")
    plot_parameter_figure(per_tile, args.assets / "supplementary_parameter_sensitivity")
    write_tables(per_tile, label_quality, args.assets / "reviewer_response_tables.md")


if __name__ == "__main__":
    main()
