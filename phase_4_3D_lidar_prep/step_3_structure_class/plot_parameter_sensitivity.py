#!/usr/bin/env python3
"""Supplementary figures and tables from `parameter_sensitivity.py`'s per-tile CSV.

Writes into --output-dir (default: beside the CSV):

    supplementary_voxel_size_sensitivity.png / .pdf   the voxel sweep, 5-25 cm
    supplementary_parameter_sensitivity.png / .pdf    radius, height-prior weight,
                                                      min_conf, min_flip_ratio
    parameter_sensitivity_tables.md                   the numbers the README quotes

Every curve is the median across a site's tiles, with the inter-quartile range
shaded; the default value is marked. No number is computed anywhere else, so
rerun this after any rerun of the sweep.

Usage
-----
python plot_parameter_sensitivity.py --csv assets/parameter_sensitivity/parameter_sensitivity_per_tile.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

from parameter_sensitivity import DEFAULTS, SITE_CONFIG, SWEEPS  # noqa: E402

# Categorical slots 1-2 of the reference palette, validated all-pairs on white.
SITE_COLOURS = {"Medium": "#2a78d6", "Low": "#eb6834"}
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"

PARAMETER_LABELS = {
    "voxel_size": "Voxel size (cm)",
    "height_match_radius": "Matching radius (m)",
    "height_prior_out_of_range_weight": "Out-of-range weight",
    "min_conf": "min_conf",
    "min_flip_ratio": "min_flip_ratio",
}

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
OTHER_PARAMETERS = ("height_match_radius", "height_prior_out_of_range_weight", "min_conf", "min_flip_ratio")

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

# How densely the labelled returns fill the grid. A voxel holding a single labelled
# return has nothing to merge, so the transfer there reduces to copying the 2D raster.
DENSITY_COLUMNS = [
    ("Structure voxels", "Occupied voxels", 0),
    ("Structure points per voxel, median", "Labelled points per voxel", 1),
    ("Single-point voxels (%)", "Single-point voxels (%)", 1),
    ("Mixed-class voxels (%)", "Mixed-class voxels (%)", 2),
    ("Stage 2 vs input Footprint recall (%)", "Footprint recall (%)", 2),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--csv", type=Path, required=True,
                        help="parameter_sensitivity_per_tile.csv")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="Default: the CSV's directory.")
    return parser.parse_args()


def style_axes(ax: plt.Axes) -> None:
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
        ax.spines[side].set_linewidth(0.8)
    ax.tick_params(colors=INK_SECONDARY, labelsize=7.5, length=3, width=0.8)
    ax.grid(axis="y", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def quartiles(frame: pd.DataFrame, column: str, worst: bool = False) -> pd.DataFrame:
    """Median and IQR across tiles per value, or with `worst` the largest tile value (no band)."""
    grouped = frame.groupby("Value")[column]
    if worst:
        largest = grouped.max()
        return pd.DataFrame({"q1": largest, "median": largest, "q3": largest}).sort_index()
    return pd.DataFrame({"q1": grouped.quantile(0.25), "median": grouped.median(),
                         "q3": grouped.quantile(0.75)}).sort_index()


def draw_sweep(ax: plt.Axes, per_tile: pd.DataFrame, parameter: str, column: str,
               positions: Dict[float, float] | None = None, worst: bool = False) -> Dict[str, pd.DataFrame]:
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
    x_default = positions[default] if positions else (default * 100.0 if parameter == "voxel_size" else default)
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


def save(fig: plt.Figure, path: Path) -> None:
    for suffix in (".png", ".pdf"):
        fig.savefig(path.with_suffix(suffix), dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Wrote: {path.with_suffix('.png')} (+ .pdf)")


def figure_legend(fig: plt.Figure, default_label: str) -> None:
    handles = [
        Line2D([], [], color=colour, linewidth=1.6, marker="o", markersize=5.5,
               markeredgecolor="white", markeredgewidth=1.2, label=f"{condition} site")
        for condition, colour in SITE_COLOURS.items()
    ]
    handles.append(Line2D([], [], color=INK_MUTED, linewidth=0.8, label=default_label))
    fig.legend(handles=handles, loc="outside upper center", ncol=len(handles), frameon=False,
               fontsize=8, labelcolor=INK_SECONDARY)


def plot_voxel_figure(per_tile: pd.DataFrame, path: Path) -> None:
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
    save(fig, path)


def plot_parameter_figure(per_tile: pd.DataFrame, path: Path) -> None:
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
    save(fig, path)


# What each parameter decides, for the one-line answer to the three review comments.
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


def fmt(value: float, digits: int) -> str:
    return "–" if pd.isna(value) else f"{value:.{digits}f}"


def mean_median(frame: pd.DataFrame, column: str, digits: int) -> str:
    return f"{fmt(frame[column].mean(), digits)} / {fmt(frame[column].median(), digits)}"


def table(per_tile: pd.DataFrame, parameter: str, columns: Sequence[Tuple[str, str, int]]) -> str:
    """Markdown table: one row per (site, value), `mean / median` across tiles."""
    header = "| Site | " + PARAMETER_LABELS[parameter] + " | " + " | ".join(label for _, label, _ in columns) + " |"
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


def answer_table(per_tile: pd.DataFrame) -> str:
    """One row per parameter: the largest change any tested value made to the final
    labels on any tile. The whole answer to the three review comments in one table."""
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
        lines.append(f"| {name} | {default} | {decides} | {tested} | "
                     + " | ".join(cells) + " |")
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
            shown = f"**{size * 100:g}**" if np.isclose(size, DEFAULTS["voxel_size"]) else f"{size * 100:g}"
            lines.append(f"| {condition} | {shown} | {labelled[size]:.1f} | {outside[size]:.2f} "
                         f"| {gained:+.2f} | {share} |")
    return "\n".join(lines)


def auto_resolution(site: pd.DataFrame) -> str:
    """How many tiles `auto` crown detection resolves to each (min_tree_dist, sigma) pair."""
    pairs = site.groupby(["Auto min_tree_dist (m)", "Auto smoothing_sigma (px)"])["Tile"].nunique()
    return "; ".join(f"{d:g} m / {s:.2f} px on {n}" for (d, s), n in pairs.items())


def input_table(per_tile: pd.DataFrame) -> str:
    """Per-site checks of what the pipeline is given, from the `input` rows. Each is the
    extreme across tiles the claim needs, not an average."""
    inputs = per_tile[per_tile["Parameter"] == "input"]
    lines = ["| Check | " + " | ".join(SITE_COLOURS) + " |", "|---|" + "---:|" * len(SITE_COLOURS)]
    checks = [
        ("Tiles", lambda s: str(s["Tile"].nunique())),
        ("Structure points with an exact Z_norm twin, lowest tile (%)",
         lambda s: fmt(s["Structure points with an exact twin (%)"].min(), 3)),
        ("Largest XY match distance, any tile (m)", lambda s: fmt(s["Match distance XY max (m)"].max(), 3)),
        ("Structure points beyond the 0.30 m radius, any tile (%)",
         lambda s: fmt(s["Structure points beyond match radius (%)"].max(), 3)),
        ("Heights taken from another return in the same column, range over tiles (%)",
         lambda s: f"{fmt(s['Stage-1 heights from another return (%)'].min(), 1)}–"
                   f"{fmt(s['Stage-1 heights from another return (%)'].max(), 1)}"),
        ("Largest class-range shift that causes, any tile (m)",
         lambda s: fmt(s["Stage-1 range shift, max (m)"].max(), 3)),
        ("Lidar points on labelled 5 cm cells missing from the structure cloud, all tiles",
         lambda s: f"{int((s['Points on labelled 5 cm cells'] - s['Structure points']).sum()):,} "
                   f"({100 * (s['Points on labelled 5 cm cells'] - s['Structure points']).sum() / s['Points on labelled 5 cm cells'].sum():.3f}%)"),
        ("Plan-view mIoU of the 5 cm labels themselves (mean / median)",
         lambda s: mean_median(s, "Final vs input mIoU", 3)),
        ("5 cm labels lying outside the 1 cm footprint (mean / median, %)",
         lambda s: mean_median(s, "Final vs 1 cm Outside footprint (%)", 1)),
        ("Crown detection in the sweep (as the delivered labels were made)",
         lambda s: "`auto`" if SITE_CONFIG[s["Site"].iloc[0]].get("min_tree_dist") == "auto" else "defaults"),
        ("`auto` resolves to (min_tree_dist / smoothing_sigma), tiles", auto_resolution),
        ("Canopy point spacing `auto` starts from, range over tiles (m)",
         lambda s: (f"{fmt(s['Canopy point spacing, 60th percentile (m)'].min(), 3)}–"
                    f"{fmt(s['Canopy point spacing, 60th percentile (m)'].max(), 3)}")),
    ]
    for label, cell in checks:
        lines.append(f"| {label} | " + " | ".join(cell(inputs[inputs["Site condition"] == c])
                                                   for c in SITE_COLOURS) + " |")
    return "\n".join(lines)


def crown_table(per_tile: pd.DataFrame) -> str:
    """Stages 3-5 at the default configuration, as counts over a site's tiles."""
    defaults = per_tile[(per_tile["Parameter"] == "voxel_size")
                        & np.isclose(per_tile["Value"], DEFAULTS["voxel_size"])]
    lines = ["| At the defaults | " + " | ".join(SITE_COLOURS) + " |", "|---|" + "---:|" * len(SITE_COLOURS)]
    rows = [
        ("Tiles with a detected crown", lambda s: f"{int(s['Stage-3 crowns'].gt(0).sum())} of {s['Tile'].nunique()}"),
        # Not `Stage-3 crowns`: watershed labels can skip IDs, so its max overcounts.
        ("Crowns holding lidar points, put to the Stage-5 vote, all tiles",
         lambda s: f"{int(s['Stage-5 crowns voted'].sum())}"),
        ("… of which more than half mallee, so turned wholly mallee",
         lambda s: f"{int(s['Stage-5 crowns with mallee majority'].sum())}"),
        ("Mallee share of the least-mallee crown, any tile (%)", lambda s: fmt(s["Stage-5 least-mallee crown (%)"].min(), 1)),
        ("Lidar points inside crowns (mean / median, %)", lambda s: mean_median(s, "Stage-3 points in crowns (%)", 1)),
        ("Stage-4 labels changed (mean / median, % of points)", lambda s: mean_median(s, "Stage-4 changed points (%)", 3)),
        ("Stage-4 mean confidence, tiles with crowns (mean)", lambda s: fmt(s["Stage-4 mean confidence"].mean(), 4)),
        ("Points turned mallee by the vote (mean / median, %)", lambda s: mean_median(s, "Stage-5 crown-majority points (%)", 2)),
    ]
    for label, cell in rows:
        lines.append(f"| {label} | " + " | ".join(cell(defaults[defaults["Site condition"] == c])
                                                   for c in SITE_COLOURS) + " |")
    return "\n".join(lines)


def worst_table(per_tile: pd.DataFrame) -> str:
    """Largest final-label change on any tile, per parameter and site."""
    lines = ["| Parameter | Values tested | " + " | ".join(f"{c}: worst tile (%)" for c in SITE_COLOURS) + " |",
             "|---|---|" + "---:|" * len(SITE_COLOURS)]
    for parameter in OTHER_PARAMETERS:
        rows = per_tile[per_tile["Parameter"] == parameter]
        cells = [fmt(rows[rows["Site condition"] == c]["Final changed vs default (%)"].max(), 4)
                 for c in SITE_COLOURS]
        values = ", ".join(f"{v:g}" for v in SWEEPS[parameter])
        lines.append(f"| {PARAMETER_LABELS[parameter]} | {values} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def stage4_table(per_tile: pd.DataFrame, condition: str) -> str:
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
    """Evaluated points below `min_conf` on the worst tile, per value: whether the gate binds anywhere."""
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


def class_iou_table(per_tile: pd.DataFrame) -> str:
    """Stage-2 plan-view IoU against the 2D labels, per class and voxel size (mean across
    tiles), with the relative loss from 10 to 20 cm: which growth forms a coarser grid costs most."""
    voxel = per_tile[per_tile["Parameter"] == "voxel_size"]
    classes = ["Forb", "Woody debris", "Hummock grass", "Shrub", "Mallee"]
    sizes = sorted(voxel["Value"].unique())
    header = "| Site | Class | " + " | ".join(f"{s * 100:g} cm" for s in sizes) + " | 10 → 20 cm |"
    lines = [header, "|---|---|" + "---:|" * (len(sizes) + 1)]
    for condition in SITE_COLOURS:
        site = voxel[voxel["Site condition"] == condition]
        means = site.groupby("Value")[[f"Stage 2 vs input IoU {c}" for c in classes]].mean()
        for name in classes:
            column = means[f"Stage 2 vs input IoU {name}"]
            if column.isna().all():
                continue
            ten, twenty = column.loc[np.isclose(column.index, 0.10)].iloc[0], column.loc[np.isclose(column.index, 0.20)].iloc[0]
            cells = [f"**{fmt(v, 3)}**" if np.isclose(s, DEFAULTS["voxel_size"]) else fmt(v, 3)
                     for s, v in column.items()]
            lines.append(f"| {condition} | {name} | " + " | ".join(cells) + f" | {100 * (twenty - ten) / ten:+.0f}% |")
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


def write_tables(per_tile: pd.DataFrame, path: Path) -> None:
    sections: List[str] = [
        "# Parameter sensitivity tables", "",
        "Generated by `plot_parameter_sensitivity.py` from `parameter_sensitivity_per_tile.csv`. "
        "Cells are `mean / median` across a site's tiles unless the row says otherwise. "
        "Defaults in bold.", "",
        "## The answer in brief", "", answer_table(per_tile), "",
        "## Inputs", "", input_table(per_tile), "",
        "## Voxel size: the transfer (Stage 2)", "", table(per_tile, "voxel_size", TRANSFER_COLUMNS), "",
        voxel_extremes(per_tile), "",
        "## Voxel size: point density in the grid", "", table(per_tile, "voxel_size", DENSITY_COLUMNS), "",
        "## Voxel size: what a larger voxel buys and costs", "", coverage_cost_table(per_tile), "",
        "## Voxel size: agreement per class (Stage 2, mean across tiles)", "", class_iou_table(per_tile), "",
        "## Voxel size: the final labels", "", table(per_tile, "voxel_size", FINAL_COLUMNS), "",
        "## Stages 3-5 at the defaults", "", crown_table(per_tile), "",
        "## Matching radius, height-prior weight, min_conf, min_flip_ratio (Medium site)", "",
        stage4_table(per_tile, "Medium"), "",
        min_conf_extremes(per_tile), "",
        "## Final labels: the worst tile", "", worst_table(per_tile), "",
        invariance_sentences(per_tile), "",
    ]
    path.write_text("\n".join(sections))
    print(f"Wrote: {path}")


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir or args.csv.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    per_tile = pd.read_csv(args.csv)
    plot_voxel_figure(per_tile, output_dir / "supplementary_voxel_size_sensitivity")
    plot_parameter_figure(per_tile, output_dir / "supplementary_parameter_sensitivity")
    write_tables(per_tile, output_dir / "parameter_sensitivity_tables.md")


if __name__ == "__main__":
    main()
