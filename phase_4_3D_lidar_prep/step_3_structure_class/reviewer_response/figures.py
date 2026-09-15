#!/usr/bin/env python3
"""Create the reviewer figures and compact result tables.

Run sensitivity.py first, then:

    python figures.py

Outputs
-------
* supplementary_voxel_sensitivity.png/.pdf
* supplementary_parameter_sensitivity.png/.pdf
* supplementary_pipeline_stage_effects.png/.pdf
* reviewer_summary.md
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from sensitivity import DEFAULTS, OUTPUT_DIR, SWEEPS  # noqa: E402


SITE_COLOURS = {"Medium": "#2a78d6", "Low": "#eb6834"}
PARAMETER_LABELS = {
    "voxel_size": "Voxel size (cm)",
    "height_match_radius": "Matching radius (m)",
    "height_prior_out_of_range_weight": "Out-of-range height weight",
    "min_conf": "Confidence threshold",
    "min_flip_ratio": "Flip ratio",
}
OTHER_PARAMETERS = (
    "height_match_radius",
    "height_prior_out_of_range_weight",
    "min_conf",
    "min_flip_ratio",
)
STAGE_ORDER = (
    "1 Height matching",
    "2 Voxel transfer",
    "3 Crown detection",
    "4 Polar refinement",
    "5 Consolidation",
)


def style_axis(ax: plt.Axes) -> None:
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#dddddd", linewidth=0.7)
    ax.set_axisbelow(True)


def save_figure(fig: plt.Figure, output: Path) -> None:
    for suffix in (".png", ".pdf"):
        fig.savefig(
            output.with_suffix(suffix),
            dpi=300,
            bbox_inches="tight",
            facecolor="white",
        )
    plt.close(fig)
    print(f"Wrote {output.with_suffix('.png')} and {output.with_suffix('.pdf')}")


def mean_sd(
    data: pd.DataFrame, parameter: str, metric: str, condition: str
) -> pd.DataFrame:
    rows = data[
        (data["Parameter"] == parameter)
        & (data["Site condition"] == condition)
    ]
    return rows.groupby("Value")[metric].agg(["mean", "std"]).sort_index()


def draw_parameter(
    ax: plt.Axes,
    data: pd.DataFrame,
    parameter: str,
    metric: str,
    ylabel: str,
    scale: float = 1.0,
) -> None:
    for condition, colour in SITE_COLOURS.items():
        summary = mean_sd(data, parameter, metric, condition)
        x = summary.index.to_numpy(dtype=float) * scale
        y = summary["mean"].to_numpy(dtype=float)
        error = summary["std"].fillna(0).to_numpy(dtype=float)
        ax.errorbar(
            x,
            y,
            yerr=error,
            marker="o",
            markersize=4,
            linewidth=1.5,
            capsize=2,
            color=colour,
            label=condition,
        )
    ax.axvline(
        DEFAULTS[parameter] * scale,
        color="#555555",
        linestyle="--",
        linewidth=1,
        label="Delivered value",
    )
    ax.set_xlabel(PARAMETER_LABELS[parameter])
    ax.set_ylabel(ylabel)
    style_axis(ax)


def voxel_figure(data: pd.DataFrame, output_dir: Path) -> None:
    panels = (
        ("Reference mIoU (%)", "mIoU at source-labelled points (%)"),
        ("Stage 2 labelled points (%)", "Lidar points labelled after transfer (%)"),
        ("Final changed from default (%)", "Final labels changed from 10 cm (%)"),
    )
    fig, axes = plt.subplots(1, 3, figsize=(9.2, 3.1), constrained_layout=True)
    for index, (ax, (metric, ylabel)) in enumerate(zip(axes, panels)):
        draw_parameter(ax, data, "voxel_size", metric, ylabel, scale=100.0)
        ax.set_title(f"({chr(97 + index)})", loc="left")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False)
    save_figure(fig, output_dir / "supplementary_voxel_sensitivity")


def parameter_figure(data: pd.DataFrame, output_dir: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(7.6, 5.6), constrained_layout=True)
    for index, (ax, parameter) in enumerate(zip(axes.ravel(), OTHER_PARAMETERS)):
        draw_parameter(
            ax,
            data,
            parameter,
            "Final changed from default (%)",
            "Final labels changed from delivered setting (%)",
        )
        ax.set_title(
            f"({chr(97 + index)}) {PARAMETER_LABELS[parameter]}", loc="left"
        )
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False)
    save_figure(fig, output_dir / "supplementary_parameter_sensitivity")


def stage_figure(stages: pd.DataFrame, output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.4, 3.6), constrained_layout=True)
    x = np.arange(len(STAGE_ORDER), dtype=float)
    width = 0.36
    for offset, (condition, colour) in zip((-0.5, 0.5), SITE_COLOURS.items()):
        site = stages[stages["Site condition"] == condition].set_index("Stage")
        values = site.reindex(STAGE_ORDER)["mean"].to_numpy(dtype=float)
        errors = site.reindex(STAGE_ORDER)["std"].fillna(0).to_numpy(dtype=float)
        ax.bar(
            x + offset * width,
            values,
            width,
            yerr=errors,
            capsize=2,
            color=colour,
            label=condition,
        )
    ax.set_xticks(x)
    ax.set_xticklabels(
        [stage.split(" ", 1)[1] for stage in STAGE_ORDER],
        rotation=20,
        ha="right",
    )
    ax.set_ylabel("Points affected (%)")
    ax.legend(frameon=False)
    style_axis(ax)
    ax.text(
        0.01,
        0.98,
        "Stage 3 is exposure to crown processing, not an error estimate.",
        transform=ax.transAxes,
        va="top",
        fontsize=8,
    )
    save_figure(fig, output_dir / "supplementary_pipeline_stage_effects")


def markdown_table(headers: Iterable[str], rows: Iterable[Iterable[str]]) -> str:
    headers = list(headers)
    lines = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join("---" for _ in headers) + "|",
    ]
    lines.extend("| " + " | ".join(map(str, row)) + " |" for row in rows)
    return "\n".join(lines)


def value_list(parameter: str) -> str:
    scale = 100.0 if parameter == "voxel_size" else 1.0
    return ", ".join(f"{value * scale:g}" for value in SWEEPS[parameter])


def voxel_table(data: pd.DataFrame) -> str:
    rows: List[List[str]] = []
    subset = data[data["Parameter"] == "voxel_size"]
    for condition in SITE_COLOURS:
        site = subset[subset["Site condition"] == condition]
        for value, group in site.groupby("Value"):
            rows.append(
                [
                    condition,
                    f"{value * 100:g}",
                    f"{group['Reference mIoU (%)'].mean():.2f}",
                    f"{group['Stage 2 labelled points (%)'].mean():.2f}",
                    f"{group['Final changed from default (%)'].mean():.2f}",
                ]
            )
    return markdown_table(
        [
            "Site",
            "Voxel (cm)",
            "Reference mIoU (%)",
            "Stage-2 labelled (%)",
            "Final change from 10 cm (%)",
        ],
        rows,
    )


def parameter_table(data: pd.DataFrame) -> str:
    rows: List[List[str]] = []
    for parameter in SWEEPS:
        subset = data[data["Parameter"] == parameter]
        scale = 100.0 if parameter == "voxel_size" else 1.0
        maxima = []
        for condition in SITE_COLOURS:
            site = subset[subset["Site condition"] == condition]
            maxima.append(f"{site['Final changed from default (%)'].max():.3f}")
        rows.append(
            [
                PARAMETER_LABELS[parameter],
                f"{DEFAULTS[parameter] * scale:g}",
                value_list(parameter),
                *maxima,
            ]
        )
    return markdown_table(
        [
            "Parameter",
            "Delivered",
            "Tested",
            "Medium: maximum change (%)",
            "Low: maximum change (%)",
        ],
        rows,
    )


def radius_table(data: pd.DataFrame) -> str:
    rows: List[List[str]] = []
    subset = data[data["Parameter"] == "height_match_radius"]
    for condition in SITE_COLOURS:
        site = subset[subset["Site condition"] == condition]
        for value, group in site.groupby("Value"):
            rows.append(
                [
                    condition,
                    f"{value:g}",
                    f"{group['Match coverage (%)'].mean():.3f}",
                    f"{group['Final changed from default (%)'].mean():.3f}",
                ]
            )
    return markdown_table(
        [
            "Site",
            "Radius (m)",
            "Matched structure points (%)",
            "Final change from 0.30 m (%)",
        ],
        rows,
    )


def stage_table(stages: pd.DataFrame) -> str:
    rows: List[List[str]] = []
    for stage in STAGE_ORDER:
        block = stages[stages["Stage"] == stage]
        if block.empty:
            continue
        row = block.iloc[0]
        values: Dict[str, str] = {}
        for condition in SITE_COLOURS:
            site = block[block["Site condition"] == condition]
            if site.empty:
                values[condition] = "–"
            else:
                values[condition] = (
                    f"{site.iloc[0]['mean']:.3f} ± {site.iloc[0]['std']:.3f}"
                )
        rows.append(
            [
                stage,
                row["Measure"],
                row["Measure type"],
                values["Medium"],
                values["Low"],
            ]
        )
    return markdown_table(
        [
            "Stage",
            "Quantity reported",
            "Interpretation",
            "Medium (%)",
            "Low (%)",
        ],
        rows,
    )


def write_summary(data: pd.DataFrame, stages: pd.DataFrame, output_dir: Path) -> None:
    text = f"""# Reviewer sensitivity analysis

## Voxel-size comparison

{voxel_table(data)}

## All tested parameters

{parameter_table(data)}

### Rationale for the delivered values

| Setting | Simple rationale |
|---|---|
| 10 cm voxel | Matches the 10 cm structural analysis grid and provides a middle value between fine voxels that transfer fewer labels and coarse voxels that spread labels farther. |
| 0.30 m matching radius | Used as a maximum search distance to tolerate small residual XY displacement while limiting matches across neighbouring plants. The measured coverage and sensitivity are reported above. |
| Height weights 1.0 and 0.1 | Classes inside their empirical height range retain full weight (1.0); classes outside it retain a small weight (0.1) rather than being excluded completely. |
| Confidence threshold 0.3 | Allows confident filling of unlabelled points while leaving weak predictions unchanged. |
| Flip ratio 1.5 | Protects an existing transferred label unless the alternative class is at least 50% more probable. |

The in-range height weight is 1.0 by definition. The delivered out-of-range
weight (0.1) retains a small contribution from classes outside their empirical
height range. The confidence threshold (0.3) controls filling of empty labels,
whereas the flip ratio (1.5) requires the alternative class to be 50% stronger
before an existing label is replaced.

## Matching-radius check

{radius_table(data)}

The 0.30 m value is an upper search limit, not an assumed registration error.
The table reports the observed proportion of structure points captured at each
tested radius and the resulting change in the final labels.

## Step-wise pipeline effects

{stage_table(stages)}

These percentages are deliberately described as error proxies or label changes.
They quantify propagation through the pipeline but do not replace validation
against independent 3D ground truth. Crown detection is characterised by the
proportion of points exposed to crown-based processing because no independent
crown delineation is available. Step percentages should not be added because a
later stage can retain or reverse a change made by an earlier stage.
"""
    path = output_dir / "reviewer_summary.md"
    path.write_text(text, encoding="utf-8")
    print(f"Wrote {path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    per_tile = pd.read_csv(args.input_dir / "parameter_sensitivity_per_tile.csv")
    stages = pd.read_csv(args.input_dir / "pipeline_stage_effects.csv")
    voxel_figure(per_tile, args.output_dir)
    parameter_figure(per_tile, args.output_dir)
    stage_figure(stages, args.output_dir)
    write_summary(per_tile, stages, args.output_dir)


if __name__ == "__main__":
    main()
