#!/usr/bin/env python3
"""Plot statistical summaries from validation CSV outputs."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

TITLE_FONTSIZE = 34
LABEL_FONTSIZE = 28
TICK_FONTSIZE = 24
LEGEND_FONTSIZE = 22
ANNOT_FONTSIZE = 19
SUPTITLE_FONTSIZE = 40

# Colours from 04_maskCoreg_site01_L1.ipynb palette
# 64 Forb / 66 Hummock share #d9b44a; 65 Woody debris #4C00A4; 67 Shrub #694714; 68 Mallee #2e8b57
CLASS_PALETTE = {
    64: "#d9b44a",  # Forb  — same as Hummock
    65: "#4C00A4",  # Woody debris
    66: "#d9b44a",  # Hummock
    67: "#694714",  # Shrub
    68: "#2e8b57",  # Mallee
}

IOU_COLS = [
    "IoU Woody debris",
    "IoU Hummock",
    "IoU Shrub",
    "IoU Mallee",
]
IOU_COLORS = [CLASS_PALETTE[65], CLASS_PALETTE[66], CLASS_PALETTE[67], CLASS_PALETTE[68]]
TABLE6_CLASSES = ["Hummock", "Shrub", "Mallee"]
TABLE6_COLORS = {
    "Hummock": CLASS_PALETTE[66],
    "Shrub":   CLASS_PALETTE[67],
    "Mallee":  CLASS_PALETTE[68],
}
IN_RANGE_COLOR = "#2ca25f"
OUT_RANGE_COLOR = "#e34a33"
RANGE_BAND_COLOR = "#a8d5b5"
RETENTION_COLOR = "#d40082"
OA_COLOR = "#d40082"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot validation statistics from generated CSV summaries."
    )
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, help="Optional combined output PNG.")
    parser.add_argument("--show", action="store_true", help="Display figures interactively.")
    return parser.parse_args()


def load_required_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing: {path}")
    return pd.read_csv(path)


def _save(fig: plt.Figure, path: Path, show: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)
    print(f"Wrote: {path}")


# ── Table 5 ──────────────────────────────────────────────────────────────────

def make_table5_figure(table5: pd.DataFrame) -> plt.Figure:
    """Per-tile IoU bars (Hummock, Shrub, Mallee) with OA line overlay.

    Forb and Woody debris are excluded from the plot — both are workflow design
    outcomes (Forb absent; Woody debris consolidated into Mallee at Step 5) and
    are explained in the figure caption rather than shown as zero bars.
    Tiles where Shrub was consolidated to zero receive a hatched n/a bar.
    """
    fig, ax = plt.subplots(figsize=(22, 9), constrained_layout=True)

    # Only the three ecologically informative classes
    plot_cols   = ["IoU Hummock", "IoU Shrub", "IoU Mallee"]
    plot_colors = [CLASS_PALETTE[66], CLASS_PALETTE[67], CLASS_PALETTE[68]]
    plot_labels = ["Hummock (0/66)", "Shrub (1/67)", "Mallee (2/68)"]

    tiles = table5["Tile"].astype(str).tolist()
    x = np.arange(len(tiles))
    n = len(plot_cols)
    width = 0.22

    for i, (col, color, label) in enumerate(zip(plot_cols, plot_colors, plot_labels)):
        raw = pd.to_numeric(table5[col], errors="coerce")
        vals = raw.to_numpy(dtype=float) * 100.0  # NaN stays NaN

        bar_x = x + (i - (n - 1) / 2.0) * width

        for xi, v in zip(bar_x, vals):
            if np.isnan(v):
                # Hatched n/a bar — tile had no Shrub after consolidation
                ax.bar(xi, 9.5, width=width, bottom=90, color=color,
                       alpha=0.25, edgecolor=color, linewidth=1.2,
                       hatch="////", zorder=2)
                ax.text(xi, 90 + 9.5 / 2, "n/a", ha="center", va="center",
                        fontsize=ANNOT_FONTSIZE - 4, color=color, fontweight="bold")
            else:
                ax.bar(xi, v - 90, width=width, bottom=90, color=color,
                       edgecolor="white", linewidth=0.4, zorder=2)
                # Label: honest decimal — rotated 90° to avoid inter-bar overlap
                ax.text(xi, max(v + 0.2, 90.4), f"{v:.1f}",
                        ha="center", va="bottom", rotation=90,
                        fontsize=ANNOT_FONTSIZE + 4, color=color)

        # Invisible full bar for legend handle only
        ax.bar(0, 0, color=color, label=label)

    OA_LINE_COLOR = OA_COLOR  # #d40082 pink
    oa = pd.to_numeric(table5["OA (%)"], errors="coerce").to_numpy()
    ax2 = ax.twinx()
    ax2.plot(x, oa, color=OA_LINE_COLOR, linewidth=3, marker="D", markersize=10,
             zorder=5, label="OA (%)")
    # Labels above line, offset to avoid bar tops — skip if too close to previous
    prev_v = -np.inf
    for xi, v in zip(x, oa):
        offset = 0.06 if abs(v - prev_v) > 0.2 else -0.25
        ax2.text(xi, v + offset, f"{v:.1f}", ha="center",
                 va="bottom" if offset >= 0 else "top",
                 fontsize=ANNOT_FONTSIZE - 3, color="black", fontweight="bold")
        prev_v = v
    ax2.set_ylim(96, 100)
    ax2.set_ylabel("Overall Accuracy (%)", fontsize=LABEL_FONTSIZE, color=OA_LINE_COLOR)
    ax2.tick_params(axis="y", labelsize=TICK_FONTSIZE, colors=OA_LINE_COLOR)
    ax2.spines["right"].set_edgecolor(OA_LINE_COLOR)

    ax.set_ylabel("IoU (%)", fontsize=LABEL_FONTSIZE)
    ax.set_ylim(90, 100)
    ax.set_xticks(x)
    ax.set_xticklabels(tiles, rotation=0, ha="center", fontsize=TICK_FONTSIZE + 6)
    ax.tick_params(axis="y", labelsize=TICK_FONTSIZE + 6)
    ax.grid(axis="y", alpha=0.2)

    iou_handles, iou_labels = ax.get_legend_handles_labels()
    oa_handle = Line2D([0], [0], color=OA_LINE_COLOR, linewidth=3,
                       marker="D", markersize=10, label="OA (%)")
    na_handle = mpatches.Patch(facecolor="#aaaaaa", alpha=0.4, hatch="////",
                               edgecolor="#aaaaaa", label="n/a — class absent after Step 5")
    fig.legend(
        iou_handles + [oa_handle, na_handle],
        iou_labels + ["OA (%)", "n/a — class absent after Step 5"],
        fontsize=LEGEND_FONTSIZE, ncol=5, frameon=False,
        loc="lower center", bbox_to_anchor=(0.5, -0.08),
    )
    return fig


# ── Table 6 ──────────────────────────────────────────────────────────────────

def make_table6_figure(table6_tile: pd.DataFrame) -> plt.Figure:
    """Per-tile dot-and-range chart: median Znorm vs configured height band.

    Dots are green (in range) or red (outside range). % outside annotated above
    each dot. The configured [min, max] band is shaded.
    """
    classes = [c for c in TABLE6_CLASSES if c in table6_tile["Class"].values]
    n = len(classes)
    fig, axes = plt.subplots(1, n, figsize=(10 * n, 10), constrained_layout=True)
    if n == 1:
        axes = [axes]

    for ax, cls in zip(axes, classes):
        sub = table6_tile[table6_tile["Class"] == cls].copy()
        sub = sub.dropna(subset=["Median Znorm (m)"])
        if sub.empty:
            ax.set_title(f"{cls}\n(no data)", fontsize=TITLE_FONTSIZE)
            continue

        zmin = float(sub["Configured min Znorm (m)"].iloc[0])
        zmax = float(sub["Configured max Znorm (m)"].iloc[0])
        medians = sub["Median Znorm (m)"].to_numpy(dtype=float)
        iqr_vals = sub["IQR (m)"].fillna(0).to_numpy(dtype=float)
        pct_out = sub["Outside configured range (%)"].to_numpy(dtype=float)
        tiles = sub["Tile"].astype(str).tolist()
        x = np.arange(len(tiles))

        # Shaded configured range
        ax.axhspan(zmin, zmax, color=RANGE_BAND_COLOR, alpha=0.35, zorder=0,
                   label=f"Configured range [{zmin:.2f}, {zmax:.2f}] m")
        ax.axhline(zmin, color="green", linewidth=1, linestyle="--", alpha=0.7)
        ax.axhline(zmax, color="green", linewidth=1, linestyle="--", alpha=0.7)

        # Per-tile dots + IQR error bars
        in_range = (medians >= zmin) & (medians <= zmax)
        for i in range(len(tiles)):
            color = IN_RANGE_COLOR if in_range[i] else OUT_RANGE_COLOR
            half_iqr = iqr_vals[i] / 2.0
            ax.errorbar(x[i], medians[i], yerr=half_iqr, fmt="o",
                        color=color, markersize=7, capsize=4, elinewidth=1.5, zorder=3)
            # % outside annotated above the error bar cap
            label_y = medians[i] + half_iqr + 0.04
            ax.text(x[i], label_y, f"{pct_out[i]:.0f}%",
                    ha="center", va="bottom", fontsize=ANNOT_FONTSIZE,
                    color=color, fontweight="bold")

        # Summary stat in subtitle
        mean_out = float(np.nanmean(pct_out))
        ax.set_title(
            f"{cls}\navg {mean_out:.0f}% of points\noutside configured range",
            fontsize=TITLE_FONTSIZE,
        )
        ax.set_xticks(x)
        ax.set_xticklabels(tiles, rotation=45, ha="right", fontsize=TICK_FONTSIZE)
        ax.set_ylabel("Median Znorm (m)  [error bar = IQR]", fontsize=LABEL_FONTSIZE)
        ax.tick_params(axis="y", labelsize=TICK_FONTSIZE)
        ax.grid(axis="y", alpha=0.2)

        in_patch = Line2D([0], [0], marker="o", color="w", markerfacecolor=IN_RANGE_COLOR,
                          markersize=8, label="Median in range")
        out_patch = Line2D([0], [0], marker="o", color="w", markerfacecolor=OUT_RANGE_COLOR,
                           markersize=8, label="Median outside range")
        range_patch = mpatches.Patch(color=RANGE_BAND_COLOR, alpha=0.6,
                                     label=f"Configured [{zmin:.2f}, {zmax:.2f}] m")
        ax.legend(handles=[range_patch, in_patch, out_patch],
                  fontsize=LEGEND_FONTSIZE, frameon=False)

    fig.suptitle(
        "Table 6: Refined label height plausibility — median Znorm vs configured range per tile",
        fontsize=SUPTITLE_FONTSIZE,
    )
    return fig


# ── TreeID ───────────────────────────────────────────────────────────────────

def make_treeid_figure(treeid: pd.DataFrame) -> plt.Figure:
    """Crown count bars + retention % line; Tile 106 annotated."""
    fig, ax = plt.subplots(figsize=(24, 10), constrained_layout=True)

    tiles = treeid["Tile"].astype(str).tolist()
    x = np.arange(len(tiles))
    width = 0.3

    step4 = pd.to_numeric(treeid["Watershed crowns (Step 4)"], errors="coerce").to_numpy()
    step9 = pd.to_numeric(treeid["Final non-zero treeIDs (Step 9)"], errors="coerce").to_numpy()
    retention = pd.to_numeric(treeid["Crown retention (%)"], errors="coerce").to_numpy()
    mean_pts = pd.to_numeric(treeid["Mean points per treeID"], errors="coerce").to_numpy()

    bars4 = ax.bar(x - width / 2, step4, width=width, color="#7bccc4",
                   edgecolor="white", linewidth=0.4, label="Step 4: Watershed crowns")
    bars9 = ax.bar(x + width / 2, step9, width=width, color="#2b8cbe",
                   edgecolor="white", linewidth=0.4, label="Step 9: Final treeIDs")

    for bar, v in zip(bars4, step4):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.4, str(int(v)),
                ha="center", va="bottom", fontsize=ANNOT_FONTSIZE, color="#2b7a78")
    for bar, v, r in zip(bars9, step9, retention):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.4, str(int(v)),
                ha="center", va="bottom", fontsize=ANNOT_FONTSIZE, color="#185a7d")

    # Retention % on twin axis
    ax2 = ax.twinx()
    ax2.plot(x, retention, color=RETENTION_COLOR, linewidth=2, marker="o",
             markersize=6, zorder=5, label="Crown retention (%)")
    ax2.axhline(100, color=RETENTION_COLOR, linewidth=0.8, linestyle=":", alpha=0.5)
    for xi, v in zip(x, retention):
        ax2.text(xi, v + 0.6, f"{v:.0f}%", ha="center", va="bottom",
                 fontsize=ANNOT_FONTSIZE, color=RETENTION_COLOR)
    ax2.set_ylim(75, 115)
    ax2.set_ylabel("Crown retention (%)", fontsize=LABEL_FONTSIZE, color=RETENTION_COLOR)
    ax2.tick_params(axis="y", labelsize=TICK_FONTSIZE, colors=RETENTION_COLOR)

    # Annotate Tile 106 (crown count increased)
    delta = pd.to_numeric(treeid["Crown count delta (Step 9 - Step 4)"], errors="coerce").to_numpy()
    for i, (tile, d) in enumerate(zip(tiles, delta)):
        if np.isfinite(d) and d > 0:
            ax.annotate(
                f"+{int(d)} crown",
                xy=(x[i] + width / 2, step9[i]),
                xytext=(x[i] + width / 2 + 0.4, step9[i] + 5),
                fontsize=ANNOT_FONTSIZE,
                color="#c0392b",
                arrowprops=dict(arrowstyle="->", color="#c0392b", lw=1),
            )

    ax.set_title(
        "TreeID: Step 4 watershed crowns vs Step 9 final treeIDs with crown retention",
        fontsize=TITLE_FONTSIZE,
    )
    ax.set_ylabel("Crown / treeID count", fontsize=LABEL_FONTSIZE)
    ax.set_xticks(x)
    ax.set_xticklabels(tiles, rotation=45, ha="right", fontsize=TICK_FONTSIZE)
    ax.tick_params(axis="y", labelsize=TICK_FONTSIZE)
    ax.grid(axis="y", alpha=0.2)

    h1, l1 = ax.get_legend_handles_labels()
    ret_handle = Line2D([0], [0], color=RETENTION_COLOR, linewidth=2,
                        marker="o", markersize=6, label="Crown retention (%)")
    ax.legend(h1 + [ret_handle], l1 + ["Crown retention (%)"],
              fontsize=LEGEND_FONTSIZE, ncol=3, frameon=False)
    fig.suptitle("Validation: TreeID Instance Summary", fontsize=SUPTITLE_FONTSIZE)
    return fig


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()
    results_dir = args.results_dir.resolve()
    show = args.show

    table5 = load_required_csv(results_dir / "table5_label_agreement_per_tile.csv")
    table6_tile = load_required_csv(results_dir / "table6_height_plausibility_per_tile.csv")

    treeid_path = results_dir / "treeid_instance_summary_per_tile.csv"
    treeid = load_required_csv(treeid_path) if treeid_path.exists() else None

    fig5 = make_table5_figure(table5)
    _save(fig5, results_dir / "validation_table5_label_agreement.png", show)

    fig6 = make_table6_figure(table6_tile)
    _save(fig6, results_dir / "validation_table6_height_plausibility.png", show)

    if treeid is not None:
        figt = make_treeid_figure(treeid)
        _save(figt, results_dir / "validation_treeid_instance_summary.png", show)
    else:
        print("Skipping treeID panel (file not found).")

    if args.output is not None:
        out = args.output.resolve()
        n_panels = 3 if treeid is not None else 2
        fig, axes = plt.subplots(n_panels, 1, figsize=(18, 8 * n_panels), constrained_layout=True)
        axes = list(axes)

        # Re-draw each panel onto the combined figure axes
        # (simpler: just save individual PNGs and note combined isn't restructured)
        # For the combined figure we embed the saved PNGs as images.
        import matplotlib.image as mpimg
        panels = [
            results_dir / "validation_table5_label_agreement.png",
            results_dir / "validation_table6_height_plausibility.png",
        ]
        if treeid is not None:
            panels.append(results_dir / "validation_treeid_instance_summary.png")

        for ax, panel_path in zip(axes, panels):
            img = mpimg.imread(str(panel_path))
            ax.imshow(img)
            ax.axis("off")

        fig.suptitle("Validation statistics summary", fontsize=SUPTITLE_FONTSIZE)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=200, bbox_inches="tight")
        if show:
            plt.show()
        plt.close(fig)
        print(f"Wrote: {out}")


if __name__ == "__main__":
    main()
