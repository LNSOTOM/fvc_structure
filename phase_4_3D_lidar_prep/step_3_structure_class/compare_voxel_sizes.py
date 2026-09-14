#!/usr/bin/env python3
"""Sweep voxel sizes and export a supplementary sensitivity figure."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import laspy
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import colors, gridspec

from fvc3dstruct.pipeline import process_tile


DEFAULT_SIZES = (0.05, 0.10, 0.15, 0.20, 0.25)
DEFAULT_Z_GLOB = "z_norm_csf_tiles_multispectral.*.las"
METRICS = [
    ("stage2_labeled_pct", "Stage 2 labeled points (%)"),
    ("stage2_nonzero_structure_voxels", "Non-zero structure voxels"),
    ("stage4_changed_pct", "Stage 4 label changes (%)"),
    ("stage4_mean_confidence", "Stage 4 mean label confidence"),
    ("final_veg_pct", "Final vegetation retention (%)"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the Step-3 fusion pipeline over multiple voxel sizes and "
            "export per-tile summaries plus a supplementary figure."
        )
    )
    parser.add_argument("--z-dir", type=Path, required=True, help="Directory with Z_norm LAS tiles.")
    parser.add_argument("--s-dir", type=Path, required=True, help="Directory with structure LAS tiles.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory for CSVs, figure, and optional LAS outputs.")
    parser.add_argument("--site-name", default="", help="Optional prefix for written LAS filenames.")
    parser.add_argument(
        "--sizes",
        default=",".join(f"{size:.2f}" for size in DEFAULT_SIZES),
        help="Comma-separated voxel sizes in meters. Default: 0.05,0.10,0.15,0.20,0.25",
    )
    parser.add_argument(
        "--tile-ids",
        default="",
        help="Optional comma-separated tile ids to limit the sweep.",
    )
    parser.add_argument(
        "--max-tiles",
        type=int,
        default=None,
        help="Optional cap on number of matched tiles to process.",
    )
    parser.add_argument(
        "--z-glob",
        default=DEFAULT_Z_GLOB,
        help=f"Glob for Z_norm LAS tiles. Default: {DEFAULT_Z_GLOB}",
    )
    parser.add_argument(
        "--write-las",
        action="store_true",
        help="Write LAS outputs for each voxel-size run. Default is summary-only.",
    )
    parser.add_argument(
        "--plot-tile-id",
        default="",
        help="Optional tile id to export as a per-tile 4-panel voxel-sweep chart.",
    )
    parser.add_argument(
        "--plot-side-view-tile-id",
        default="",
        help="Optional tile id to export a sampled LiDAR side-view figure from the Z_norm LAS.",
    )
    parser.add_argument(
        "--plot-side-view-sweep-tile-id",
        default="",
        help="Optional tile id to export side-view comparisons across the requested voxel sizes.",
    )
    parser.add_argument(
        "--plot-3d-sweep-tile-id",
        default="",
        help="Optional tile id to export 3D voxel-volume comparisons across the requested voxel sizes.",
    )
    return parser.parse_args()


def parse_sizes(raw_sizes: str) -> list[float]:
    sizes = []
    for token in raw_sizes.split(","):
        token = token.strip()
        if not token:
            continue
        size = float(token)
        if size <= 0:
            raise ValueError(f"Voxel size must be positive, got {size}")
        sizes.append(size)
    if not sizes:
        raise ValueError("At least one voxel size is required")
    return sizes


def detect_tile_id(path: Path) -> str | None:
    match = re.search(r"(\d+)(?=\.las$)", path.name)
    if match:
        return match.group(1)
    return None


def find_matched_tiles(z_dir: Path, s_dir: Path, z_glob: str, wanted_tiles: set[str] | None) -> list[tuple[str, Path, Path]]:
    matched = []
    for z_path in sorted(z_dir.glob(z_glob)):
        tile_id = detect_tile_id(z_path)
        if tile_id is None:
            continue
        if wanted_tiles and tile_id not in wanted_tiles:
            continue
        s_path = s_dir / f"mask_tiles_multispectral.{tile_id}.las"
        if not s_path.exists():
            continue
        matched.append((tile_id, z_path, s_path))
    return matched


def _metric_grid_shape(n_metrics: int) -> tuple[int, int]:
    cols = 2
    rows = -(-n_metrics // cols)  # ceil division
    return rows, cols


def _hide_unused_axes(axes: np.ndarray, n_used: int) -> None:
    for ax in axes[n_used:]:
        ax.set_visible(False)


def _save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote: {path}")


def write_outputs(records: list[dict], output_dir: Path) -> None:
    if not records:
        return

    per_tile_df = pd.DataFrame.from_records(records)
    per_tile_path = output_dir / "voxel_size_sweep_per_tile.csv"
    per_tile_df.to_csv(per_tile_path, index=False)
    print(f"Wrote: {per_tile_path}")

    average_df = (
        per_tile_df.groupby(["voxel_size_xy", "voxel_size_cm"], as_index=False)
        .agg(
            tile_count=("tile_id", "nunique"),
            stage2_labeled_pct_mean=("stage2_labeled_pct", "mean"),
            stage2_labeled_pct_std=("stage2_labeled_pct", "std"),
            stage2_nonzero_structure_voxels_mean=("stage2_nonzero_structure_voxels", "mean"),
            stage2_nonzero_structure_voxels_std=("stage2_nonzero_structure_voxels", "std"),
            stage4_changed_pct_mean=("stage4_changed_pct", "mean"),
            stage4_changed_pct_std=("stage4_changed_pct", "std"),
            stage4_mean_confidence_mean=("stage4_mean_confidence", "mean"),
            stage4_mean_confidence_std=("stage4_mean_confidence", "std"),
            stage4_low_confidence_pct_mean=("stage4_low_confidence_pct", "mean"),
            stage4_low_confidence_pct_std=("stage4_low_confidence_pct", "std"),
            final_veg_pct_mean=("final_veg_pct", "mean"),
            final_veg_pct_std=("final_veg_pct", "std"),
            final_treeids_mean=("final_treeids", "mean"),
            final_treeids_std=("final_treeids", "std"),
        )
        .sort_values("voxel_size_cm")
    )
    average_path = output_dir / "voxel_size_sweep_average.csv"
    average_df.to_csv(average_path, index=False)
    print(f"Wrote: {average_path}")

    figure_path = output_dir / "supplementary_voxel_size_sensitivity.png"
    plot_voxel_sweep(average_df, figure_path)


def plot_voxel_sweep(average_df: pd.DataFrame, output_path: Path) -> None:
    rows, cols = _metric_grid_shape(len(METRICS))
    fig, axes = plt.subplots(rows, cols, figsize=(8 * cols, 5 * rows), constrained_layout=True)
    axes = np.ravel(axes)

    x = average_df["voxel_size_cm"].to_numpy()
    n_tiles = int(average_df["tile_count"].max()) if not average_df.empty else 0

    for ax, (metric, label) in zip(axes, METRICS):
        y = average_df[f"{metric}_mean"].to_numpy()
        yerr = average_df[f"{metric}_std"].fillna(0.0).to_numpy()

        ax.plot(x, y, marker="o", linewidth=2.5, color="#2e8b57")
        ax.fill_between(x, y - yerr, y + yerr, color="#2e8b57", alpha=0.18)
        ax.set_title(label, fontsize=13)
        ax.set_xlabel("Voxel size (cm)")
        ax.set_ylabel(label)
        ax.grid(alpha=0.25)

        for xi, yi in zip(x, y):
            ax.annotate(f"{yi:.1f}", (xi, yi), textcoords="offset points", xytext=(0, 6), ha="center", fontsize=9)

    _hide_unused_axes(axes, len(METRICS))
    fig.suptitle(
        f"Voxel-size sensitivity analysis for Step-3 label fusion (n={n_tiles} tiles)",
        fontsize=16,
    )
    _save(fig, output_path)


def plot_single_tile_sweep(tile_df: pd.DataFrame, output_path: Path, tile_id: str) -> None:
    rows, cols = _metric_grid_shape(len(METRICS))
    fig, axes = plt.subplots(rows, cols, figsize=(8 * cols, 5 * rows), constrained_layout=True)
    axes = np.ravel(axes)

    sorted_df = tile_df.sort_values("voxel_size_cm")
    x = sorted_df["voxel_size_cm"].to_numpy()

    for ax, (metric, label) in zip(axes, METRICS):
        y = sorted_df[metric].to_numpy()
        ax.plot(x, y, marker="o", linewidth=2.5, color="#c45b12")
        ax.set_title(label, fontsize=13)
        ax.set_xlabel("Voxel size (cm)")
        ax.set_ylabel(label)
        ax.grid(alpha=0.25)

        for xi, yi in zip(x, y):
            if metric == "stage2_nonzero_structure_voxels":
                text = f"{yi:,.0f}"
            else:
                text = f"{yi:.1f}"
            ax.annotate(text, (xi, yi), textcoords="offset points", xytext=(0, 6), ha="center", fontsize=9)

    _hide_unused_axes(axes, len(METRICS))
    fig.suptitle(f"Voxel-size sensitivity for tile {tile_id}", fontsize=16)
    _save(fig, output_path)


def sample_las_side_view_points(
    las_path: Path,
    max_points: int = 120_000,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    las = laspy.read(str(las_path))
    x_values = np.asarray(las.x, dtype=np.float32)
    y_values = np.asarray(las.y, dtype=np.float32)
    z_values = np.asarray(las.z, dtype=np.float32)

    n_points = len(x_values)
    if n_points == 0:
        return x_values, y_values, z_values

    if n_points > max_points:
        rng = np.random.default_rng(seed)
        keep = rng.choice(n_points, size=max_points, replace=False)
        x_values = x_values[keep]
        y_values = y_values[keep]
        z_values = z_values[keep]

    return x_values, y_values, z_values


def sample_las_points_3d(
    las_path: Path,
    max_points: int = 120_000,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return sample_las_side_view_points(las_path, max_points=max_points, seed=seed)


def plot_las_side_views(las_path: Path, output_path: Path, tile_id: str) -> None:
    x_values, y_values, z_values = sample_las_side_view_points(las_path)

    fig, axes = plt.subplots(1, 2, figsize=(15, 6), constrained_layout=True)
    scatter_style = {
        "s": 0.8,
        "c": z_values,
        "cmap": "viridis",
        "alpha": 0.65,
        "linewidths": 0,
    }

    axes[0].scatter(x_values, z_values, **scatter_style)
    axes[0].set_title(f"Tile {tile_id} side view (X-Z)", fontsize=13)
    axes[0].set_xlabel("X")
    axes[0].set_ylabel("Elevation / Z_norm")
    axes[0].grid(alpha=0.2)
    axes[0].ticklabel_format(style="plain", axis="x", useOffset=False)

    side_scatter = axes[1].scatter(y_values, z_values, **scatter_style)
    axes[1].set_title(f"Tile {tile_id} side view (Y-Z)", fontsize=13)
    axes[1].set_xlabel("Y")
    axes[1].set_ylabel("Elevation / Z_norm")
    axes[1].grid(alpha=0.2)
    axes[1].ticklabel_format(style="plain", axis="x", useOffset=False)

    cbar = fig.colorbar(side_scatter, ax=axes, shrink=0.92)
    cbar.set_label("Elevation / Z_norm")
    fig.suptitle(f"Sampled LiDAR point cloud side views for tile {tile_id}", fontsize=16)
    _save(fig, output_path)


def voxelise_side_view(
    x_values: np.ndarray,
    z_values: np.ndarray,
    voxel_size: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if len(x_values) == 0:
        return x_values, z_values, z_values

    x0 = float(x_values.min())
    z0 = float(z_values.min())
    x_idx = np.floor((x_values - x0) / voxel_size).astype(np.int32)
    z_idx = np.floor((z_values - z0) / voxel_size).astype(np.int32)
    voxel_idx = np.column_stack((x_idx, z_idx))
    unique_voxels, _, counts = np.unique(
        voxel_idx,
        axis=0,
        return_index=True,
        return_counts=True,
    )

    x_centers = x0 + (unique_voxels[:, 0].astype(np.float32) + 0.5) * voxel_size
    z_centers = z0 + (unique_voxels[:, 1].astype(np.float32) + 0.5) * voxel_size
    density = counts.astype(np.float32)
    order = np.argsort(density)
    return x_centers[order], z_centers[order], density[order]


def voxelise_points_3d(
    x_values: np.ndarray,
    y_values: np.ndarray,
    z_values: np.ndarray,
    voxel_size: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if len(x_values) == 0:
        return x_values, y_values, z_values, z_values

    x0 = float(x_values.min())
    y0 = float(y_values.min())
    z0 = float(z_values.min())
    x_idx = np.floor((x_values - x0) / voxel_size).astype(np.int32)
    y_idx = np.floor((y_values - y0) / voxel_size).astype(np.int32)
    z_idx = np.floor((z_values - z0) / voxel_size).astype(np.int32)
    voxel_idx = np.column_stack((x_idx, y_idx, z_idx))
    unique_voxels, counts = np.unique(voxel_idx, axis=0, return_counts=True)

    x_centers = x0 + (unique_voxels[:, 0].astype(np.float32) + 0.5) * voxel_size
    y_centers = y0 + (unique_voxels[:, 1].astype(np.float32) + 0.5) * voxel_size
    z_centers = z0 + (unique_voxels[:, 2].astype(np.float32) + 0.5) * voxel_size
    density = counts.astype(np.float32)
    order = np.argsort(density)
    return x_centers[order], y_centers[order], z_centers[order], density[order]


def plot_las_side_view_sweep(
    las_path: Path,
    output_path: Path,
    tile_id: str,
    sizes: list[float],
) -> None:
    x_values, _, z_values = sample_las_side_view_points(las_path, max_points=180_000)

    n_panels = len(sizes)
    fig, axes = plt.subplots(1, n_panels, figsize=(5 * n_panels, 5.5), constrained_layout=True)
    if n_panels == 1:
        axes = [axes]

    x_limits = (float(x_values.min()), float(x_values.max())) if len(x_values) else (0.0, 1.0)
    z_limits = (float(z_values.min()), float(z_values.max())) if len(z_values) else (0.0, 1.0)
    last_scatter = None

    for ax, size in zip(axes, sizes):
        x_centers, z_centers, density = voxelise_side_view(x_values, z_values, size)
        size_cm = round(size * 100)
        last_scatter = ax.scatter(
            x_centers,
            z_centers,
            c=density,
            s=10,
            cmap="plasma",
            alpha=0.9,
            linewidths=0,
        )
        ax.set_title(f"{size_cm} cm", fontsize=13)
        ax.set_xlabel("X")
        ax.set_ylabel("Elevation / Z_norm")
        ax.set_xlim(*x_limits)
        ax.set_ylim(*z_limits)
        ax.grid(alpha=0.2)
        ax.ticklabel_format(style="plain", axis="x", useOffset=False)

    if last_scatter is not None:
        cbar = fig.colorbar(last_scatter, ax=axes, shrink=0.92)
        cbar.set_label("Points per X-Z voxel")
    fig.suptitle(f"Tile {tile_id} side-view comparison across voxel sizes", fontsize=16)
    _save(fig, output_path)


def plot_las_3d_voxel_sweep(
    las_path: Path,
    output_path: Path,
    tile_id: str,
    sizes: list[float],
) -> None:
    x_values, y_values, z_values = sample_las_points_3d(las_path, max_points=180_000)

    n_panels = len(sizes)
    fig = plt.figure(figsize=(5 * n_panels, 6.5), constrained_layout=True)
    x_limits = (float(x_values.min()), float(x_values.max())) if len(x_values) else (0.0, 1.0)
    y_limits = (float(y_values.min()), float(y_values.max())) if len(y_values) else (0.0, 1.0)
    z_limits = (float(z_values.min()), float(z_values.max())) if len(z_values) else (0.0, 1.0)
    voxel_sets = [voxelise_points_3d(x_values, y_values, z_values, size) for size in sizes]
    density_sets = [density for _, _, _, density in voxel_sets if len(density)]
    if density_sets:
        merged_density = np.concatenate(density_sets)
        vmin = max(1.0, float(np.percentile(merged_density, 5)))
        vmax = max(vmin + 1.0, float(np.percentile(merged_density, 99.5)))
        density_norm = colors.LogNorm(vmin=vmin, vmax=vmax)
    else:
        density_norm = None
    axes = []
    last_scatter = None

    for index, (size, voxel_set) in enumerate(zip(sizes, voxel_sets), start=1):
        ax = fig.add_subplot(1, n_panels, index, projection="3d")
        axes.append(ax)
        x_centers, y_centers, z_centers, density = voxel_set
        size_cm = round(size * 100)
        last_scatter = ax.scatter(
            x_centers,
            y_centers,
            z_centers,
            c=density,
            s=8,
            cmap="plasma",
            norm=density_norm,
            alpha=0.88,
            linewidths=0,
        )
        ax.set_title(f"{size_cm} cm\n{len(density):,} occupied voxels", fontsize=13)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.set_xlim(*x_limits)
        ax.set_ylim(*y_limits)
        ax.set_zlim(*z_limits)
        ax.view_init(elev=24, azim=-62)
        ax.ticklabel_format(style="plain", axis="x", useOffset=False)
        ax.ticklabel_format(style="plain", axis="y", useOffset=False)
        ax.set_box_aspect((
            max(x_limits[1] - x_limits[0], 1.0),
            max(y_limits[1] - y_limits[0], 1.0),
            max(z_limits[1] - z_limits[0], 1.0),
        ))

    if last_scatter is not None and axes:
        cbar = fig.colorbar(last_scatter, ax=axes, shrink=0.84)
        cbar.set_label("Points per 3D voxel")
    # Voxels are counted over the sampled returns, not the whole tile.
    fig.suptitle(
        f"Tile {tile_id} 3D voxel-volume comparison across voxel sizes "
        f"(random sample of {len(x_values):,} returns)",
        fontsize=16,
    )
    _save(fig, output_path)


def plot_tile_summary_with_side_view_sweep(
    tile_df: pd.DataFrame,
    las_path: Path,
    output_path: Path,
    tile_id: str,
    sizes: list[float],
) -> None:
    sorted_df = tile_df.sort_values("voxel_size_cm")
    x_values = sorted_df["voxel_size_cm"].to_numpy()
    las_x, _, las_z = sample_las_side_view_points(las_path, max_points=180_000)
    x_limits = (float(las_x.min()), float(las_x.max())) if len(las_x) else (0.0, 1.0)
    z_limits = (float(las_z.min()), float(las_z.max())) if len(las_z) else (0.0, 1.0)

    rows, cols = _metric_grid_shape(len(METRICS))
    metrics_height = 5.5 * rows
    side_height = 6.5
    fig = plt.figure(figsize=(24, metrics_height + side_height), constrained_layout=True)
    outer = gridspec.GridSpec(2, 1, figure=fig, height_ratios=[metrics_height, side_height])
    metrics_grid = outer[0].subgridspec(rows, cols)
    side_grid = outer[1].subgridspec(1, len(sizes))

    for index, (metric, label) in enumerate(METRICS):
        ax = fig.add_subplot(metrics_grid[index // cols, index % cols])
        y_values = sorted_df[metric].to_numpy()
        ax.plot(x_values, y_values, marker="o", linewidth=2.5, color="#c45b12")
        ax.set_title(label, fontsize=13)
        ax.set_xlabel("Voxel size (cm)")
        ax.set_ylabel(label)
        ax.grid(alpha=0.25)

        for x_value, y_value in zip(x_values, y_values):
            text = f"{y_value:,.0f}" if metric == "stage2_nonzero_structure_voxels" else f"{y_value:.1f}"
            ax.annotate(text, (x_value, y_value), textcoords="offset points", xytext=(0, 6), ha="center", fontsize=9)

    last_scatter = None
    side_axes = []
    for index, size in enumerate(sizes):
        ax = fig.add_subplot(side_grid[0, index])
        side_axes.append(ax)
        x_centers, z_centers, density = voxelise_side_view(las_x, las_z, size)
        size_cm = round(size * 100)
        last_scatter = ax.scatter(
            x_centers,
            z_centers,
            c=density,
            s=10,
            cmap="plasma",
            alpha=0.9,
            linewidths=0,
        )
        ax.set_title(f"{size_cm} cm side view", fontsize=13)
        ax.set_xlabel("X")
        ax.set_ylabel("Elevation / Z_norm")
        ax.set_xlim(*x_limits)
        ax.set_ylim(*z_limits)
        ax.grid(alpha=0.2)
        ax.ticklabel_format(style="plain", axis="x", useOffset=False)

    if last_scatter is not None and side_axes:
        cbar = fig.colorbar(last_scatter, ax=side_axes, shrink=0.82)
        cbar.set_label("Points per X-Z voxel")

    fig.suptitle(f"Tile {tile_id}: metric sensitivity and side-view comparison across voxel sizes", fontsize=18)
    _save(fig, output_path)


def main() -> None:
    args = parse_args()
    sizes = parse_sizes(args.sizes)
    wanted_tiles = {tile.strip() for tile in args.tile_ids.split(",") if tile.strip()} or None

    z_dir = args.z_dir.resolve()
    s_dir = args.s_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    matched_tiles = find_matched_tiles(z_dir, s_dir, args.z_glob, wanted_tiles)
    if args.max_tiles is not None:
        matched_tiles = matched_tiles[: args.max_tiles]

    if not matched_tiles:
        raise SystemExit("No matched tiles found for the requested inputs.")

    site_prefix = f"{args.site_name}_" if args.site_name else ""
    records = []

    print(f"Found {len(matched_tiles)} matched tile pairs")
    print(f"Voxel sizes: {', '.join(f'{size * 100:.0f} cm' for size in sizes)}")

    try:
        for size in sizes:
            size_cm = round(size * 100)
            run_dir = output_dir / f"voxel_{size_cm:02d}cm"
            las_dir = run_dir / "las"
            if args.write_las:
                las_dir.mkdir(parents=True, exist_ok=True)

            print("=" * 72)
            print(f"Running voxel sweep for {size_cm} cm")
            for tile_id, z_path, s_path in matched_tiles:
                out_path = las_dir / f"{site_prefix}tile{tile_id}_struct_polar.las"
                summary = process_tile(
                    z_path=z_path,
                    s_path=s_path,
                    out_path=out_path,
                    config={
                        "voxel_size_xy": size,
                        "voxel_size_z": size,
                        "write_outputs": args.write_las,
                    },
                )
                summary["tile_id"] = tile_id
                summary["voxel_size_cm"] = float(size * 100.0)
                records.append(summary)
                write_outputs(records, output_dir)
    except KeyboardInterrupt:
        print("Interrupted. Writing partial sweep outputs...")
        write_outputs(records, output_dir)
        raise

    write_outputs(records, output_dir)

    if args.plot_tile_id:
        per_tile_df = pd.DataFrame.from_records(records)
        tile_df = per_tile_df[per_tile_df["tile_id"] == args.plot_tile_id].copy()
        if tile_df.empty:
            raise SystemExit(f"Tile id {args.plot_tile_id} was not found in the processed results.")
        tile_figure_path = output_dir / f"tile_{args.plot_tile_id}_voxel_size_sensitivity.png"
        plot_single_tile_sweep(tile_df, tile_figure_path, args.plot_tile_id)

    if args.plot_side_view_tile_id:
        matched_by_tile = {tile_id: z_path for tile_id, z_path, _ in matched_tiles}
        z_path = matched_by_tile.get(args.plot_side_view_tile_id)
        if z_path is None:
            raise SystemExit(f"Tile id {args.plot_side_view_tile_id} was not found in the matched Z_norm LAS inputs.")
        side_view_path = output_dir / f"tile_{args.plot_side_view_tile_id}_side_views.png"
        plot_las_side_views(z_path, side_view_path, args.plot_side_view_tile_id)

    if args.plot_side_view_sweep_tile_id:
        matched_by_tile = {tile_id: z_path for tile_id, z_path, _ in matched_tiles}
        z_path = matched_by_tile.get(args.plot_side_view_sweep_tile_id)
        if z_path is None:
            raise SystemExit(f"Tile id {args.plot_side_view_sweep_tile_id} was not found in the matched Z_norm LAS inputs.")
        side_view_sweep_path = output_dir / f"tile_{args.plot_side_view_sweep_tile_id}_side_view_voxel_sweep.png"
        plot_las_side_view_sweep(z_path, side_view_sweep_path, args.plot_side_view_sweep_tile_id, sizes)

        per_tile_df = pd.DataFrame.from_records(records)
        tile_df = per_tile_df[per_tile_df["tile_id"] == args.plot_side_view_sweep_tile_id].copy()
        if not tile_df.empty:
            combined_path = output_dir / f"tile_{args.plot_side_view_sweep_tile_id}_metrics_and_side_view_voxel_sweep.png"
            plot_tile_summary_with_side_view_sweep(
                tile_df,
                z_path,
                combined_path,
                args.plot_side_view_sweep_tile_id,
                sizes,
            )

    if args.plot_3d_sweep_tile_id:
        matched_by_tile = {tile_id: z_path for tile_id, z_path, _ in matched_tiles}
        z_path = matched_by_tile.get(args.plot_3d_sweep_tile_id)
        if z_path is None:
            raise SystemExit(f"Tile id {args.plot_3d_sweep_tile_id} was not found in the matched Z_norm LAS inputs.")
        volume_path = output_dir / f"tile_{args.plot_3d_sweep_tile_id}_3d_voxel_sweep.png"
        plot_las_3d_voxel_sweep(z_path, volume_path, args.plot_3d_sweep_tile_id, sizes)


if __name__ == "__main__":
    main()