#!/usr/bin/env python3
"""Sweep voxel sizes and export a supplementary sensitivity figure."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from fvc3dstruct.pipeline import process_tile


DEFAULT_SIZES = (0.05, 0.10, 0.15, 0.20, 0.25)
DEFAULT_Z_GLOB = "z_norm_csf_tiles_multispectral.*.las"
METRICS = [
    ("stage2_labeled_pct", "Stage 2 labeled points (%)"),
    ("stage2_nonzero_structure_voxels", "Non-zero structure voxels"),
    ("stage4_changed_pct", "Stage 4 label changes (%)"),
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


def _save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote: {path}")


def plot_voxel_sweep(average_df: pd.DataFrame, output_path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(16, 10), constrained_layout=True)
    axes = axes.ravel()

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

    fig.suptitle(
        f"Voxel-size sensitivity analysis for Step-3 label fusion (n={n_tiles} tiles)",
        fontsize=16,
    )
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

    for size in sizes:
        size_cm = int(round(size * 100))
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


if __name__ == "__main__":
    main()