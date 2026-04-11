"""Create a four-panel validation figure for the Phase 4 2D-to-3D label workflow.

Panels:
  (a) Original 2D structure mask derived from multispectral imagery
  (b) Initial projection of 2D labels onto the LiDAR point cloud
  (c) Refined 3D labels after voxel-based aggregation and polar refinement
  (d) Vertical view of the refined point cloud showing height-based separation
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import laspy
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from matplotlib.colors import BoundaryNorm, ListedColormap


CLASS_LABELS = {
    0: "Background",
    64: "Forb",
    65: "Woody debris",
    66: "Hummock",
    67: "Shrub",
    68: "Mallee",
}

CLASS_COLORS = {
    0: "#000000",
    64: "#66c2a5",
    65: "#8c6d31",
    66: "#ffd92f",
    67: "#fc8d62",
    68: "#1f78b4",
}


def build_class_cmap(class_codes: Iterable[int]) -> Tuple[ListedColormap, BoundaryNorm]:
    codes = list(class_codes)
    colors = [CLASS_COLORS[code] for code in codes]
    cmap = ListedColormap(colors)
    bounds = np.arange(len(codes) + 1) - 0.5
    norm = BoundaryNorm(bounds, cmap.N)
    return cmap, norm


def remap_to_index(arr: np.ndarray, class_codes: Iterable[int], fill_value: int = 0) -> np.ndarray:
    lut = {code: idx for idx, code in enumerate(class_codes)}
    out = np.full(arr.shape, fill_value, dtype=np.int32)
    for code, idx in lut.items():
        out[arr == code] = idx
    return out


def read_mask(mask_path: str) -> Tuple[np.ndarray, rasterio.Affine]:
    with rasterio.open(mask_path) as src:
        return src.read(1), src.transform


def _read_dimension(las: laspy.LasData, name: str) -> np.ndarray:
    dims = set(las.point_format.dimension_names)
    if name in dims:
        return np.asarray(las[name])
    if hasattr(las, name):
        return np.asarray(getattr(las, name))
    raise KeyError(f"LAS does not contain dimension '{name}'")


def read_las_points(
    las_path: str,
    class_field: str = "classification",
    z_norm_field: Optional[str] = "Z_norm",
) -> Dict[str, np.ndarray | None]:
    las = laspy.read(las_path)

    cls = _read_dimension(las, class_field)
    z_norm = None
    if z_norm_field is not None:
        try:
            z_norm = _read_dimension(las, z_norm_field)
        except KeyError:
            z_norm = None

    return {
        "x": np.asarray(las.x),
        "y": np.asarray(las.y),
        "z": np.asarray(las.z),
        "class": np.asarray(cls),
        "z_norm": None if z_norm is None else np.asarray(z_norm),
    }


def raster_extent_from_transform(mask_shape: Tuple[int, int], transform: rasterio.Affine) -> list[float]:
    height, width = mask_shape
    left = transform.c
    top = transform.f
    right = left + width * transform.a
    bottom = top + height * transform.e
    return [left, right, bottom, top]


def filter_points_by_class(
    points: Dict[str, np.ndarray | None],
    class_codes: Iterable[int],
) -> Dict[str, np.ndarray | None]:
    codes = np.array(list(class_codes))
    mask = np.isin(points["class"], codes)
    return {key: (None if value is None else value[mask]) for key, value in points.items()}


def sample_points(
    points: Dict[str, np.ndarray | None],
    max_points: int = 200_000,
    seed: int = 42,
) -> Dict[str, np.ndarray | None]:
    n_points = len(points["x"])
    if n_points <= max_points:
        return points

    rng = np.random.default_rng(seed)
    idx = rng.choice(n_points, size=max_points, replace=False)
    return {key: (None if value is None else value[idx]) for key, value in points.items()}


def _select_horizontal_axis(points: Dict[str, np.ndarray | None]) -> Tuple[np.ndarray, str]:
    x_range = float(np.ptp(points["x"])) if len(points["x"]) else 0.0
    y_range = float(np.ptp(points["y"])) if len(points["y"]) else 0.0
    if y_range > x_range:
        return points["y"], "Y"
    return points["x"], "X"


def plot_top_view(
    ax,
    points: Dict[str, np.ndarray | None],
    class_codes: Iterable[int],
    title: str,
    point_size: float = 0.5,
):
    codes = list(class_codes)
    cls_idx = remap_to_index(points["class"], codes)
    cmap, norm = build_class_cmap(codes)

    sc = ax.scatter(
        points["x"],
        points["y"],
        c=cls_idx,
        cmap=cmap,
        norm=norm,
        s=point_size,
        linewidths=0,
        alpha=0.8,
    )
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_aspect("equal")
    return sc


def plot_side_view(
    ax,
    points: Dict[str, np.ndarray | None],
    class_codes: Iterable[int],
    title: str,
    point_size: float = 0.5,
    use_z_norm: bool = True,
):
    zvals = points["z_norm"] if (use_z_norm and points["z_norm"] is not None) else points["z"]
    horiz, horiz_label = _select_horizontal_axis(points)
    codes = list(class_codes)
    cls_idx = remap_to_index(points["class"], codes)
    cmap, norm = build_class_cmap(codes)

    sc = ax.scatter(
        horiz,
        zvals,
        c=cls_idx,
        cmap=cmap,
        norm=norm,
        s=point_size,
        linewidths=0,
        alpha=0.8,
    )
    ax.set_title(title, fontsize=11)
    ax.set_xlabel(horiz_label)
    ax.set_ylabel("Z_norm" if (use_z_norm and points["z_norm"] is not None) else "Z")
    return sc


def add_class_legend(fig, class_codes: Iterable[int]) -> None:
    handles = []
    labels = []
    for code in class_codes:
        if code not in CLASS_LABELS:
            continue
        handles.append(
            plt.Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor=CLASS_COLORS[code],
                markersize=8,
                linestyle="None",
            )
        )
        labels.append(f"{code}: {CLASS_LABELS[code]}")
    fig.legend(handles, labels, loc="lower center", ncol=min(len(labels), 6), frameon=False)


def make_label_transfer_validation_figure(
    mask_path: str,
    projected_las_path: str,
    refined_las_path: str,
    output_path: str,
    class_codes: Tuple[int, ...] = (64, 65, 66, 67, 68),
    max_points: int = 150_000,
    point_size: float = 0.5,
    show_background_mask: bool = False,
    sample_seed: int = 42,
) -> None:
    mask, transform = read_mask(mask_path)

    projected_pts = read_las_points(projected_las_path)
    refined_pts = read_las_points(refined_las_path)

    projected_pts = sample_points(
        filter_points_by_class(projected_pts, class_codes),
        max_points=max_points,
        seed=sample_seed,
    )
    refined_pts = sample_points(
        filter_points_by_class(refined_pts, class_codes),
        max_points=max_points,
        seed=sample_seed,
    )

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    ax = axes[0, 0]
    if not show_background_mask:
        mask = np.where(np.isin(mask, class_codes), mask, 0)
    mask_codes = [0] + list(class_codes)
    cmap, norm = build_class_cmap(mask_codes)
    mask_idx = remap_to_index(mask, mask_codes)
    extent = raster_extent_from_transform(mask.shape, transform)

    ax.imshow(mask_idx, cmap=cmap, norm=norm, extent=extent, origin="upper")
    ax.set_title("(a) Original 2D structure mask", fontsize=11)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_aspect("equal")

    plot_top_view(
        axes[0, 1],
        projected_pts,
        class_codes,
        "(b) Initial projection of 2D labels onto LiDAR",
        point_size=point_size,
    )

    plot_top_view(
        axes[1, 0],
        refined_pts,
        class_codes,
        "(c) Refined 3D labels after voxel/polar refinement",
        point_size=point_size,
    )

    plot_side_view(
        axes[1, 1],
        refined_pts,
        class_codes,
        "(d) Vertical view of refined point cloud",
        point_size=point_size,
        use_z_norm=True,
    )

    add_class_legend(fig, mask_codes)
    fig.suptitle("Validation of the 2D--3D label transfer workflow", fontsize=14, y=0.98)
    fig.tight_layout(rect=[0, 0.06, 1, 0.96])

    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create the four-panel Phase 4 label-transfer validation figure."
    )
    parser.add_argument("--mask", required=True, help="Path to the original 2D structure mask raster.")
    parser.add_argument(
        "--projected-las",
        required=True,
        help="Path to the initial LiDAR labels after 2D->3D projection.",
    )
    parser.add_argument(
        "--refined-las",
        required=True,
        help="Path to the refined LAS after voxel merge and polar refinement.",
    )
    parser.add_argument("--output", required=True, help="Path to the output PNG figure.")
    parser.add_argument(
        "--class-codes",
        nargs="+",
        type=int,
        default=[64, 65, 66, 67, 68],
        help="Structure class codes to visualize.",
    )
    parser.add_argument("--max-points", type=int, default=150_000, help="Maximum points per cloud panel.")
    parser.add_argument("--point-size", type=float, default=0.5, help="Scatter marker size.")
    parser.add_argument(
        "--show-background-mask",
        action="store_true",
        help="Keep original non-structure codes in the mask panel as background.",
    )
    parser.add_argument("--sample-seed", type=int, default=42, help="Random seed for point sampling.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    make_label_transfer_validation_figure(
        mask_path=args.mask,
        projected_las_path=args.projected_las,
        refined_las_path=args.refined_las,
        output_path=args.output,
        class_codes=tuple(args.class_codes),
        max_points=args.max_points,
        point_size=args.point_size,
        show_background_mask=args.show_background_mask,
        sample_seed=args.sample_seed,
    )


if __name__ == "__main__":
    main()