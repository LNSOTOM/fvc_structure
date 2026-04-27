"""Create a validation gallery for the Phase 4 2D-to-3D label workflow.

Panels:
  (a) Original 2D structure mask
  (b) Reference raster / multispectral tile
  (c) Co-registered LiDAR point cloud
  (d) Raster-LiDAR overlay
  (e) Initial projection of 2D labels
  (f) 3D view of initial 2D-to-3D label projection
  (g) Refined 3D labels
  (h) Vertical view of refined point cloud
  (i) Step 4 watershed crown regions used to initialise mallee tree IDs
  (j) Step 9 final mallee canopy points coloured by non-zero treeID
"""

from __future__ import annotations

import argparse
from pathlib import Path
import re
from typing import Dict, Iterable, Optional, Sequence, Tuple

import laspy
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.colors import Normalize
from matplotlib.ticker import ScalarFormatter


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

TREEID_SINGLE_COLOR = np.array([0.7922, 0.1216, 0.4824, 0.95], dtype=np.float32)


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


def resolve_input_path(path_str: str, label: str) -> Path:
    candidate = Path(path_str).expanduser()
    if path_str.strip() == "...":
        raise FileNotFoundError(f"{label} path is still set to placeholder '...'. Replace it with a real file path.")
    if re.match(r"^[A-Za-z]:[\\/]", path_str):
        raise FileNotFoundError(
            f"{label} path '{path_str}' looks like a Windows path, but this script is running on Linux. "
            "Use the mounted Linux path instead."
        )
    if not candidate.exists():
        raise FileNotFoundError(f"{label} not found: {candidate}")
    return candidate


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


def validation_z(points: Dict[str, np.ndarray | None]) -> np.ndarray:
    z_norm = points.get("z_norm")
    if z_norm is not None:
        return np.asarray(z_norm, dtype=np.float32)
    z_values = np.asarray(points["z"], dtype=np.float32)
    baseline = float(np.nanpercentile(z_values, 1))
    return z_values - baseline


def combined_z_limits(*points_sets: Optional[Dict[str, np.ndarray | None]]) -> tuple[float, float]:
    finite_values = []
    for points in points_sets:
        if points is None:
            continue
        z_values = validation_z(points)
        z_values = z_values[np.isfinite(z_values)]
        if z_values.size:
            finite_values.append(z_values)
    if not finite_values:
        return (0.0, 1.0)
    merged = np.concatenate(finite_values)
    z_min = float(np.min(merged))
    z_max = float(np.max(merged))
    padding = 0.05 * max(z_max - z_min, 1.0)
    return (z_min - padding, z_max + padding)


def get_las_z_norm(las: laspy.LasData) -> np.ndarray | None:
    dims = set(las.point_format.dimension_names)
    for name in ("Z_norm", "z_norm"):
        if name in dims:
            return np.asarray(las[name], dtype=np.float32)
    for name in ("Z_norm", "z_norm"):
        if hasattr(las, name):
            return np.asarray(getattr(las, name), dtype=np.float32)
    return None


def format_planar_panel(ax, extent: Sequence[float]) -> None:
    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])
    ax.set_aspect("equal")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")


def normalize_band(band: np.ndarray) -> np.ndarray:
    finite = band[np.isfinite(band)]
    if finite.size == 0:
        return np.zeros_like(band, dtype=np.float32)
    lo, hi = np.percentile(finite, [2, 98])
    scaled = np.clip((band - lo) / (hi - lo + 1e-6), 0, 1)
    scaled[~np.isfinite(scaled)] = 0
    return scaled.astype(np.float32)


def load_raster_rgb(raster_path: str) -> tuple[np.ndarray, tuple[float, float, float, float]]:
    with rasterio.open(raster_path) as src:
        if src.count >= 5:
            bands = (5, 3, 1)
            rgb = np.stack([normalize_band(src.read(b)) for b in bands], axis=-1)
        elif src.count >= 3:
            bands = (1, 2, 3)
            rgb = np.stack([normalize_band(src.read(b)) for b in bands], axis=-1)
        else:
            gray = normalize_band(src.read(1))
            rgb = np.stack([gray, gray, gray], axis=-1)
        extent = (src.bounds.left, src.bounds.right, src.bounds.bottom, src.bounds.top)
    return rgb, extent


def sample_display_las(las_path: str, max_points: int = 180_000, seed: int = 42) -> dict[str, np.ndarray]:
    las = laspy.read(las_path)
    n_points = len(las.x)
    if n_points == 0:
        return {
            "x": np.array([], dtype=np.float64),
            "y": np.array([], dtype=np.float64),
            "colors": np.empty((0, 4), dtype=np.float32),
        }

    idx = np.arange(n_points)
    if n_points > max_points:
        rng = np.random.default_rng(seed)
        idx = rng.choice(n_points, size=max_points, replace=False)

    x = np.asarray(las.x)[idx]
    y = np.asarray(las.y)[idx]

    dims = set(las.point_format.dimension_names)
    if {"red", "green", "blue"}.issubset(dims):
        colors = np.stack(
            [
                np.asarray(las["red"])[idx],
                np.asarray(las["green"])[idx],
                np.asarray(las["blue"])[idx],
            ],
            axis=-1,
        ).astype(np.float32)
        denom = 65535.0 if np.max(colors) > 255 else 255.0
        colors = np.clip(colors / denom, 0, 1)
        alpha = np.full((colors.shape[0], 1), 0.65, dtype=np.float32)
        colors = np.concatenate([colors, alpha], axis=1)
    else:
        colors = np.tile(np.array([[0.2, 0.2, 0.2, 0.45]], dtype=np.float32), (len(x), 1))

    return {"x": x, "y": y, "colors": colors}


def validation_colors(class_values: np.ndarray) -> np.ndarray:
    return np.array([CLASS_COLORS.get(int(code), "#8f8f8f") for code in class_values], dtype=object)


def plot_las_only_panel(ax, las_display: dict[str, np.ndarray], title: str) -> None:
    ax.scatter(
        las_display["x"],
        las_display["y"],
        s=0.25,
        c=las_display["colors"],
        linewidths=0,
    )
    ax.set_title(title, fontsize=11)
    format_planar_panel(
        ax,
        [
            float(np.min(las_display["x"])) if len(las_display["x"]) else 0.0,
            float(np.max(las_display["x"])) if len(las_display["x"]) else 1.0,
            float(np.min(las_display["y"])) if len(las_display["y"]) else 0.0,
            float(np.max(las_display["y"])) if len(las_display["y"]) else 1.0,
        ],
    )


def plot_raster_las_overlay_panel(
    ax,
    rgb_overlay: np.ndarray,
    overlay_extent: Sequence[float],
    las_display: dict[str, np.ndarray],
    title: str,
) -> None:
    ax.imshow(rgb_overlay, extent=overlay_extent, origin="upper")
    ax.scatter(
        las_display["x"],
        las_display["y"],
        s=0.25,
        c=las_display["colors"],
        linewidths=0,
    )
    ax.set_title(title, fontsize=11)
    format_planar_panel(ax, overlay_extent)


def plot_overlay_panel(
    ax,
    rgb_overlay: np.ndarray,
    overlay_extent: Sequence[float],
    points: Dict[str, np.ndarray | None],
    title: str,
    point_size: float = 0.35,
) -> None:
    ax.imshow(rgb_overlay, extent=overlay_extent, origin="upper")
    class_values = np.asarray(points["class"])
    ax.scatter(
        points["x"],
        points["y"],
        s=point_size,
        c=validation_colors(class_values),
        alpha=0.75,
        linewidths=0,
    )
    ax.set_title(title, fontsize=11)
    format_planar_panel(ax, overlay_extent)


def plot_3d_panel(
    ax,
    points: Dict[str, np.ndarray | None],
    title: str,
    z_limits: tuple[float, float],
    extent: Sequence[float],
) -> None:
    z_vals = validation_z(points)
    class_values = np.asarray(points["class"])
    colors = validation_colors(class_values)
    order = np.argsort(z_vals)

    x_vals = np.asarray(points["x"])[order]
    y_vals = np.asarray(points["y"])[order]
    z_vals = z_vals[order]
    colors = colors[order]

    ax.scatter(x_vals, y_vals, z_vals, s=0.8, c=colors, alpha=0.78, linewidths=0, depthshade=False)
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z_norm" if points.get("z_norm") is not None else "Height proxy")
    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])
    ax.set_zlim(*z_limits)
    ax.view_init(elev=24, azim=-58)
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(ScalarFormatter(useOffset=False))
    ax.yaxis.set_major_formatter(ScalarFormatter(useOffset=False))
    ax.zaxis.set_major_formatter(ScalarFormatter(useOffset=False))
    ax.tick_params(labelsize=8)


def load_watershed_tree_data(projected_las_path: str) -> dict[str, object] | None:
    try:
        from fvc3dstruct.watershed import get_mallee_centroids_watershed

        projected_path = Path(projected_las_path)
        structure_las = laspy.read(str(projected_path))
        tile_id_match = re.search(r"(\d+)$", projected_path.stem)
        tile_id = tile_id_match.group(1) if tile_id_match else None

        clipped_root = projected_path.parent.parent.parent.parent
        fallback_points = None
        if tile_id is not None:
            znorm_las_path = (
                clipped_root / "raster" / "z_norm" / "csf" / f"z_norm_csf_tiles_multispectral.{tile_id}.las"
            )
            if znorm_las_path.exists():
                znorm_las = laspy.read(str(znorm_las_path))
                znorm_values = get_las_z_norm(znorm_las)
                if znorm_values is None:
                    z_values = np.asarray(znorm_las.z, dtype=np.float32)
                    znorm_values = z_values - float(np.percentile(z_values, 1))
                fallback_points = (znorm_las.x, znorm_las.y, znorm_values)

        if fallback_points is not None:
            labels, centroids, min_x, min_y = get_mallee_centroids_watershed(
                structure_las,
                grid_res=0.10,
                min_tree_dist=0.80,
                smoothing_sigma=1.33,
                fallback_points=fallback_points,
                fallback_min_height=1.0,
            )
        else:
            labels, centroids, min_x, min_y = get_mallee_centroids_watershed(
                structure_las,
                grid_res=0.10,
                min_tree_dist=0.80,
                smoothing_sigma=1.33,
            )

        extent = (
            float(min_x),
            float(min_x) + labels.shape[1] * 0.10,
            float(min_y),
            float(min_y) + labels.shape[0] * 0.10,
        )
        return {
            "labels": labels,
            "centroids": centroids,
            "extent": extent,
            "count": int(labels.max()),
        }
    except Exception as exc:
        print(f"Watershed crown display skipped: {exc}")
        return None


def sample_treeid_points(
    las_path: str,
    max_points: int = 140_000,
    seed: int = 42,
) -> dict[str, np.ndarray] | None:
    las = laspy.read(las_path)
    dims = set(las.point_format.dimension_names)
    if "treeID" not in dims:
        return None

    tree_ids = np.asarray(las["treeID"], dtype=np.int32)
    class_values = np.asarray(las.classification, dtype=np.int16)
    keep = (tree_ids > 0) & (class_values == 68)
    if not np.any(keep):
        return None

    x_values = np.asarray(las.x)[keep]
    y_values = np.asarray(las.y)[keep]
    z_values = np.asarray(las.z, dtype=np.float32)[keep]
    tree_ids = tree_ids[keep]

    z_norm_all = get_las_z_norm(las)
    if z_norm_all is not None:
        z_norm = np.asarray(z_norm_all, dtype=np.float32)[keep]
    else:
        z_norm = z_values - float(np.percentile(z_values, 1))

    if len(x_values) > max_points:
        rng = np.random.default_rng(seed)
        sample_idx = rng.choice(len(x_values), size=max_points, replace=False)
        x_values = x_values[sample_idx]
        y_values = y_values[sample_idx]
        z_values = z_values[sample_idx]
        z_norm = z_norm[sample_idx]
        tree_ids = tree_ids[sample_idx]

    return {
        "x": x_values,
        "y": y_values,
        "z": z_values,
        "z_norm": z_norm,
        "treeID": tree_ids,
    }


def treeid_colors(tree_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    unique_ids = np.unique(tree_ids)
    if len(unique_ids) == 0:
        return np.empty((0, 4), dtype=np.float32), unique_ids
    if len(unique_ids) == 1:
        return np.tile(TREEID_SINGLE_COLOR, (len(tree_ids), 1)), unique_ids

    id_to_rank = {int(tree_id): idx for idx, tree_id in enumerate(unique_ids)}
    ranks = np.array([id_to_rank[int(tree_id)] for tree_id in tree_ids], dtype=np.float32)
    colors = plt.cm.nipy_spectral(ranks / max(len(unique_ids) - 1, 1))
    colors[:, 3] = 0.92
    return colors, unique_ids


def add_treeid_colorbar(fig, cax, treeid_points: dict[str, np.ndarray] | None) -> None:
    if treeid_points is None:
        cax.axis("off")
        return

    unique_ids = np.unique(np.asarray(treeid_points["treeID"], dtype=np.int32))
    n_ids = len(unique_ids)
    if n_ids == 0:
        cax.axis("off")
        return

    if n_ids == 1:
        cmap = ListedColormap([TREEID_SINGLE_COLOR])
        norm = BoundaryNorm([-0.5, 0.5], cmap.N)
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = fig.colorbar(sm, cax=cax)
        cbar.set_ticks([0])
        cbar.set_ticklabels([str(unique_ids[0])])
        cbar.set_label("Final treeID", fontsize=10)
        cbar.ax.tick_params(labelsize=9)
        return

    norm = Normalize(vmin=0, vmax=n_ids - 1)
    sm = plt.cm.ScalarMappable(cmap=plt.cm.nipy_spectral, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cax)
    tick_idx = np.linspace(0, n_ids - 1, min(8, n_ids), dtype=int)
    tick_idx = np.unique(tick_idx)
    cbar.set_ticks(tick_idx)
    cbar.set_ticklabels([str(unique_ids[idx]) for idx in tick_idx])
    cbar.set_label("Final treeID", fontsize=10)
    cbar.ax.tick_params(labelsize=9)


def plot_watershed_treeid_panel(ax, extent: Sequence[float], watershed_data: dict[str, object] | None, title: str) -> None:
    if watershed_data is None:
        ax.text(
            0.5,
            0.5,
            "Watershed crowns unavailable",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=10,
        )
    else:
        labels = np.asarray(watershed_data["labels"])
        ws_extent = watershed_data["extent"]
        centroids = watershed_data["centroids"]
        crown_grid = np.ma.masked_where(labels == 0, labels)
        ax.imshow(crown_grid, origin="lower", cmap="nipy_spectral", extent=ws_extent, alpha=0.8)
        if centroids is not None and len(centroids) > 1:
            ax.scatter(
                centroids[1:, 0],
                centroids[1:, 1],
                c="white",
                edgecolors="black",
                marker="x",
                s=24,
                linewidths=0.8,
                zorder=5,
            )
        ax.text(
            0.02,
            0.98,
            f"{int(watershed_data['count'])} watershed crowns",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=9,
            bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": "#b7c1c9", "alpha": 0.9},
        )

    ax.set_title(title, fontsize=11)
    format_planar_panel(ax, extent)


def plot_treeid_panel_3d(
    ax,
    treeid_points: dict[str, np.ndarray] | None,
    extent: Sequence[float],
    title: str,
    z_limits: tuple[float, float],
) -> None:
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z_norm")
    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])
    ax.set_zlim(*z_limits)
    ax.view_init(elev=24, azim=-58)

    if treeid_points is None:
        ax.text2D(0.5, 0.5, "No non-zero treeID values found", ha="center", va="center", transform=ax.transAxes)
        return

    x_vals = np.asarray(treeid_points["x"])
    y_vals = np.asarray(treeid_points["y"])
    z_vals = np.asarray(treeid_points["z_norm"], dtype=np.float32)
    tree_ids = np.asarray(treeid_points["treeID"])
    colors, unique_ids = treeid_colors(tree_ids)

    order = np.argsort(z_vals)
    x_vals = x_vals[order]
    y_vals = y_vals[order]
    z_vals = z_vals[order]
    colors = colors[order]

    x_span = max(extent[1] - extent[0], 1e-6)
    y_span = max(extent[3] - extent[2], 1e-6)
    z_span = max(z_limits[1] - z_limits[0], 1e-6)

    ax.scatter(x_vals, y_vals, z_vals, s=3.2, c="black", alpha=0.08, linewidths=0, depthshade=False)
    ax.scatter(x_vals, y_vals, z_vals, s=0.7, c=colors, alpha=0.92, linewidths=0, depthshade=False)
    ax.text2D(
        0.02,
        0.98,
        f"{len(unique_ids)} final treeIDs",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9,
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": "#b7c1c9", "alpha": 0.9},
    )
    ax.grid(True, alpha=0.3)
    ax.xaxis.pane.set_alpha(0.05)
    ax.yaxis.pane.set_alpha(0.05)
    ax.zaxis.pane.set_alpha(0.08)
    ax.xaxis.set_major_formatter(ScalarFormatter(useOffset=False))
    ax.yaxis.set_major_formatter(ScalarFormatter(useOffset=False))
    ax.zaxis.set_major_formatter(ScalarFormatter(useOffset=False))
    ax.tick_params(labelsize=8)
    ax.set_box_aspect((x_span, y_span, z_span * 1.18))


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
    overlay_raster_path: Optional[str] = None,
    class_codes: Tuple[int, ...] = (64, 65, 66, 67, 68),
    max_points: int = 150_000,
    point_size: float = 0.5,
    show_background_mask: bool = False,
    sample_seed: int = 42,
) -> None:
    mask_path_resolved = resolve_input_path(mask_path, "Mask raster")
    projected_las_path_resolved = resolve_input_path(projected_las_path, "Projected LAS")
    refined_las_path_resolved = resolve_input_path(refined_las_path, "Refined LAS")
    overlay_source_resolved = (
        resolve_input_path(overlay_raster_path, "Overlay raster") if overlay_raster_path else mask_path_resolved
    )

    mask, transform = read_mask(str(mask_path_resolved))
    validation_extent = raster_extent_from_transform(mask.shape, transform)
    rgb_overlay, overlay_extent = load_raster_rgb(str(overlay_source_resolved))

    projected_pts = sample_points(
        filter_points_by_class(read_las_points(str(projected_las_path_resolved)), class_codes),
        max_points=max_points,
        seed=sample_seed,
    )
    refined_pts = sample_points(
        filter_points_by_class(read_las_points(str(refined_las_path_resolved)), class_codes),
        max_points=max_points,
        seed=sample_seed,
    )
    projected_las_display = sample_display_las(
        str(projected_las_path_resolved), max_points=max_points, seed=sample_seed
    )

    watershed_tree_data = load_watershed_tree_data(str(projected_las_path_resolved))
    refined_treeid_points = sample_treeid_points(
        str(refined_las_path_resolved),
        max_points=min(max_points, 140_000),
        seed=sample_seed,
    )
    z_limits = combined_z_limits(projected_pts, refined_pts)
    treeid_z_limits = combined_z_limits(refined_pts, refined_treeid_points)

    fig = plt.figure(figsize=(21, 33))
    grid = fig.add_gridspec(
        10,
        5,
        height_ratios=(0.10, 0.95, 0.10, 0.95, 0.10, 1.18, 0.10, 1.18, 0.10, 0.95),
        width_ratios=(1.0, 1.0, 0.08, 1.0, 1.0),
        hspace=0.18,
        wspace=0.18,
    )

    row_tags = {
        "r1": fig.add_subplot(grid[0, :]),
        "r2": fig.add_subplot(grid[2, :]),
        "r3": fig.add_subplot(grid[4, :]),
        "r4": fig.add_subplot(grid[6, :]),
        "r5": fig.add_subplot(grid[8, :]),
    }
    for ax in row_tags.values():
        ax.axis("off")

    axes = {
        "a": fig.add_subplot(grid[1, 0:2]),
        "b": fig.add_subplot(grid[1, 3:5]),
        "c": fig.add_subplot(grid[3, 0:2]),
        "d": fig.add_subplot(grid[3, 3:5]),
        "e": fig.add_subplot(grid[5, 0:2]),
        "f": fig.add_subplot(grid[5, 3:5], projection="3d"),
        "g": fig.add_subplot(grid[7, 0:2]),
        "h": fig.add_subplot(grid[7, 3:5], projection="3d"),
        "i": fig.add_subplot(grid[9, 0:2]),
        "j": fig.add_subplot(grid[9, 3:5], projection="3d"),
    }
    cbar_axes = {
        "a": fig.add_subplot(grid[1, 2]),
        "e": fig.add_subplot(grid[5, 2]),
        "g": fig.add_subplot(grid[7, 2]),
        "j": fig.add_subplot(grid[9, 2]),
    }
    spacer_ax = fig.add_subplot(grid[3, 2])
    spacer_ax.axis("off")

    row_tags["r1"].text(0.0, 0.5, "2D reference inputs", fontsize=13, fontweight="bold", ha="left", va="center")
    row_tags["r2"].text(0.0, 0.5, "Spatial alignment assessment", fontsize=13, fontweight="bold", ha="left", va="center")
    row_tags["r3"].text(0.0, 0.5, "2D-to-3D transfer and refinement", fontsize=13, fontweight="bold", ha="left", va="center")
    row_tags["r4"].text(0.0, 0.5, "Structural plausibility from Z-normalised LiDAR", fontsize=13, fontweight="bold", ha="left", va="center")
    row_tags["r5"].text(0.0, 0.5, "Step 4 and Step 9 mallee crown instance IDs", fontsize=13, fontweight="bold", ha="left", va="center")

    if not show_background_mask:
        mask = np.where(np.isin(mask, class_codes), mask, 0)
    mask_codes = [0] + list(class_codes)
    mask_cmap, mask_norm = build_class_cmap(mask_codes)
    mask_idx = remap_to_index(mask, mask_codes)

    axes["a"].imshow(mask_idx, cmap=mask_cmap, norm=mask_norm, extent=validation_extent, origin="upper")
    axes["a"].set_title("(a) Original 2D structure mask derived from multispectral imagery", fontsize=11)
    format_planar_panel(axes["a"], validation_extent)

    axes["b"].imshow(rgb_overlay, extent=overlay_extent, origin="upper")
    axes["b"].set_title("(b) Reference raster used to derive the 2D structure mask", fontsize=11)
    format_planar_panel(axes["b"], overlay_extent)

    plot_las_only_panel(axes["c"], projected_las_display, "(c) Co-registered LiDAR point cloud before label transfer")
    plot_raster_las_overlay_panel(
        axes["d"],
        rgb_overlay,
        overlay_extent,
        projected_las_display,
        "(d) Raster-LiDAR overlay for spatial alignment assessment",
    )
    plot_overlay_panel(
        axes["e"],
        rgb_overlay,
        overlay_extent,
        projected_pts,
        "(e) Initial projection of 2D labels onto the LiDAR point cloud",
        point_size=max(point_size * 0.7, 0.25),
    )
    plot_3d_panel(
        axes["f"],
        projected_pts,
        "(f) 3D view of the initial 2D-to-3D label projection",
        z_limits,
        overlay_extent,
    )
    plot_overlay_panel(
        axes["g"],
        rgb_overlay,
        overlay_extent,
        refined_pts,
        "(g) Refined 3D labels after voxel-based aggregation and polar refinement",
        point_size=max(point_size * 0.7, 0.25),
    )
    plot_3d_panel(
        axes["h"],
        refined_pts,
        "(h) Vertical view of the refined point cloud showing height-based separation of vegetation classes",
        z_limits,
        overlay_extent,
    )
    plot_watershed_treeid_panel(
        axes["i"],
        overlay_extent,
        watershed_tree_data,
        "(i) Step 4: Watershed crown regions used to initialise mallee tree IDs",
    )
    plot_treeid_panel_3d(
        axes["j"],
        refined_treeid_points,
        overlay_extent,
        "(j) Step 9: Final mallee canopy points coloured by non-zero treeID",
        treeid_z_limits,
    )

    class_sm = plt.cm.ScalarMappable(
        cmap=ListedColormap([CLASS_COLORS[code] for code in class_codes]),
        norm=BoundaryNorm(np.arange(-0.5, len(class_codes) + 0.5, 1.0), len(class_codes)),
    )
    class_sm.set_array([])
    for key in ("a", "e", "g"):
        cbar = fig.colorbar(class_sm, cax=cbar_axes[key])
        cbar.set_ticks(np.arange(len(class_codes)))
        cbar.set_ticklabels([CLASS_LABELS[code] for code in class_codes])
        cbar.set_label("Structure class", fontsize=10)
        cbar.ax.tick_params(labelsize=9)

    add_treeid_colorbar(fig, cbar_axes["j"], refined_treeid_points)

    fig.suptitle("Validation of the 2D--3D label transfer workflow", fontsize=15, y=0.992)
    fig.subplots_adjust(left=0.05, right=0.985, top=0.975, bottom=0.04)

    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create the full Phase 4 label-transfer validation gallery, including the treeID row."
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
    parser.add_argument(
        "--overlay-raster",
        help="Optional reference raster or multispectral image. Defaults to --mask when omitted.",
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
        overlay_raster_path=args.overlay_raster,
        class_codes=tuple(args.class_codes),
        max_points=args.max_points,
        point_size=args.point_size,
        show_background_mask=args.show_background_mask,
        sample_seed=args.sample_seed,
    )


if __name__ == "__main__":
    main()
