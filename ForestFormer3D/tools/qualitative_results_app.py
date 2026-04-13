from __future__ import annotations

import argparse
import hashlib
import io
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import laspy
import numpy as np
import plotly.graph_objects as go
from dash import Dash, Input, Output, State, ctx, dcc, html, no_update
from PIL import Image, ImageDraw, ImageFont
from plotly.colors import qualitative
from plyfile import PlyData


DEFAULT_WORK_DIRS_ROOT = Path(
    "/media/laura/8402326D023263F8/calperumResearch/site1_1_DD0001/lidar/ForestFormer3D/data/ForAINetV2/work_dirs"
)
SUPPORTED_SUFFIXES = {".ply", ".las"}
UI_COLORS = {
    "background": "#F3EFE7",
    "panel": "#FFFDF8",
    "ink": "#1F2933",
    "muted": "#5B6470",
    "border": "#D9D1C3",
    "accent": "#1F5C5B",
    "accent_dark": "#15403F",
    "chip_bg": "#EEE7DA",
}
OVERLAY_CLASS_COLORS = {
    64: (0.851, 0.706, 0.290),  # Forb      → #D9B44A (ground-like)
    65: (0.412, 0.220, 0.082),  # Woody debris → #694714
    66: (0.851, 0.706, 0.290),  # Hummock   → #D9B44A (ground-like)
    67: (0.412, 0.220, 0.082),  # Shrub     → #694714 (darker brown)
    68: (0.180, 0.545, 0.341),  # Mallee    → #2E8B57 (tree analogue)
    69: (0.851, 0.706, 0.290),  # Ground    → #D9B44A
    70: (0.412, 0.220, 0.082),  # Wood      → #694714 (darker brown)
    71: (0.180, 0.545, 0.341),  # Tree      → #2E8B57
}
OVERLAY_CLASS_LABELS = {
    64: "Forb",
    65: "Woody debris",
    66: "Hummock",
    67: "Shrub",
    68: "Mallee",
    69: "Ground",
    70: "Wood",
    71: "Tree",
}
SITE_SEMANTIC_OVERLAY_ALIASES = {
    "low": {0: 64, 1: 67, 2: 68},
    "medium": {0: 66, 1: 67, 2: 68},
    "forest": {0: 69, 1: 70, 2: 71},
    "unknown": {0: 64, 1: 67, 2: 68},
}
IGNORE_COLOR = "#B8BEC7"
INSTANCE_PALETTE = (
    qualitative.Dark24
    + qualitative.Light24
    + qualitative.Alphabet
    + qualitative.Safe
)
EXPORT_TITLE_FONT_SIZE = 40
EXPORT_LEGEND_FONT_SIZE = 30
EXPORT_COMPARISON_HEADER_FONT_SIZE = 44
EXPORT_MARKER_SIZE_MULTIPLIER = 2.0  # doubles point size on export
CAMERA_EYE = {"x": 1.45, "y": -1.55, "z": 0.78}
EXPORT_CAMERA_EYE = {"x": 1.55, "y": -1.65, "z": 1.20}
PLOT_TOP_MARGIN = 72


@dataclass(frozen=True)
class SceneEntry:
    experiment_name: str
    scene_name: str
    scene_path: str


def color_tuple_to_hex(color: tuple[float, float, float]) -> str:
    red, green, blue = color
    return f"#{int(red * 255):02x}{int(green * 255):02x}{int(blue * 255):02x}"


def color_tuple_to_rgba(color: tuple[float, float, float], alpha: float) -> str:
    red, green, blue = color
    return f"rgba({int(red * 255)}, {int(green * 255)}, {int(blue * 255)}, {alpha})"


SEMANTIC_COLORS = {
    class_id: color_tuple_to_hex(color)
    for class_id, color in OVERLAY_CLASS_COLORS.items()
}


def infer_site_kind(scene_name: str, experiment_name: str = "") -> str:
    if "_l_" in scene_name:
        return "low"
    if "_m_" in scene_name:
        return "medium"
    if any(tag in experiment_name for tag in ("zero_shot_forest", "zero_shot_culs", "zero_shot_scion")):
        return "forest"
    return "unknown"


def semantic_overlay_id(label_id: int, site_kind: str) -> int | None:
    if label_id == -1:
        return None
    return SITE_SEMANTIC_OVERLAY_ALIASES.get(site_kind, SITE_SEMANTIC_OVERLAY_ALIASES["unknown"]).get(label_id)


def semantic_color(label_id: int, site_kind: str) -> str:
    overlay_class_id = semantic_overlay_id(label_id, site_kind)
    if overlay_class_id is None:
        return IGNORE_COLOR
    return SEMANTIC_COLORS.get(overlay_class_id, "#7F7F7F")


def instance_base_color(semantic_id: int, site_kind: str) -> str:
    if semantic_id == -1:
        return IGNORE_COLOR
    return semantic_color(semantic_id, site_kind)


def semantic_label(label_id: int, site_kind: str) -> str:
    overlay_class_id = semantic_overlay_id(label_id, site_kind)
    if overlay_class_id is None:
        return "Ignore (-1)"
    overlay_name = OVERLAY_CLASS_LABELS.get(overlay_class_id)
    if overlay_name is None:
        return str(label_id)
    if label_id in SITE_SEMANTIC_OVERLAY_ALIASES.get(site_kind, {}):
        return f"{overlay_name} ({label_id}->{overlay_class_id})"
    return f"{overlay_name} ({overlay_class_id})"


def site_class_summary(site_kind: str) -> str:
    if site_kind == "medium":
        return "Medium site mapping: 0 = Hummock, 1 = Shrub, 2 = Mallee"
    if site_kind == "low":
        return "Low site mapping: 0 = Forb, 1 = Shrub, 2 = Mallee"
    if site_kind == "forest":
        return "Forest site: 0 = Ground, 1 = Wood, 2 = Tree"
    return "Site mapping: 0 = Forb/Hummock, 1 = Shrub, 2 = Mallee"


def site_heading(site_kind: str) -> str:
    if site_kind == "medium":
        return "Qualitative results · Medium site"
    if site_kind == "low":
        return "Qualitative results · Low site"
    if site_kind == "forest":
        return "Qualitative results · Forest site"
    return "Qualitative results"


def make_site_banner(scene_name: str, experiment_name: str = "") -> html.Div:
    site_kind = infer_site_kind(scene_name, experiment_name)
    return html.Div(
        site_class_summary(site_kind),
        style={
            "display": "inline-block",
            "margin": "0 auto 16px auto",
            "padding": "8px 14px",
            "borderRadius": "999px",
            "backgroundColor": UI_COLORS["chip_bg"],
            "border": f"1px solid {UI_COLORS['border']}",
            "color": UI_COLORS["ink"],
            "fontSize": "14px",
            "fontWeight": 600,
        },
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Launch a Dash app to browse ForestFormer3D qualitative results across "
            "evaluation experiments."
        )
    )
    parser.add_argument(
        "--work-dirs-root",
        type=Path,
        default=DEFAULT_WORK_DIRS_ROOT,
        help="Root directory containing ForestFormer3D experiment work_dirs.",
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Host interface for the Dash server.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8052,
        help="Port for the Dash server.",
    )
    parser.add_argument(
        "--max-points",
        type=int,
        default=120000,
        help="Maximum number of points to render per scene.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Base seed for deterministic downsampling.",
    )
    parser.add_argument(
        "--marker-size",
        type=float,
        default=2.0,
        help="Scatter marker size for the rendered point clouds.",
    )
    parser.add_argument(
        "--list-only",
        action="store_true",
        help="Print discovered experiments and scenes, then exit.",
    )
    return parser.parse_args()


def scan_experiments(work_dirs_root: Path) -> dict[str, list[SceneEntry]]:
    if not work_dirs_root.exists():
        raise FileNotFoundError(f"work_dirs root does not exist: {work_dirs_root}")

    experiments: dict[str, list[SceneEntry]] = {}
    for experiment_dir in sorted(work_dirs_root.iterdir()):
        if not experiment_dir.is_dir():
            continue
        scene_paths = sorted(
            path
            for path in experiment_dir.iterdir()
            if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES
        )
        if not scene_paths:
            continue
        experiments[experiment_dir.name] = [
            SceneEntry(
                experiment_name=experiment_dir.name,
                scene_name=scene_path.stem,
                scene_path=str(scene_path),
            )
            for scene_path in scene_paths
        ]
    if not experiments:
        raise RuntimeError(f"No scene files were found under {work_dirs_root}")
    return experiments


def print_experiment_summary(experiments: dict[str, list[SceneEntry]]) -> None:
    for experiment_name, scenes in experiments.items():
        print(f"{experiment_name}: {len(scenes)} scenes")
        for scene in scenes:
            print(f"  - {scene.scene_name}")


def _read_required_columns_from_ply(scene_path: Path) -> dict[str, np.ndarray]:
    ply = PlyData.read(scene_path)
    vertex = ply["vertex"].data
    names = set(vertex.dtype.names or [])
    required = {"x", "y", "z", "semantic_pred", "instance_pred", "semantic_gt", "instance_gt"}
    missing = sorted(required - names)
    if missing:
        raise ValueError(f"Missing required fields in {scene_path}: {missing}")
    return {
        "x": np.asarray(vertex["x"], dtype=np.float32),
        "y": np.asarray(vertex["y"], dtype=np.float32),
        "z": np.asarray(vertex["z"], dtype=np.float32),
        "semantic_pred": np.asarray(vertex["semantic_pred"], dtype=np.int32),
        "instance_pred": np.asarray(vertex["instance_pred"], dtype=np.int32),
        "semantic_gt": np.asarray(vertex["semantic_gt"], dtype=np.int32),
        "instance_gt": np.asarray(vertex["instance_gt"], dtype=np.int32),
    }


def _read_required_columns_from_las(scene_path: Path) -> dict[str, np.ndarray]:
    las = laspy.read(scene_path)
    names = {name.lower(): name for name in las.point_format.dimension_names}
    required_names = {
        "semantic_pred": "semantic_pred",
        "instance_pred": "instance_pred",
        "semantic_gt": "semantic_gt",
        "instance_gt": "instance_gt",
    }
    missing = [key for key in required_names if key not in names]
    if missing:
        raise ValueError(f"Missing required LAS dimensions in {scene_path}: {missing}")
    return {
        "x": np.asarray(las.x, dtype=np.float32),
        "y": np.asarray(las.y, dtype=np.float32),
        "z": np.asarray(las.z, dtype=np.float32),
        "semantic_pred": np.asarray(las[names["semantic_pred"]], dtype=np.int32),
        "instance_pred": np.asarray(las[names["instance_pred"]], dtype=np.int32),
        "semantic_gt": np.asarray(las[names["semantic_gt"]], dtype=np.int32),
        "instance_gt": np.asarray(las[names["instance_gt"]], dtype=np.int32),
    }


@lru_cache(maxsize=128)
def load_scene(scene_path_str: str) -> dict[str, np.ndarray]:
    scene_path = Path(scene_path_str)
    suffix = scene_path.suffix.lower()
    if suffix == ".ply":
        return _read_required_columns_from_ply(scene_path)
    if suffix == ".las":
        return _read_required_columns_from_las(scene_path)
    raise ValueError(f"Unsupported scene suffix for {scene_path}")


def make_sample_indices(scene_path_str: str, total_points: int, max_points: int, base_seed: int) -> np.ndarray:
    if total_points <= max_points:
        return np.arange(total_points, dtype=np.int64)
    digest = hashlib.sha1(scene_path_str.encode("utf-8")).digest()
    scene_seed = int.from_bytes(digest[:8], byteorder="big", signed=False)
    rng = np.random.default_rng(scene_seed ^ base_seed)
    return np.sort(rng.choice(total_points, size=max_points, replace=False))


def semantic_colors(labels: np.ndarray, site_kind: str) -> np.ndarray:
    return np.asarray([semantic_color(int(label), site_kind) for label in labels], dtype=object)


def instance_colors(instance_ids: np.ndarray, semantic_ids: np.ndarray, site_kind: str) -> np.ndarray:
    colors: list[str] = []
    palette_len = len(INSTANCE_PALETTE)
    for instance_id, semantic_id in zip(instance_ids, semantic_ids):
        instance_id = int(instance_id)
        semantic_id = int(semantic_id)
        if instance_id > 0:
            colors.append(INSTANCE_PALETTE[instance_id % palette_len])
        else:
            colors.append("rgba(0, 0, 0, 0)")
    return np.asarray(colors, dtype=object)


def semantic_legend_entries(label_ids: np.ndarray, site_kind: str) -> list[tuple[str, str]]:
    entries: list[tuple[str, str]] = []
    for label_id in sorted({int(value) for value in np.unique(label_ids)}):
        entries.append((semantic_label(label_id, site_kind), semantic_color(label_id, site_kind)))
    return entries


def instance_legend_entries(semantic_ids: np.ndarray, site_kind: str) -> list[tuple[str, str]]:
    del semantic_ids, site_kind
    return [("Tree ID > 0", INSTANCE_PALETTE[0])]


def create_ground_mesh(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    semantic_labels: np.ndarray | None = None,
    ground_label: int = 0,
    grid_resolution: int = 30,
):
    """Create a triangulated terrain mesh from ground-labelled points."""
    from scipy.interpolate import griddata
    from scipy.spatial import Delaunay

    # Use semantic ground points if available, else bottom 5%
    if semantic_labels is not None:
        ground_mask = semantic_labels == ground_label
    else:
        ground_mask = z <= np.percentile(z, 5)

    if not np.any(ground_mask):
        ground_mask = z <= np.percentile(z, 5)

    x_ground = x[ground_mask]
    y_ground = y[ground_mask]
    z_ground = z[ground_mask]

    x_min, x_max = x.min(), x.max()
    y_min, y_max = y.min(), y.max()
    xi = np.linspace(x_min, x_max, grid_resolution)
    yi = np.linspace(y_min, y_max, grid_resolution)
    xi_grid, yi_grid = np.meshgrid(xi, yi)

    points = np.column_stack([x_ground, y_ground])
    zi_grid = griddata(points, z_ground, (xi_grid, yi_grid), method="linear")

    # Fill NaN edges with nearest
    if np.any(np.isnan(zi_grid)):
        zi_grid_nearest = griddata(points, z_ground, (xi_grid, yi_grid), method="nearest")
        zi_grid = np.where(np.isnan(zi_grid), zi_grid_nearest, zi_grid)

    # Slightly below to avoid z-fighting with ground points
    zi_grid -= 0.05

    x_flat = xi_grid.flatten()
    y_flat = yi_grid.flatten()
    z_flat = zi_grid.flatten()

    tri = Delaunay(np.column_stack([x_flat, y_flat]))
    return x_flat, y_flat, z_flat, tri.simplices


def build_figure(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    colors: np.ndarray,
    title: str,
    marker_size: float,
    legend_entries: list[tuple[str, str]],
    legend_note: str | None = None,
    semantic_labels: np.ndarray | None = None,
    site_kind: str = "unknown",
) -> go.Figure:
    fig = go.Figure()

    # Apply z-gradient for forest sites (ground + tree), and for low/medium (forb/hummock + mallee)
    if semantic_labels is not None:
        # Ground-like class = 0 (Forb / Hummock / Ground)
        ground_mask = semantic_labels == 0
        # Tree-like class = 2 (Mallee / Tree)
        tree_mask = semantic_labels == 2
        # Mid class = 1 (Shrub / Wood)
        mid_mask = semantic_labels == 1
        # Ignore = -1
        ignore_mask = ~(ground_mask | tree_mask | mid_mask)

        # --- Ground / Forb / Hummock: warm golden-yellow gradient ---
        if np.any(ground_mask):
            z_g = z[ground_mask]
            z_norm = (z_g - z_g.min()) / (z_g.max() - z_g.min() + 1e-6)
            fig.add_trace(
                go.Scatter3d(
                    x=x[ground_mask], y=y[ground_mask], z=z_g,
                    mode="markers",
                    marker={
                        "size": marker_size,
                        "color": z_norm,
                        "colorscale": [
                            [0.0, "#A07820"],  # deep golden-brown (low)
                            [0.35, "#C9973A"], # warm amber
                            [0.65, "#D9B44A"], # golden yellow (mid)
                            [1.0, "#EDD87A"],  # pale yellow (high)
                        ],
                        "opacity": 0.95,
                        "showscale": False,
                    },
                    hoverinfo="skip", showlegend=False,
                )
            )

        # --- Tree / Mallee: forest green gradient ---
        if np.any(tree_mask):
            z_t = z[tree_mask]
            z_norm = (z_t - z_t.min()) / (z_t.max() - z_t.min() + 1e-6)
            fig.add_trace(
                go.Scatter3d(
                    x=x[tree_mask], y=y[tree_mask], z=z_t,
                    mode="markers",
                    marker={
                        "size": marker_size,
                        "color": z_norm,
                        "colorscale": [
                            [0.0, "#1A5C35"],  # deep forest green (low/trunk)
                            [0.35, "#2E8B57"], # forest green
                            [0.65, "#3AAB68"], # mid green
                            [1.0, "#5DC98A"],  # bright canopy green (high)
                        ],
                        "opacity": 1.0,
                        "showscale": False,
                    },
                    hoverinfo="skip", showlegend=False,
                )
            )

        # --- Shrub / Wood: flat brown ---
        if np.any(mid_mask):
            fig.add_trace(
                go.Scatter3d(
                    x=x[mid_mask], y=y[mid_mask], z=z[mid_mask],
                    mode="markers",
                    marker={
                        "size": marker_size,
                        "color": colors[mid_mask],
                        "opacity": 0.95,
                    },
                    hoverinfo="skip", showlegend=False,
                )
            )

        # --- Ignore points ---
        if np.any(ignore_mask):
            fig.add_trace(
                go.Scatter3d(
                    x=x[ignore_mask], y=y[ignore_mask], z=z[ignore_mask],
                    mode="markers",
                    marker={
                        "size": marker_size,
                        "color": colors[ignore_mask],
                        "opacity": 0.95,
                    },
                    hoverinfo="skip", showlegend=False,
                )
            )
    else:
        fig.add_trace(
            go.Scatter3d(
                x=x, y=y, z=z,
                mode="markers",
                marker={"size": marker_size, "color": colors, "opacity": 0.95},
                hoverinfo="skip", showlegend=False,
            )
        )

    for label, color in legend_entries:
        fig.add_trace(
            go.Scatter3d(
                x=[None], y=[None], z=[None],
                mode="markers",
                marker={"size": max(marker_size * 4, 8), "color": color, "opacity": 1.0},
                name=label, showlegend=True, hoverinfo="skip",
            )
        )

    annotations = []
    if legend_note:
        annotations.append(
            {
                "xref": "paper", "yref": "paper",
                "x": 0.98, "y": 0.02,
                "xanchor": "right", "yanchor": "bottom",
                "align": "left",
                "text": legend_note,
                "showarrow": False,
                "font": {"size": 11, "color": UI_COLORS["muted"]},
                "bgcolor": "rgba(255,255,255,0.90)",
                "bordercolor": UI_COLORS["border"],
                "borderwidth": 1, "borderpad": 6,
            }
        )

    fig.update_layout(
        title={"text": title, "x": 0.02, "xanchor": "left", "y": 0.98, "yanchor": "top", "pad": {"t": 10, "b": 4}},
        margin={"l": 0, "r": 0, "t": PLOT_TOP_MARGIN, "b": 0},
        paper_bgcolor="white", plot_bgcolor="white",
        showlegend=True,
        legend={
            "x": 0.98, "y": 0.98, "xanchor": "right", "yanchor": "top",
            "bgcolor": "rgba(255,255,255,0.92)",
            "bordercolor": UI_COLORS["border"], "borderwidth": 1,
            "font": {"size": 12, "color": UI_COLORS["ink"]},
            "itemsizing": "constant",
        },
        annotations=annotations,
        scene={
            "xaxis": {"visible": False}, "yaxis": {"visible": False}, "zaxis": {"visible": False},
            "aspectmode": "data", "camera": {"eye": CAMERA_EYE}, "bgcolor": "white",
        },
        uirevision=title,
    )
    return fig


def build_instance_figure(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    instance_ids: np.ndarray,
    title: str,
    marker_size: float,
    cmin: int = 1,
    cmax: int = 1,
    legend_note: str | None = None,
    semantic_labels: np.ndarray | None = None,
) -> go.Figure:
    mask = instance_ids > 0

    n_colors = max(cmax - cmin + 1, 1)
    discrete_colorscale = []
    for i in range(n_colors):
        frac_start = i / n_colors
        frac_end = (i + 1) / n_colors
        t = i / max(n_colors - 1, 1)
        hue = int(t * 300)
        discrete_colorscale.append([frac_start, f"hsl({hue}, 90%, 50%)"])
        discrete_colorscale.append([frac_end, f"hsl({hue}, 90%, 50%)"])

    tick_vals = [cmin]
    current = ((cmin // 5) + 1) * 5
    while current < cmax:
        tick_vals.append(current)
        current += 5
    if cmax not in tick_vals:
        tick_vals.append(cmax)

    fig = go.Figure()

    if np.any(mask):
        visible_ids = instance_ids[mask].astype(np.int32)
        fig.add_trace(
            go.Scatter3d(
                x=x[mask],
                y=y[mask],
                z=z[mask],
                mode="markers",
                marker={
                    "size": marker_size,
                    "color": visible_ids,
                    "colorscale": discrete_colorscale,
                    "opacity": 0.95,
                    "showscale": True,
                    "cmin": cmin - 0.5,
                    "cmax": cmax + 0.5,
                    "colorbar": {
                        "title": f"Tree ID ({cmin}→{cmax})",
                        "orientation": "h",
                        "x": 0.5,
                        "xanchor": "center",
                        "y": 0.96,
                        "yanchor": "top",
                        "len": 0.5,
                        "thickness": 14,
                        "bgcolor": "rgba(255,255,255,0.82)",
                        "outlinecolor": UI_COLORS["border"],
                        "outlinewidth": 1,
                        "tickmode": "array",
                        "tickvals": tick_vals,
                        "ticktext": [str(v) for v in tick_vals],
                    },
                },
                hoverinfo="skip",
                showlegend=False,
            )
        )
    else:
        legend_note = "No tree ids > 0 are present in this view."

    annotations = []
    if legend_note:
        annotations.append(
            {
                "xref": "paper",
                "yref": "paper",
                "x": 0.98,
                "y": 0.02,
                "xanchor": "right",
                "yanchor": "bottom",
                "align": "left",
                "text": legend_note,
                "showarrow": False,
                "font": {"size": 11, "color": UI_COLORS["muted"]},
                "bgcolor": "rgba(255,255,255,0.90)",
                "bordercolor": UI_COLORS["border"],
                "borderwidth": 1,
                "borderpad": 6,
            }
        )

    fig.update_layout(
        title={
            "text": title,
            "x": 0.02,
            "xanchor": "left",
            "y": 0.98,
            "yanchor": "top",
            "pad": {"t": 10, "b": 4},
        },
        margin={"l": 0, "r": 0, "t": 88, "b": 0},
        paper_bgcolor="white",
        plot_bgcolor="white",
        showlegend=False,
        annotations=annotations,
        scene={
            "xaxis": {"visible": False},
            "yaxis": {"visible": False},
            "zaxis": {"visible": False},
            "aspectmode": "data",
            "camera": {
                "eye": CAMERA_EYE,
            },
            "bgcolor": "white",
        },
        uirevision=title,
    )
    return fig


def tree_id_range_text(instance_ids: np.ndarray) -> str:
    visible_ids = instance_ids[instance_ids > 0]
    if visible_ids.size == 0:
        return "none"
    return f"{int(visible_ids.min())} to {int(visible_ids.max())}"


def format_scene_summary(scene_arrays: dict[str, np.ndarray], sample_count: int, site_kind: str) -> html.Div:
    semantic_gt = scene_arrays["semantic_gt"]
    semantic_pred = scene_arrays["semantic_pred"]
    instance_gt = scene_arrays["instance_gt"]
    instance_pred = scene_arrays["instance_pred"]

    gt_instances = int(np.unique(instance_gt[instance_gt > 0]).size)
    pred_instances = int(np.unique(instance_pred[instance_pred > 0]).size)
    gt_tree_id_range = tree_id_range_text(instance_gt)
    pred_tree_id_range = tree_id_range_text(instance_pred)
    gt_semantics = sorted({semantic_label(int(label), site_kind) for label in np.unique(semantic_gt)})
    pred_semantics = sorted({semantic_label(int(label), site_kind) for label in np.unique(semantic_pred)})

    items = [
        html.Span(f"Sampled points: {sample_count:,}"),
        html.Span(f"GT instances: {gt_instances:,}"),
        html.Span(f"Pred instances: {pred_instances:,}"),
        html.Span(f"GT tree IDs > 0: {gt_tree_id_range}"),
        html.Span(f"Pred tree IDs > 0: {pred_tree_id_range}"),
        html.Span(f"GT semantic labels: {', '.join(gt_semantics)}"),
        html.Span(f"Pred semantic labels: {', '.join(pred_semantics)}"),
    ]
    return html.Div(
        items,
        style={
            "display": "flex",
            "gap": "16px",
            "flexWrap": "wrap",
            "justifyContent": "center",
            "fontSize": "14px",
            "marginBottom": "16px",
            "color": UI_COLORS["muted"],
        },
    )


def make_legend_chip(label: str, color: str) -> html.Div:
    return html.Div(
        [
            html.Span(
                style={
                    "display": "inline-block",
                    "width": "12px",
                    "height": "12px",
                    "borderRadius": "999px",
                    "backgroundColor": color,
                    "border": f"1px solid {UI_COLORS['border']}",
                }
            ),
            html.Span(label),
        ],
        style={
            "display": "inline-flex",
            "alignItems": "center",
            "gap": "8px",
            "padding": "6px 10px",
            "borderRadius": "999px",
            "backgroundColor": UI_COLORS["chip_bg"],
            "color": UI_COLORS["ink"],
            "fontSize": "13px",
        },
    )


def make_mode_legend(mode: str, scene_name: str, experiment_name: str = "") -> html.Div:
    site_kind = infer_site_kind(scene_name, experiment_name)
    if mode == "semantic":
        items = [
            make_legend_chip(semantic_label(0, site_kind), semantic_color(0, site_kind)),
            make_legend_chip(semantic_label(1, site_kind), semantic_color(1, site_kind)),
            make_legend_chip(semantic_label(2, site_kind), semantic_color(2, site_kind)),
        ]
    else:
        items = [make_legend_chip("Tree ID > 0 color bar", INSTANCE_PALETTE[0])]

    return html.Div(
        items,
        style={
            "display": "flex",
            "gap": "10px",
            "justifyContent": "center",
            "flexWrap": "wrap",
            "marginBottom": "16px",
        },
    )


def make_panel(title: str, graph_id: str) -> html.Div:
    return html.Div(
        [
            html.Div(
                title,
                style={
                    "position": "absolute",
                    "top": "8px",
                    "left": "8px",
                    "zIndex": 1,
                    "backgroundColor": UI_COLORS["accent_dark"],
                    "color": "white",
                    "fontWeight": 600,
                    "fontSize": "14px",
                    "padding": "4px 10px",
                    "borderRadius": "999px",
                    "letterSpacing": "0.02em",
                },
            ),
            dcc.Graph(id=graph_id, style={"height": "520px"}, config={"displayModeBar": True}),
        ],
        style={
            "position": "relative",
            "flex": "1 1 0",
            "minWidth": "420px",
            "backgroundColor": UI_COLORS["panel"],
            "border": f"1px solid {UI_COLORS['border']}",
            "borderRadius": "12px",
            "boxShadow": "0 10px 30px rgba(54, 46, 33, 0.08)",
            "padding": "8px",
        },
    )


def build_scene_views(
    scene_path: str,
    scene_name: str,
    experiment_name: str,
    mode: str,
    max_points: int,
    seed: int,
    marker_size: float,
) -> tuple[go.Figure, go.Figure, html.Div]:
    arrays = load_scene(scene_path)
    site_kind = infer_site_kind(scene_name, experiment_name)
    indices = make_sample_indices(scene_path, len(arrays["x"]), max_points=max_points, base_seed=seed)

    x = arrays["x"][indices]
    y = arrays["y"][indices]
    z = arrays["z"][indices]
    semantic_gt = arrays["semantic_gt"][indices]
    semantic_pred = arrays["semantic_pred"][indices]
    instance_gt = arrays["instance_gt"][indices]
    instance_pred = arrays["instance_pred"][indices]

    gt_title = f"GT {mode} [{site_kind}] ({scene_name})"
    pred_title = f"Prediction {mode} [{site_kind}] ({scene_name})"
    gt_legend_note = None
    pred_legend_note = None

    if mode == "semantic":
        gt_colors = semantic_colors(semantic_gt, site_kind)
        pred_colors = semantic_colors(semantic_pred, site_kind)
        gt_legend_entries = semantic_legend_entries(semantic_gt, site_kind)
        pred_legend_entries = semantic_legend_entries(semantic_pred, site_kind)

        gt_fig = build_figure(
            x, y, z, gt_colors, gt_title, marker_size, gt_legend_entries, gt_legend_note,
            semantic_labels=semantic_gt, site_kind=site_kind,
        )
        pred_fig = build_figure(
            x, y, z, pred_colors, pred_title, marker_size, pred_legend_entries, pred_legend_note,
            semantic_labels=semantic_pred, site_kind=site_kind,
        )
    else:
        gt_ids = instance_gt[instance_gt > 0]
        pred_ids = instance_pred[instance_pred > 0]

        gt_cmin = int(gt_ids.min()) if gt_ids.size > 0 else 1
        gt_cmax = int(gt_ids.max()) if gt_ids.size > 0 else 1
        pred_cmin = int(pred_ids.min()) if pred_ids.size > 0 else 1
        pred_cmax = int(pred_ids.max()) if pred_ids.size > 0 else 1

        gt_fig = build_instance_figure(
            x, y, z, instance_gt, gt_title, marker_size,
            cmin=gt_cmin, cmax=gt_cmax, legend_note=gt_legend_note,
            semantic_labels=semantic_gt,
        )
        pred_fig = build_instance_figure(
            x, y, z, instance_pred, pred_title, marker_size,
            cmin=pred_cmin, cmax=pred_cmax, legend_note=pred_legend_note,
            semantic_labels=semantic_pred,
        )

    summary = format_scene_summary(arrays, len(indices), site_kind)
    return gt_fig, pred_fig, summary


def prepare_figure_for_export(fig: go.Figure) -> go.Figure:
    export_fig = go.Figure(fig)
    export_fig.update_layout(
        title={"font": {"size": EXPORT_TITLE_FONT_SIZE}},
        legend={"font": {"size": EXPORT_LEGEND_FONT_SIZE, "color": UI_COLORS["ink"]}},
        scene={
            "xaxis": {"visible": False},
            "yaxis": {"visible": False},
            "zaxis": {"visible": False},
            "aspectmode": "data",
            "camera": {"eye": EXPORT_CAMERA_EYE},
            "bgcolor": "white",
        },
    )

    if export_fig.layout.annotations:
        for annotation in export_fig.layout.annotations:
            annotation.font = {
                **(annotation.font.to_plotly_json() if annotation.font else {}),
                "size": EXPORT_LEGEND_FONT_SIZE,
            }

    for trace in export_fig.data:
        marker = getattr(trace, "marker", None)
        if marker is None:
            continue
        current_size = getattr(marker, "size", None)
        if isinstance(current_size, (int, float)):
            trace.update(marker={"size": current_size * EXPORT_MARKER_SIZE_MULTIPLIER})
        colorbar = getattr(marker, "colorbar", None)
        if colorbar is not None:
            showscale = bool(getattr(marker, "showscale", False))
            trace.update(
                marker={
                    "showscale": showscale,
                    "colorbar": {
                        "title": {"font": {"size": EXPORT_LEGEND_FONT_SIZE}},
                        "tickfont": {"size": EXPORT_LEGEND_FONT_SIZE},
                    },
                }
            )

    return export_fig


def get_export_header_font() -> ImageFont.ImageFont | ImageFont.FreeTypeFont:
    font_candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf",
    ]
    for font_path in font_candidates:
        try:
            return ImageFont.truetype(font_path, EXPORT_COMPARISON_HEADER_FONT_SIZE)
        except OSError:
            continue
    return ImageFont.load_default()


def make_png_download(fig: go.Figure, filename: str):
    def writer(buffer) -> None:
        export_fig = prepare_figure_for_export(fig)
        buffer.write(export_fig.to_image(format="png", width=1800, height=1200, scale=2))

    return dcc.send_bytes(writer, filename)


def make_comparison_png_download(
    gt_fig: go.Figure,
    pred_fig: go.Figure,
    filename: str,
    title: str,
):
    def writer(buffer) -> None:
        export_gt_fig = prepare_figure_for_export(gt_fig)
        export_pred_fig = prepare_figure_for_export(pred_fig)
        gt_image = Image.open(io.BytesIO(export_gt_fig.to_image(format="png", width=1600, height=1100, scale=2))).convert("RGB")
        pred_image = Image.open(io.BytesIO(export_pred_fig.to_image(format="png", width=1600, height=1100, scale=2))).convert("RGB")

        padding = 32
        header_height = 72
        canvas_width = gt_image.width + pred_image.width + padding * 3
        canvas_height = max(gt_image.height, pred_image.height) + padding * 2 + header_height

        canvas = Image.new("RGB", (canvas_width, canvas_height), "white")
        draw = ImageDraw.Draw(canvas)
        draw.text((padding, 20), title, fill=UI_COLORS["ink"], font=get_export_header_font())

        top = header_height + padding // 2
        canvas.paste(gt_image, (padding, top))
        canvas.paste(pred_image, (padding * 2 + gt_image.width, top))
        canvas.save(buffer, format="PNG")

    return dcc.send_bytes(writer, filename)


def create_app(
    experiments: dict[str, list[SceneEntry]],
    max_points: int,
    seed: int,
    marker_size: float,
) -> Dash:
    app = Dash(__name__)
    experiment_names = list(experiments)
    default_experiment = experiment_names[0]
    default_scene = experiments[default_experiment][0].scene_name

    scene_lookup = {
        experiment_name: {scene.scene_name: scene.scene_path for scene in scenes}
        for experiment_name, scenes in experiments.items()
    }

    app.layout = html.Div(
        [
            html.H1(
                id="site-heading",
                children="Qualitative results",
                style={
                    "textAlign": "center",
                    "fontFamily": "Helvetica, Arial, sans-serif",
                    "marginBottom": "18px",
                    "marginTop": "12px",
                    "color": UI_COLORS["ink"],
                },
            ),
            html.Div(
                id="site-class-banner",
                style={"display": "flex", "justifyContent": "center"},
            ),
            html.Div(
                [
                    html.Div(
                        [
                            html.Label("Experiment", style={"fontWeight": 600}),
                            dcc.Dropdown(
                                id="experiment-dropdown",
                                options=[
                                    {"label": name, "value": name}
                                    for name in experiment_names
                                ],
                                value=default_experiment,
                                clearable=False,
                            ),
                        ],
                        style={"minWidth": "420px", "flex": "1 1 420px"},
                    ),
                    html.Div(
                        [
                            html.Label("Scene", style={"fontWeight": 600}),
                            dcc.Dropdown(id="scene-dropdown", value=default_scene, clearable=False),
                        ],
                        style={"minWidth": "320px", "flex": "1 1 320px"},
                    ),
                ],
                style={
                    "display": "flex",
                    "gap": "16px",
                    "justifyContent": "center",
                    "flexWrap": "wrap",
                    "marginBottom": "12px",
                },
            ),
            dcc.Tabs(
                id="mode-tabs",
                value="semantic",
                children=[
                    dcc.Tab(
                        label="Instance seg.",
                        value="instance",
                        style={
                            "backgroundColor": UI_COLORS["chip_bg"],
                            "border": f"1px solid {UI_COLORS['border']}",
                            "color": UI_COLORS["muted"],
                        },
                        selected_style={
                            "backgroundColor": UI_COLORS["accent"],
                            "border": f"1px solid {UI_COLORS['accent']}",
                            "color": "white",
                            "fontWeight": 700,
                        },
                    ),
                    dcc.Tab(
                        label="Semantic seg.",
                        value="semantic",
                        style={
                            "backgroundColor": UI_COLORS["chip_bg"],
                            "border": f"1px solid {UI_COLORS['border']}",
                            "color": UI_COLORS["muted"],
                        },
                        selected_style={
                            "backgroundColor": UI_COLORS["accent"],
                            "border": f"1px solid {UI_COLORS['accent']}",
                            "color": "white",
                            "fontWeight": 700,
                        },
                    ),
                ],
                style={"maxWidth": "420px", "margin": "0 auto 16px auto"},
            ),
            html.Div(id="mode-legend"),
            html.Div(
                [
                    html.Button(
                        "Download GT snapshot",
                        id="download-gt-button",
                        n_clicks=0,
                        style={
                            "padding": "10px 16px",
                            "border": "none",
                            "borderRadius": "999px",
                            "backgroundColor": UI_COLORS["accent"],
                            "color": "white",
                            "fontWeight": 600,
                            "cursor": "pointer",
                        },
                    ),
                    html.Button(
                        "Download prediction snapshot",
                        id="download-pred-button",
                        n_clicks=0,
                        style={
                            "padding": "10px 16px",
                            "border": "none",
                            "borderRadius": "999px",
                            "backgroundColor": UI_COLORS["accent_dark"],
                            "color": "white",
                            "fontWeight": 600,
                            "cursor": "pointer",
                        },
                    ),
                    html.Button(
                        "Download comparison snapshot",
                        id="download-comparison-button",
                        n_clicks=0,
                        style={
                            "padding": "10px 16px",
                            "border": "none",
                            "borderRadius": "999px",
                            "backgroundColor": UI_COLORS["ink"],
                            "color": "white",
                            "fontWeight": 600,
                            "cursor": "pointer",
                        },
                    ),
                    dcc.Download(id="download-gt"),
                    dcc.Download(id="download-pred"),
                    dcc.Download(id="download-comparison"),
                ],
                style={
                    "display": "flex",
                    "gap": "12px",
                    "justifyContent": "center",
                    "flexWrap": "wrap",
                    "marginBottom": "16px",
                },
            ),
            html.Div(id="scene-summary"),
            html.Div(
                [
                    make_panel("GT", "gt-graph"),
                    make_panel("Prediction", "pred-graph"),
                ],
                style={
                    "display": "flex",
                    "gap": "12px",
                    "flexWrap": "wrap",
                    "justifyContent": "center",
                },
            ),
        ],
        style={
            "padding": "12px 16px 24px 16px",
            "backgroundColor": UI_COLORS["background"],
            "minHeight": "100vh",
            "fontFamily": "Helvetica, Arial, sans-serif",
        },
    )

    @app.callback(
        Output("scene-dropdown", "options"),
        Output("scene-dropdown", "value"),
        Input("experiment-dropdown", "value"),
    )
    def update_scene_options(experiment_name: str) -> tuple[list[dict[str, str]], str]:
        scenes = experiments[experiment_name]
        options = [{"label": scene.scene_name, "value": scene.scene_name} for scene in scenes]
        return options, scenes[0].scene_name

    @app.callback(
        Output("site-heading", "children"),
        Output("gt-graph", "figure"),
        Output("pred-graph", "figure"),
        Output("scene-summary", "children"),
        Output("mode-legend", "children"),
        Output("site-class-banner", "children"),
        Input("experiment-dropdown", "value"),
        Input("scene-dropdown", "value"),
        Input("mode-tabs", "value"),
    )
    def update_graphs(
        experiment_name: str,
        scene_name: str,
        mode: str,
    ) -> tuple[str, go.Figure, go.Figure, html.Div, html.Div, html.Div]:
        scene_path = scene_lookup[experiment_name][scene_name]
        site_kind = infer_site_kind(scene_name, experiment_name)
        gt_fig, pred_fig, summary = build_scene_views(
            scene_path, scene_name, experiment_name, mode, max_points, seed, marker_size
        )
        return (
            site_heading(site_kind),
            gt_fig,
            pred_fig,
            summary,
            make_mode_legend(mode, scene_name, experiment_name),
            make_site_banner(scene_name, experiment_name),
        )

    @app.callback(
        Output("download-gt", "data"),
        Output("download-pred", "data"),
        Output("download-comparison", "data"),
        Input("download-gt-button", "n_clicks"),
        Input("download-pred-button", "n_clicks"),
        Input("download-comparison-button", "n_clicks"),
        State("experiment-dropdown", "value"),
        State("scene-dropdown", "value"),
        State("mode-tabs", "value"),
        prevent_initial_call=True,
    )
    def download_snapshot(
        gt_clicks: int,
        pred_clicks: int,
        comparison_clicks: int,
        experiment_name: str,
        scene_name: str,
        mode: str,
    ):
        del gt_clicks, pred_clicks, comparison_clicks
        scene_path = scene_lookup[experiment_name][scene_name]
        gt_fig, pred_fig, _ = build_scene_views(
            scene_path, scene_name, experiment_name, mode, max_points, seed, marker_size
        )
        mode_slug = "semantic" if mode == "semantic" else "instance"

        if ctx.triggered_id == "download-gt-button":
            filename = f"{experiment_name}__{scene_name}__gt_{mode_slug}.png"
            return make_png_download(gt_fig, filename), no_update, no_update

        if ctx.triggered_id == "download-pred-button":
            filename = f"{experiment_name}__{scene_name}__prediction_{mode_slug}.png"
            return no_update, make_png_download(pred_fig, filename), no_update

        if ctx.triggered_id == "download-comparison-button":
            filename = f"{experiment_name}__{scene_name}__comparison_{mode_slug}.png"
            title = f"{experiment_name} | {scene_name} | {mode_slug}"
            return no_update, no_update, make_comparison_png_download(gt_fig, pred_fig, filename, title)

        return no_update, no_update, no_update

    return app


def main() -> None:
    args = parse_args()
    experiments = scan_experiments(args.work_dirs_root)
    if args.list_only:
        print_experiment_summary(experiments)
        return

    app = create_app(
        experiments=experiments,
        max_points=args.max_points,
        seed=args.seed,
        marker_size=args.marker_size,
    )
    app.run(host=args.host, port=args.port, debug=False)


if __name__ == "__main__":
    main()