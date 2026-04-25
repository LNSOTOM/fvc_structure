from __future__ import annotations

import argparse
import hashlib
import io
import re
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
DEFAULT_NORMALIZED_HEIGHT_ROOT = Path(
    "/media/laura/8402326D023263F8/calperumResearch/site1_1_DD0001/lidar/clipped/raster/z_norm/csf"
)
NORMALIZED_HEIGHT_PREFIXES = (
    "z_norm_csf_tiles_multispectral.",
    "tiles_multispectral.",
)
DEFAULT_SEGMENT_ANY_TREE_EXPERIMENTS = {
    "low_L1": Path(
        "/media/laura/8402326D023263F8/calperumResearch/site1_1_DD0001/lidar/SegmentAnyTree/calperum/low_L1_20260312_013732/results"
    ),
    "low_L2": Path(
        "/media/laura/8402326D023263F8/calperumResearch/site1_1_DD0001/lidar/SegmentAnyTree/calperum/low_L2_20260312_020346/results"
    ),
    "medium_L1": Path(
        "/media/laura/8402326D023263F8/calperumResearch/site1_1_DD0001/lidar/SegmentAnyTree/calperum/medium_L1_20260312_010211/results"
    ),
    "medium_L2": Path(
        "/media/laura/8402326D023263F8/calperumResearch/site1_1_DD0001/lidar/SegmentAnyTree/calperum/medium_L2_20260312_012138/results"
    ),
    "forest": Path(
        "/media/laura/8402326D023263F8/calperumResearch/site1_1_DD0001/lidar/SegmentAnyTree/ForAINetV2/forest/results"
    ),
}
SUPPORTED_SUFFIXES = {".ply", ".las", ".laz"}
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
    71: (0.180, 0.545, 0.341),  # Leaf      → #2E8B57
}
OVERLAY_CLASS_LABELS = {
    64: "Forb",
    65: "Woody debris",
    66: "Hummock",
    67: "Shrub",
    68: "Mallee",
    69: "Ground",
    70: "Wood",
    71: "Leaf",
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
EXPORT_DOWNLOAD_TITLE_FONT_SIZE = 56
EXPORT_COMPARISON_TITLE_FONT_SIZE = EXPORT_TITLE_FONT_SIZE * 2
EXPORT_COMPARISON_PANEL_TITLE_FONT_SIZE = 104
EXPORT_COMPARISON_LEGEND_FONT_SIZE = EXPORT_LEGEND_FONT_SIZE * 4
EXPORT_COMPARISON_COLORBAR_THICKNESS = 42
EXPORT_INSTANCE_COMPARISON_LEGEND_FONT_SIZE = 72
EXPORT_INSTANCE_COMPARISON_COLORBAR_THICKNESS = 32
EXPORT_INSTANCE_COMPARISON_COLORBAR_TITLE_FONT_SIZE = 96
EXPORT_INSTANCE_COMPARISON_COLORBAR_TICK_FONT_SIZE = 88
EXPORT_COMPARISON_HEADER_FONT_SIZE = 88
EXPORT_MARKER_SIZE_MULTIPLIER = 2.0  # doubles point size on export
EXPORT_LEGEND_MARKER_SIZE_MULTIPLIER = 2.0
EXPORT_COMPARISON_CUSTOM_LEGEND_DOT_RADIUS = 54
CAMERA_EYE = {"x": 1.35, "y": -1.45, "z": 1.10}
EXPORT_CAMERA_EYE = {"x": 1.25, "y": -1.35, "z": 1.60}
PLOT_TOP_MARGIN = 72
DEFAULT_PREFERRED_SCENE = "SCION_SCION_plot_61_annotated_test_out"
PREFERRED_DEFAULT_SCENES = {
    "ForestFormer3D": "SCION_SCION_plot_31_annotated_test",
}


@dataclass(frozen=True)
class SceneEntry:
    experiment_name: str
    scene_name: str
    scene_path: str


@dataclass(frozen=True)
class HeightSceneEntry:
    tile_id: str
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


def semantic_overlay_class_id(label_id: int, site_kind: str) -> int | None:
    if label_id < 0:
        return None
    aliases = SITE_SEMANTIC_OVERLAY_ALIASES.get(site_kind, SITE_SEMANTIC_OVERLAY_ALIASES["unknown"])
    return aliases.get(label_id)


def semantic_color(label_id: int, site_kind: str) -> str:
    overlay_id = semantic_overlay_class_id(label_id, site_kind)
    if overlay_id is None:
        return IGNORE_COLOR
    return SEMANTIC_COLORS.get(overlay_id, IGNORE_COLOR)


def semantic_label(label_id: int, site_kind: str) -> str:
    if label_id < 0:
        return "Ignore"
    overlay_id = semantic_overlay_class_id(label_id, site_kind)
    if overlay_id is None:
        return f"Unknown ({label_id})"
    class_name = OVERLAY_CLASS_LABELS.get(overlay_id, f"Class {overlay_id}")
    return f"{class_name} ({label_id}->{overlay_id})"


def site_class_summary(site_kind: str) -> str:
    if site_kind == "low":
        return "Low site mapping: 0 = Forb, 1 = Shrub, 2 = Mallee"
    if site_kind == "medium":
        return "Medium site mapping: 0 = Hummock, 1 = Shrub, 2 = Mallee"
    if site_kind == "forest":
        return "Forest site mapping: 0 = Ground, 1 = Wood, 2 = Leaf"
    return "Semantic mapping unavailable"


def infer_site_kind(scene_name: str, experiment_name: str = "") -> str:
    scene_name_lower = scene_name.lower()
    experiment_name_lower = experiment_name.lower()

    if "medium" in scene_name_lower or "medium" in experiment_name_lower or "_m_" in scene_name_lower:
        return "medium"
    if "low" in scene_name_lower or "low" in experiment_name_lower or "_l_" in scene_name_lower:
        return "low"
    if "forest" in scene_name_lower or "forest" in experiment_name_lower:
        return "forest"
    return "unknown"


def site_heading(site_kind: str) -> str:
    if site_kind == "low":
        return "Qualitative results · Low site"
    if site_kind == "medium":
        return "Qualitative results · Medium site"
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
        "--normalized-height-root",
        type=Path,
        default=DEFAULT_NORMALIZED_HEIGHT_ROOT,
        help=(
            "Root directory containing normalized-height LAS/LAZ/PLY tiles named like "
            "'z_norm_csf_tiles_multispectral.<tile_id>.*'."
        ),
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


def scan_named_experiments(experiment_dirs: dict[str, Path]) -> dict[str, list[SceneEntry]]:
    experiments: dict[str, list[SceneEntry]] = {}

    for experiment_name, experiment_dir in experiment_dirs.items():
        if not experiment_dir.exists():
            continue

        scene_paths = sorted(
            path
            for path in experiment_dir.rglob("*")
            if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES
        )
        if not scene_paths:
            continue

        experiments[experiment_name] = [
            SceneEntry(
                experiment_name=experiment_name,
                scene_name=scene_path.stem,
                scene_path=str(scene_path),
            )
            for scene_path in scene_paths
        ]

    return experiments


def _normalized_height_priority(scene_path: Path) -> tuple[int, str]:
    path_lower = str(scene_path).lower()
    if "/z_norm/csf/" in path_lower or path_lower.endswith("/z_norm/csf"):
        return (0, path_lower)
    if "coreg/icp" in path_lower:
        return (1, path_lower)
    if "coreg" in path_lower:
        return (2, path_lower)
    if "original" in path_lower:
        return (4, path_lower)
    return (3, path_lower)


def extract_scene_tile_id(scene_name: str) -> str | None:
    matches = re.findall(r"(\d+)", scene_name)
    if not matches:
        return None
    return matches[-1]


def compact_experiment_label(experiment_name: str) -> str:
    return re.sub(r"^zero_shot_", "", experiment_name)


def compact_scene_label(scene_name: str) -> str:
    compact_name = re.sub(r"^(?:SCION_)+(?=plot_)", "", scene_name, flags=re.IGNORECASE)
    return compact_name


def scan_normalized_height_tiles(root: Path) -> dict[str, HeightSceneEntry]:
    if not root.exists():
        return {}

    tiles: dict[str, HeightSceneEntry] = {}
    candidates = sorted(
        path
        for path in root.rglob("*")
        if path.is_file()
        and path.suffix.lower() in SUPPORTED_SUFFIXES
        and path.stem.startswith(NORMALIZED_HEIGHT_PREFIXES)
    )
    for scene_path in candidates:
        tile_id = scene_path.stem.split(".")[-1]
        current = tiles.get(tile_id)
        if current is None or _normalized_height_priority(scene_path) < _normalized_height_priority(Path(current.scene_path)):
            tiles[tile_id] = HeightSceneEntry(tile_id=tile_id, scene_path=str(scene_path))
    return tiles


def print_experiment_summary(experiments: dict[str, list[SceneEntry]]) -> None:
    for experiment_name, scenes in experiments.items():
        print(f"{experiment_name}: {len(scenes)} scenes")
        for scene in scenes:
            print(f"  - {scene.scene_name}")


def _first_available_ply_field(vertex, candidates: tuple[str, ...]) -> np.ndarray | None:
    names = set(vertex.dtype.names or [])
    for candidate in candidates:
        if candidate in names:
            return np.asarray(vertex[candidate])
    lower_map = {name.lower(): name for name in names}
    for candidate in candidates:
        matched = lower_map.get(candidate.lower())
        if matched is not None:
            return np.asarray(vertex[matched])
    return None


def _first_available_las_field(las, candidates: tuple[str, ...]) -> np.ndarray | None:
    names = {name.lower(): name for name in las.point_format.dimension_names}
    for candidate in candidates:
        matched = names.get(candidate.lower())
        if matched is not None:
            return np.asarray(las[matched])
    return None


def _read_required_columns_from_ply(scene_path: Path) -> dict[str, np.ndarray]:
    ply = PlyData.read(scene_path)
    vertex = ply["vertex"].data
    names = set(vertex.dtype.names or [])

    required_xyz = {"x", "y", "z"}
    missing_xyz = sorted(required_xyz - names)
    if missing_xyz:
        raise ValueError(f"Missing required fields in {scene_path}: {missing_xyz}")

    x = np.asarray(vertex["x"], dtype=np.float32)
    y = np.asarray(vertex["y"], dtype=np.float32)
    z = np.asarray(vertex["z"], dtype=np.float32)
    total = len(x)

    semantic_pred = _first_available_ply_field(vertex, ("semantic_pred", "pred_semantic", "semantic"))
    semantic_gt = _first_available_ply_field(vertex, ("semantic_gt", "gt_semantic"))
    instance_pred = _first_available_ply_field(vertex, ("instance_pred", "PredInstance", "pred_instance", "predinstance"))
    instance_gt = _first_available_ply_field(vertex, ("instance_gt", "treeID", "tree_id", "TreeID", "gt_tree_id"))

    if instance_pred is None:
        raise ValueError(f"Missing prediction instance field in {scene_path}. Expected one of: instance_pred, PredInstance")
    if instance_gt is None:
        raise ValueError(f"Missing GT instance field in {scene_path}. Expected one of: instance_gt, treeID")

    if semantic_pred is None:
        semantic_pred = np.full(total, -1, dtype=np.int32)
    if semantic_gt is None:
        semantic_gt = np.full(total, -1, dtype=np.int32)

    return {
        "x": x,
        "y": y,
        "z": z,
        "semantic_pred": np.asarray(semantic_pred, dtype=np.int32),
        "instance_pred": np.asarray(instance_pred, dtype=np.int32),
        "semantic_gt": np.asarray(semantic_gt, dtype=np.int32),
        "instance_gt": np.asarray(instance_gt, dtype=np.int32),
    }


def _read_required_columns_from_las(scene_path: Path) -> dict[str, np.ndarray]:
    las = laspy.read(scene_path)

    x = np.asarray(las.x, dtype=np.float32)
    y = np.asarray(las.y, dtype=np.float32)
    z = np.asarray(las.z, dtype=np.float32)
    total = len(x)

    semantic_pred = _first_available_las_field(las, ("semantic_pred", "pred_semantic", "semantic"))
    semantic_gt = _first_available_las_field(las, ("semantic_gt", "gt_semantic"))
    instance_pred = _first_available_las_field(las, ("instance_pred", "PredInstance", "pred_instance", "predinstance"))
    instance_gt = _first_available_las_field(las, ("instance_gt", "treeID", "tree_id", "TreeID", "gt_tree_id"))

    if instance_pred is None:
        raise ValueError(f"Missing prediction instance LAS dimension in {scene_path}. Expected one of: instance_pred, PredInstance")
    if instance_gt is None:
        raise ValueError(f"Missing GT instance LAS dimension in {scene_path}. Expected one of: instance_gt, treeID")

    if semantic_pred is None:
        semantic_pred = np.full(total, -1, dtype=np.int32)
    if semantic_gt is None:
        semantic_gt = np.full(total, -1, dtype=np.int32)

    return {
        "x": x,
        "y": y,
        "z": z,
        "semantic_pred": np.asarray(semantic_pred, dtype=np.int32),
        "instance_pred": np.asarray(instance_pred, dtype=np.int32),
        "semantic_gt": np.asarray(semantic_gt, dtype=np.int32),
        "instance_gt": np.asarray(instance_gt, dtype=np.int32),
    }


def _read_xyz_from_ply(scene_path: Path) -> dict[str, np.ndarray]:
    ply = PlyData.read(scene_path)
    vertex = ply["vertex"].data
    names = set(vertex.dtype.names or [])

    required_xyz = {"x", "y", "z"}
    missing_xyz = sorted(required_xyz - names)
    if missing_xyz:
        raise ValueError(f"Missing required fields in {scene_path}: {missing_xyz}")

    return {
        "x": np.asarray(vertex["x"], dtype=np.float32),
        "y": np.asarray(vertex["y"], dtype=np.float32),
        "z": np.asarray(vertex["z"], dtype=np.float32),
    }


def _read_xyz_from_las(scene_path: Path) -> dict[str, np.ndarray]:
    las = laspy.read(scene_path)
    return {
        "x": np.asarray(las.x, dtype=np.float32),
        "y": np.asarray(las.y, dtype=np.float32),
        "z": np.asarray(las.z, dtype=np.float32),
    }


@lru_cache(maxsize=128)
def load_scene(scene_path_str: str) -> dict[str, np.ndarray]:
    scene_path = Path(scene_path_str)
    suffix = scene_path.suffix.lower()
    if suffix == ".ply":
        return _read_required_columns_from_ply(scene_path)
    if suffix in {".las", ".laz"}:
        return _read_required_columns_from_las(scene_path)
    raise ValueError(f"Unsupported scene suffix for {scene_path}")


@lru_cache(maxsize=128)
def load_xyz_scene(scene_path_str: str) -> dict[str, np.ndarray]:
    scene_path = Path(scene_path_str)
    suffix = scene_path.suffix.lower()
    if suffix == ".ply":
        return _read_xyz_from_ply(scene_path)
    if suffix in {".las", ".laz"}:
        return _read_xyz_from_las(scene_path)
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


def remap_instance_ids(instance_ids: np.ndarray, reference_ids: np.ndarray) -> tuple[np.ndarray, int]:
    unique_ids = np.sort(np.unique(reference_ids[reference_ids > 0]))
    if unique_ids.size == 0:
        return np.zeros_like(instance_ids, dtype=np.int32), 0

    mapped_ids = np.zeros_like(instance_ids, dtype=np.int32)
    mask = instance_ids > 0
    mapped_ids[mask] = np.searchsorted(unique_ids, instance_ids[mask]) + 1
    return mapped_ids, int(unique_ids.size)


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

    # Apply z-gradient for forest sites (ground + leaf), and for low/medium (forb/hummock + mallee)
    if semantic_labels is not None:
        # Ground-like class = 0 (Forb / Hummock / Ground)
        ground_mask = semantic_labels == 0
        # Canopy-like class = 2 (Mallee / Leaf)
        canopy_mask = semantic_labels == 2
        # Mid class = 1 (Shrub / Wood)
        mid_mask = semantic_labels == 1
        # Ignore = -1
        ignore_mask = ~(ground_mask | canopy_mask | mid_mask)

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

        # --- Leaf / Mallee: forest green gradient ---
        if np.any(canopy_mask):
            z_t = z[canopy_mask]
            z_norm = (z_t - z_t.min()) / (z_t.max() - z_t.min() + 1e-6)
            fig.add_trace(
                go.Scatter3d(
                    x=x[canopy_mask], y=y[canopy_mask], z=z_t,
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
        uirevision=None,  # always reset view on scene/experiment change
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
        t = 1 - (i / max(n_colors - 1, 1))
        hue = int(t * 300)
        discrete_colorscale.append([frac_start, f"hsl({hue}, 90%, 50%)"])
        discrete_colorscale.append([frac_end, f"hsl({hue}, 90%, 50%)"])

    middle_value = cmin + (cmax - cmin) // 2
    tick_vals = sorted({cmin, middle_value, cmax})

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
            "camera": {"eye": CAMERA_EYE},
            "bgcolor": "white",
        },
        uirevision=None,  # always reset view on scene/experiment change
    )
    return fig


def build_message_figure(title: str, message: str) -> go.Figure:
    fig = go.Figure()
    fig.update_layout(
        title={"text": title, "x": 0.02, "xanchor": "left"},
        margin={"l": 0, "r": 0, "t": PLOT_TOP_MARGIN, "b": 0},
        paper_bgcolor="white",
        plot_bgcolor="white",
        annotations=[
            {
                "xref": "paper",
                "yref": "paper",
                "x": 0.5,
                "y": 0.5,
                "text": message,
                "showarrow": False,
                "font": {"size": 15, "color": UI_COLORS["muted"]},
                "align": "center",
            }
        ],
        scene={
            "xaxis": {"visible": False},
            "yaxis": {"visible": False},
            "zaxis": {"visible": False},
            "aspectmode": "data",
            "camera": {"eye": CAMERA_EYE},
            "bgcolor": "white",
        },
    )
    return fig


def build_height_figure(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    title: str,
    marker_size: float,
) -> go.Figure:
    if z.size == 0:
        return build_message_figure(title, "No normalized-height points are available for this tile.")

    z_norm = (z - z.min()) / (z.max() - z.min() + 1e-6)
    fig = go.Figure(
        data=[
            go.Scatter3d(
                x=x,
                y=y,
                z=z,
                mode="markers",
                marker={
                    "size": marker_size,
                    "color": z_norm,
                    "colorscale": [
                        [0.0, "#5b3f2a"],
                        [0.2, "#8f6b44"],
                        [0.45, "#d2b46c"],
                        [0.7, "#8ebf6a"],
                        [1.0, "#1f5c5b"],
                    ],
                    "opacity": 0.95,
                    "showscale": True,
                    "colorbar": {
                        "title": "Norm. height",
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
                        "tickvals": [0.0, 0.5, 1.0],
                        "ticktext": [f"{z.min():.1f}", f"{(z.min() + z.max()) / 2:.1f}", f"{z.max():.1f}"],
                    },
                },
                hoverinfo="skip",
                showlegend=False,
            )
        ]
    )
    fig.update_layout(
        title={"text": title, "x": 0.02, "xanchor": "left", "y": 0.98, "yanchor": "top"},
        margin={"l": 0, "r": 0, "t": 88, "b": 0},
        paper_bgcolor="white",
        plot_bgcolor="white",
        showlegend=False,
        scene={
            "xaxis": {"visible": False},
            "yaxis": {"visible": False},
            "zaxis": {"visible": False},
            "aspectmode": "data",
            "camera": {"eye": CAMERA_EYE},
            "bgcolor": "white",
        },
        uirevision=None,
    )
    return fig


def tree_id_range_text(instance_ids: np.ndarray) -> str:
    visible_ids = instance_ids[instance_ids > 0]
    if visible_ids.size == 0:
        return "none"
    return f"{int(visible_ids.min())}→{int(visible_ids.max())}"


def tree_id_gap_count(instance_ids: np.ndarray) -> int:
    visible_ids = instance_ids[instance_ids > 0]
    if visible_ids.size == 0:
        return 0
    return max(0, int(visible_ids.max()) - int(visible_ids.min()) + 1 - np.unique(visible_ids).size)


def format_scene_summary(scene_arrays: dict[str, np.ndarray], sample_count: int, site_kind: str) -> html.Div:
    semantic_gt = scene_arrays["semantic_gt"]
    semantic_pred = scene_arrays["semantic_pred"]
    instance_gt = scene_arrays["instance_gt"]
    instance_pred = scene_arrays["instance_pred"]

    gt_instances = int(np.unique(instance_gt[instance_gt > 0]).size)
    pred_instances = int(np.unique(instance_pred[instance_pred > 0]).size)
    gt_tree_id_range = tree_id_range_text(instance_gt)
    pred_tree_id_range = tree_id_range_text(instance_pred)
    gt_tree_id_gaps = tree_id_gap_count(instance_gt)
    pred_tree_id_gaps = tree_id_gap_count(instance_pred)
    gt_semantics = sorted({semantic_label(int(label), site_kind) for label in np.unique(semantic_gt)})
    pred_semantics = sorted({semantic_label(int(label), site_kind) for label in np.unique(semantic_pred)})

    items = [
        html.Span(f"Sampled points: {sample_count:,}"),
        html.Span(f"GT unique instances: {gt_instances:,} | Original Tree ID range: {gt_tree_id_range} | gaps: {gt_tree_id_gaps}"),
        html.Span(f"Pred unique instances: {pred_instances:,} | Original Tree ID range: {pred_tree_id_range} | gaps: {pred_tree_id_gaps}"),
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
            dcc.Graph(
                id=graph_id,
                style={"height": "520px"},
                config={"displayModeBar": True},
                responsive=True,
            ),
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


@lru_cache(maxsize=24)
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
        gt_display_ids, gt_count = remap_instance_ids(instance_gt, arrays["instance_gt"])
        pred_display_ids, pred_count = remap_instance_ids(instance_pred, arrays["instance_pred"])

        gt_cmin = 1
        gt_cmax = gt_count if gt_count > 0 else 1
        pred_cmin = 1
        pred_cmax = pred_count if pred_count > 0 else 1

        gt_fig = build_instance_figure(
            x, y, z, gt_display_ids, gt_title, marker_size,
            cmin=gt_cmin, cmax=gt_cmax, legend_note=gt_legend_note,
            semantic_labels=semantic_gt,
        )
        pred_fig = build_instance_figure(
            x, y, z, pred_display_ids, pred_title, marker_size,
            cmin=pred_cmin, cmax=pred_cmax, legend_note=pred_legend_note,
            semantic_labels=semantic_pred,
        )

    summary = format_scene_summary(arrays, len(indices), site_kind)
    return gt_fig, pred_fig, summary


@lru_cache(maxsize=48)
def build_normalized_height_view(
    scene_path: str,
    scene_label: str,
    max_points: int,
    seed: int,
    marker_size: float,
) -> go.Figure:
    arrays = load_xyz_scene(scene_path)
    indices = make_sample_indices(scene_path, len(arrays["x"]), max_points=max_points, base_seed=seed)
    x = arrays["x"][indices]
    y = arrays["y"][indices]
    z = arrays["z"][indices]
    return build_height_figure(x, y, z, scene_label, marker_size)


def prepare_figure_for_export(
    fig: go.Figure,
    title_font_size: int = EXPORT_TITLE_FONT_SIZE,
    legend_font_size: int = EXPORT_LEGEND_FONT_SIZE,
    legend_marker_size_multiplier: float = EXPORT_LEGEND_MARKER_SIZE_MULTIPLIER,
    colorbar_thickness: int | None = None,
    colorbar_len: float | None = None,
    colorbar_xpad: int | None = None,
    show_legend: bool = True,
) -> go.Figure:
    export_fig = go.Figure(fig)
    existing_margin = export_fig.layout.margin.to_plotly_json() if export_fig.layout.margin else {}
    export_top_margin = max(int(existing_margin.get("t", 0)), title_font_size + 40)
    export_fig.update_layout(
        title={"font": {"size": title_font_size}},
        legend={
            "font": {"size": legend_font_size, "color": UI_COLORS["ink"]},
            "itemsizing": "trace",
        },
        showlegend=show_legend,
        margin={**existing_margin, "t": export_top_margin},
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
                "size": legend_font_size,
            }

    def is_legend_only_trace(trace: go.BaseTraceType) -> bool:
        return bool(getattr(trace, "showlegend", False)) and all(
            values is not None and all(value is None for value in values)
            for values in (getattr(trace, "x", None), getattr(trace, "y", None), getattr(trace, "z", None))
        )

    if not show_legend:
        export_fig.data = tuple(trace for trace in export_fig.data if not is_legend_only_trace(trace))

    for trace in export_fig.data:
        marker = getattr(trace, "marker", None)
        if marker is None:
            continue
        current_size = getattr(marker, "size", None)
        if isinstance(current_size, (int, float)):
            size_multiplier = (
                legend_marker_size_multiplier
                if is_legend_only_trace(trace)
                else EXPORT_MARKER_SIZE_MULTIPLIER
            )
            trace.update(marker={"size": current_size * size_multiplier})
        colorbar = getattr(marker, "colorbar", None)
        if colorbar is not None:
            showscale = bool(getattr(marker, "showscale", False))
            existing_thickness = getattr(colorbar, "thickness", None)
            trace.update(
                marker={
                    "showscale": showscale,
                    "colorbar": {
                        "title": {"font": {"size": legend_font_size}},
                        "tickfont": {"size": legend_font_size},
                        "thickness": colorbar_thickness if colorbar_thickness is not None else existing_thickness,
                        "len": colorbar_len if colorbar_len is not None else getattr(colorbar, "len", None),
                        "xpad": colorbar_xpad if colorbar_xpad is not None else getattr(colorbar, "xpad", None),
                    },
                }
            )

    return export_fig


def get_export_font(size: int, bold: bool = False) -> ImageFont.ImageFont | ImageFont.FreeTypeFont:
    font_candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    ]
    for font_path in font_candidates:
        try:
            return ImageFont.truetype(font_path, size)
        except OSError:
            continue
    return ImageFont.load_default()


def get_export_header_font() -> ImageFont.ImageFont | ImageFont.FreeTypeFont:
    return get_export_font(EXPORT_COMPARISON_HEADER_FONT_SIZE, bold=True)


def get_export_panel_title_font() -> ImageFont.ImageFont | ImageFont.FreeTypeFont:
    return get_export_font(EXPORT_COMPARISON_PANEL_TITLE_FONT_SIZE, bold=True)


def normalize_export_figure_coordinates(fig: go.Figure) -> go.Figure:
    normalized_fig = go.Figure(fig)
    axes: list[np.ndarray] = []
    for trace in normalized_fig.data:
        for axis_name in ("x", "y", "z"):
            axis_values = getattr(trace, axis_name, None)
            if axis_values is None:
                continue
            axis_array = np.asarray(axis_values, dtype=float)
            finite_values = axis_array[np.isfinite(axis_array)]
            if finite_values.size:
                axes.append(finite_values)

    if not axes:
        return normalized_fig

    global_min = min(float(axis.min()) for axis in axes)
    global_max = max(float(axis.max()) for axis in axes)
    global_span = max(global_max - global_min, 1.0)

    for trace in normalized_fig.data:
        updates: dict[str, np.ndarray] = {}
        for axis_name in ("x", "y", "z"):
            axis_values = getattr(trace, axis_name, None)
            if axis_values is None:
                continue
            axis_array = np.asarray(axis_values, dtype=float)
            normalized_axis = np.where(
                np.isfinite(axis_array),
                (axis_array - global_min) / global_span,
                axis_array,
            )
            updates[axis_name] = normalized_axis
        if updates:
            trace.update(**updates)

    normalized_fig.update_layout(
        scene={
            "aspectmode": "cube",
            "xaxis": {"range": [0.0, 1.0], "visible": False},
            "yaxis": {"range": [0.0, 1.0], "visible": False},
            "zaxis": {"range": [0.0, 1.0], "visible": False},
        }
    )
    return normalized_fig


def render_export_figure_image(
    fig: go.Figure,
    *,
    width: int,
    height: int,
    scale: int,
) -> bytes:
    try:
        return fig.to_image(format="png", width=width, height=height, scale=scale)
    except ValueError as exc:
        if "axis scaling" not in str(exc):
            raise

    # Older Kaleido releases sometimes fail on 3D scene axis scaling; retry with simpler aspect modes.
    fallback_scene_updates = (
        {"scene": {"aspectmode": "cube"}},
        {"scene": {"aspectmode": "manual", "aspectratio": {"x": 1, "y": 1, "z": 0.6}}},
    )
    last_error: ValueError | None = None
    for layout_update in fallback_scene_updates:
        retry_fig = go.Figure(fig)
        retry_fig.update_layout(**layout_update)
        try:
            return retry_fig.to_image(format="png", width=width, height=height, scale=scale)
        except ValueError as exc:
            last_error = exc
            if "axis scaling" not in str(exc):
                raise

    if last_error is not None:
        normalized_fig = normalize_export_figure_coordinates(fig)
        return normalized_fig.to_image(format="png", width=width, height=height, scale=scale)
    raise RuntimeError("Unreachable export fallback state")


def extract_legend_entries(fig: go.Figure) -> list[tuple[str, str]]:
    legend_entries: list[tuple[str, str]] = []
    for trace in fig.data:
        if not bool(getattr(trace, "showlegend", False)):
            continue
        name = getattr(trace, "name", None)
        marker = getattr(trace, "marker", None)
        color = getattr(marker, "color", None) if marker is not None else None
        if not name or not isinstance(color, str):
            continue
        legend_entries.append((name, color))
    return legend_entries


def draw_custom_legend(
    image: Image.Image,
    legend_entries: list[tuple[str, str]],
) -> None:
    if not legend_entries:
        return

    draw = ImageDraw.Draw(image)
    font = get_export_font(EXPORT_COMPARISON_LEGEND_FONT_SIZE)
    dot_radius = EXPORT_COMPARISON_CUSTOM_LEGEND_DOT_RADIUS
    row_gap = 18
    text_gap = 18
    box_padding = 18

    text_sizes = [draw.textbbox((0, 0), label, font=font) for label, _ in legend_entries]
    text_width = max((bbox[2] - bbox[0]) for bbox in text_sizes)
    text_height = max((bbox[3] - bbox[1]) for bbox in text_sizes)
    row_height = max(dot_radius * 2, text_height)
    box_width = box_padding * 2 + dot_radius * 2 + text_gap + text_width
    box_height = box_padding * 2 + len(legend_entries) * row_height + max(0, len(legend_entries) - 1) * row_gap
    left = image.width - box_width - 48
    top = 48
    right = left + box_width
    bottom = top + box_height

    draw.rectangle(
        [(left, top), (right, bottom)],
        fill=(255, 255, 255),
        outline=UI_COLORS["border"],
        width=2,
    )

    for index, (label, color) in enumerate(legend_entries):
        row_top = top + box_padding + index * (row_height + row_gap)
        dot_center_x = left + box_padding + dot_radius
        dot_center_y = row_top + row_height // 2
        draw.ellipse(
            [
                (dot_center_x - dot_radius, dot_center_y - dot_radius),
                (dot_center_x + dot_radius, dot_center_y + dot_radius),
            ],
            fill=color,
            outline=color,
        )
        text_x = left + box_padding + dot_radius * 2 + text_gap
        text_y = row_top + (row_height - text_height) // 2 - 2
        draw.text((text_x, text_y), label, fill=UI_COLORS["ink"], font=font)


def extract_instance_colorbar_labels(fig: go.Figure) -> tuple[int, int, int] | None:
    for trace in fig.data:
        marker = getattr(trace, "marker", None)
        if marker is None or not bool(getattr(marker, "showscale", False)):
            continue
        cmin = getattr(marker, "cmin", None)
        cmax = getattr(marker, "cmax", None)
        if not isinstance(cmin, (int, float)) or not isinstance(cmax, (int, float)):
            continue
        min_value = int(round(cmin + 0.5))
        max_value = int(round(cmax - 0.5))
        middle_value = min_value + (max_value - min_value) // 2
        return min_value, middle_value, max_value
    return None


def hide_export_colorbar(fig: go.Figure) -> None:
    for trace in fig.data:
        marker = getattr(trace, "marker", None)
        if marker is None or not bool(getattr(marker, "showscale", False)):
            continue
        trace.update(marker={"showscale": False})


def draw_custom_instance_colorbar(
    image: Image.Image,
    labels: tuple[int, int, int] | None,
) -> None:
    if labels is None:
        return

    draw = ImageDraw.Draw(image)
    title_font = get_export_font(EXPORT_INSTANCE_COMPARISON_COLORBAR_TITLE_FONT_SIZE, bold=False)
    tick_font = get_export_font(EXPORT_INSTANCE_COMPARISON_COLORBAR_TICK_FONT_SIZE, bold=False)
    min_value, middle_value, max_value = labels

    title = f"Tree ID ({min_value}\u2192{max_value})"
    title_bbox = draw.textbbox((0, 0), title, font=title_font)
    title_width = title_bbox[2] - title_bbox[0]
    title_height = title_bbox[3] - title_bbox[1]

    bar_width = 400
    bar_height = 46
    tick_gap = 22
    content_width = title_width + 48 + bar_width
    start_x = max(56, (image.width - content_width) // 2)
    title_y = 88
    bar_x = start_x + title_width + 48
    bar_y = title_y + max(0, (title_height - bar_height) // 2) + 8

    draw.text((start_x, title_y), title, fill=UI_COLORS["ink"], font=title_font)

    for offset in range(bar_width):
        t = offset / max(bar_width - 1, 1)
        hue = int((1.0 - t) * 300)
        draw.line(
            [(bar_x + offset, bar_y), (bar_x + offset, bar_y + bar_height)],
            fill=f"hsl({hue}, 90%, 50%)",
            width=1,
        )

    draw.rectangle(
        [(bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height)],
        outline=UI_COLORS["border"],
        width=1,
    )

    tick_positions = [bar_x, bar_x + bar_width // 2, bar_x + bar_width]
    tick_texts = [str(min_value), str(middle_value), str(max_value)]
    tick_y = bar_y + bar_height + tick_gap

    for index, (position, text) in enumerate(zip(tick_positions, tick_texts)):
        tick_bbox = draw.textbbox((0, 0), text, font=tick_font)
        tick_width = tick_bbox[2] - tick_bbox[0]
        if index == 0:
            text_x = position
        elif index == len(tick_positions) - 1:
            text_x = position - tick_width
        else:
            text_x = position - tick_width // 2
        draw.text((text_x, tick_y), text, fill=UI_COLORS["ink"], font=tick_font)


def make_png_download(fig: go.Figure, filename: str, title: str | None = None):
    def writer(buffer) -> None:
        export_fig = prepare_figure_for_export(fig, title_font_size=EXPORT_DOWNLOAD_TITLE_FONT_SIZE)
        if title is not None:
            export_fig.update_layout(title={"text": title, "font": {"size": EXPORT_DOWNLOAD_TITLE_FONT_SIZE}})
        buffer.write(render_export_figure_image(export_fig, width=1800, height=1200, scale=2))

    return dcc.send_bytes(writer, filename)


def make_comparison_png_download(
    gt_fig: go.Figure,
    pred_fig: go.Figure,
    filename: str,
    title: str | None = None,
    gt_title: str | None = None,
    pred_title: str | None = None,
):
    def writer(buffer) -> None:
        gt_legend_entries = extract_legend_entries(gt_fig)
        pred_legend_entries = extract_legend_entries(pred_fig)
        has_semantic_legend = bool(gt_legend_entries or pred_legend_entries)
        gt_instance_colorbar = None if has_semantic_legend else extract_instance_colorbar_labels(gt_fig)
        pred_instance_colorbar = None if has_semantic_legend else extract_instance_colorbar_labels(pred_fig)
        comparison_legend_font_size = (
            EXPORT_COMPARISON_LEGEND_FONT_SIZE if has_semantic_legend else EXPORT_INSTANCE_COMPARISON_LEGEND_FONT_SIZE
        )
        comparison_colorbar_thickness = (
            EXPORT_COMPARISON_COLORBAR_THICKNESS if has_semantic_legend else EXPORT_INSTANCE_COMPARISON_COLORBAR_THICKNESS
        )
        comparison_colorbar_len = None if has_semantic_legend else 0.5
        comparison_colorbar_xpad = None if has_semantic_legend else 28
        export_gt_fig = prepare_figure_for_export(
            gt_fig,
            title_font_size=EXPORT_TITLE_FONT_SIZE,
            legend_font_size=comparison_legend_font_size,
            legend_marker_size_multiplier=EXPORT_LEGEND_MARKER_SIZE_MULTIPLIER * 2,
            colorbar_thickness=comparison_colorbar_thickness,
            colorbar_len=comparison_colorbar_len,
            colorbar_xpad=comparison_colorbar_xpad,
            show_legend=not gt_legend_entries,
        )
        export_pred_fig = prepare_figure_for_export(
            pred_fig,
            title_font_size=EXPORT_TITLE_FONT_SIZE,
            legend_font_size=comparison_legend_font_size,
            legend_marker_size_multiplier=EXPORT_LEGEND_MARKER_SIZE_MULTIPLIER * 2,
            colorbar_thickness=comparison_colorbar_thickness,
            colorbar_len=comparison_colorbar_len,
            colorbar_xpad=comparison_colorbar_xpad,
            show_legend=not pred_legend_entries,
        )
        export_gt_fig.update_layout(title={"text": ""})
        export_pred_fig.update_layout(title={"text": ""})
        if not has_semantic_legend:
            hide_export_colorbar(export_gt_fig)
            hide_export_colorbar(export_pred_fig)
        gt_image = Image.open(io.BytesIO(render_export_figure_image(export_gt_fig, width=1600, height=1100, scale=2))).convert("RGB")
        pred_image = Image.open(io.BytesIO(render_export_figure_image(export_pred_fig, width=1600, height=1100, scale=2))).convert("RGB")

        draw_custom_legend(gt_image, gt_legend_entries)
        draw_custom_legend(pred_image, pred_legend_entries)
        if not has_semantic_legend:
            draw_custom_instance_colorbar(gt_image, gt_instance_colorbar)
            draw_custom_instance_colorbar(pred_image, pred_instance_colorbar)

        padding = 32
        header_height = 72 if title else 0
        panel_title_font = get_export_panel_title_font()
        panel_title_height = 0
        if gt_title or pred_title:
            panel_title_bbox = ImageDraw.Draw(Image.new("RGB", (1, 1), "white")).textbbox((0, 0), gt_title or pred_title or "", font=panel_title_font)
            panel_title_height = (panel_title_bbox[3] - panel_title_bbox[1]) + 24
        canvas_width = gt_image.width + pred_image.width + padding * 3
        canvas_height = max(gt_image.height, pred_image.height) + padding * 2 + header_height + panel_title_height

        canvas = Image.new("RGB", (canvas_width, canvas_height), "white")
        draw = ImageDraw.Draw(canvas)
        if title:
            draw.text((padding, 20), title, fill=UI_COLORS["ink"], font=get_export_header_font())

        panel_title_top = header_height + padding // 2
        top = panel_title_top + panel_title_height
        if gt_title:
            draw.text((padding, panel_title_top), gt_title, fill=UI_COLORS["ink"], font=panel_title_font)
        if pred_title:
            pred_title_x = padding * 2 + gt_image.width
            draw.text((pred_title_x, panel_title_top), pred_title, fill=UI_COLORS["ink"], font=panel_title_font)
        canvas.paste(gt_image, (padding, top))
        canvas.paste(pred_image, (padding * 2 + gt_image.width, top))
        canvas.save(buffer, format="PNG")

    return dcc.send_bytes(writer, filename)


def create_app(
    experiment_families: dict[str, dict[str, list[SceneEntry]]],
    max_points: int,
    seed: int,
    marker_size: float,
    normalized_height_root: Path | None = DEFAULT_NORMALIZED_HEIGHT_ROOT,
) -> Dash:
    app = Dash(__name__)
    normalized_height_tiles = (
        scan_normalized_height_tiles(normalized_height_root)
        if normalized_height_root is not None
        else {}
    )

    def preferred_default_scene_for_family(family_name: str) -> str:
        return PREFERRED_DEFAULT_SCENES.get(family_name, DEFAULT_PREFERRED_SCENE)

    def preferred_experiment_name(family_name: str) -> str:
        experiments = experiment_families[family_name]
        preferred_scene = preferred_default_scene_for_family(family_name)
        for experiment_name, scenes in experiments.items():
            if any(scene.scene_name == preferred_scene for scene in scenes):
                return experiment_name
        return next(iter(experiments))

    def preferred_scene_name(family_name: str, experiment_name: str) -> str:
        preferred_scene = preferred_default_scene_for_family(family_name)
        for scene in experiment_families[family_name][experiment_name]:
            if scene.scene_name == preferred_scene:
                return scene.scene_name
        return experiment_families[family_name][experiment_name][0].scene_name

    family_names = list(experiment_families)
    default_family = family_names[0]
    default_experiment = preferred_experiment_name(default_family)
    default_scene = preferred_scene_name(default_family, default_experiment)

    scene_lookup = {
        family_name: {
            experiment_name: {scene.scene_name: scene.scene_path for scene in scenes}
            for experiment_name, scenes in experiments.items()
        }
        for family_name, experiments in experiment_families.items()
    }

    app.layout = html.Div(
        [
            dcc.Tabs(
                id="family-tabs",
                value=default_family,
                children=[
                    dcc.Tab(
                        label=family_name,
                        value=family_name,
                        style={
                            "backgroundColor": UI_COLORS["chip_bg"],
                            "border": f"1px solid {UI_COLORS['border']}",
                            "color": UI_COLORS["muted"],
                        },
                        selected_style={
                            "backgroundColor": UI_COLORS["ink"],
                            "border": f"1px solid {UI_COLORS['ink']}",
                            "color": "white",
                            "fontWeight": 700,
                        },
                    )
                    for family_name in family_names
                ],
                style={"maxWidth": "620px", "margin": "0 auto 16px auto"},
            ),
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
            html.Div(
                [
                    make_panel("Normalized Height", "height-graph"),
                    html.Div(
                        id="height-panel-note",
                        style={
                            "textAlign": "center",
                            "color": UI_COLORS["muted"],
                            "fontSize": "13px",
                            "marginTop": "10px",
                        },
                    ),
                ],
                style={
                    "marginTop": "16px",
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
        Output("experiment-dropdown", "options"),
        Output("experiment-dropdown", "value"),
        Input("family-tabs", "value"),
    )
    def update_experiment_options(family_name: str) -> tuple[list[dict[str, str]], str]:
        experiment_names = list(experiment_families[family_name])
        options = [{"label": name, "value": name} for name in experiment_names]
        return options, preferred_experiment_name(family_name)

    @app.callback(
        Output("scene-dropdown", "options"),
        Output("scene-dropdown", "value"),
        Input("family-tabs", "value"),
        Input("experiment-dropdown", "value"),
    )
    def update_scene_options(family_name: str, experiment_name: str) -> tuple[list[dict[str, str]], str]:
        scenes = experiment_families[family_name][experiment_name]
        options = [{"label": scene.scene_name, "value": scene.scene_name} for scene in scenes]
        return options, preferred_scene_name(family_name, experiment_name)

    @app.callback(
        Output("site-heading", "children"),
        Output("gt-graph", "figure"),
        Output("pred-graph", "figure"),
        Output("height-graph", "figure"),
        Output("scene-summary", "children"),
        Output("mode-legend", "children"),
        Output("site-class-banner", "children"),
        Output("height-panel-note", "children"),
        Input("family-tabs", "value"),
        Input("experiment-dropdown", "value"),
        Input("scene-dropdown", "value"),
        Input("mode-tabs", "value"),
    )
    def update_graphs(
        family_name: str,
        experiment_name: str,
        scene_name: str,
        mode: str,
    ) -> tuple[str, go.Figure, go.Figure, go.Figure, html.Div, html.Div, html.Div, str]:
        scene_path = scene_lookup[family_name][experiment_name][scene_name]
        site_kind = infer_site_kind(scene_name, experiment_name)
        gt_fig, pred_fig, summary = build_scene_views(
            scene_path, scene_name, experiment_name, mode, max_points, seed, marker_size
        )
        tile_id = extract_scene_tile_id(scene_name)
        height_note = "Normalized-height tile unavailable for this scene."
        height_fig = build_message_figure("Normalized height", height_note)
        if tile_id is not None:
            height_entry = normalized_height_tiles.get(tile_id)
            if height_entry is not None:
                height_note = f"Source: {height_entry.scene_path}"
                height_fig = build_normalized_height_view(
                    height_entry.scene_path,
                    f"Normalized height [tile {tile_id}] ({scene_name})",
                    max_points,
                    seed,
                    marker_size,
                )
        return (
            site_heading(site_kind),
            gt_fig,
            pred_fig,
            height_fig,
            summary,
            make_mode_legend(mode, scene_name, experiment_name),
            make_site_banner(scene_name, experiment_name),
            height_note,
        )

    @app.callback(
        Output("download-gt", "data"),
        Output("download-pred", "data"),
        Output("download-comparison", "data"),
        Input("download-gt-button", "n_clicks"),
        Input("download-pred-button", "n_clicks"),
        Input("download-comparison-button", "n_clicks"),
        State("family-tabs", "value"),
        State("experiment-dropdown", "value"),
        State("scene-dropdown", "value"),
        State("mode-tabs", "value"),
        prevent_initial_call=True,
    )
    def download_snapshot(
        gt_clicks: int,
        pred_clicks: int,
        comparison_clicks: int,
        family_name: str,
        experiment_name: str,
        scene_name: str,
        mode: str,
    ):
        del gt_clicks, pred_clicks, comparison_clicks
        scene_path = scene_lookup[family_name][experiment_name][scene_name]
        gt_fig, pred_fig, _ = build_scene_views(
            scene_path, scene_name, experiment_name, mode, max_points, seed, marker_size
        )
        mode_slug = "semantic" if mode == "semantic" else "instance"
        gt_export_title = f"GT {mode}"
        pred_export_title = f"Prediction {mode}"

        if ctx.triggered_id == "download-gt-button":
            filename = f"{experiment_name}__{scene_name}__gt_{mode_slug}.png"
            return make_png_download(gt_fig, filename, title=gt_export_title), no_update, no_update

        if ctx.triggered_id == "download-pred-button":
            filename = f"{experiment_name}__{scene_name}__prediction_{mode_slug}.png"
            return no_update, make_png_download(pred_fig, filename, title=pred_export_title), no_update

        if ctx.triggered_id == "download-comparison-button":
            filename = f"{experiment_name}__{scene_name}__comparison_{mode_slug}.png"
            return (
                no_update,
                no_update,
                make_comparison_png_download(
                    gt_fig,
                    pred_fig,
                    filename,
                    None,
                    gt_title=gt_export_title,
                    pred_title=pred_export_title,
                ),
            )

        return no_update, no_update, no_update

    return app


def main() -> None:
    args = parse_args()

    segment_any_tree_experiments = scan_named_experiments(DEFAULT_SEGMENT_ANY_TREE_EXPERIMENTS)
    forestformer_experiments = scan_experiments(args.work_dirs_root)

    experiment_families: dict[str, dict[str, list[SceneEntry]]] = {}
    if segment_any_tree_experiments:
        experiment_families["SegmentAnyTree"] = segment_any_tree_experiments
    if forestformer_experiments:
        experiment_families["ForestFormer3D"] = forestformer_experiments

    if not experiment_families:
        raise RuntimeError("No experiments were found.")

    if args.list_only:
        for family_name, experiments in experiment_families.items():
            print(f"\n[{family_name}]")
            print_experiment_summary(experiments)
        return

    app = create_app(
        experiment_families=experiment_families,
        max_points=args.max_points,
        seed=args.seed,
        marker_size=args.marker_size,
        normalized_height_root=args.normalized_height_root,
    )
    app.run(host=args.host, port=args.port, debug=False)


if __name__ == "__main__":
    main()
