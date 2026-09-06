from pathlib import Path
import re
import sys

import laspy
from matplotlib.colors import BoundaryNorm, ListedColormap, Normalize
from matplotlib.ticker import ScalarFormatter
import matplotlib.pyplot as plt
import numpy as np
import rasterio


def find_repo_root(start: Path) -> Path:
    script_relpath = Path('phase_4_3D_lidar_prep/step_3_structure_class/make_label_transfer_validation_figure.py')
    for candidate in [start, *start.parents]:
        if (candidate / script_relpath).exists():
            return candidate
    raise FileNotFoundError(f'Could not locate repository root from {start}')


repo_root = find_repo_root(Path.cwd().resolve())
script_dir = repo_root / 'phase_4_3D_lidar_prep' / 'step_3_structure_class'
if str(script_dir) not in sys.path:
    sys.path.insert(0, str(script_dir))

from make_label_transfer_validation_figure import (
    filter_points_by_class,
    raster_extent_from_transform,
    read_las_points,
    read_mask,
    remap_to_index,
    sample_points,
)

OVERLAY_CLASS_COLORS = {
    64: (0.851, 0.706, 0.290),  # Forb / #d9b44a
    65: (1.0, 0.0, 1.0),  # Woody debris / #FF00FF (magenta)
    66: (0.851, 0.706, 0.290),  # Hummock / #d9b44a
    67: (0.412, 0.220, 0.082),  # Shrub / #694714
    68: (0.180, 0.545, 0.341),  # Mallee / #2e8b57
}

OVERLAY_CLASS_LABELS = {
    64: 'Forb',
    65: 'Woody\ndebris',
    66: 'Hummock',
    67: 'Shrub',
    68: 'Mallee',
}

HEIGHT_RANGES = {
    64: (0.00, 0.40),
    65: (0.05, 1.20),
    66: (0.40, 3.00),
    67: (0.00, 0.80),
    68: (1.50, 8.00),
}

VALIDATION_CLASS_CODES = (64, 65, 66, 67, 68)


def wsl_path(path: str) -> Path:
    match = re.match(r'^([A-Za-z]):\\(.*)$', path)
    if match:
        drive, rest = match.groups()
        rest_path = rest.replace('\\', '/')
        mounted_path = Path('/mnt') / drive.lower() / rest_path
        mirror_path = Path('/tmp') / f'wsl_drive_{drive.lower()}' / rest_path
        return mirror_path if mirror_path.parent.exists() else mounted_path
    return Path(path)

# mask_path = Path('/media/laura/8402326D023263F8/calperumResearch/site2_1_DD0011/inputs/masks/tiles_3072/raw/structure_class/annotation_raster/downsample_5cm_nearest/mask_structure_3072.33.tif')
# projected_las_path = Path('/media/laura/8402326D023263F8/calperumResearch/site2_1_DD0011/lidar/clipped/mask/coreg/icp/mask_tiles_multispectral.33.las')
# refined_las_root = Path('/media/laura/8402326D023263F8/calperumResearch/site2_1_DD0011/lidar/clipped/3d_structure_mask_malleid')
# multispectral_dir = Path('/media/laura/8402326D023263F8/calperumResearch/site2_1_DD0011/inputs/predictors/tiles_3072/stacked')

mask_path = wsl_path('I:\\calperumResearch\\site2_1_DD0011\\inputs\\masks\\tiles_3072\\raw\\structure_class\\annotation_raster\\downsample_5cm_nearest\\mask_structure_3072.33.tif')
projected_las_path = wsl_path('I:\\calperumResearch\\site2_1_DD0011\\lidar\\clipped\\mask\\coreg\\icp\\mask_tiles_multispectral.33.las')
refined_las_root = wsl_path('I:\\calperumResearch\\site2_1_DD0011\\lidar\\clipped\\3d_structure_mask_malleid')
multispectral_dir = wsl_path('I:\\calperumResearch\\site2_1_DD0011\\inputs\\predictors\\tiles_3072\\stacked')

results_output_path = wsl_path('I:\\calperumResearch\\site2_1_DD0011\\lidar\\clipped\\validation\\label_transfer_validation_results_tile33.png')
gallery_output_path = wsl_path('I:\\calperumResearch\\site2_1_DD0011\\lidar\\clipped\\validation\\label_transfer_validation_gallery_tile33.png')
height_output_path = wsl_path('I:\\calperumResearch\\site2_1_DD0011\\lidar\\clipped\\validation\\label_transfer_validation_height_qc_tile33.png')


def find_site_root(path: Path) -> Path:
    for candidate in [path, *path.parents]:
        if candidate.name.startswith('site'):
            return candidate
    raise FileNotFoundError(f'Could not infer site root from {path}')


def detect_tile_id(*paths: Path) -> str:
    for path in paths:
        match = re.search(r'(\d+)$', path.stem)
        if match:
            return match.group(1)
    raise ValueError('Could not detect tile id from the configured paths')


def find_multispectral_raster(site_root: Path, tile_id: str) -> Path | None:
    candidates = [
        multispectral_dir / f'tiles_multispectral_{tile_id}.tif',
        multispectral_dir / f'tiles_multispectral.{tile_id}.tif',
        site_root / 'imagery' / 'inputs' / 'predictors' / 'tiles_3072' / 'stacked' / f'tiles_multispectral.{tile_id}.tif',
        site_root / 'imagery' / 'clipped' / 'raster' / f'tiles_multispectral.{tile_id}.tif',
        site_root / 'imagery' / 'clipped' / 'raster' / 'new' / f'tiles_multispectral.{tile_id}.tif',
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate

    imagery_root = site_root / 'imagery'
    if imagery_root.exists():
        preferred = []
        fallback = []
        for path in imagery_root.rglob(f'*{tile_id}*.tif'):
            path_str = str(path).lower()
            if 'multispectral' in path_str and 'mask' not in path_str:
                preferred.append(path)
            elif 'mask' not in path_str:
                fallback.append(path)
        matches = sorted(preferred) or sorted(fallback)
        if matches:
            return matches[0]

    if multispectral_dir.exists():
        preferred = sorted(multispectral_dir.glob(f'*{tile_id}*.tif'))
        if preferred:
            return preferred[0]

    return None


def find_refined_las(refined_root: Path, tile_id: str) -> Path:
    candidates = [
        refined_root / f'znorm_3Dlabels_{tile_id}.las',
        refined_root / f'znorm_3Dlabels_l_{tile_id}.las',
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate

    matches = sorted(refined_root.glob(f'*{tile_id}*.las'))
    if matches:
        return matches[0]

    raise FileNotFoundError(f'Could not find refined LAS for tile {tile_id} under {refined_root}')


def normalize_band(band: np.ndarray) -> np.ndarray:
    finite = band[np.isfinite(band)]
    if finite.size == 0:
        return np.zeros_like(band, dtype=np.float32)
    lo, hi = np.percentile(finite, [2, 98])
    scaled = np.clip((band - lo) / (hi - lo + 1e-6), 0, 1)
    scaled[~np.isfinite(scaled)] = 0
    return scaled.astype(np.float32)


def load_raster_rgb(raster_path: Path) -> tuple[np.ndarray, tuple[float, float, float, float], tuple[int, ...]]:
    with rasterio.open(raster_path) as src:
        if src.count >= 5:
            bands = (5, 3, 1)
            rgb = np.stack([normalize_band(src.read(b)) for b in bands], axis=-1)
        elif src.count >= 3:
            bands = (1, 2, 3)
            rgb = np.stack([normalize_band(src.read(b)) for b in bands], axis=-1)
        else:
            bands = (1,)
            gray = normalize_band(src.read(1))
            rgb = np.stack([gray, gray, gray], axis=-1)
        extent = (src.bounds.left, src.bounds.right, src.bounds.bottom, src.bounds.top)
    return rgb, extent, bands


tile_id = detect_tile_id(mask_path, projected_las_path)
site_root = find_site_root(mask_path)
refined_las_path = find_refined_las(refined_las_root, tile_id)
multispectral_raster_path = find_multispectral_raster(site_root, tile_id)
overlay_raster_path = multispectral_raster_path if multispectral_raster_path is not None else mask_path
rgb_overlay, overlay_extent, overlay_bands = load_raster_rgb(overlay_raster_path)

print(f'Tile id: {tile_id}')
print(f'Site root: {site_root}')
print(f'Configured multispectral directory: {multispectral_dir}')
print(f'Refined LAS: {refined_las_path}')
if multispectral_raster_path is None:
    print('No multispectral TIFF was found automatically; using the mask raster as the overlay background.')
else:
    print(f'Multispectral raster: {multispectral_raster_path}')
print(f'Overlay raster: {overlay_raster_path}')
print(f'Bands used for display: {overlay_bands}')
print(f'Results figure output: {results_output_path}')
print(f'Gallery output: {gallery_output_path}')
print(f'Height QC output: {height_output_path}')

# ----------------------------
# style constants
# ----------------------------
EXCLUDED_HEIGHT_CODES = {64}

MAIN_TITLE_FONTSIZE = 18
ROW_TAG_FONTSIZE = 13
PANEL_TITLE_FONTSIZE = 13
AXIS_LABEL_FONTSIZE = 11
AXIS_TICK_FONTSIZE = 8
INFOBOX_FONTSIZE = 36
HEIGHT_GUIDE_FONTSIZE = 36
THREED_BOX_ZOOM = 1.12
COLORBAR_LABEL_FONTSIZE = 10
COLORBAR_TICK_FONTSIZE = 8
DISPLAY_Z_MODE = 'z_norm_preferred'


# ----------------------------
# validation inputs
# ----------------------------
validation_mask, validation_transform = read_mask(str(mask_path))
validation_mask = np.where(np.isin(validation_mask, VALIDATION_CLASS_CODES), validation_mask, 0)
validation_mask_codes = [0, *VALIDATION_CLASS_CODES]
validation_mask_idx = remap_to_index(validation_mask, validation_mask_codes)
validation_extent = raster_extent_from_transform(validation_mask.shape, validation_transform)


def build_validation_mask_style(class_codes: tuple[int, ...]) -> tuple[ListedColormap, BoundaryNorm]:
    colors = [(0.0, 0.0, 0.0, 0.0)]
    colors.extend([(*OVERLAY_CLASS_COLORS[code], 1.0) for code in class_codes])
    cmap = ListedColormap(colors)
    bounds = np.arange(-0.5, len(colors) + 0.5, 1.0)
    norm = BoundaryNorm(bounds, cmap.N)
    return cmap, norm


def build_class_colorbar_style(class_codes: tuple[int, ...]) -> tuple[ListedColormap, BoundaryNorm]:
    colors = [OVERLAY_CLASS_COLORS[code] for code in class_codes]
    cmap = ListedColormap(colors)
    bounds = np.arange(-0.5, len(class_codes) + 0.5, 1.0)
    norm = BoundaryNorm(bounds, cmap.N)
    return cmap, norm


validation_cmap, validation_norm = build_validation_mask_style(VALIDATION_CLASS_CODES)
classbar_cmap, classbar_norm = build_class_colorbar_style(VALIDATION_CLASS_CODES)

projected_validation = sample_points(
    filter_points_by_class(read_las_points(str(projected_las_path)), VALIDATION_CLASS_CODES),
    max_points=180_000,
    seed=42,
)
refined_validation = sample_points(
    filter_points_by_class(read_las_points(str(refined_las_path)), VALIDATION_CLASS_CODES),
    max_points=180_000,
    seed=42,
)


def collect_present_class_codes() -> tuple[int, ...]:
    present: set[int] = set()
    present.update(int(code) for code in np.unique(validation_mask) if int(code) in OVERLAY_CLASS_COLORS)
    for points in (projected_validation, refined_validation):
        if points is None:
            continue
        class_arr = np.asarray(points.get('class', []))
        if class_arr.size:
            present.update(int(code) for code in np.unique(class_arr) if int(code) in OVERLAY_CLASS_COLORS)
    return tuple(code for code in VALIDATION_CLASS_CODES if code in present)


PRESENT_CLASS_CODES = collect_present_class_codes() or VALIDATION_CLASS_CODES
classbar_cmap, classbar_norm = build_class_colorbar_style(PRESENT_CLASS_CODES)

top_reference_present = {
    int(code) for code in np.unique(validation_mask) if int(code) in OVERLAY_CLASS_COLORS
}
top_reference_present.update(
    int(code)
    for code in np.unique(np.asarray(projected_validation['class']))
    if int(code) in OVERLAY_CLASS_COLORS
)
TOP_REFERENCE_CODES = tuple(code for code in VALIDATION_CLASS_CODES if code in top_reference_present) or VALIDATION_CLASS_CODES


def validation_colors(class_values: np.ndarray) -> np.ndarray:
    return np.array(
        [OVERLAY_CLASS_COLORS.get(int(code), '#8f8f8f') for code in class_values],
        dtype=object,
    )


def validation_z(points: dict[str, np.ndarray | None]) -> np.ndarray:
    z_norm = points.get('z_norm')
    if DISPLAY_Z_MODE == 'z_norm_preferred' and z_norm is not None:
        return np.asarray(z_norm, dtype=np.float32)
    return np.asarray(points['z'], dtype=np.float32)


def validation_z_label(points: dict[str, np.ndarray | None]) -> str:
    z_norm = points.get('z_norm')
    if DISPLAY_Z_MODE == 'z_norm_preferred' and z_norm is not None:
        return 'Z_norm (m)'
    return 'Z (m, absolute)'


def z_range_text(values: np.ndarray, label: str) -> str:
    finite_values = np.asarray(values, dtype=np.float32)
    finite_values = finite_values[np.isfinite(finite_values)]
    if finite_values.size == 0:
        return f'{label}: unavailable'
    return f'{label}: {float(np.min(finite_values)):.2f} to {float(np.max(finite_values)):.2f} m'


def z_range_lines(values: np.ndarray, label: str) -> list[str]:
    finite_values = np.asarray(values, dtype=np.float32)
    finite_values = finite_values[np.isfinite(finite_values)]
    if finite_values.size == 0:
        return [label, 'unavailable']
    return [label, f'{float(np.min(finite_values)):.2f} to {float(np.max(finite_values)):.2f} m']


def add_z_extra_info(ax, points: dict[str, np.ndarray | None]) -> None:
    lines = z_range_lines(validation_z(points), f'Display {validation_z_label(points)}')

    z_norm = points.get('z_norm')
    if z_norm is not None:
        lines.extend(z_range_lines(np.asarray(points['z'], dtype=np.float32), 'Absolute Z'))

    ax.text2D(
        0.02,
        0.98,
        '\n'.join(lines),
        transform=ax.transAxes,
        ha='left',
        va='top',
        fontsize=INFOBOX_FONTSIZE,
        bbox={'boxstyle': 'round,pad=0.25', 'facecolor': 'white', 'edgecolor': '#b7c1c9', 'alpha': 0.9},
    )


def set_stable_3d_box_aspect(ax, z_limits: tuple[float, float]) -> None:
    x_span = max(overlay_extent[1] - overlay_extent[0], 1e-6)
    y_span = max(overlay_extent[3] - overlay_extent[2], 1e-6)
    xy_span = max(x_span, y_span)
    z_span = max(z_limits[1] - z_limits[0], 1e-6)
    z_ratio = float(np.clip(z_span / xy_span, 0.35, 0.9))
    ax.set_box_aspect((1.0, 1.0, z_ratio))


def combined_z_limits(*points_sets: dict[str, np.ndarray | None]) -> tuple[float, float]:
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
    for dim_name in las.point_format.dimension_names:
        if str(dim_name).lower() == 'z_norm':
            return np.asarray(las[dim_name], dtype=np.float32)

    for attr_name in ('z_norm', 'Z_norm'):
        if hasattr(las, attr_name):
            return np.asarray(getattr(las, attr_name), dtype=np.float32)

    return None


def sample_display_las(
    las_path: Path,
    max_points: int = 200_000,
    seed: int = 42,
) -> dict[str, np.ndarray | bool]:
    las = laspy.read(str(las_path))
    x_values = np.asarray(las.x)
    y_values = np.asarray(las.y)

    use_rgb = hasattr(las, 'red') and hasattr(las, 'green') and hasattr(las, 'blue')
    if use_rgb:
        red = np.asarray(las.red, dtype=np.float32)
        green = np.asarray(las.green, dtype=np.float32)
        blue = np.asarray(las.blue, dtype=np.float32)
        colors = np.stack([red, green, blue], axis=-1)
        denom = 65535.0 if float(np.nanmax(colors)) > 255.0 else 255.0
        colors = np.clip(colors / denom, 0, 1)
    else:
        z_values = np.asarray(las.z, dtype=np.float32)
        z_lo, z_hi = np.percentile(z_values, [2, 98])
        colors = np.clip((z_values - z_lo) / (z_hi - z_lo + 1e-6), 0, 1)

    if len(x_values) > max_points:
        rng = np.random.default_rng(seed)
        keep = rng.choice(len(x_values), size=max_points, replace=False)
        x_values = x_values[keep]
        y_values = y_values[keep]
        colors = colors[keep]

    return {
        'x': x_values,
        'y': y_values,
        'colors': colors,
        'use_rgb': use_rgb,
    }


def format_planar_panel(ax, extent: tuple[float, float, float, float]) -> None:
    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])
    ax.set_aspect('equal')
    ax.set_xlabel('X', fontsize=AXIS_LABEL_FONTSIZE)
    ax.set_ylabel('Y', fontsize=AXIS_LABEL_FONTSIZE)
    ax.tick_params(labelsize=AXIS_TICK_FONTSIZE)
    ax.ticklabel_format(style='plain', axis='both', useOffset=False)


def add_row_tag(ax, text: str) -> None:
    ax.axis('off')
    ax.text(
        0.5,
        0.5,
        text,
        ha='center',
        va='center',
        fontsize=ROW_TAG_FONTSIZE,
        fontweight='bold',
        bbox={
            'boxstyle': 'round,pad=0.28',
            'facecolor': '#f1f4f6',
            'edgecolor': '#b7c1c9',
            'alpha': 0.95,
        },
    )


def add_height_reference_guides(ax, present_codes=None) -> None:
    if DISPLAY_Z_MODE != 'z_norm_preferred':
        return

    x_min, x_max = overlay_extent[0], overlay_extent[1]
    y_min, y_max = overlay_extent[2], overlay_extent[3]
    x_mesh = np.array([[x_min, x_max], [x_min, x_max]], dtype=np.float32)
    y_mesh = np.array([[y_min, y_min], [y_max, y_max]], dtype=np.float32)

    if present_codes is None:
        present_codes = VALIDATION_CLASS_CODES

    present_codes = [int(code) for code in present_codes if int(code) not in EXCLUDED_HEIGHT_CODES]

    legend_lines = []
    for code in present_codes:
        if code not in HEIGHT_RANGES:
            continue

        z_min, z_max = HEIGHT_RANGES[code]
        color = OVERLAY_CLASS_COLORS[code]

        for z_level, alpha in ((z_min, 0.08), (z_max, 0.16)):
            z_mesh = np.full_like(x_mesh, z_level, dtype=np.float32)
            ax.plot_wireframe(
                x_mesh,
                y_mesh,
                z_mesh,
                color=color,
                linewidth=0.6,
                alpha=alpha,
            )

        legend_lines.append(f"{OVERLAY_CLASS_LABELS[code]}: {z_min:.1f}-{z_max:.1f} m")

    if legend_lines:
        ax.text2D(
            1.05,
            0.98,
            "Expected height bands\n" + "\n".join(legend_lines),
            transform=ax.transAxes,
            va='top',
            ha='left',
            fontsize=HEIGHT_GUIDE_FONTSIZE,
            bbox={
                'boxstyle': 'round,pad=0.25',
                'facecolor': 'white',
                'edgecolor': '#b7c1c9',
                'alpha': 0.92,
            },
        )


def plot_3d_panel(
    ax,
    points: dict[str, np.ndarray | None],
    title: str,
    z_limits: tuple[float, float],
) -> None:
    z_values = validation_z(points)
    class_values = np.asarray(points['class'])
    present_codes = sorted(
        int(code) for code in np.unique(class_values).astype(int)
        if int(code) not in EXCLUDED_HEIGHT_CODES
    )

    order = np.argsort(z_values)
    x_vals = np.asarray(points['x'])[order]
    y_vals = np.asarray(points['y'])[order]
    z_vals = np.asarray(z_values)[order]
    point_colors = validation_colors(class_values[order])

    ax.scatter(
        x_vals,
        y_vals,
        z_vals,
        s=3.4,
        c='black',
        alpha=0.08,
        linewidths=0,
        depthshade=False,
    )

    ax.scatter(
        x_vals,
        y_vals,
        z_vals,
        s=0.65,
        c=point_colors,
        alpha=1.0,
        linewidths=0,
        depthshade=False,
    )

    add_height_reference_guides(ax, present_codes=present_codes)
    add_z_extra_info(ax, points)

    ax.set_title(title, fontsize=PANEL_TITLE_FONTSIZE)
    ax.set_xlabel('X', fontsize=AXIS_LABEL_FONTSIZE)
    ax.set_ylabel('Y', fontsize=AXIS_LABEL_FONTSIZE)
    ax.set_zlabel(validation_z_label(points), fontsize=AXIS_LABEL_FONTSIZE)
    ax.set_xlim(overlay_extent[0], overlay_extent[1])
    ax.set_ylim(overlay_extent[2], overlay_extent[3])
    ax.set_zlim(*z_limits)
    ax.view_init(elev=24, azim=-58)
    ax.grid(True, alpha=0.3)
    ax.xaxis.pane.set_alpha(0.05)
    ax.yaxis.pane.set_alpha(0.05)
    ax.zaxis.pane.set_alpha(0.08)
    ax.xaxis.set_major_formatter(ScalarFormatter(useOffset=False))
    ax.yaxis.set_major_formatter(ScalarFormatter(useOffset=False))
    ax.zaxis.set_major_formatter(ScalarFormatter(useOffset=False))
    ax.tick_params(labelsize=AXIS_TICK_FONTSIZE)
    set_stable_3d_box_aspect(ax, z_limits)


def plot_las_only_panel(ax, las_display: dict[str, np.ndarray | bool], title: str) -> None:
    colors = np.asarray(las_display['colors'])
    if bool(las_display['use_rgb']):
        ax.scatter(las_display['x'], las_display['y'], s=0.5, c=colors, alpha=1.0, linewidths=0)
    else:
        ax.scatter(las_display['x'], las_display['y'], s=0.5, c=colors, cmap='viridis', alpha=1.0, linewidths=0)

    ax.set_title(title, fontsize=PANEL_TITLE_FONTSIZE)
    format_planar_panel(ax, overlay_extent)


def plot_raster_las_overlay_panel(ax, las_display: dict[str, np.ndarray | bool], title: str) -> None:
    ax.imshow(rgb_overlay, extent=overlay_extent, origin='upper')
    colors = np.asarray(las_display['colors'])

    if bool(las_display['use_rgb']):
        ax.scatter(las_display['x'], las_display['y'], s=1.2, c=colors, alpha=1.0, linewidths=0)
    else:
        ax.scatter(las_display['x'], las_display['y'], s=1.2, c=colors, cmap='viridis', alpha=1.0, linewidths=0)

    ax.set_title(title, fontsize=PANEL_TITLE_FONTSIZE)
    format_planar_panel(ax, overlay_extent)


def plot_overlay_panel(ax, points: dict[str, np.ndarray | None], title: str) -> None:
    ax.imshow(rgb_overlay, extent=overlay_extent, origin='upper')

    ax.scatter(
        points['x'],
        points['y'],
        s=4.0,
        c='black',
        alpha=0.55,
        linewidths=0,
        zorder=2,
    )

    ax.scatter(
        points['x'],
        points['y'],
        s=2.6,
        c=validation_colors(np.asarray(points['class'])),
        alpha=1.0,
        linewidths=0,
        zorder=3,
    )

    ax.set_title(title, fontsize=PANEL_TITLE_FONTSIZE)
    format_planar_panel(ax, overlay_extent)


def load_watershed_tree_data() -> dict[str, object] | None:
    try:
        from fvc3dstruct.watershed import get_mallee_centroids_watershed

        structure_las = laspy.read(str(projected_las_path))
        clipped_root = projected_las_path.parent.parent.parent.parent
        znorm_las_path = (
            clipped_root / 'raster' / 'z_norm' / 'csf'
            / f'z_norm_csf_tiles_multispectral.{tile_id}.las'
        )

        fallback_points = None
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
            'labels': labels,
            'centroids': centroids,
            'extent': extent,
            'count': int(labels.max()),
        }
    except Exception as exc:
        print(f'Watershed crown display skipped: {exc}')
        return None


def sample_treeid_points(
    las_path: Path,
    max_points: int = 140_000,
    seed: int = 42,
) -> dict[str, np.ndarray] | None:
    las = laspy.read(str(las_path))
    if 'treeID' not in las.point_format.dimension_names:
        return None

    tree_ids = np.asarray(las['treeID'], dtype=np.int32)
    class_values = np.asarray(las.classification, dtype=np.int16)
    keep = (tree_ids > 0) & (class_values == 68)
    if not np.any(keep):
        return None

    x_values = np.asarray(las.x)[keep]
    y_values = np.asarray(las.y)[keep]
    z_values = np.asarray(las.z, dtype=np.float32)[keep]
    tree_ids = tree_ids[keep]

    z_norm_all = get_las_z_norm(las)
    z_norm = None if z_norm_all is None else np.asarray(z_norm_all, dtype=np.float32)[keep]
    z_plot = np.asarray(z_norm if z_norm is not None else z_values, dtype=np.float32)

    if len(x_values) > max_points:
        rng = np.random.default_rng(seed)
        sample_idx = rng.choice(len(x_values), size=max_points, replace=False)
        x_values = x_values[sample_idx]
        y_values = y_values[sample_idx]
        z_values = z_values[sample_idx]
        if z_norm is not None:
            z_norm = z_norm[sample_idx]
        z_plot = z_plot[sample_idx]
        tree_ids = tree_ids[sample_idx]

    return {
        'x': x_values,
        'y': y_values,
        'z': z_values,
        'z_norm': z_norm,
        'z_plot': z_plot,
        'treeID': tree_ids,
    }


def treeid_colors(tree_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    unique_ids = np.unique(tree_ids)
    if len(unique_ids) == 0:
        return np.empty((0, 4)), unique_ids

    if len(unique_ids) == 1:
        return np.tile(np.array([[0.7922, 0.1216, 0.4824, 0.95]]), (len(tree_ids), 1)), unique_ids

    id_to_rank = {int(tree_id): idx for idx, tree_id in enumerate(unique_ids)}
    ranks = np.array([id_to_rank[int(tree_id)] for tree_id in tree_ids], dtype=np.float32)
    colors = plt.cm.nipy_spectral(ranks / max(len(unique_ids) - 1, 1))
    colors[:, 3] = 0.92
    return colors, unique_ids


def compact_treeid_ticks(n_ids: int, max_labels: int = 8) -> np.ndarray:
    if n_ids <= 0:
        return np.array([], dtype=int)
    if n_ids <= max_labels:
        return np.arange(1, n_ids + 1, dtype=int)

    tick_values = np.linspace(1, n_ids, max_labels, dtype=int)
    tick_values = np.unique(np.concatenate(([1], tick_values, [n_ids]))).astype(int)
    return tick_values


def add_treeid_colorbar(fig, cax, treeid_points: dict[str, np.ndarray] | None) -> None:
    cax.clear()

    if treeid_points is None:
        cax.axis('off')
        return

    unique_ids = np.unique(np.asarray(treeid_points['treeID'], dtype=np.int32))
    n_ids = len(unique_ids)

    if n_ids == 0:
        cax.axis('off')
        return

    if n_ids == 1:
        cmap = ListedColormap([[0.7922, 0.1216, 0.4824, 0.95]])
        norm = BoundaryNorm([0.5, 1.5], cmap.N)
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = fig.colorbar(sm, cax=cax)
        cbar.set_ticks([1])
        cbar.set_ticklabels(['1'])
        cbar.set_label('Final treeID', fontsize=COLORBAR_LABEL_FONTSIZE)
        cbar.ax.tick_params(labelsize=COLORBAR_TICK_FONTSIZE)
        return

    norm = Normalize(vmin=1, vmax=n_ids)
    sm = plt.cm.ScalarMappable(cmap=plt.cm.nipy_spectral, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cax)

    tick_values = compact_treeid_ticks(n_ids)
    cbar.set_ticks(tick_values)
    cbar.set_ticklabels([str(value) for value in tick_values])
    cbar.set_label('Final treeID', fontsize=COLORBAR_LABEL_FONTSIZE)
    cbar.ax.tick_params(labelsize=COLORBAR_TICK_FONTSIZE)


def plot_watershed_treeid_panel(ax, watershed_data: dict[str, object] | None, title: str) -> None:
    ax.imshow(rgb_overlay, extent=overlay_extent, origin='upper')
    if watershed_data is None:
        ax.text(
            0.5,
            0.5,
            'Watershed crowns unavailable',
            ha='center',
            va='center',
            transform=ax.transAxes,
            fontsize=INFOBOX_FONTSIZE,
            color='#345548',
            bbox={'boxstyle': 'round,pad=0.25', 'facecolor': 'white', 'edgecolor': '#b7c1c9', 'alpha': 0.9},
        )
    else:
        labels = np.asarray(watershed_data['labels'])
        extent = watershed_data['extent']
        centroids = watershed_data['centroids']
        crown_grid = np.ma.masked_where(labels == 0, labels)
        ax.imshow(crown_grid, origin='lower', cmap='nipy_spectral', extent=extent, alpha=0.68)

        if centroids is not None and len(centroids) > 1:
            ax.scatter(
                centroids[1:, 0],
                centroids[1:, 1],
                c='white',
                marker='x',
                s=26,
                linewidths=0.9,
                zorder=5,
            )

        ax.text(
            0.02,
            1.02,
            f"{int(watershed_data['count'])} watershed crowns",
            transform=ax.transAxes,
            ha='left',
            va='bottom',
            fontsize=INFOBOX_FONTSIZE,
            bbox={'boxstyle': 'round,pad=0.25', 'facecolor': 'white', 'edgecolor': '#b7c1c9', 'alpha': 0.9},
        )

    ax.set_title(title, fontsize=PANEL_TITLE_FONTSIZE)
    format_planar_panel(ax, overlay_extent)


def plot_treeid_panel_3d(
    ax,
    treeid_points: dict[str, np.ndarray] | None,
    title: str,
    z_limits: tuple[float, float],
) -> None:
    if treeid_points is None:
        ax.text2D(
            0.5,
            0.5,
            'No non-zero treeID values found',
            ha='center',
            va='center',
            transform=ax.transAxes,
            fontsize=INFOBOX_FONTSIZE,
            color='#345548',
            bbox={'boxstyle': 'round,pad=0.25', 'facecolor': 'white', 'edgecolor': '#b7c1c9', 'alpha': 0.9},
        )
        ax.set_title(title, fontsize=PANEL_TITLE_FONTSIZE)
        ax.set_xlabel('X', fontsize=AXIS_LABEL_FONTSIZE)
        ax.set_ylabel('Y', fontsize=AXIS_LABEL_FONTSIZE)
        ax.set_zlabel(validation_z_label({'z': np.array([], dtype=np.float32), 'z_norm': None}), fontsize=AXIS_LABEL_FONTSIZE)
        ax.set_xlim(overlay_extent[0], overlay_extent[1])
        ax.set_ylim(overlay_extent[2], overlay_extent[3])
        ax.set_zlim(*z_limits)
        ax.view_init(elev=24, azim=-58)
        return

    x_vals = np.asarray(treeid_points['x'])
    y_vals = np.asarray(treeid_points['y'])
    z_vals = np.asarray(treeid_points['z_plot'], dtype=np.float32)
    tree_ids = np.asarray(treeid_points['treeID'])

    colors, unique_ids = treeid_colors(tree_ids)

    order = np.argsort(z_vals)
    x_vals = x_vals[order]
    y_vals = y_vals[order]
    z_vals = z_vals[order]
    colors = colors[order]

    ax.scatter(
        x_vals,
        y_vals,
        z_vals,
        s=3.4,
        c='black',
        alpha=0.08,
        linewidths=0,
        depthshade=False,
    )

    ax.scatter(
        x_vals,
        y_vals,
        z_vals,
        s=0.70,
        c=colors,
        alpha=1.0,
        linewidths=0,
        depthshade=False,
    )

    ax.text2D(
        0.02,
        0.98,
        f'{len(unique_ids)} final treeIDs',
        transform=ax.transAxes,
        ha='left',
        va='top',
        fontsize=INFOBOX_FONTSIZE,
        bbox={'boxstyle': 'round,pad=0.25', 'facecolor': 'white', 'edgecolor': '#b7c1c9', 'alpha': 0.9},
    )
    add_z_extra_info(ax, treeid_points)

    ax.set_title(title, fontsize=PANEL_TITLE_FONTSIZE)
    ax.set_xlabel('X', fontsize=AXIS_LABEL_FONTSIZE)
    ax.set_ylabel('Y', fontsize=AXIS_LABEL_FONTSIZE)
    ax.set_zlabel(validation_z_label(treeid_points), fontsize=AXIS_LABEL_FONTSIZE)
    ax.set_xlim(overlay_extent[0], overlay_extent[1])
    ax.set_ylim(overlay_extent[2], overlay_extent[3])
    ax.set_zlim(*z_limits)
    ax.view_init(elev=24, azim=-58)
    ax.grid(True, alpha=0.3)
    ax.xaxis.pane.set_alpha(0.05)
    ax.yaxis.pane.set_alpha(0.05)
    ax.zaxis.pane.set_alpha(0.08)
    ax.xaxis.set_major_formatter(ScalarFormatter(useOffset=False))
    ax.yaxis.set_major_formatter(ScalarFormatter(useOffset=False))
    ax.zaxis.set_major_formatter(ScalarFormatter(useOffset=False))
    ax.tick_params(labelsize=AXIS_TICK_FONTSIZE)
    set_stable_3d_box_aspect(ax, z_limits)


projected_las_display = sample_display_las(projected_las_path, max_points=800_000)
watershed_tree_data = load_watershed_tree_data()
refined_treeid_points = sample_treeid_points(refined_las_path)
z_limits = combined_z_limits(projected_validation, refined_validation)
treeid_z_limits = (
    combined_z_limits(refined_validation, refined_treeid_points)
    if refined_treeid_points is not None
    else z_limits
)

gallery_output_path = gallery_output_path.with_suffix('.png')
gallery_output_path.parent.mkdir(parents=True, exist_ok=True)
gallery_top_output_path = gallery_output_path.with_name(f'{gallery_output_path.stem}_rows1_2.png')
gallery_bottom_output_path = gallery_output_path.with_name(f'{gallery_output_path.stem}_rows3_5.png')


def add_structure_class_colorbar(fig, cax) -> None:
    sm = plt.cm.ScalarMappable(cmap=classbar_cmap, norm=classbar_norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cax)
    cbar.set_ticks(np.arange(len(PRESENT_CLASS_CODES)))
    cbar.set_ticklabels([OVERLAY_CLASS_LABELS[code] for code in PRESENT_CLASS_CODES])
    cbar.ax.tick_params(labelsize=COLORBAR_TICK_FONTSIZE)
    cbar.set_label('Structure class', fontsize=COLORBAR_LABEL_FONTSIZE)


fig_top = plt.figure(figsize=(28, 20))
grid_top = fig_top.add_gridspec(
    4,
    5,
    height_ratios=(0.12, 0.95, 0.12, 0.95),
    width_ratios=(1.0, 1.0, 0.08, 1.0, 1.0),
    hspace=0.18,
    wspace=0.18,
)

top_row_tags = {
    'r1': fig_top.add_subplot(grid_top[0, :]),
    'r2': fig_top.add_subplot(grid_top[2, :]),
}

top_axes = {
    'a': fig_top.add_subplot(grid_top[1, 0:2]),
    'b': fig_top.add_subplot(grid_top[1, 3:5]),
    'c': fig_top.add_subplot(grid_top[3, 0:2]),
    'd': fig_top.add_subplot(grid_top[3, 3:5]),
}

top_cbar_axes = {'a': fig_top.add_subplot(grid_top[1, 2])}
top_spacer_ax = fig_top.add_subplot(grid_top[3, 2])
top_spacer_ax.axis('off')

TOP_ROW_TAG_FONTSIZE = 30
TOP_TICK_FONTSIZE = 30
TOP_COLORBAR_TICK_FONTSIZE = 36
TOP_COLORBAR_LABEL_FONTSIZE = 36


def add_top_row_tag(ax, text: str) -> None:
    ax.axis('off')
    ax.text(
        0.5,
        0.5,
        text,
        ha='center',
        va='center',
        fontsize=TOP_ROW_TAG_FONTSIZE,
        fontweight='bold',
        bbox={
            'boxstyle': 'round,pad=0.28',
            'facecolor': '#f1f4f6',
            'edgecolor': '#b7c1c9',
            'alpha': 0.95,
        },
    )


add_top_row_tag(top_row_tags['r1'], '2D reference inputs')
add_top_row_tag(top_row_tags['r2'], 'Spatial alignment assessment')

top_axes['a'].imshow(
    validation_mask_idx,
    cmap=validation_cmap,
    norm=validation_norm,
    extent=validation_extent,
    origin='upper',
    interpolation='nearest',
)
format_planar_panel(top_axes['a'], validation_extent)

top_axes['b'].imshow(rgb_overlay, extent=overlay_extent, origin='upper')
format_planar_panel(top_axes['b'], overlay_extent)

plot_las_only_panel(top_axes['c'], projected_las_display, '')
top_axes['c'].clear()
_c_colors = np.asarray(projected_las_display['colors'])
if bool(projected_las_display['use_rgb']):
    top_axes['c'].scatter(projected_las_display['x'], projected_las_display['y'], s=1.4, c=_c_colors, alpha=1.0, linewidths=0)
else:
    top_axes['c'].scatter(projected_las_display['x'], projected_las_display['y'], s=1.4, c=_c_colors, cmap='viridis', alpha=1.0, linewidths=0)
format_planar_panel(top_axes['c'], overlay_extent)
top_axes['c'].set_title('')
plot_raster_las_overlay_panel(top_axes['d'], projected_las_display, '')
top_axes['d'].set_title('')

for ax in top_axes.values():
    ax.tick_params(axis='both', labelsize=TOP_TICK_FONTSIZE)
    ax.set_xlabel('', fontsize=TOP_TICK_FONTSIZE)
    ax.set_ylabel('', fontsize=TOP_TICK_FONTSIZE)

top_classbar_cmap, top_classbar_norm = build_class_colorbar_style(TOP_REFERENCE_CODES)
sm_top = plt.cm.ScalarMappable(cmap=top_classbar_cmap, norm=top_classbar_norm)
sm_top.set_array([])
cbar_top = fig_top.colorbar(sm_top, cax=top_cbar_axes['a'])
cbar_top.set_ticks(np.arange(len(TOP_REFERENCE_CODES)))
cbar_top.set_ticklabels([OVERLAY_CLASS_LABELS[code] for code in TOP_REFERENCE_CODES])
cbar_top.ax.tick_params(labelsize=TOP_COLORBAR_TICK_FONTSIZE)
cbar_top.ax.yaxis.set_ticks_position('left')
cbar_top.ax.yaxis.set_label_position('left')

_pos = top_cbar_axes['a'].get_position()
_left_pos = top_axes['a'].get_position()
_right_pos = top_axes['b'].get_position()
_new_x = (_left_pos.x1 + _right_pos.x0) / 2 - _pos.width / 2
top_cbar_axes['a'].set_position([_new_x, _pos.y0, _pos.width, _pos.height])

fig_top.subplots_adjust(left=0.05, right=0.985, top=0.97, bottom=0.05)


def add_scale_bar(ax, length_m: float = 5.0, fontsize: int = 30) -> None:
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    span_x = x1 - x0
    span_y = y1 - y0
    pad_x = span_x * 0.05
    pad_y = span_y * 0.06
    bar_y = y0 + pad_y
    bar_x_start = x0 + pad_x
    bar_x_end = bar_x_start + length_m
    ax.plot(
        [bar_x_start, bar_x_end],
        [bar_y, bar_y],
        color='black',
        linewidth=6,
        solid_capstyle='butt',
        zorder=10,
    )
    ax.text(
        (bar_x_start + bar_x_end) / 2.0,
        bar_y + span_y * 0.02,
        f'{length_m:g} m',
        ha='center',
        va='bottom',
        fontsize=fontsize,
        fontweight='bold',
        color='black',
        zorder=11,
        bbox={'boxstyle': 'round,pad=0.2', 'facecolor': 'white', 'edgecolor': 'none', 'alpha': 0.85},
    )


add_scale_bar(top_axes['b'], length_m=5.0, fontsize=30)

for key in ('a', 'c', 'd'):
    top_axes[key].set_xticklabels([])
    top_axes[key].set_yticklabels([])
    top_axes[key].tick_params(axis='both', length=0)
fig_top.savefig(
    gallery_top_output_path,
    dpi=300,
    bbox_inches='tight',
    facecolor='white',
    format='png',
)


fig_bottom = plt.figure(figsize=(36, 32))
grid_bottom = fig_bottom.add_gridspec(
    6,
    5,
    height_ratios=(0.12, 1.18, 0.12, 1.18, 0.12, 1.18),
    width_ratios=(1.0, 1.0, 0.08, 1.6, 1.6),
    hspace=0.18,
    wspace=0.30,
)

bottom_row_tags = {
    'r3': fig_bottom.add_subplot(grid_bottom[0, :]),
    'r4': fig_bottom.add_subplot(grid_bottom[2, :]),
    'r5': fig_bottom.add_subplot(grid_bottom[4, :]),
}

bottom_axes = {
    'e': fig_bottom.add_subplot(grid_bottom[1, 0:2]),
    'f': fig_bottom.add_subplot(grid_bottom[1, 3:5], projection='3d'),
    'g': fig_bottom.add_subplot(grid_bottom[3, 0:2]),
    'h': fig_bottom.add_subplot(grid_bottom[3, 3:5], projection='3d'),
    'i': fig_bottom.add_subplot(grid_bottom[5, 0:2]),
    'j': fig_bottom.add_subplot(grid_bottom[5, 3:5], projection='3d'),
}

bottom_cbar_axes = {
    'e': fig_bottom.add_subplot(grid_bottom[1, 2]),
    'g': fig_bottom.add_subplot(grid_bottom[3, 2]),
    'j': fig_bottom.add_subplot(grid_bottom[5, 2]),
}

add_top_row_tag(bottom_row_tags['r3'], 'Initial label transfer')
add_top_row_tag(bottom_row_tags['r4'], 'Refinement and structural validation')
add_top_row_tag(bottom_row_tags['r5'], 'Mallee crown instance IDs')


def filter_validation_points(points: dict[str, np.ndarray | None], codes: tuple[int, ...]) -> dict[str, np.ndarray | None]:
    class_arr = np.asarray(points['class'])
    keep = np.isin(class_arr, list(codes))
    out: dict[str, np.ndarray | None] = {}
    for key, value in points.items():
        if value is None:
            out[key] = None
        else:
            arr = np.asarray(value)
            out[key] = arr[keep] if arr.shape[:1] == class_arr.shape[:1] else value
    return out


def compute_present_codes(class_arrays: list[np.ndarray]) -> tuple[int, ...]:
    present: set[int] = set()
    for arr in class_arrays:
        if arr is None:
            continue
        a = np.asarray(arr)
        if a.size:
            present.update(int(c) for c in np.unique(a) if int(c) in OVERLAY_CLASS_COLORS)
    return tuple(c for c in VALIDATION_CLASS_CODES if c in present)


refined_present = filter_validation_points(refined_validation, PRESENT_CLASS_CODES)
row3_codes = compute_present_codes([np.asarray(projected_validation['class'])]) or PRESENT_CLASS_CODES
row4_codes = compute_present_codes([np.asarray(refined_present['class'])]) or PRESENT_CLASS_CODES

plot_overlay_panel(bottom_axes['e'], projected_validation, '')
plot_3d_panel(bottom_axes['f'], projected_validation, '', z_limits)
plot_overlay_panel(bottom_axes['g'], refined_present, '')
plot_3d_panel(bottom_axes['h'], refined_present, '', z_limits)
plot_watershed_treeid_panel(
    bottom_axes['i'],
    watershed_tree_data,
    '',
)
plot_treeid_panel_3d(
    bottom_axes['j'],
    refined_treeid_points,
    '',
    treeid_z_limits,
)

for key in ('e', 'f', 'g', 'h', 'i', 'j'):
    bottom_axes[key].set_title('')

for cbar_key, codes in (('e', row3_codes), ('g', row4_codes)):
    row_cmap, row_norm = build_class_colorbar_style(codes)
    sm_b = plt.cm.ScalarMappable(cmap=row_cmap, norm=row_norm)
    sm_b.set_array([])
    cbar_b = fig_bottom.colorbar(sm_b, cax=bottom_cbar_axes[cbar_key])
    cbar_b.set_ticks(np.arange(len(codes)))
    cbar_b.set_ticklabels([OVERLAY_CLASS_LABELS[code] for code in codes])
    cbar_b.ax.tick_params(labelsize=TOP_COLORBAR_TICK_FONTSIZE)
    cbar_b.ax.yaxis.set_ticks_position('left')
    cbar_b.ax.yaxis.set_label_position('left')

add_treeid_colorbar(fig_bottom, bottom_cbar_axes['j'], refined_treeid_points)
bottom_cbar_axes['j'].tick_params(labelsize=TOP_COLORBAR_TICK_FONTSIZE)
for label in bottom_cbar_axes['j'].yaxis.get_label(),:
    label.set_fontsize(TOP_COLORBAR_LABEL_FONTSIZE)

for _cb_key in ('e', 'g', 'j'):
    _pos_b = bottom_cbar_axes[_cb_key].get_position()
    bottom_cbar_axes[_cb_key].set_position([_pos_b.x0 + 0.025, _pos_b.y0, _pos_b.width, _pos_b.height])

for key in ('e', 'f', 'g', 'h', 'i', 'j'):
    ax = bottom_axes[key]
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    if hasattr(ax, 'set_zticklabels'):
        try:
            ax.set_zticklabels([])
        except Exception:
            pass
    ax.set_xlabel('')
    ax.set_ylabel('')
    if hasattr(ax, 'set_zlabel'):
        try:
            ax.set_zlabel('')
        except Exception:
            pass
    ax.tick_params(axis='both', length=0, labelsize=TOP_TICK_FONTSIZE)
    if key == 'j' and hasattr(ax, 'tick_params'):
        try:
            ax.tick_params(axis='z', length=0, labelsize=TOP_TICK_FONTSIZE)
        except Exception:
            pass

for key in ('f', 'h', 'j'):
    ax3d = bottom_axes[key]
    try:
        ax3d.set_box_aspect(ax3d.get_box_aspect(), zoom=THREED_BOX_ZOOM)
    except TypeError:
        pos = ax3d.get_position()
        dx = pos.width * 0.18
        dy = pos.height * 0.18
        ax3d.set_position([pos.x0 - dx * 0.5, pos.y0 - dy * 0.5, pos.width + dx, pos.height + dy])

fig_bottom.subplots_adjust(left=0.05, right=0.985, top=0.97, bottom=0.04)
fig_bottom.savefig(
    gallery_bottom_output_path,
    dpi=300,
    bbox_inches='tight',
    facecolor='white',
    format='png',
)

plt.show()

if DISPLAY_Z_MODE == 'z_norm_preferred':
    print('Using Z_norm for 3D display when available, with absolute LAS Z reported as extra info.')
else:
    print('Using absolute LAS Z values for the 3D panels.')

print(f"Projected validation points: {len(projected_validation['x']):,}")
print(f"Refined validation points: {len(refined_validation['x']):,}")
print(z_range_text(validation_z(projected_validation), f"Projected display {validation_z_label(projected_validation)}"))
print(z_range_text(np.asarray(projected_validation['z'], dtype=np.float32), 'Projected absolute Z'))
print(z_range_text(validation_z(refined_validation), f"Refined display {validation_z_label(refined_validation)}"))
print(z_range_text(np.asarray(refined_validation['z'], dtype=np.float32), 'Refined absolute Z'))
if refined_treeid_points is not None:
    print(f"Final non-zero treeIDs shown: {len(np.unique(refined_treeid_points['treeID'])):,}")
    print(z_range_text(validation_z(refined_treeid_points), f"TreeID display {validation_z_label(refined_treeid_points)}"))
    print(z_range_text(np.asarray(refined_treeid_points['z'], dtype=np.float32), 'TreeID absolute Z'))
print(f"Saved gallery rows 1-2 figure: {gallery_top_output_path.resolve()}")
print(f"Saved gallery rows 3-5 figure: {gallery_bottom_output_path.resolve()}")
