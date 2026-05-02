import os
import sys
from pathlib import Path

FVC_STRUCTURE_ENV = Path(sys.prefix)
proj_data_dir = FVC_STRUCTURE_ENV / "share" / "proj"
if (proj_data_dir / "proj.db").exists():
    os.environ.setdefault("PROJ_DATA", str(proj_data_dir))
    os.environ.setdefault("PROJ_LIB", str(proj_data_dir))

MASKCLIP_PARENT = Path("/home/laura/Documents/code/fvc_structure/phase_4_3D_lidar_prep/step_1_clipping")
if str(MASKCLIP_PARENT) not in sys.path:
    sys.path.insert(0, str(MASKCLIP_PARENT))

from maskclip import classify_single
print("maskclip OK")

from pathlib import Path

# --- CONFIGURATION: Site 1 (DD0001) ---
SITE_NAME = 'site1_DD0001'


# LiDAR ICP registered + mask classification
LAS_DIR = Path("/media/laura/8402326D023263F8/calperumResearch/site2_1_DD0011/lidar/clipped/raster/coreg/icp")
AOI_RASTER_DIR = Path("/media/laura/8402326D023263F8/calperumResearch/site2_1_DD0011/inputs/predictors/tiles_3072/stacked")
LIDAR_RASTER_PATTERN = 'tiles_multispectral*.tif'
MASK_DIR = Path("/media/laura/8402326D023263F8/calperumResearch/site2_1_DD0011/inputs/masks/tiles_3072/raw/structure_class/annotation_raster/downsample_5cm_nearest")


# If LAS_DIR was set to a specific LAS file path, normalize to its parent directory
if LAS_DIR.is_file():
    print(f"Note: LAS_DIR points to a file. Using parent folder: {LAS_DIR.parent}")
    LAS_DIR = LAS_DIR.parent

# Pattern for LAS tiles produced by rasterclip
RASTER_PATTERN = 'tiles_multispectral*.las'

# Single tile example
TILE_ID = '33'
A_TILE = AOI_RASTER_DIR / f'tiles_multispectral.{TILE_ID}.tif'

# Output directory for mask classification
OUT_DIR = Path("/media/laura/8402326D023263F8/calperumResearch/site2_1_DD0011/lidar/clipped/mask/coreg/icp_top")
OUT_DIR.mkdir(parents=True, exist_ok=True)

print('--- Configuration ---')
print(f'Raster dir  : {AOI_RASTER_DIR}')
print(f'Mask dir    : {MASK_DIR}')
print(f'LAS dir     : {LAS_DIR}')
print(f'Pattern     : {RASTER_PATTERN}')
print(f'Single tile : {TILE_ID}')
print(f'Output dir  : {OUT_DIR}')
print('----------------------')

##################
# --- View mask + LAS + multispectral overlay ---
import laspy
import numpy as np
import matplotlib.pyplot as plt
import rasterio
from matplotlib.patches import Patch
from matplotlib.colors import ListedColormap, BoundaryNorm

# Global font size
plt.rcParams.update({"font.size": 30})

VALID_CLASS_CODES = {64, 65, 66, 67, 68}
PALETTE = {
    64: (0.851, 0.706, 0.290),  # Forb / #d9b44a
    65: (1.0, 0.0, 1.0),  # Woody debris / #FF00FF (magenta)
    66: (0.851, 0.706, 0.290),  # Hummock / #d9b44a
    67: (0.412, 0.220, 0.082),  # Shrub / #694714
    68: (0.180, 0.545, 0.341),  # Mallee / #2e8b57
}
CLASS_LABELS = {
    64: "Forb=0",
    66: "Hummock=0",
    65: "Woody debris=1",
    67: "Shrub=1",
    68: "Mallee=2",
}


def resolve_sample_las(tile_id):
    preferred_candidates = [
        OUT_DIR / f"mask_tiles_multispectral_{tile_id}.las",
        OUT_DIR / f"tiles_multispectral_{tile_id}.las",
        OUT_DIR / f"mask_tiles_multispectral.{tile_id}.las",
        OUT_DIR / f"tiles_multispectral.{tile_id}.las",
        LAS_DIR / f"tiles_multispectral_{tile_id}.las",
        LAS_DIR / f"tiles_multispectral.{tile_id}.las",
    ]
    for candidate in preferred_candidates:
        if candidate.exists():
            return candidate

    fallback_patterns = [
        (OUT_DIR, "mask_tiles_multispectral*.las"),
        (OUT_DIR, "tiles_multispectral*.las"),
        (LAS_DIR, "tiles_multispectral*.las"),
    ]
    for folder, pattern in fallback_patterns:
        matches = sorted(folder.glob(pattern))
        if matches:
            return matches[0]

    return None

# -----------------------------
# 1) Get sample_out from configured TILE_ID or find first available output
# -----------------------------
try:
    sample_out = resolve_sample_las(TILE_ID)
except NameError:
    sample_out = resolve_sample_las(None)

if sample_out is None:
    raise FileNotFoundError(
        "No LAS files found in either OUT_DIR or LAS_DIR. Run classification or check the configured paths."
    )

if sample_out.parent == LAS_DIR:
    print(f"Using source LAS tile because no classified output was found: {sample_out}")

# -----------------------------
# 2) Read LAS
# -----------------------------
las = laspy.read(sample_out)
x = las.x
y = las.y

# -----------------------------
# 3) Resolve tile id
# -----------------------------
stem_parts = sample_out.stem.replace(".", "_").split("_")
tile_id = stem_parts[-1]

# -----------------------------
# 4) Load multispectral raster for extent and RGB overlay
# -----------------------------
raster_candidates = [
    AOI_RASTER_DIR / f"tiles_multispectral_{tile_id}.tif",
    AOI_RASTER_DIR / f"tiles_multispectral.{tile_id}.tif",
]
raster_extent_path = next((path for path in raster_candidates if path.exists()), None)

if raster_extent_path is not None:
    with rasterio.open(raster_extent_path) as src:
        extent = (src.bounds.left, src.bounds.right, src.bounds.bottom, src.bounds.top)

        if src.count < 5:
            raise RuntimeError(f"Raster has {src.count} bands; need >= 5 for 5/3/1 RGB.")

        b5 = src.read(5).astype(np.float32)
        b3 = src.read(3).astype(np.float32)
        b1 = src.read(1).astype(np.float32)

        def norm_band(b):
            valid = np.isfinite(b)
            lo, hi = np.percentile(b[valid], [2, 98])
            out = np.clip((b - lo) / (hi - lo + 1e-6), 0, 1)
            out[~valid] = 0
            return out

        rgb = np.stack([norm_band(b5), norm_band(b3), norm_band(b1)], axis=-1)
else:
    extent = (x.min(), x.max(), y.min(), y.max())
    rgb = None
    print(f"Raster tile not found for tile_id={tile_id}; checked: {[str(p) for p in raster_candidates]}")

# -----------------------------
# 5) Classification colors or Z colors
# -----------------------------
has_classification = hasattr(las, "classification")
classified_codes = set()

if has_classification:
    cls = np.asarray(las.classification, dtype=np.uint8)
    classified_codes = {int(code) for code in np.unique(cls) if int(code) in VALID_CLASS_CODES}
    has_classification = bool(classified_codes)

if has_classification:
    colors = np.array([PALETTE.get(int(c), (0.5, 0.5, 0.5)) for c in cls], dtype=np.float32)
else:
    z = np.asarray(las.z, dtype=np.float32)
    z_lo, z_hi = np.percentile(z, [2, 98])
    colors = np.clip((z - z_lo) / (z_hi - z_lo + 1e-6), 0, 1)

# -----------------------------
# 6) Load mask raster for plot 1 only
# -----------------------------
mask_candidates = [
    MASK_DIR / f"mask_structure_3072.{tile_id}.tif",
    MASK_DIR / f"mask_tiles_multispectral.{tile_id}.tif",
    MASK_DIR / f"mask_tiles_multispectral_{tile_id}.tif",
]
mask_path = next((path for path in mask_candidates if path.exists()), None)

mask_plot = None
mask_extent = extent

if mask_path is not None:
    with rasterio.open(mask_path) as src:
        mask_arr = src.read(1).astype(np.float32)
        mask_extent = (src.bounds.left, src.bounds.right, src.bounds.bottom, src.bounds.top)
        mask_nodata = src.nodata

    valid_classes = [64, 65, 66, 67, 68]
    valid_mask = np.isin(mask_arr, valid_classes)
    mask_plot = mask_arr.copy()
    mask_plot[~valid_mask] = np.nan
    if mask_nodata is not None:
        mask_plot[mask_arr == mask_nodata] = np.nan

    mask_cmap = ListedColormap([PALETTE[c] for c in valid_classes])
    mask_cmap.set_bad((1, 1, 1, 0))
    mask_norm = BoundaryNorm([63.5, 64.5, 65.5, 66.5, 67.5, 68.5], mask_cmap.N)
elif mask_path is None:
    print(f"Mask tile not found for tile_id={tile_id}; checked: {[str(p) for p in mask_candidates]}")

# -----------------------------
# 7) Subsample if needed
# -----------------------------
n = len(x)
max_pts = 600000

if n > max_pts:
    idx = np.random.choice(n, size=max_pts, replace=False)
    x = x[idx]
    y = y[idx]
    colors = colors[idx]
    if has_classification:
        cls = cls[idx]
else:
    idx = None

# -----------------------------
# 8) Create figure
# -----------------------------
fig, axes = plt.subplots(1, 3, figsize=(42, 12), dpi=1000)
plt.subplots_adjust(wspace=0.01, hspace=0.05, bottom=0.25)


for ax in axes:
    ax.set_aspect("equal")
    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])

    ax.tick_params(axis="x", labelrotation=30, labelsize=30)
    for label in ax.get_xticklabels():
        label.set_horizontalalignment("right")

    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_title("")
    ax.tick_params(axis="both", labelsize=30)

# -----------------------------
# 9) Plot 1: mask raster
# -----------------------------
ax = axes[0]
if mask_plot is not None:
    ax.imshow(mask_plot, extent=mask_extent, origin="upper", cmap=mask_cmap, norm=mask_norm)
else:
    ax.text(0.5, 0.5, "Mask not available", ha="center", va="center",
            transform=ax.transAxes, fontsize=20)

# -----------------------------
# 10) Plot 2: RGB coloring
# -----------------------------
ax = axes[1]
if hasattr(las, "red") and hasattr(las, "green") and hasattr(las, "blue"):
    if idx is not None:
        r = np.asarray(las.red[idx], dtype=np.float32) / 65535.0
        g = np.asarray(las.green[idx], dtype=np.float32) / 65535.0
        b = np.asarray(las.blue[idx], dtype=np.float32) / 65535.0
    else:
        r = np.asarray(las.red, dtype=np.float32) / 65535.0
        g = np.asarray(las.green, dtype=np.float32) / 65535.0
        b = np.asarray(las.blue, dtype=np.float32) / 65535.0

    rgb_colors = np.stack([r, g, b], axis=-1)
    ax.scatter(x, y, s=2, c=rgb_colors, alpha=0.8)
else:
    ax.text(0.5, 0.5, "RGB not available", ha="center", va="center",
            transform=ax.transAxes, fontsize=20)

# -----------------------------
# 11) Plot 3: overlay + legend
# -----------------------------
ax = axes[2]

if rgb is not None:
    ax.imshow(rgb, extent=extent, origin="upper")

if colors.ndim == 2:
    ax.scatter(x, y, s=2, c=colors, alpha=0.7)
else:
    ax.scatter(x, y, s=2, c=colors, cmap="viridis", alpha=0.7)

if has_classification:
    unique_classes = sorted(classified_codes)
    legend_elements = [
        Patch(
            facecolor=PALETTE.get(int(c), (0.5, 0.5, 0.5)),
            label=f"{CLASS_LABELS.get(int(c), f'Class {int(c)}')} ({int(c)})"
        )
        for c in unique_classes if int(c) in PALETTE
    ]

    fig.legend(
        handles=legend_elements,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.02),
        ncol=len(legend_elements),
        frameon=True,
        fontsize=30
    )

# -----------------------------
# 12) Save BEFORE show
# -----------------------------
out_png = OUT_DIR / f"{sample_out.stem}_three_panel_view.png"

plt.savefig(out_png, dpi=500, format="png", bbox_inches="tight", pad_inches=0.05)
print(f"Saved to: {out_png}")

plt.show()
# plt.close(fig)
