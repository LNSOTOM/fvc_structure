# --- Visualization: Original + CSF Ground + HAG ---
# Set specific tile ID for visualization (or use test_tile from previous cell)
TILE_ID = '109'

# Load specific tile if TILE_ID is set, otherwise use test_tile from previous cell
if TILE_ID:
    test_tile = INPUT_DIR / f"tiles_multispectral.{TILE_ID}.las"
    tile_name = test_tile.stem
    csf_ground = GROUND_CSF_DIR / f"csf_{tile_name}.las"
    znorm_csf = ZNORM_CSF_DIR / f"z_norm_csf_{tile_name}.las"
    
    if not test_tile.exists():
        print(f"⚠️ Warning: Tile {TILE_ID} not found at {test_tile}")
        print("   Using test_tile from previous cell instead...")
    elif not csf_ground.exists() or not znorm_csf.exists():
        print(f"⚠️ Warning: Processed outputs for tile {TILE_ID} not found")
        print(f"   CSF ground exists: {csf_ground.exists()}")
        print(f"   Z-norm exists: {znorm_csf.exists()}")
        print("   Run processing cell first or use test_tile from previous cell")
    else:
        print(f"✅ Visualizing tile: {tile_name}")

max_pts = 50000

def downsample_idx(n: int, max_pts: int) -> np.ndarray:
    if n > max_pts:
        return np.random.choice(n, size=max_pts, replace=False)
    return np.arange(n)

# Load data
las_orig = laspy.read(test_tile)
las_csf_ground = laspy.read(csf_ground)
las_znorm = laspy.read(znorm_csf)

# Get extent from multispectral raster tile
tile_id = test_tile.stem.split('.')[-1]
raster_extent_path = AOI_RASTER_DIR / f'tiles_multispectral.{tile_id}.tif'
if raster_extent_path.exists():
    with rasterio.open(raster_extent_path) as src:
        extent = (src.bounds.left, src.bounds.right, src.bounds.bottom, src.bounds.top)
else:
    extent = None
    print(f"⚠️ Warning: Raster tile not found at {raster_extent_path}")

# Extract data for original
idx0 = downsample_idx(len(las_orig.points), max_pts)
x_orig = las_orig.x[idx0]
y_orig = las_orig.y[idx0]
z_orig = las_orig.z[idx0]

# Extract data for ground classification
x_csf = np.asarray(las_csf_ground.x)
y_csf = np.asarray(las_csf_ground.y)
cls_csf = np.asarray(las_csf_ground.classification)
idx1 = downsample_idx(len(x_csf), max_pts)
x_csf, y_csf, cls_csf = x_csf[idx1], y_csf[idx1], cls_csf[idx1]

# Extract HAG
z_norm = np.asarray(las_znorm.Z_norm, dtype=np.float32)
idx2 = downsample_idx(len(z_norm), max_pts)
x_hag = las_znorm.x[idx2]
y_hag = las_znorm.y[idx2]
hag = z_norm[idx2]

# Create figure
fig, axes = plt.subplots(1, 3, figsize=(24, 8), dpi=150)
fig.subplots_adjust(wspace=0.25)

# (1) Original point cloud (Z-colored)
ax = axes[0]
sc0 = ax.scatter(x_orig, y_orig, s=1, c=z_orig, cmap='terrain', alpha=0.5)
ax.set_title('Original Point Cloud', fontsize=14)
ax.set_xlabel('X (m)')
ax.set_ylabel('Y (m)')
ax.set_aspect('equal')
if extent:
    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])
divider0 = make_axes_locatable(ax)
cax0 = divider0.append_axes("right", size="3%", pad=0.08)
cbar0 = fig.colorbar(sc0, cax=cax0)
cbar0.set_label('Z (m)')

# (2) CSF ground classification
ax = axes[1]
is_ground = (cls_csf == 2).astype(np.uint8)
class_cmap = ListedColormap(["lightgray", "limegreen"])
class_norm = BoundaryNorm([0, 0.5, 1.0], class_cmap.N)
sc1 = ax.scatter(x_csf, y_csf, s=1, c=is_ground, cmap=class_cmap, norm=class_norm, alpha=0.9)
ax.set_title('CSF Ground Classification', fontsize=14)
ax.set_xlabel('X (m)')
ax.set_ylabel('Y (m)')
ax.set_aspect('equal')
if extent:
    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])
divider1 = make_axes_locatable(ax)
cax1 = divider1.append_axes("right", size="3%", pad=0.08)
class_mappable = ScalarMappable(cmap=class_cmap, norm=class_norm)
class_mappable.set_array([])
cbar1 = fig.colorbar(class_mappable, cax=cax1)
cbar1.set_ticks([0.25, 0.75])
cbar1.set_ticklabels(["non-ground", "ground (class 2)"])

# (3) Height Above Ground (HAG)
ax = axes[2]
colors_bgyr = ['blue', 'green', 'yellow', 'red']
cmap_bgyr = LinearSegmentedColormap.from_list('bgyr', colors_bgyr, N=256)
sc2 = ax.scatter(x_hag, y_hag, s=1, c=hag, cmap=cmap_bgyr, alpha=0.6, vmin=0, vmax=5)
ax.set_title('Height Above Ground (HAG)', fontsize=14)
ax.set_xlabel('X (m)')
ax.set_ylabel('Y (m)')
ax.set_aspect('equal')
if extent:
    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])
divider2 = make_axes_locatable(ax)
cax2 = divider2.append_axes("right", size="3%", pad=0.08)
cbar2 = fig.colorbar(sc2, cax=cax2)
cbar2.set_label('HAG (m)')

plt.suptitle(f'Ground Classification & Height Normalisation: {tile_name}', fontsize=16, y=0.98)
plt.show()

# Statistics
print("\n=== HAG Statistics ===")
print(f"Min: {z_norm.min():.3f} m")
print(f"Mean: {z_norm.mean():.3f} m")
print(f"Median: {np.median(z_norm):.3f} m")
print(f"Max: {z_norm.max():.3f} m")
print(f"Std: {z_norm.std():.3f} m")