def add_height_reference_guides(ax, present_codes=None) -> None:
    x_min, x_max = overlay_extent[0], overlay_extent[1]
    y_min, y_max = overlay_extent[2], overlay_extent[3]
    x_mesh = np.array([[x_min, x_max], [x_min, x_max]], dtype=np.float32)
    y_mesh = np.array([[y_min, y_min], [y_max, y_max]], dtype=np.float32)

    if present_codes is None:
        present_codes = VALIDATION_CLASS_CODES

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
            0.02,
            0.98,
            "Expected height bands\n" + "\n".join(legend_lines),
            transform=ax.transAxes,
            va='top',
            ha='left',
            fontsize=7,
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
    present_codes = sorted(np.unique(class_values).astype(int))

    x_span = max(overlay_extent[1] - overlay_extent[0], 1e-6)
    y_span = max(overlay_extent[3] - overlay_extent[2], 1e-6)
    z_span = max(z_limits[1] - z_limits[0], 1e-6)

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
        alpha=0.92,
        linewidths=0,
        depthshade=False,
    )

    add_height_reference_guides(ax, present_codes=present_codes)

    ax.set_title(title, fontsize=10)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z_norm (m)')
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
    ax.tick_params(labelsize=8)
    ax.set_box_aspect((x_span, y_span, z_span * 1.18))


projected_las_display = sample_display_las(projected_las_path)
z_limits = combined_z_limits(projected_validation, refined_validation)

fig = plt.figure(figsize=(21, 26))
grid = fig.add_gridspec(
    8,
    5,
    height_ratios=(
        0.10, 0.95,
        0.10, 0.95,
        0.10, 1.18,
        0.10, 1.18,
    ),
    width_ratios=(1.0, 1.0, 0.08, 1.0, 1.0),
    hspace=0.18,
    wspace=0.18,
)

row_tags = {
    'r1': fig.add_subplot(grid[0, :]),
    'r2': fig.add_subplot(grid[2, :]),
    'r3': fig.add_subplot(grid[4, :]),
    'r4': fig.add_subplot(grid[6, :]),
}

axes = {
    'a': fig.add_subplot(grid[1, 0:2]),
    'b': fig.add_subplot(grid[1, 3:5]),
    'c': fig.add_subplot(grid[3, 0:2]),
    'd': fig.add_subplot(grid[3, 3:5]),
    'e': fig.add_subplot(grid[5, 0:2]),
    'f': fig.add_subplot(grid[5, 3:5], projection='3d'),
    'g': fig.add_subplot(grid[7, 0:2]),
    'h': fig.add_subplot(grid[7, 3:5], projection='3d'),
}

# dedicated colorbar axes so A, C, E, G stay aligned
cbar_axes = {
    'a': fig.add_subplot(grid[1, 2]),
    'e': fig.add_subplot(grid[5, 2]),
    'g': fig.add_subplot(grid[7, 2]),
}

# empty spacer for row 2 to preserve alignment
blank_cbar_ax = fig.add_subplot(grid[3, 2])
blank_cbar_ax.axis('off')

add_row_tag(row_tags['r1'], '2D reference inputs')
add_row_tag(row_tags['r2'], 'Spatial alignment assessment')
add_row_tag(row_tags['r3'], '2D-to-3D transfer and refinement')
add_row_tag(row_tags['r4'], 'Structural plausibility from Z-normalised LiDAR')

axes['a'].imshow(
    validation_mask_idx,
    cmap=validation_cmap,
    norm=validation_norm,
    extent=validation_extent,
    origin='upper',
    interpolation='nearest',
)
axes['a'].set_title('(a) Original 2D structure mask derived from multispectral imagery', fontsize=10)
axes['a'].set_xlabel('X')
axes['a'].set_ylabel('Y')
format_planar_panel(axes['a'], validation_extent)

axes['b'].imshow(rgb_overlay, extent=overlay_extent, origin='upper')
axes['b'].set_title('(b) Multispectral tile used to derive the 2D structure mask', fontsize=10)
axes['b'].set_xlabel('X')
axes['b'].set_ylabel('Y')
format_planar_panel(axes['b'], overlay_extent)

plot_las_only_panel(axes['c'], projected_las_display, '(c) Co-registered LiDAR point cloud before label transfer')
plot_raster_las_overlay_panel(axes['d'], projected_las_display, '(d) Raster-LiDAR overlay for spatial alignment assessment')
plot_overlay_panel(axes['e'], projected_validation, '(e) Initial projection of 2D labels onto the LiDAR point cloud')
plot_3d_panel(axes['f'], projected_validation, '(f) 3D view of the initial 2D-to-3D label projection', z_limits)
plot_overlay_panel(axes['g'], refined_validation, '(g) Refined 3D labels after voxel-based aggregation and polar refinement')
plot_3d_panel(axes['h'], refined_validation, '(h) Vertical view of the refined point cloud showing height-based separation of vegetation classes', z_limits)

# colorbars only for A, E, G using dedicated axes
for key in ('a', 'e', 'g'):
    sm = plt.cm.ScalarMappable(cmap=classbar_cmap, norm=classbar_norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cbar_axes[key])
    cbar.set_ticks(np.arange(len(VALIDATION_CLASS_CODES)))
    cbar.set_ticklabels([OVERLAY_CLASS_LABELS[code] for code in VALIDATION_CLASS_CODES])
    cbar.ax.tick_params(labelsize=8)
    cbar.set_label('Structure class', fontsize=9)

fig.suptitle(f'Tile {tile_id} validation of 2D-to-3D label transfer', fontsize=15, y=0.992)
fig.subplots_adjust(left=0.05, right=0.985, top=0.97, bottom=0.05)

gallery_output_path.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(gallery_output_path, dpi=300, bbox_inches='tight')
plt.show()

if projected_validation.get('z_norm') is None:
    print('Projected LAS does not contain Z_norm; using a local height proxy computed as Z minus the 1st percentile of tile elevation.')

print(f"Projected validation points: {len(projected_validation['x']):,}")
print(f"Refined validation points: {len(refined_validation['x']):,}")
print(f"Saved gallery figure: {gallery_output_path}")