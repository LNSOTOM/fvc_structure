"""Quick-look figures for label-transfer results.

Needs matplotlib: ``pip install "fvc3d[plot]"``. Every function draws on an
existing Matplotlib ``ax`` if given (else a new figure) and returns the axes,
so figures can be combined and saved as usual::

    import fvc3d.plot

    result = fvc3d.transfer_labels("labels.tif", "lidar.laz")
    fvc3d.plot.summary(result).savefig("tile.png", dpi=200)
    fvc3d.plot.points(result, view="side")

Colours follow the paper's figures (forb, woody debris, hummock, shrub,
mallee); pass ``colors={code: "#rrggbb"}`` to change them.
"""

from __future__ import annotations

import numpy as np

from .classes import PAPER_CLASSES, ClassScheme
from .io import LabelRaster, read_labels

#: Class colours used in the paper's figures.
PAPER_COLORS: dict[int, str] = {
    64: "#66c2a5",  # forb
    65: "#8c6d31",  # woody debris
    66: "#ffd92f",  # hummock
    67: "#fc8d62",  # shrub
    68: "#1f78b4",  # mallee
}
#: Colour of a single crown when only one treeID is present (paper figures).
SINGLE_CROWN_COLOR = "#ca1f7b"
#: Colours for classes outside the paper's scheme, in order.
_EXTRA_COLORS = ["#66c2a5", "#fc8d62", "#8da0cb", "#e78ac3", "#a6d854", "#ffd92f", "#e5c494", "#b3b3b3"]


def _plt():
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise ImportError('fvc3d.plot needs matplotlib: pip install "fvc3d[plot]"') from exc
    return plt


def class_colors(classes=None, output: bool = False) -> dict[int, str]:
    """Colour per class code for a scheme (default: the paper's).

    With ``output=True``, colours are for the scheme's output classes (e.g.
    FOR-instanceV2's ground/wood/leaf); each takes the colour of the first
    growth form mapped to it.
    """
    scheme = ClassScheme.from_any(classes)
    base = {c: PAPER_COLORS.get(c, _EXTRA_COLORS[i % len(_EXTRA_COLORS)]) for i, c in enumerate(scheme.codes)}
    if not output or scheme.output_map is None:
        return base
    out: dict[int, str] = {}
    for src, dst in scheme.output_map.items():
        out.setdefault(dst, base[src])
    return out


def _new_ax(ax, figsize=(6, 6)):
    plt = _plt()
    if ax is None:
        _, ax = plt.subplots(figsize=figsize)
    return ax


def _legend(ax, codes, names, colors, loc="lower center", outside=True):
    plt = _plt()
    handles = [plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=colors[c], markersize=8, linestyle="None")
               for c in codes]
    labels = [f"{c}: {names[c].replace('_', ' ').capitalize()}" for c in codes]
    kw = {"bbox_to_anchor": (0.5, -0.12), "loc": "upper center"} if outside else {"loc": loc}
    ax.legend(handles, labels, ncol=min(len(labels), 3), frameon=False, fontsize=9, **kw)


def _style(ax, title, xlabel, ylabel):
    ax.set_title(title, fontsize=11)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    for axis in ("x", "y"):  # plain coordinates on numeric axes (categorical axes have no such formatter)
        try:
            ax.ticklabel_format(axis=axis, useOffset=False, style="plain")
        except AttributeError:
            pass
    ax.tick_params(labelsize=8)


def _sample(n, max_points, seed):
    if max_points is None or n <= max_points:
        return np.arange(n)
    return np.sort(np.random.default_rng(seed).choice(n, size=max_points, replace=False))


def labels(labels, classes=None, class_map=None, colors: dict | None = None, ax=None, title="Label raster"):
    """Show a label raster (a :class:`~fvc3d.io.LabelRaster` or a GeoTIFF path) in class colours."""
    from matplotlib.colors import BoundaryNorm, ListedColormap

    scheme = ClassScheme.from_any(classes)
    raster = labels if isinstance(labels, LabelRaster) else read_labels(labels, class_map=class_map, classes=scheme)
    colors = colors or class_colors(scheme)
    codes = [c for c in scheme.codes if np.any(raster.data == c)]
    index = np.full(raster.shape, np.nan)
    for i, c in enumerate(codes):
        index[raster.data == c] = i
    ax = _new_ax(ax)
    left, bottom, right, top = raster.bounds
    ax.imshow(index, extent=(left, right, bottom, top), cmap=ListedColormap([colors[c] for c in codes]),
              norm=BoundaryNorm(np.arange(len(codes) + 1) - 0.5, len(codes)), interpolation="nearest")
    ax.set_aspect("equal")
    _style(ax, title, "X (m)", "Y (m)")
    _legend(ax, codes, scheme.classes, colors)
    return ax


def points(result, view: str = "top", color_by: str = "class", labelled_only: bool = True,
           max_points: int | None = 200_000, seed: int = 42, colors: dict | None = None,
           point_size: float = 0.5, ax=None, title: str | None = None):
    """Scatter the labelled points of a transfer result.

    Parameters
    ----------
    result:
        A :class:`~fvc3d.pipeline.TransferResult`.
    view:
        ``"top"`` (X, Y) or ``"side"`` (the longer horizontal axis against ``Z_norm``).
    color_by:
        ``"class"`` (output classes), ``"treeID"`` (crowns) or ``"z_norm"`` (height).
    labelled_only:
        Show only points with a class (default), or every point.
    max_points:
        Random subsample for speed (``None`` for all points).
    """
    if view not in ("top", "side"):
        raise ValueError('view must be "top" or "side"')
    if color_by not in ("class", "treeID", "z_norm"):
        raise ValueError('color_by must be "class", "treeID" or "z_norm"')
    xyz, cls, ids, zn = result.points, result.classification, result.tree_id, result.z_norm
    keep = np.flatnonzero(cls != 0) if labelled_only else np.arange(len(cls))
    if color_by == "treeID":
        keep = keep[ids[keep] > 0]
    keep = keep[_sample(len(keep), max_points, seed)]
    x, y = xyz[keep, 0], xyz[keep, 1]
    if view == "top":
        h, v, xlabel, ylabel = x, y, "X (m)", "Y (m)"
    else:
        wide_x = np.ptp(x) >= np.ptp(y) if len(x) else True
        h, xlabel = (x, "X (m)") if wide_x else (y, "Y (m)")
        v, ylabel = zn[keep], "Z_norm (m)"
    ax = _new_ax(ax, figsize=(6, 6) if view == "top" else (8, 3.5))
    kw = {"s": point_size, "linewidths": 0, "alpha": 0.8, "rasterized": True}
    scheme = result.classes
    if color_by == "class":
        colors = colors or class_colors(scheme, output=True)
        names = scheme.final_classes
        codes = [c for c in names if np.any(cls[keep] == c)]
        ax.scatter(h, v, c=[colors.get(int(c), "#8f8f8f") for c in cls[keep]], **kw)
        _legend(ax, codes, names, colors)
    elif color_by == "treeID":
        tid = ids[keep]
        n = len(np.unique(tid))
        if n <= 1:
            ax.scatter(h, v, c=SINGLE_CROWN_COLOR, **kw)
        else:
            _, order = np.unique(tid, return_inverse=True)  # crown IDs -> 0..n-1, adjacent IDs get distant colours
            ax.scatter(h, v, c=(order * 0.618034) % 1.0, cmap="nipy_spectral", vmin=0, vmax=1, **kw)
        ax.text(0.02, 0.98, f"{n} crowns", transform=ax.transAxes, va="top", fontsize=9,
                bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": "#b7c1c9", "alpha": 0.9})
    else:
        sc = ax.scatter(h, v, c=zn[keep], cmap="viridis", **kw)
        ax.figure.colorbar(sc, ax=ax, label="Z_norm (m)", shrink=0.8)
    if view == "top":
        ax.set_aspect("equal")
    default = {"class": "Labelled points", "treeID": "Crowns (treeID)", "z_norm": "Height above ground"}[color_by]
    _style(ax, title or f"{default}: {view} view", xlabel, ylabel)
    return ax


def crowns(result, **kwargs):
    """Top view of the crown points, one colour per ``treeID``."""
    kwargs.setdefault("title", f"{result.classes.name(result.classes.crown_class).capitalize()} crowns (treeID)"
                      if result.classes.crown_class is not None else "Crowns (treeID)")
    return points(result, view="top", color_by="treeID", **kwargs)


def heights(result, colors: dict | None = None, ax=None, title="Height plausibility of refined classes"):
    """``Z_norm`` per class (box plots) against each class's configured height range (shaded).

    Uses the method's own classes, which the configured ranges belong to.
    """
    scheme = result.classes
    cls = result.classification_method if result.classification_method is not None else result.classification
    colors = colors or class_colors(scheme)
    codes = [c for c in scheme.codes if np.any(cls == c)]
    ax = _new_ax(ax, figsize=(1.6 * max(len(codes), 2) + 1.5, 4))
    data = [result.z_norm[cls == c] for c in codes]
    for i, c in enumerate(codes):
        lo, hi = scheme.height_ranges[c]
        ax.fill_between([i + 0.6, i + 1.4], lo, hi, color=colors[c], alpha=0.25, linewidth=0)
    bp = ax.boxplot(data, widths=0.5, showfliers=False, patch_artist=True, medianprops={"color": "#251b36"})
    for patch, c in zip(bp["boxes"], codes):
        patch.set_facecolor(colors[c])
        patch.set_edgecolor("#251b36")
    ax.set_xticks(range(1, len(codes) + 1), [scheme.name(c).replace("_", " ").capitalize() for c in codes])
    from matplotlib.patches import Patch
    ax.legend([Patch(facecolor="#9aa0a6", alpha=0.35)], ["Configured height range"], frameon=False, fontsize=9,
              loc="upper left")
    ax.grid(axis="y", alpha=0.3)
    _style(ax, title, "", "Z_norm (m)")
    return ax


def alignment(result, canopy_height: float = 1.0, ax=None, title="Alignment: labels vs LiDAR canopy"):
    """Overlay of crown-class pixels (labels) and LiDAR canopy (``Z_norm`` ≥ ``canopy_height``).

    Agreement shows where both coincide; a systematic band of one colour on one
    side of the crowns suggests a horizontal offset. Uses the same 5 cm (or
    coarser) grid as :func:`~fvc3d.registration.check_alignment`.
    """
    from matplotlib.colors import ListedColormap
    from .registration import _block_any, _block_factor, _point_blocks

    raster = result.labels
    crown = result.classes.crown_class
    f = _block_factor(raster)
    target = raster.data == crown if crown is not None else raster.data > 0
    lab = _block_any(target, f)
    lid, _ = _point_blocks(raster, result.points[:, 0], result.points[:, 1], f, keep=result.z_norm >= canopy_height)
    img = np.full(lab.shape, np.nan)
    img[lab & ~lid], img[~lab & lid], img[lab & lid] = 0, 1, 2
    colors = [PAPER_COLORS.get(crown, "#1f78b4"), "#fc8d62", "#555555"]
    ax = _new_ax(ax)
    left, bottom, right, top = raster.bounds
    ax.imshow(img, extent=(left, right, bottom, top), cmap=ListedColormap(colors), vmin=0, vmax=2,
              interpolation="nearest")
    ax.set_aspect("equal")
    plt = _plt()
    handles = [plt.Line2D([0], [0], marker="s", color="w", markerfacecolor=c, markersize=9, linestyle="None")
               for c in colors]
    name = result.classes.name(crown) if crown is not None else "labelled"
    ax.legend(handles, [f"{name.capitalize()} label only", f"LiDAR ≥ {canopy_height:g} m only", "Both"],
              bbox_to_anchor=(0.5, -0.12), loc="upper center", ncol=3, frameon=False, fontsize=9)
    rep = result.alignment
    if rep is not None and rep.shift is not None:
        ax.text(0.02, 0.98, f"offset {rep.shift:.2f} m", transform=ax.transAxes, va="top", fontsize=9,
                bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": "#b7c1c9", "alpha": 0.9})
    _style(ax, title, "X (m)", "Y (m)")
    return ax


def summary(result, max_points: int | None = 200_000):
    """Four-panel overview: label raster, labelled points (top and side views) and class heights.

    Returns the Matplotlib figure.
    """
    plt = _plt()
    fig = plt.figure(figsize=(13, 11), layout="constrained")
    grid = fig.add_gridspec(2, 2, height_ratios=[1.25, 1])
    labels(result.labels, classes=result.classes, ax=fig.add_subplot(grid[0, 0]))
    points(result, view="top", max_points=max_points, ax=fig.add_subplot(grid[0, 1]))
    points(result, view="side", max_points=max_points, ax=fig.add_subplot(grid[1, 0]))
    heights(result, ax=fig.add_subplot(grid[1, 1]))
    m = result.metrics
    crowns_txt = (f" · {m['final_treeids']} {result.classes.name(result.classes.crown_class)} crowns"
                  if result.classes.crown_class is not None else "")
    fig.suptitle(f"{m.get('tile') or 'Tile'}: {m['final_veg_points']:,} labelled points{crowns_txt}", fontsize=13)
    return fig
