"""Structure-aware refinement of transferred labels (Stages 1 and 3–5).

* Stage 1 :func:`class_height_ranges` — data-driven ``Z_norm`` range per class.
* Stage 3 :func:`detect_crowns` — watershed mallee crowns and ``treeID`` grid.
* Stage 4 :func:`polar_refine` — crown-centric polar bins with height priors.
* Stage 5 :func:`consolidate` — height rules and crown majority vote.

All functions take plain NumPy arrays. Class-specific rules (which class
forms crowns, which classes merge) are arguments, defaulting to the paper's
classes; with those defaults behaviour matches ``fvc3dstruct`` 1.0.0.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree

from .classes import PAPER_CLASSES
from .io import CLASS_NAMES, STRUCTURE_CODES

log = logging.getLogger("fvc3d")

#: The paper's fallback ``Z_norm`` range (m) for a class with fewer than 20 matched points.
DEFAULT_HEIGHT_RANGES: dict[int, tuple[float, float]] = dict(PAPER_CLASSES.height_ranges)

MALLEE = 68
WOODY_DEBRIS = 65


# --------------------------------------------------------------------------
# Stage 1: class height ranges
# --------------------------------------------------------------------------
def match_heights(source_xy: np.ndarray, target_xy: np.ndarray, target_z_norm: np.ndarray,
                  max_dist: float = 0.30) -> np.ndarray:
    """``Z_norm`` of the XY-nearest target point for each source point (NaN beyond ``max_dist``)."""
    tree = cKDTree(target_xy)
    dist, idx = tree.query(source_xy, k=1, distance_upper_bound=max_dist)
    out = np.full(len(source_xy), np.nan)
    ok = np.isfinite(dist) & (idx < len(target_z_norm))
    out[ok] = np.asarray(target_z_norm, dtype=float)[idx[ok]]
    return out


def class_height_ranges(heights: np.ndarray, classes: np.ndarray,
                        low: float = 15.0, high: float = 85.0,
                        defaults: dict | None = None) -> dict[int, tuple[float, float]]:
    """Per-class (low, high) percentile range of ``heights``, after dropping the 5% tails.

    Classes are the keys of ``defaults`` (the fallback ranges; default: the paper's).
    """
    defaults = DEFAULT_HEIGHT_RANGES if defaults is None else defaults
    ranges = {}
    for cls in defaults:
        mask = (classes == cls) & np.isfinite(heights)
        if int(mask.sum()) < 20:
            ranges[cls] = defaults[cls]
            continue
        h = heights[mask]
        p5, p95 = np.percentile(h, [5, 95])
        clipped = h[(h >= p5) & (h <= p95)]
        if len(clipped) >= 10:
            h = clipped
        lo, hi = np.percentile(h, [low, high])
        ranges[cls] = (max(0, float(lo)), float(hi))
    return ranges


def mallee_min_height(ranges: dict[int, tuple[float, float]], crown_class: int = MALLEE) -> float:
    """Minimum crown height for ``treeID``: 0.7 × the crown class's low bound, at least 0.6 m."""
    fallback = DEFAULT_HEIGHT_RANGES.get(crown_class, (0.0, 0.0))
    return float(max(0.6, ranges.get(crown_class, fallback)[0] * 0.7))


# --------------------------------------------------------------------------
# Stage 3: crown detection
# --------------------------------------------------------------------------
@dataclass
class Crowns:
    """Watershed crown labels on a regular grid. ``centroids[i]`` is crown ``i``'s (x, y)."""

    grid: np.ndarray
    centroids: np.ndarray
    min_x: float
    min_y: float
    grid_res: float
    source: str = ""
    #: If True, points beyond the grid take the id of the nearest edge cell
    #: (fvc3dstruct 1.0.0 behaviour). If False they get 0.
    clamp_outside: bool = False

    @property
    def count(self) -> int:
        return int(self.grid.max()) if self.grid.size else 0

    def lookup(self, x, y) -> np.ndarray:
        """Crown id under each (x, y); 0 outside crowns."""
        h, w = self.grid.shape
        ix = ((np.asarray(x) - self.min_x) / self.grid_res).astype(int)
        iy = ((np.asarray(y) - self.min_y) / self.grid_res).astype(int)
        ids = self.grid[np.clip(iy, 0, h - 1), np.clip(ix, 0, w - 1)]
        if not self.clamp_outside:
            # int() truncates toward zero, so test the scaled coordinate, not ix/iy.
            fx = (np.asarray(x) - self.min_x) / self.grid_res
            fy = (np.asarray(y) - self.min_y) / self.grid_res
            ids = np.where((fx < 0) | (fx >= w) | (fy < 0) | (fy >= h), 0, ids)
        return ids


def _empty_crowns(x, y, grid_res, source) -> Crowns:
    min_x, max_x, min_y, max_y = float(np.min(x)), float(np.max(x)), float(np.min(y)), float(np.max(y))
    w = int(np.ceil((max_x - min_x) / grid_res)) + 1
    h = int(np.ceil((max_y - min_y) / grid_res)) + 1
    return Crowns(np.zeros((h, w), dtype=int), np.array([]), min_x, min_y, grid_res, source)


def auto_crown_params(source_xy_mallee: np.ndarray, fallback_xy: np.ndarray,
                      grid_res: float = 0.10, seed: int = 0) -> tuple[float, float]:
    """Estimate (min_tree_dist m, smoothing_sigma px) from canopy point spacing."""
    pts = source_xy_mallee if len(source_xy_mallee) else fallback_xy
    if len(pts) < 100:
        return 1.5, 1.0
    if len(pts) > 50_000:
        pts = pts[np.random.default_rng(seed).choice(len(pts), 50_000, replace=False)]
    dists, _ = cKDTree(pts).query(pts, k=2, workers=-1)
    min_tree_dist = float(np.clip(np.percentile(dists[:, 1], 60), 0.8, 3.0))
    sigma = float(np.clip((min_tree_dist / grid_res) / 6.0, 0.8, 2.5))
    return min_tree_dist, sigma


def detect_crowns(
    source_x, source_y, source_z, source_classes,
    fallback_x=None, fallback_y=None, fallback_height=None,
    grid_res: float = 0.10,
    min_tree_dist: float = 1.5,
    smoothing_sigma: float = 1.0,
    fallback_min_height: float = 1.0,
    crown_class: int = MALLEE,
) -> Crowns:
    """Detect mallee crowns by marker-controlled watershed on a canopy surface.

    The surface is built from mallee-labelled source points (their ``z``). If
    there are none, it uses fallback points (target cloud ``x, y, Z_norm``)
    above ``fallback_min_height``, relaxing the threshold until 100 points
    qualify. Every connected canopy blob gets at least one marker.
    """
    from skimage.feature import peak_local_max
    from skimage.measure import label as blob_label, regionprops
    from skimage.segmentation import watershed

    source_x, source_y = np.asarray(source_x), np.asarray(source_y)
    is_mallee = np.asarray(source_classes) == crown_class
    source = "structure"

    if np.any(is_mallee):
        use_x, use_y = source_x[is_mallee], source_y[is_mallee]
        use_h = np.asarray(source_z, dtype=float)[is_mallee]
    else:
        if fallback_height is None:
            return _empty_crowns(source_x, source_y, grid_res, "none")
        fb_x, fb_y = np.asarray(fallback_x), np.asarray(fallback_y)
        fb_h = np.asarray(fallback_height, dtype=float)
        if (float(np.nanmax(fb_h)) if fb_h.size else 0.0) < fallback_min_height:
            return _empty_crowns(fb_x, fb_y, grid_res, "none")
        thresholds = [fallback_min_height, max(0.8 * fallback_min_height, 0.6),
                      max(0.6 * fallback_min_height, 0.5), max(0.4 * fallback_min_height, 0.4), 0.3]
        chosen, chosen_th = None, fallback_min_height
        for th in thresholds:
            if np.count_nonzero(fb_h >= th) >= 100:
                chosen, chosen_th = fb_h >= th, th
                break
        if chosen is None:
            chosen, chosen_th = fb_h >= 0.1, 0.1
            if not np.any(chosen):
                return _empty_crowns(fb_x, fb_y, grid_res, "none")
        use_x, use_y, use_h = fb_x[chosen], fb_y[chosen], fb_h[chosen]
        source = f"Z_norm fallback (>= {chosen_th:.2f} m)"

    min_x, max_x = float(np.min(use_x)), float(np.max(use_x))
    min_y, max_y = float(np.min(use_y)), float(np.max(use_y))
    w = max(1, int(np.ceil((max_x - min_x) / grid_res)) + 1)
    h = max(1, int(np.ceil((max_y - min_y) / grid_res)) + 1)
    mx = np.clip(((use_x - min_x) / grid_res).astype(int), 0, w - 1)
    my = np.clip(((use_y - min_y) / grid_res).astype(int), 0, h - 1)

    order = np.argsort(use_h)
    surface = np.zeros((h, w), dtype=np.float32)
    surface[my[order], mx[order]] = use_h[order]  # highest point wins per cell
    canopy = surface > 0
    smoothed = ndimage.gaussian_filter(surface, sigma=smoothing_sigma)

    peaks = peak_local_max(smoothed, min_distance=max(1, int(min_tree_dist / grid_res)),
                           threshold_abs=0.1, labels=canopy)
    is_peak = np.zeros_like(smoothed, dtype=bool)
    if len(peaks):
        is_peak[peaks[:, 0], peaks[:, 1]] = True
    markers = ndimage.label(is_peak)[0]

    blobs = blob_label(canopy)
    peak_coords = np.argwhere(is_peak)
    next_id = markers.max() + 1
    for blob in regionprops(blobs, intensity_image=surface):
        r0, c0, r1, c1 = blob.bbox
        near = peak_coords[(peak_coords[:, 0] >= r0) & (peak_coords[:, 0] < r1)
                           & (peak_coords[:, 1] >= c0) & (peak_coords[:, 1] < c1)]
        if not any(blobs[py, px] == blob.label for py, px in near):
            cy, cx = map(int, blob.weighted_centroid)
            markers[np.clip(cy, 0, h - 1), np.clip(cx, 0, w - 1)] = next_id
            next_id += 1

    grid = watershed(-smoothed, markers, mask=canopy)
    n = int(grid.max())
    if n == 0:
        return Crowns(grid, np.array([]), min_x, min_y, grid_res, source)
    coms = ndimage.center_of_mass(canopy, grid, range(1, n + 1))
    centroids = np.zeros((n + 1, 2))
    for i, (cy, cx) in enumerate(coms):
        centroids[i + 1] = [min_x + cx * grid_res, min_y + cy * grid_res]
    return Crowns(grid, centroids, min_x, min_y, grid_res, source)


# --------------------------------------------------------------------------
# Stage 4: crown-centric polar refinement
# --------------------------------------------------------------------------
def _polar_bins(crowns: Crowns, x, y, r_bin, theta_bin_deg):
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    crown = crowns.lookup(x, y)
    ir = np.full(len(x), -1, dtype=np.int64)
    it = np.full(len(x), -1, dtype=np.int64)
    ok = (crown > 0) & (crown < len(crowns.centroids))
    if np.any(ok):
        c = crowns.centroids[crown[ok]]
        dx, dy = x[ok] - c[:, 0], y[ok] - c[:, 1]
        r = np.sqrt(dx ** 2 + dy ** 2)
        theta = np.mod(np.arctan2(dy, dx), 2 * np.pi)
        ir[ok] = np.floor(r / r_bin).astype(np.int64)
        it[ok] = np.floor(theta / np.deg2rad(theta_bin_deg)).astype(np.int64)
    return crown.astype(np.int64), ir, it


def polar_refine(
    crowns: Crowns,
    target_x, target_y, target_z_norm, classification,
    source_x, source_y, source_classes,
    height_ranges: dict[int, tuple[float, float]],
    r_bin: float = 0.10,
    theta_bin_deg: float = 10.0,
    min_conf: float = 0.3,
    min_flip_ratio: float = 1.5,
    out_of_range_weight: float = 0.1,
    min_samples: int = 3,
    codes=STRUCTURE_CODES,
) -> tuple[np.ndarray, np.ndarray]:
    """Revise labels inside crowns from the class mix of their polar bin.

    Each (crown, r, θ) bin holds counts of source labels. For a target point
    in a bin with at least ``min_samples`` labels, class probabilities are the
    bin's class shares, weighted by 1 if the point's ``Z_norm`` lies in the
    class height range and ``out_of_range_weight`` otherwise, renormalised.
    Unlabelled points take the best class if its probability is at least
    ``min_conf``; labelled points switch only if the best class beats their
    current class by ``min_flip_ratio``. On a tie, the class seen first in
    the bin wins.

    Returns the refined labels and the best-class probability per point (NaN
    where not evaluated).
    """
    classification = np.asarray(classification)
    out = classification.copy()
    confidence = np.full(len(classification), np.nan)
    if crowns.count == 0:
        return out, confidence

    crown_s, ir_s, it_s = _polar_bins(crowns, source_x, source_y, r_bin, theta_bin_deg)
    crown_t, ir_t, it_t = _polar_bins(crowns, target_x, target_y, r_bin, theta_bin_deg)
    source_classes = np.asarray(source_classes)
    codes = np.asarray(codes, dtype=np.int64)
    column = np.full(256, -1, dtype=np.int64)  # class code -> column in the count table
    column[codes] = np.arange(len(codes))
    valid_s = (crown_s > 0) & (ir_s >= 0) & (it_s >= 0) & np.isin(source_classes, codes)
    valid_t = (crown_t > 0) & (ir_t >= 0) & (it_t >= 0)
    if not np.any(valid_s) or not np.any(valid_t):
        return out, confidence

    n_ir = int(max(ir_s.max(), ir_t.max())) + 1
    n_it = int(max(it_s.max(), it_t.max())) + 1

    def encode(c, r, t):
        return (c * n_ir + r) * n_it + t

    s_idx = np.flatnonzero(valid_s)
    keys_s = encode(crown_s[s_idx], ir_s[s_idx], it_s[s_idx])
    bins, inverse = np.unique(keys_s, return_inverse=True)
    col = column[source_classes[s_idx].astype(np.int64)]
    counts = np.zeros((len(bins), len(codes)), dtype=np.int64)
    np.add.at(counts, (inverse, col), 1)
    first_seen = np.full(counts.shape, np.iinfo(np.int64).max)
    np.minimum.at(first_seen, (inverse, col), s_idx)

    t_idx = np.flatnonzero(valid_t)
    keys_t = encode(crown_t[t_idx], ir_t[t_idx], it_t[t_idx])
    pos = np.clip(np.searchsorted(bins, keys_t), 0, len(bins) - 1)
    found = bins[pos] == keys_t
    t_idx, pos = t_idx[found], pos[found]
    total = counts[pos].sum(axis=1)
    enough = total >= min_samples
    t_idx, pos, total = t_idx[enough], pos[enough], total[enough]
    if len(t_idx) == 0:
        return out, confidence

    c = counts[pos]
    zn = np.asarray(target_z_norm, dtype=float)[t_idx]
    lo = np.array([height_ranges[k][0] for k in codes])
    hi = np.array([height_ranges[k][1] for k in codes])
    in_range = (zn[:, None] >= lo) & (zn[:, None] <= hi)
    weights = np.where(in_range, 1.0, out_of_range_weight)
    probs = np.where(c > 0, (c / total[:, None]) * weights, 0.0)

    # Sum in first-seen order so totals match the reference implementation bit for bit.
    seen = first_seen[pos]
    order = np.argsort(seen, axis=1, kind="stable")
    prob_sum = np.zeros(len(t_idx))
    for j in range(order.shape[1]):
        prob_sum = prob_sum + np.take_along_axis(probs, order[:, j:j + 1], axis=1)[:, 0]
    positive = prob_sum > 0
    t_idx, probs, seen, prob_sum = t_idx[positive], probs[positive], seen[positive], prob_sum[positive]
    probs = probs / prob_sum[:, None]

    best_p = probs.max(axis=1)
    tied = (probs == best_p[:, None]) & (seen != np.iinfo(np.int64).max)
    best_col = np.argmin(np.where(tied, seen, np.iinfo(np.int64).max), axis=1)
    best_cls = codes[best_col]
    best_conf = probs[np.arange(len(best_col)), best_col]
    confidence[t_idx] = best_conf

    current = classification[t_idx].astype(np.int64)
    unlabelled = ~np.isin(current, codes)
    cur_col = np.where(unlabelled, 0, column[np.clip(current, 0, 255)])
    current_conf = np.where(unlabelled, 0.0, probs[np.arange(len(cur_col)), cur_col])
    fill = unlabelled & (best_conf >= min_conf)
    flip = ~unlabelled & (best_cls != current) & (best_conf > min_flip_ratio * current_conf)
    change = fill | flip
    out[t_idx[change]] = best_cls[change]
    return out, confidence


# --------------------------------------------------------------------------
# Stage 5: consolidation
# --------------------------------------------------------------------------
def consolidate(
    classification: np.ndarray,
    z_norm: np.ndarray,
    crown_ids: np.ndarray,
    mallee_min: float,
    merge_65_to_68_znorm: float = 0.5,
    promote_unlabelled_znorm: float = 1.3,
    mallee_force_min_znorm: float | None = None,
    collapse_treeid_to_one: bool = False,
    crown_class: int | None = MALLEE,
    merge: dict | None = None,
    promote_unlabelled: bool = True,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Apply the height rules and crown vote; return (labels, treeID, stats).

    With the paper's classes (the defaults):

    1. Woody debris above ``merge_65_to_68_znorm`` becomes mallee; below, unlabelled.
    2. Unlabelled points at or above ``promote_unlabelled_znorm`` become mallee.
    3. A crown with more than 50% mallee points becomes wholly mallee.
    4. Optionally, crown points at or above ``mallee_force_min_znorm`` become mallee.
    5. ``treeID`` is kept only for mallee points at or above ``mallee_min``.

    For other schemes, ``merge`` ({from: to}) replaces rule 1, ``crown_class``
    replaces mallee, and ``crown_class=None`` skips rules 2–5.
    """
    merge = {WOODY_DEBRIS: MALLEE} if merge is None else merge
    cls = np.asarray(classification).copy()
    zn = np.asarray(z_norm, dtype=float)
    ids = np.asarray(crown_ids).astype(np.uint16)

    for frm, to in merge.items():
        source = cls == frm
        cls[source & (zn > merge_65_to_68_znorm)] = to
        cls[source & (zn <= merge_65_to_68_znorm)] = 0

    promote = np.zeros(len(cls), dtype=bool)
    if promote_unlabelled and crown_class is not None:
        promote = (cls == 0) & (zn >= promote_unlabelled_znorm)
        cls[promote] = crown_class

    n_reclassified, shares = 0, np.array([])
    if crown_class is not None and ids.max(initial=0) > 0:
        n_ids = int(ids.max()) + 1
        n_total = np.bincount(ids, minlength=n_ids)
        n_crown = np.bincount(ids, weights=(cls == crown_class), minlength=n_ids)
        occupied = np.arange(n_ids) > 0
        occupied &= n_total > 0
        shares = n_crown[occupied] / n_total[occupied]
        majority = np.zeros(n_ids, dtype=bool)
        majority[np.flatnonzero(occupied)[shares > 0.5]] = True
        convert = majority[ids] & (cls != crown_class)
        n_reclassified = int(np.count_nonzero(convert))
        cls[convert] = crown_class

    n_forced = 0
    if crown_class is not None and mallee_force_min_znorm is not None and ids.max(initial=0) > 0:
        force = (ids > 0) & (zn >= float(mallee_force_min_znorm))
        n_forced = int(np.count_nonzero(force & (cls != crown_class)))
        cls[force] = crown_class

    tree_id = ids.copy()
    if crown_class is None:
        tree_id[:] = 0
    else:
        tree_id[~((cls == crown_class) & (zn >= mallee_min))] = 0
        cls[tree_id > 0] = crown_class
    if collapse_treeid_to_one:
        tree_id[tree_id > 0] = 1

    stats = {
        "stage5_promoted_points": int(np.count_nonzero(promote)),
        "stage5_crowns_voted": int(len(shares)),
        "stage5_crowns_mallee_majority": int(np.count_nonzero(shares > 0.5)),
        "stage5_min_crown_mallee_pct": float(100.0 * shares.min()) if len(shares) else float("nan"),
        "stage5_majority_reclassified_points": n_reclassified,
        "stage5_forced_points": n_forced,
    }
    return cls, tree_id, stats


def describe_ranges(ranges: dict[int, tuple[float, float]], names: dict | None = None) -> str:
    names = CLASS_NAMES if names is None else names
    return ", ".join(f"{names.get(c, c)} {lo:.2f}–{hi:.2f} m" for c, (lo, hi) in ranges.items())
