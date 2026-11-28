"""One-call label transfer: ``transfer_labels(labels, point_cloud, output)``."""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Mapping

import numpy as np

from .classes import ClassScheme
from .io import LabelRaster, PointCloud, read_labels, read_point_cloud, write_point_cloud
from .label_transfer import drape_labels, resolve_z_norm, voxel_transfer
from .quality_control import QualityReport, quality_report
from .refinement import (
    auto_crown_params,
    class_height_ranges,
    Crowns,
    consolidate,
    describe_ranges,
    detect_crowns,
    mallee_min_height,
    match_heights,
    polar_refine,
)
from .registration import AlignmentError, AlignmentReport, check_alignment

log = logging.getLogger("fvc3d")


@dataclass
class TransferConfig:
    """Method parameters. Defaults are those used in the paper.

    Names match ``fvc3dstruct.DEFAULT_CONFIG`` so settings carry over. The one
    difference: crown detection defaults to ``"auto"``, which is how the
    published label-transfer outputs were made. ``fvc3dstruct.DEFAULT_CONFIG``
    lists 1.5 m / 1.0 instead.
    """

    #: Stage 2 voxel size (m).
    voxel_size_xy: float = 0.10
    voxel_size_z: float = 0.10
    #: Stage 1: XY radius (m) for matching a labelled point to a height.
    height_match_radius: float = 0.30
    #: Stage 3: crown grid cell (m), peak spacing (m or "auto"), smoothing (px or "auto").
    #: "auto" estimates both from canopy point spacing (0.80 m / 1.33 px on DD0001).
    crown_grid_res: float = 0.10
    min_tree_dist: float | str = "auto"
    smoothing_sigma: float | str = "auto"
    #: Stage 3/5: override the derived minimum mallee crown height (m).
    mallee_min_znorm: float | None = None
    #: Stage 4: polar bin size and update thresholds.
    r_bin: float = 0.10
    theta_bin_deg: float = 10.0
    min_conf: float = 0.3
    min_flip_ratio: float = 1.5
    height_prior_out_of_range_weight: float = 0.1
    #: Stage 5: height rules (m).
    merge_65_to_68_znorm: float = 0.5
    promote_unlabelled_znorm: float = 1.3
    mallee_force_min_znorm: float | None = None
    #: Output: write every non-zero treeID as 1.
    collapse_treeid_to_one: bool = False
    #: True reproduces fvc3dstruct 1.0.0 exactly, including its handling of
    #: points beyond the crown grid: they take the nearest edge crown's id and
    #: can be turned to mallee by the crown vote. False gives them no crown.
    legacy_crown_clamp: bool = False

    @classmethod
    def from_any(cls, value) -> "TransferConfig":
        """Build from ``None``, a ``TransferConfig``, a dict, or a YAML file path."""
        if value is None:
            return cls()
        if isinstance(value, cls):
            return value
        if isinstance(value, (str, Path)):
            import yaml

            with open(value) as fh:
                value = yaml.safe_load(fh) or {}
        if not isinstance(value, Mapping):
            raise TypeError(f"config must be a TransferConfig, dict or YAML path, not {type(value).__name__}")
        known = {f.name for f in fields(cls)}
        unknown = sorted(set(value) - known)
        if unknown:
            raise ValueError(f"Unknown config keys {unknown}; valid keys: {sorted(known)}")
        return cls(**value)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class FusionResult:
    """Per-point outputs of :func:`fuse_labels`, aligned with the target points.

    ``classification`` is in the scheme's output classes (e.g. ground/wood/leaf
    for ``"FOR-instanceV2"``); the other label arrays are in the method's classes.
    """

    classification: np.ndarray
    tree_id: np.ndarray
    confidence: np.ndarray
    classification_voxel: np.ndarray
    classification_refined: np.ndarray
    height_ranges: dict
    metrics: dict
    classification_method: np.ndarray | None = None


def fuse_labels(
    target_xyz: np.ndarray,
    target_z_norm: np.ndarray,
    source_xyz: np.ndarray,
    source_classes: np.ndarray,
    config: TransferConfig | Mapping | None = None,
    classes: ClassScheme | Mapping | str | Path | None = None,
) -> FusionResult:
    """Run Stages 1–5 on arrays.

    ``target_*`` is the cloud to label (with ``Z_norm``); ``source_*`` is the
    labelled cloud, coded with the class scheme ``classes`` (default: the
    paper's 64–68). Use this when your data is already in memory.
    """
    cfg = TransferConfig.from_any(config)
    scheme = ClassScheme.from_any(classes)
    crown = scheme.crown_class
    tx, ty, tz = (np.asarray(target_xyz, dtype=float)[:, i] for i in range(3))
    sx, sy, sz = (np.asarray(source_xyz, dtype=float)[:, i] for i in range(3))
    zn = np.asarray(target_z_norm, dtype=float)
    s_cls = np.asarray(source_classes)
    n = len(tx)

    # Stage 1: class height ranges
    heights = match_heights(np.column_stack((sx, sy)), np.column_stack((tx, ty)), zn, cfg.height_match_radius)
    ranges = class_height_ranges(heights, s_cls, defaults=scheme.height_ranges)
    if cfg.mallee_min_znorm is not None:
        mallee_min = float(cfg.mallee_min_znorm)
    else:
        mallee_min = mallee_min_height(ranges, crown) if crown is not None else 0.0
    log.info("Stage 1 height ranges: %s; crown min %.2f m", describe_ranges(ranges, scheme.classes), mallee_min)

    # Stage 2: voxel transfer
    voxel_cls, vstats = voxel_transfer(
        np.column_stack((tx, ty, tz)), np.column_stack((sx, sy, sz)), s_cls, cfg.voxel_size_xy, cfg.voxel_size_z,
        codes=scheme.codes,
    )
    log.info("Stage 2 voxel transfer: %d/%d points labelled", vstats["labeled_points"], n)

    # Stage 3: crowns (skipped when the scheme has no crown class)
    mt, sg = cfg.min_tree_dist, cfg.smoothing_sigma
    if crown is None:
        mt = float("nan") if mt == "auto" else mt
        sg = float("nan") if sg == "auto" else sg
    elif mt == "auto" or sg == "auto":
        is_m = s_cls == crown
        tall = zn >= mallee_min
        est_mt, est_sg = auto_crown_params(
            np.column_stack((sx[is_m], sy[is_m])), np.column_stack((tx[tall], ty[tall])), cfg.crown_grid_res
        )
        mt = est_mt if mt == "auto" else mt
        sg = est_sg if sg == "auto" else sg
    if crown is None:
        crowns = Crowns(np.zeros((1, 1), dtype=int), np.array([]), 0.0, 0.0, cfg.crown_grid_res, "no crown class")
    else:
        crowns = detect_crowns(
            sx, sy, sz, s_cls, tx, ty, zn,
            grid_res=cfg.crown_grid_res, min_tree_dist=float(mt), smoothing_sigma=float(sg),
            fallback_min_height=mallee_min, crown_class=crown,
        )
    crowns.clamp_outside = cfg.legacy_crown_clamp
    crown_ids = crowns.lookup(tx, ty).astype(np.uint16)
    log.info("Stage 3 crowns: %d (%s)", crowns.count, crowns.source)

    # Stage 4: polar refinement
    refined, confidence = polar_refine(
        crowns, tx, ty, zn, voxel_cls, sx, sy, s_cls, ranges,
        r_bin=cfg.r_bin, theta_bin_deg=cfg.theta_bin_deg, min_conf=cfg.min_conf,
        min_flip_ratio=cfg.min_flip_ratio, out_of_range_weight=cfg.height_prior_out_of_range_weight,
        codes=scheme.codes,
    )
    changed = int(np.count_nonzero(refined != voxel_cls))
    evaluated = confidence[np.isfinite(confidence)]
    log.info("Stage 4 polar refinement: %d labels changed", changed)

    # Stage 5: consolidation
    final, tree_id, cstats = consolidate(
        refined, zn, crown_ids, mallee_min,
        merge_65_to_68_znorm=cfg.merge_65_to_68_znorm,
        promote_unlabelled_znorm=cfg.promote_unlabelled_znorm,
        mallee_force_min_znorm=cfg.mallee_force_min_znorm,
        collapse_treeid_to_one=cfg.collapse_treeid_to_one,
        crown_class=crown,
        merge=scheme.merge,
        promote_unlabelled=scheme.promote_unlabelled,
    )
    output = scheme.to_output(final).astype(np.uint8)
    n_veg = int(np.count_nonzero(output))
    metrics = {
        "voxel_size_xy": float(cfg.voxel_size_xy),
        "voxel_size_z": float(cfg.voxel_size_z),
        "z_points": n,
        "structure_points": int(len(sx)),
        "mallee_min_znorm": float(mallee_min),
        "stage2_nonzero_structure_voxels": vstats["nonzero_structure_voxels"],
        "stage2_labeled_points": vstats["labeled_points"],
        "stage2_labeled_pct": vstats["labeled_pct"],
        "stage3_min_tree_dist": float(mt),
        "stage3_smoothing_sigma": float(sg),
        "stage3_watershed_crowns": crowns.count,
        "stage3_points_in_crowns": int(np.count_nonzero(crown_ids)),
        "stage4_changed_points": changed,
        "stage4_changed_pct": 100.0 * changed / n if n else 0.0,
        "stage4_mean_confidence": float(np.mean(evaluated)) if len(evaluated) else float("nan"),
        "stage4_low_confidence_pct": (
            100.0 * float(np.count_nonzero(evaluated < cfg.min_conf)) / len(evaluated) if len(evaluated) else 0.0
        ),
        **cstats,
        "final_veg_points": n_veg,
        "final_veg_pct": 100.0 * n_veg / n if n else 0.0,
        "final_treeids": int(len(np.unique(tree_id[tree_id > 0]))),
    }
    return FusionResult(output, tree_id, confidence, voxel_cls, refined, ranges, metrics, classification_method=final)


@dataclass
class TransferResult:
    """What :func:`transfer_labels` returns. Arrays align with the input point cloud."""

    classification: np.ndarray
    tree_id: np.ndarray
    z_norm: np.ndarray
    confidence: np.ndarray
    alignment: AlignmentReport | None
    quality: QualityReport
    metrics: dict
    height_ranges: dict
    config: TransferConfig
    output: Path | None = None
    z_norm_source: str = ""
    warnings: list[str] = field(default_factory=list)
    classes: ClassScheme = field(default_factory=lambda: ClassScheme.from_any(None))
    #: x, y, z of every input point (for plotting), shape (n, 3).
    points: np.ndarray | None = field(default=None, repr=False)
    #: The label raster used, in the scheme's own codes.
    labels: LabelRaster | None = field(default=None, repr=False)
    #: Labels in the scheme's own classes, before any output relabelling
    #: (equal to ``classification`` unless the scheme sets ``output_map``).
    classification_method: np.ndarray | None = field(default=None, repr=False)

    @property
    def vegetation(self) -> np.ndarray:
        """Boolean mask of points with a structure label."""
        return self.classification != 0

    @property
    def class_counts(self) -> dict[str, int]:
        return {name: int(np.count_nonzero(self.classification == code)) for code, name in self.classes.final_classes.items()}

    def summary(self) -> str:
        m = self.metrics
        lines = [
            f"Labelled {m['final_veg_points']:,} of {m['z_points']:,} points ({m['final_veg_pct']:.1f}%)"
            + (f"; {m['final_treeids']} {self.classes.final_classes[self.classes.final_crown_class]} crowns"
               if self.classes.final_crown_class is not None else ""),
            "  " + ", ".join(f"{k} {v:,}" for k, v in self.class_counts.items()),
            f"  Z_norm from: {self.z_norm_source}",
        ]
        if self.output:
            lines.append(f"  written to: {self.output}")
        if self.alignment is not None:
            lines.append(self.alignment.summary())
        lines.append(self.quality.summary())
        return "\n".join(lines)

    def __repr__(self) -> str:
        return f"<TransferResult {self.metrics['final_veg_points']:,} labelled points, output={self.output}>"


def transfer_labels(
    labels,
    point_cloud,
    output=None,
    *,
    validate_alignment: bool = True,
    max_shift: float = 0.50,
    class_map: Mapping | str | None = None,
    ground=None,
    config: TransferConfig | Mapping | str | Path | None = None,
    classes: ClassScheme | Mapping | str | Path | None = None,
    keep: str = "vegetation",
    overwrite: bool = False,
) -> TransferResult:
    """Transfer 2D growth-form labels onto a LiDAR point cloud.

    Parameters
    ----------
    labels:
        Label GeoTIFF path (or a :class:`~fvc3d.io.LabelRaster`). With the
        default classes, values 64–68 are forb, woody debris, hummock, shrub
        and mallee; use ``class_map`` for other codes.
    point_cloud:
        LAS/LAZ path (or a :class:`~fvc3d.io.PointCloud`) in the same CRS as
        the labels, already co-registered to the imagery. Needs a ``Z_norm``
        dimension, ground points (class 2), or ``ground``.
    output:
        Optional ``.las`` or ``.laz`` path. Points get ``classification``
        (the scheme's codes, 0 = none), ``Z_norm`` and ``treeID`` (crown instance).
    validate_alignment:
        Check CRS, overlap and horizontal offset before transferring;
        raise :class:`~fvc3d.registration.AlignmentError` on failure.
    max_shift:
        Largest canopy offset (m) accepted by the alignment check.
    class_map:
        ``{raster_value: class_name_or_code}``, e.g. ``{1: "shrub", 2: "mallee"}``,
        or ``"paper"`` for the published rasters' 0–4 coding.
    ground:
        Ground-only LAS/LAZ used to compute ``Z_norm``.
    config:
        :class:`TransferConfig`, dict or YAML path overriding method parameters.
    classes:
        Class scheme: ``"paper"`` (default, the five growth forms), a
        :class:`~fvc3d.classes.ClassScheme`, a dict, or a YAML path
        (see ``configs/classes.yaml``).
    keep:
        ``"vegetation"`` writes labelled points only; ``"all"`` writes every point.
    overwrite:
        Allow replacing an existing ``output``.
    """
    if keep not in ("vegetation", "all"):
        raise ValueError('keep must be "vegetation" or "all"')
    if output is not None:
        output = Path(output)
        if output.exists() and not overwrite:
            raise FileExistsError(f"{output} exists; pass overwrite=True to replace it")
    cfg = TransferConfig.from_any(config)
    scheme = ClassScheme.from_any(classes)

    raster = labels if isinstance(labels, LabelRaster) else read_labels(labels, class_map=class_map, classes=scheme)
    cloud = point_cloud if isinstance(point_cloud, PointCloud) else read_point_cloud(point_cloud)
    ground_cloud = None if ground is None else (ground if isinstance(ground, PointCloud) else read_point_cloud(ground))
    if len(cloud) == 0:
        raise ValueError("The point cloud is empty")

    z_norm, z_source = resolve_z_norm(cloud, ground_cloud)

    alignment = None
    if validate_alignment:
        alignment = check_alignment(raster, cloud, z_norm=z_norm, max_shift=max_shift, crown_class=scheme.crown_class)
        if not alignment.ok:
            raise AlignmentError(alignment.summary())
        log.info(alignment.summary())

    # Labelled source points: the LiDAR points under labelled pixels.
    draped = drape_labels(raster, cloud.x, cloud.y)
    src = draped > 0
    if not np.any(src):
        raise ValueError("No LiDAR points fall on labelled pixels")
    xyz = np.column_stack((cloud.x, cloud.y, cloud.z))
    fused = fuse_labels(xyz, z_norm, xyz[src], draped[src], cfg, classes=scheme)

    veg = fused.classification != 0
    quality = quality_report(
        xyz, fused.classification_voxel, fused.classification_method, z_norm,
        voxel_xy=cfg.voxel_size_xy, voxel_z=cfg.voxel_size_z,
        watershed_crowns=fused.metrics["stage3_watershed_crowns"],
        final_treeids=fused.metrics["final_treeids"],
        classes=scheme,
    )

    written = None
    if output is not None:
        written = write_point_cloud(
            cloud, output, fused.classification, z_norm, fused.tree_id,
            keep=veg if keep == "vegetation" else None,
        )

    warnings = list(quality.warnings) + ([] if alignment is None else list(alignment.notes))
    for w in warnings:
        log.warning(w)

    return TransferResult(
        classification=fused.classification,
        tree_id=fused.tree_id,
        z_norm=z_norm,
        confidence=fused.confidence,
        alignment=alignment,
        quality=quality,
        metrics={"tile": cloud.path.stem if cloud.path else "", **fused.metrics},
        height_ranges=fused.height_ranges,
        config=cfg,
        output=written,
        z_norm_source=z_source,
        warnings=warnings,
        classes=scheme,
        points=xyz,
        labels=raster,
        classification_method=fused.classification_method,
    )
