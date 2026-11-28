"""fvc3d — transfer 2D vegetation-structure labels onto LiDAR point clouds.

Quick start::

    from fvc3d import transfer_labels

    result = transfer_labels(
        labels="growth_forms.tif",
        point_cloud="lidar.laz",
        output="labelled_vegetation.laz",
        validate_alignment=True,
    )
    print(result.summary())

Method: Sotomayor et al. (2026), "From 2D labels to 3D structure",
https://doi.org/10.31223/X5948M
"""

from .classes import FORINSTANCEV2_CLASSES, PAPER_CLASSES, ClassScheme
from .io import CLASS_CODES, CLASS_NAMES, PAPER_CLASS_MAP, LabelRaster, PointCloud, read_labels, read_point_cloud, write_point_cloud
from .label_transfer import drape_labels, height_above_ground, voxel_transfer
from .pipeline import FusionResult, TransferConfig, TransferResult, fuse_labels, transfer_labels
from .quality_control import QualityReport, crown_retention, height_plausibility, label_agreement, voxel_label_agreement
from .refinement import DEFAULT_HEIGHT_RANGES, consolidate, detect_crowns, polar_refine
from .registration import AlignmentError, AlignmentReport, check_alignment

__version__ = "0.1.0"

__all__ = [
    "transfer_labels",
    "ClassScheme",
    "PAPER_CLASSES",
    "FORINSTANCEV2_CLASSES",
    "TransferConfig",
    "TransferResult",
    "fuse_labels",
    "FusionResult",
    "read_labels",
    "read_point_cloud",
    "write_point_cloud",
    "LabelRaster",
    "PointCloud",
    "check_alignment",
    "AlignmentReport",
    "AlignmentError",
    "drape_labels",
    "height_above_ground",
    "voxel_transfer",
    "detect_crowns",
    "polar_refine",
    "consolidate",
    "voxel_label_agreement",
    "crown_retention",
    "label_agreement",
    "height_plausibility",
    "QualityReport",
    "DEFAULT_HEIGHT_RANGES",
    "CLASS_NAMES",
    "CLASS_CODES",
    "PAPER_CLASS_MAP",
]
