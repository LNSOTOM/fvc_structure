"""LAS I/O utilities for LAS 1.4 format 6."""

import laspy
import numpy as np
from pathlib import Path


def ensure_las14_pf6(las_in: laspy.LasData) -> laspy.LasData:
    """
    Ensure LAS 1.4 / point format 6, preserving coordinates and extra dims.
    """
    if las_in.header.version == "1.4" and las_in.header.point_format.id >= 6:
        return las_in

    hdr = laspy.LasHeader(version="1.4", point_format=6)
    hdr.scales = las_in.header.scales
    hdr.offsets = las_in.header.offsets

    out = laspy.LasData(hdr)
    out.points = laspy.ScaleAwarePointRecord.zeros(len(las_in.points), hdr.point_format)
    out.X, out.Y, out.Z = las_in.X, las_in.Y, las_in.Z

    # Copy overlapping standard dimensions
    for dim in (set(las_in.point_format.dimension_names) &
                set(out.point_format.dimension_names)):
        if dim not in ("X", "Y", "Z"):
            out[dim] = las_in[dim]

    # Copy extra dimensions
    for eb in las_in.header.extra_dimensions:
        if eb.name not in out.point_format.dimension_names:
            out.add_extra_dim(eb)
        out[eb.name] = las_in[eb.name]

    return out


def read_las(path):
    """Read LAS file."""
    return laspy.read(str(path))


def write_las(las_obj, out_path):
    """Write LAS file."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    las_obj.write(str(out_path))


def write_stage_output(
    las_template,
    classification,
    output_path,
    zn_values,
    stage_label,
    tree_ids=None,
):
    """
    Save a pipeline stage snapshot (e.g., Stage 1, 2, or 3) containing:
      - XYZ from las_template
      - classification
      - Z_norm
      - optional treeID
    
    Always written as LAS 1.4 PF6.
    
    Args:
        las_template: Source LAS for XYZ coordinates
        classification: Classification array
        output_path: Path to write output LAS
        zn_values: Z_norm height values
        stage_label: Descriptive label (e.g., "Stage 1 (Cartesian)")
        tree_ids: Optional treeID array
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Convert to LAS 1.4 PF6 FIRST
    las_base = ensure_las14_pf6(las_template)

    n = len(las_base.points)
    cls_arr = np.asarray(classification, dtype=np.uint8)
    zn_arr = np.asarray(zn_values, dtype=np.float32)
    ids_arr = (
        np.asarray(tree_ids, dtype=np.uint16)
        if tree_ids is not None
        else np.zeros(n, dtype=np.uint16)
    )

    if len(cls_arr) != n or len(zn_arr) != n or len(ids_arr) != n:
        raise ValueError("write_stage_output(): array length mismatch.")

    # Create empty LAS with correct header
    las_out = laspy.LasData(las_base.header)

    # Add extra dimensions BEFORE assigning points
    if "Z_norm" not in las_out.point_format.dimension_names:
        las_out.add_extra_dim(laspy.ExtraBytesParams(name="Z_norm", type=np.float32))

    if "treeID" not in las_out.point_format.dimension_names:
        las_out.add_extra_dim(laspy.ExtraBytesParams(name="treeID", type=np.uint16))

    # Copy points from base, which already holds XYZ
    las_out.points = las_base.points.copy()

    # Assign fields
    las_out.classification = cls_arr
    las_out["Z_norm"] = zn_arr
    las_out["treeID"] = ids_arr

    # Write
    las_out.write(str(output_path))
    print(f"   📝 Saved {stage_label}: {output_path.name}")
