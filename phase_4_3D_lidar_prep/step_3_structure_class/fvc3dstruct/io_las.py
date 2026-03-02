"""LAS I/O utilities for LAS 1.4 format 6."""

import laspy
import numpy as np
from pathlib import Path


def _clone_las(
    las_in: laspy.LasData,
    keep_mask: np.ndarray | None = None,
) -> laspy.LasData:
    """Create a new LasData object with the same header/points.

    Some laspy versions don't provide `LasData.copy()`. This helper performs a
    safe clone that works across versions.

    If `keep_mask` is provided, clones only the selected subset of points.
    """

    if keep_mask is not None:
        keep_mask = np.asarray(keep_mask, dtype=bool)

    # Prefer header.copy() when available to avoid mutating the template header.
    try:
        hdr = las_in.header.copy()
    except Exception:
        hdr = laspy.LasHeader(
            version=str(las_in.header.version),
            point_format=las_in.header.point_format,
        )
        hdr.scales = las_in.header.scales
        hdr.offsets = las_in.header.offsets

        # Preserve extra dimension definitions when possible.
        for eb in las_in.header.extra_dimensions:
            try:
                hdr.add_extra_dim(eb)
            except Exception:
                pass

    las_out = laspy.LasData(hdr)

    pts = las_in.points if keep_mask is None else las_in.points[keep_mask]
    # Make a best-effort deep copy of the point record.
    if hasattr(pts, "copy"):
        pts = pts.copy()
    las_out.points = pts
    return las_out


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

    # Start from a clone of the base LAS so point formats always match.
    # Important: add extra dims AFTER points exist to avoid laspy format mismatch.
    las_out = _clone_las(las_base)

    if "Z_norm" not in las_out.point_format.dimension_names:
        las_out.add_extra_dim(laspy.ExtraBytesParams(name="Z_norm", type=np.float32))
    if "treeID" not in las_out.point_format.dimension_names:
        las_out.add_extra_dim(laspy.ExtraBytesParams(name="treeID", type=np.uint16))

    # Assign fields
    las_out.classification = cls_arr
    las_out["Z_norm"] = zn_arr
    las_out["treeID"] = ids_arr

    # Write
    las_out.write(str(output_path))
    print(f"   📝 Saved {stage_label}: {output_path.name}")


def write_las_with_fields(
    las_template: laspy.LasData,
    out_path,
    classification,
    zn_values,
    tree_ids=None,
    keep_mask=None,
) -> None:
    """Write LAS 1.4 PF6 with `classification`, `Z_norm`, and optional `treeID`.

    If `keep_mask` is provided, writes only the subset of points where mask is True.
    """

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    las_base = ensure_las14_pf6(las_template)
    n = len(las_base.points)

    cls_arr = np.asarray(classification, dtype=np.uint8)
    zn_arr = np.asarray(zn_values, dtype=np.float32)
    if tree_ids is None:
        ids_arr = np.zeros(n, dtype=np.uint16)
    else:
        ids_arr = np.asarray(tree_ids, dtype=np.uint16)

    if len(cls_arr) != n or len(zn_arr) != n or len(ids_arr) != n:
        raise ValueError("write_las_with_fields(): array length mismatch.")

    if keep_mask is not None:
        keep_mask = np.asarray(keep_mask, dtype=bool)
        if len(keep_mask) != n:
            raise ValueError("write_las_with_fields(): keep_mask length mismatch.")
        cls_arr = cls_arr[keep_mask]
        zn_arr = zn_arr[keep_mask]
        ids_arr = ids_arr[keep_mask]

    # Start from base LAS (PF6) to keep point formats consistent.
    las_out = _clone_las(las_base, keep_mask=keep_mask)

    # Add extra dims AFTER points exist to avoid: "Cannot set points with a different point format"
    if "Z_norm" not in las_out.point_format.dimension_names:
        las_out.add_extra_dim(laspy.ExtraBytesParams(name="Z_norm", type=np.float32))
    if "treeID" not in las_out.point_format.dimension_names:
        las_out.add_extra_dim(laspy.ExtraBytesParams(name="treeID", type=np.uint16))

    las_out.classification = cls_arr
    las_out["Z_norm"] = zn_arr
    las_out["treeID"] = ids_arr

    las_out.write(str(out_path))
