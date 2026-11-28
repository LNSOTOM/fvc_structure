"""Reading and writing label rasters and LAS/LAZ point clouds."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping

import laspy
import numpy as np
from pyproj import CRS

#: Structure classes used throughout the method (LAS classification codes).
CLASS_NAMES: dict[int, str] = {
    64: "forb",
    65: "woody_debris",
    66: "hummock",
    67: "shrub",
    68: "mallee",
}
CLASS_CODES: dict[str, int] = {name: code for code, name in CLASS_NAMES.items()}
STRUCTURE_CODES = tuple(CLASS_NAMES)


@dataclass
class LabelRaster:
    """A single-band label raster with values remapped to codes 64–68 (0 = none)."""

    data: np.ndarray
    transform: object  # affine.Affine
    crs: CRS | None
    path: Path | None = None

    @property
    def shape(self) -> tuple[int, int]:
        return self.data.shape

    @property
    def resolution(self) -> tuple[float, float]:
        return abs(self.transform.a), abs(self.transform.e)

    @property
    def bounds(self) -> tuple[float, float, float, float]:
        """(left, bottom, right, top) in CRS units."""
        rows, cols = self.data.shape
        xs = (self.transform.c, self.transform.c + cols * self.transform.a)
        ys = (self.transform.f, self.transform.f + rows * self.transform.e)
        return min(xs), min(ys), max(xs), max(ys)


@dataclass
class PointCloud:
    """Point coordinates plus the LAS record they came from (kept for writing)."""

    x: np.ndarray
    y: np.ndarray
    z: np.ndarray
    crs: CRS | None = None
    z_norm: np.ndarray | None = None
    classification: np.ndarray | None = None
    las: laspy.LasData | None = field(default=None, repr=False)
    path: Path | None = None

    def __len__(self) -> int:
        return len(self.x)


def horizontal_crs(crs) -> CRS | None:
    """Return the horizontal part of a CRS (compound CRSs carry a vertical one)."""
    if crs is None:
        return None
    crs = CRS.from_user_input(crs)
    if crs.is_compound:
        crs = crs.sub_crs_list[0]
    return crs


def _resolve_class_code(value) -> int:
    if isinstance(value, str):
        key = value.strip().lower().replace(" ", "_").replace("-", "_")
        if key not in CLASS_CODES:
            raise ValueError(f"Unknown class name {value!r}; use one of {sorted(CLASS_CODES)}")
        return CLASS_CODES[key]
    code = int(value)
    if code != 0 and code not in CLASS_NAMES:
        raise ValueError(f"Class code {code} is not one of {STRUCTURE_CODES} (or 0 for none)")
    return code


def read_labels(path, class_map: Mapping | None = None, band: int = 1) -> LabelRaster:
    """Read a label GeoTIFF and remap its values to structure codes 64–68.

    Parameters
    ----------
    path:
        Single-band raster of growth-form labels.
    class_map:
        Optional ``{raster_value: class}`` mapping, where ``class`` is a code
        64–68, a name (``"forb"``, ``"woody_debris"``, ``"hummock"``,
        ``"shrub"``, ``"mallee"``) or 0 to ignore. Without it the raster must
        already use codes 64–68. Values not mapped, and NoData, become 0.
    """
    import rasterio

    path = Path(path)
    with rasterio.open(path) as src:
        raw = src.read(band)
        nodata = src.nodata
        transform = src.transform
        crs = horizontal_crs(src.crs.to_wkt()) if src.crs else None

    data = np.zeros(raw.shape, dtype=np.uint8)
    valid = np.ones(raw.shape, dtype=bool)
    if nodata is not None:
        valid = ~np.isnan(raw) if np.isnan(nodata) else raw != nodata

    if class_map is None:
        keep = valid & np.isin(raw, STRUCTURE_CODES)
        data[keep] = raw[keep]
    else:
        for value, cls in class_map.items():
            data[valid & (raw == value)] = _resolve_class_code(cls)

    if not np.any(data):
        found = np.unique(raw[valid])[:20].tolist()
        hint = "" if class_map else " Pass class_map={raster_value: 'mallee', ...} to map them."
        raise ValueError(f"{path.name}: no pixels with structure codes 64–68 (values found: {found}).{hint}")

    return LabelRaster(data=data, transform=transform, crs=crs, path=path)


def read_point_cloud(path) -> PointCloud:
    """Read a LAS or LAZ file. A ``Z_norm`` dimension is picked up if present."""
    path = Path(path)
    try:
        las = laspy.read(str(path))
    except laspy.errors.LaspyException as exc:
        if path.suffix.lower() == ".laz":
            raise RuntimeError(
                f"Cannot read {path.name}: LAZ support needs a backend. Install it with "
                "`pip install 'laspy[lazrs]'`."
            ) from exc
        raise

    dims = set(las.point_format.dimension_names)
    z_norm = None
    for name in ("Z_norm", "z_norm"):
        if name in dims:
            z_norm = np.asarray(las[name], dtype=float)
            break

    return PointCloud(
        x=np.asarray(las.x, dtype=float),
        y=np.asarray(las.y, dtype=float),
        z=np.asarray(las.z, dtype=float),
        crs=horizontal_crs(las.header.parse_crs()),
        z_norm=z_norm,
        classification=np.asarray(las.classification),
        las=las,
        path=path,
    )


def _clone_las(las_in: laspy.LasData, keep: np.ndarray | None = None) -> laspy.LasData:
    try:
        header = las_in.header.copy()
    except Exception:
        header = laspy.LasHeader(version=str(las_in.header.version), point_format=las_in.header.point_format)
        header.scales = las_in.header.scales
        header.offsets = las_in.header.offsets
    out = laspy.LasData(header)
    points = las_in.points if keep is None else las_in.points[keep]
    out.points = points.copy() if hasattr(points, "copy") else points
    return out


def ensure_las14_pf6(las_in: laspy.LasData) -> laspy.LasData:
    """Return LAS 1.4 with point format 6 or later, preserving scale, offset and extra dims."""
    if str(las_in.header.version) == "1.4" and las_in.header.point_format.id >= 6:
        return las_in

    header = laspy.LasHeader(version="1.4", point_format=6)
    header.scales = las_in.header.scales
    header.offsets = las_in.header.offsets
    crs = las_in.header.parse_crs()
    if crs is not None:
        header.add_crs(crs)

    out = laspy.LasData(header)
    out.points = laspy.ScaleAwarePointRecord.zeros(len(las_in.points), header=header)
    out.X, out.Y, out.Z = las_in.X, las_in.Y, las_in.Z
    shared = set(las_in.point_format.dimension_names) & set(out.point_format.dimension_names)
    for dim in shared - {"X", "Y", "Z"}:
        out[dim] = las_in[dim]
    for eb in las_in.header.extra_dimensions:
        if eb.name not in out.point_format.dimension_names:
            out.add_extra_dim(eb)
        out[eb.name] = las_in[eb.name]
    return out


def write_point_cloud(
    cloud: PointCloud,
    path,
    classification: np.ndarray,
    z_norm: np.ndarray,
    tree_id: np.ndarray | None = None,
    keep: np.ndarray | None = None,
) -> Path:
    """Write ``cloud`` with new ``classification``, ``Z_norm`` and ``treeID`` fields.

    The output is LAS 1.4 (point format 6 or later); a ``.laz`` suffix writes
    compressed LAZ. ``keep`` selects a subset of points to write.
    """
    if cloud.las is None:
        raise ValueError("write_point_cloud() needs a PointCloud read from a file (cloud.las is None)")

    path = Path(path)
    if path.suffix.lower() not in (".las", ".laz"):
        raise ValueError(f"Output must end in .las or .laz: {path}")

    base = ensure_las14_pf6(cloud.las)
    n = len(base.points)
    cls = np.asarray(classification, dtype=np.uint8)
    zn = np.asarray(z_norm, dtype=np.float32)
    ids = np.zeros(n, dtype=np.uint16) if tree_id is None else np.asarray(tree_id, dtype=np.uint16)
    if not (len(cls) == len(zn) == len(ids) == n):
        raise ValueError("write_point_cloud(): array lengths do not match the point cloud")
    if keep is not None:
        keep = np.asarray(keep, dtype=bool)
        cls, zn, ids = cls[keep], zn[keep], ids[keep]

    out = _clone_las(base, keep)
    for name, dtype in (("Z_norm", np.float32), ("treeID", np.uint16)):
        if name not in out.point_format.dimension_names:
            out.add_extra_dim(laspy.ExtraBytesParams(name=name, type=dtype))
    out.classification = cls
    out["Z_norm"] = zn
    out["treeID"] = ids

    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        out.write(str(path))
    except laspy.errors.LaspyException as exc:
        if path.suffix.lower() == ".laz":
            raise RuntimeError(
                "Writing LAZ needs a backend. Install it with `pip install 'laspy[lazrs]'`, "
                "or write .las instead."
            ) from exc
        raise
    return path
