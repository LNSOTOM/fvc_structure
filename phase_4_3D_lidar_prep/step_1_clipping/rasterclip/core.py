from pathlib import Path
import json
import subprocess
from typing import Iterable, Optional, Tuple, List

import pdal
import rasterio
from shapely.geometry import Polygon
from shapely.ops import transform as shp_transform
from pyproj import CRS, Transformer


def _read_las_crs(input_las: Path) -> CRS:
    info = subprocess.check_output(["pdal", "info", "--metadata", str(input_las)])
    meta = json.loads(info.decode("utf-8"))
    wkt = meta["metadata"]["srs"].get("wkt", None)
    if not wkt:
        raise RuntimeError(f"Could not read LAS SRS from file metadata: {input_las}")
    return CRS.from_wkt(wkt)


def _raster_bounds_and_epsg(raster: Path) -> Tuple[Tuple[float, float, float, float], int, CRS]:
    with rasterio.open(raster) as src:
        crs = CRS.from_wkt(src.crs.to_wkt())
        epsg = crs.to_epsg()
        minx, miny, maxx, maxy = src.bounds
    return (minx, miny, maxx, maxy), int(epsg), crs


def _raster_bbox_in_las_crs(bounds: Tuple[float, float, float, float], raster_crs: CRS, las_crs: CRS) -> Tuple[float, float, float, float]:
    minx, miny, maxx, maxy = bounds
    pad = max((maxx - minx), (maxy - miny)) * 1e-9
    minx -= pad; miny -= pad; maxx += pad; maxy += pad

    bbox_poly = Polygon([(minx, miny), (maxx, miny), (maxx, maxy), (minx, maxy)])
    if raster_crs != las_crs:
        to_las = Transformer.from_crs(raster_crs, las_crs, always_xy=True)
        bbox_poly = shp_transform(lambda x, y, z=None: to_las.transform(x, y), bbox_poly)
    minx_las, miny_las, maxx_las, maxy_las = bbox_poly.bounds
    return minx_las, miny_las, maxx_las, maxy_las


def clip_single(input_las: Path, aoi_raster: Path, out_path: Path) -> int:
    las_crs = _read_las_crs(input_las)
    bounds, target_epsg, raster_crs = _raster_bounds_and_epsg(aoi_raster)
    minx_las, miny_las, maxx_las, maxy_las = _raster_bbox_in_las_crs(bounds, raster_crs, las_crs)
    bounds_las_str = f"([{minx_las},{maxx_las}],[{miny_las},{maxy_las}])"

    pipeline = {
        "pipeline": [
            {"type": "readers.las", "filename": str(input_las), "tag": "src"},
            {"type": "filters.crop", "bounds": bounds_las_str, "inputs": ["src"], "tag": "crop_las"},
            {"type": "filters.reprojection", "out_srs": f"EPSG:{target_epsg}", "inputs": ["crop_las"], "tag": "rpj"},
            {"type": "writers.las", "filename": str(out_path), "minor_version": 4, "inputs": ["rpj"]},
        ]
    }
    pipe = pdal.Pipeline(json.dumps(pipeline))
    cnt = pipe.execute()
    return int(cnt)


def clip_many(
    input_las: Path,
    aoi_raster_dir: Path,
    out_dir: Path,
    pattern: str = "tiles_multispectral*.tif",
    skip_existing: bool = True,
) -> List[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    rasters = sorted(aoi_raster_dir.glob(pattern))
    outputs: List[Path] = []
    for raster in rasters:
        out_path = out_dir / f"{raster.stem}.las"
        if skip_existing and out_path.exists():
            continue
        cnt = clip_single(input_las=input_las, aoi_raster=raster, out_path=out_path)
        outputs.append(out_path)
    return outputs
