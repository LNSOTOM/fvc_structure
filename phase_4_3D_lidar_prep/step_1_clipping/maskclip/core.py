import json
from pathlib import Path
from typing import Optional

import pdal
import rasterio
import laspy

def _raster_epsg_and_bounds(raster_path: Path):
    with rasterio.open(raster_path) as src:
        epsg = src.crs.to_epsg()
        minx, miny, maxx, maxy = src.bounds
    return epsg, (minx, miny, maxx, maxy)

def classify_single(las_path: Path, raster_path: Path, mask_path: Path, out_path: Path, keep_range: str = "Classification[64:68]", minor_version: int = 4) -> int:
    """
    Colorize LAS with mask raster into Classification, filter by range, and write output.
    Reprojects to raster CRS and crops to raster bounds before classification.
    Returns number of points written.
    """
    las_path = Path(las_path)
    raster_path = Path(raster_path)
    mask_path = Path(mask_path)
    out_path = Path(out_path)

    # Preserve input LAS scale/offset
    src_las = laspy.read(las_path)
    scale_x, scale_y, scale_z = src_las.header.scales
    offset_x, offset_y, offset_z = src_las.header.offsets

    # Use mask bounds for cropping to ensure all points are within mask coverage
    target_epsg, (minx, miny, maxx, maxy) = _raster_epsg_and_bounds(mask_path)
    bounds_str = f"([{minx},{maxx}],[{miny},{maxy}])"

    pipeline = {
        "pipeline": [
            {"type": "readers.las", "filename": str(las_path)},
            {"type": "filters.crop", "bounds": bounds_str},
            {"type": "filters.colorization", "raster": str(mask_path), "dimensions": "Classification"},
            {"type": "filters.range", "limits": keep_range},
            {
                "type": "writers.las",
                "filename": str(out_path),
                "minor_version": minor_version,
                "extra_dims": "all",
                # "scale_x": scale_x,
                # "scale_y": scale_y,
                # "scale_z": scale_z,
                "offset_x": offset_x,
                "offset_y": offset_y,
                "offset_z": offset_z,
            },
        ]
    }

    pipe = pdal.Pipeline(json.dumps(pipeline))
    cnt = pipe.execute()
    del pipe
    return int(cnt)
