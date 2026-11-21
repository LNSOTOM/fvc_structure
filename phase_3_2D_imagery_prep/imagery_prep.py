#!/usr/bin/env python3
"""Config-driven Phase 3 preparation using the historical label settings."""

import argparse
import re
import subprocess
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from osgeo import gdal, ogr, osr
from rasterio.features import rasterize

from phase3_config import load_config

# These match the original 2_1_structure_2D_labels.py classification.
STRUCTURE_IDS = {
    "forb": 0,
    "branch_stem_dead": 1,
    "branch_stem_standing": 1,
    "cwd": 1,
    "grass_hummock": 2,
    "grass_hummock_senescent": 2,
    "shrub_chenopod": 3,
    "shrub_chenopod_senescent": 3,
    "shrub": 3,
    "mallee": 4,
}
EXCLUDED = {"bare_ground", "shadow", "be", "unknowm"}
TYPE_COLUMNS = (
    "structure", "class", "type", "label", "vegetation",
    "class_name", "classname", "CLASS", "TYPE",
)

gdal.UseExceptions()
ogr.UseExceptions()


def section(config, name):
    value = config.get(name)
    if not isinstance(value, dict):
        raise ValueError(f"Config section '{name}' must be a mapping")
    return value


def path_setting(settings, name):
    value = settings.get(name)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Missing path setting: {name}")
    return Path(value).expanduser()


def inputs(directory, pattern):
    if not directory.is_dir():
        raise NotADirectoryError(f"Input directory not found: {directory}")
    files = sorted(directory.glob(pattern))
    if not files:
        raise ValueError(f"No files matching {pattern} in {directory}")
    return files


def output_path(directory, filename):
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / filename
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite: {path}")
    return path


def tile_id(path):
    match = re.search(r"(\d+)$", path.stem)
    if not match:
        raise ValueError(f"Filename must end in a numeric tile ID: {path}")
    return match.group(1)


def unique_ids(files):
    found = {}
    for path in files:
        ident = tile_id(path)
        if ident in found:
            raise ValueError(f"Duplicate tile ID {ident}: {found[ident]} and {path}")
        found[ident] = path
    return found


def require_metric_grid(src, path, resolution):
    if src.crs is None or not src.crs.is_projected or src.crs.linear_units_factor[1] != 1.0:
        raise ValueError(f"Expected a projected CRS in metres: {path}")
    if src.transform.b != 0 or src.transform.d != 0:
        raise ValueError(f"Rotated raster is not supported: {path}")
    if not np.allclose(src.res, (resolution, resolution), rtol=0, atol=1e-8):
        raise ValueError(f"Reference must have {resolution * 100:g} cm pixels: {path} has {src.res}")


def tile(settings):
    raster_path = path_setting(settings, "orthomosaic")
    grid_path = path_setting(settings, "grid")
    out_dir = path_setting(settings, "output_dir")
    raster = gdal.Open(str(raster_path))
    grid = ogr.Open(str(grid_path))
    if raster is None or grid is None:
        raise ValueError("Could not open orthomosaic or grid")
    layer = grid.GetLayer()
    spatial_ref = layer.GetSpatialRef()
    raster_ref = osr.SpatialReference()
    if not raster.GetProjection() or raster_ref.ImportFromWkt(raster.GetProjection()) != 0:
        raise ValueError("Orthomosaic must have a CRS")
    if spatial_ref is None or not spatial_ref.IsSame(raster_ref):
        raise ValueError("Grid and orthomosaic must have the same CRS")
    id_field = settings.get("grid_id_field", "id")
    annotated_field = settings.get("annotated_field", "annotated")
    definition = layer.GetLayerDefn()
    for field in (id_field, annotated_field):
        if not isinstance(field, str) or definition.GetFieldIndex(field) < 0:
            raise ValueError(f"Grid field not found: {field}")
    annotated_value = settings.get("annotated_value", 1)
    selected = []
    seen = set()
    for feature in layer:
        if feature.GetField(annotated_field) != annotated_value:
            continue
        ident = str(feature.GetField(id_field))
        if not ident.isdigit() or ident in seen:
            raise ValueError(f"Invalid or duplicate grid ID: {ident}")
        geom = feature.GetGeometryRef()
        if geom is None or geom.IsEmpty():
            raise ValueError(f"Grid tile {ident} has no geometry")
        seen.add(ident)
        selected.append((ident, geom.GetEnvelope()))
    if not selected:
        raise ValueError("No annotated grid cells selected")
    for ident, (min_x, max_x, min_y, max_y) in selected:
        output = output_path(out_dir, f"tiles_multispectral.{ident}.tif")
        result = gdal.Translate(
            str(output), raster,
            options=gdal.TranslateOptions(
                format="GTiff", outputType=gdal.GDT_Float32,
                projWin=[min_x, max_y, max_x, min_y],
            ),
        )
        if result is None:
            raise RuntimeError(f"Failed to clip tile {ident}")
        result = None
        print(f"Clipped {output}")


def stack(settings):
    source = path_setting(settings, "input_dir")
    target = path_setting(settings, "output_dir")
    for path in inputs(source, "*.tif"):
        with rasterio.open(path) as src:
            if src.count < 10:
                raise ValueError(f"{path} has {src.count} bands; at least 10 are required")
            profile = src.profile.copy()
            profile.update(count=5)
            data = src.read([2, 4, 6, 8, 10])
        output = output_path(target, path.name)
        with rasterio.open(output, "w", **profile) as dst:
            dst.write(data)
        print(f"Stacked {output}")


def classify(settings):
    source = path_setting(settings, "input_dir")
    target = path_setting(settings, "output_dir")
    files = inputs(source, "*.shp")
    unique_ids(files)
    for path in files:
        gdf = gpd.read_file(path)
        if gdf.crs is None or gdf.crs.is_geographic:
            raise ValueError(f"Annotations need a projected CRS for area: {path}")
        column = next((name for name in TYPE_COLUMNS if name in gdf.columns), None)
        if column is None:
            raise ValueError(f"No vegetation type field in {path}; refusing to assign a default class")
        values = gdf[column].fillna("unknown").astype(str)
        excluded = values.str.lower().str.replace(" ", "_", regex=False).isin(EXCLUDED)
        mapped = values.map(STRUCTURE_IDS)
        unknown = ~excluded & mapped.isna()
        if excluded.any():
            print(f"Excluded types in {path.name}: {values[excluded].value_counts().to_dict()}")
        if unknown.any():
            print(f"WARNING: removing unrecognized types in {path.name}: {values[unknown].value_counts().to_dict()}")
        keep = ~excluded & mapped.notna()
        if not keep.any():
            raise ValueError(f"No recognized polygons remain in {path}")
        result = gdf.loc[keep].copy()
        result["struct_id"] = mapped[keep].astype(int)
        result["area"] = result.geometry.area.astype(float)
        output = output_path(target, path.name)
        for sidecar in (".shx", ".dbf", ".prj", ".cpg"):
            if output.with_suffix(sidecar).exists():
                raise FileExistsError(f"Refusing to overwrite shapefile sidecar: {output.with_suffix(sidecar)}")
        result.to_file(output)
        print(f"Classified {path.name}: {len(gdf)} -> {len(result)} polygons; IDs {result['struct_id'].value_counts().sort_index().to_dict()}")


def make_labels(settings):
    source = path_setting(settings, "input_dir")
    reference_dir = path_setting(settings, "reference_dir")
    target = path_setting(settings, "output_dir")
    pattern = settings.get("reference_pattern")
    if not isinstance(pattern, str) or pattern.count("{id}") != 1 or Path(pattern).name != pattern:
        raise ValueError("reference_pattern must be a filename containing exactly one {id}")
    for ident, path in unique_ids(inputs(source, "*.shp")).items():
        reference = reference_dir / pattern.format(id=ident)
        shapes = gpd.read_file(path)
        if shapes.empty or "struct_id" not in shapes.columns:
            raise ValueError(f"Missing polygons or struct_id in {path}")
        if shapes["struct_id"].isna().any() or not shapes["struct_id"].isin(range(5)).all():
            raise ValueError(f"Invalid struct_id values in {path}")
        with rasterio.open(reference) as ref:
            if ref.crs is None or shapes.crs is None or shapes.crs != ref.crs:
                raise ValueError(f"CRS mismatch between {path} and {reference}")
            require_metric_grid(ref, reference, 0.01)
            profile = ref.profile.copy()
            profile.update(driver="GTiff", count=1, compress="lzw",
                           nodata=255, dtype="uint8", tiled=True,
                           blockxsize=3072, blockysize=3072)
            data = rasterize(
                zip(shapes.geometry, shapes["struct_id"]),
                out_shape=(ref.height, ref.width), transform=ref.transform,
                fill=255, all_touched=True, dtype="uint8",
            )
        output = output_path(target, f"mask_structure_3072.{ident}.tif")
        with rasterio.open(output, "w", **profile) as dst:
            dst.write(data, 1)
        print(f"Rasterized {output}")


def downsample(settings):
    source = path_setting(settings, "input_dir")
    target = path_setting(settings, "output_dir")
    files = inputs(source, "mask_structure_3072.*.tif")
    unique_ids(files)
    for path in files:
        with rasterio.open(path) as src:
            if src.crs is None or src.count != 1 or src.nodata != 255:
                raise ValueError(f"Expected single-band label raster with CRS and NoData 255: {path}")
            require_metric_grid(src, path, 0.01)
            values = np.unique(src.read(1))
            if not np.isin(values, [0, 1, 2, 3, 4, 255]).all():
                raise ValueError(f"Unexpected label values in {path}: {values.tolist()}")
            bounds = src.bounds
        output = output_path(target, path.name)
        subprocess.run([
            "gdalwarp", "-overwrite", "-q",
            "-tr", "0.05", "0.05", "-r", "nearest",
            "-te", str(bounds.left), str(bounds.bottom), str(bounds.right), str(bounds.top),
            "-ot", "Byte", "-of", "GTiff",
            "-co", "COMPRESS=LZW", "-co", "PREDICTOR=2", "-co", "BIGTIFF=YES",
            "-co", "TILED=YES", "-co", "BLOCKXSIZE=3072", "-co", "BLOCKYSIZE=3072",
            "-srcnodata", "255", "-dstnodata", "255", str(path), str(output),
        ], check=True)
        with rasterio.open(output, "r+") as dst:
            band = dst.read(1)
            remapped = band.copy()
            for value in range(5):
                remapped[band == value] = value + 64
            dst.write(remapped, 1)
            dst.nodata = 255
        print(f"Downsampled {output}")


STAGES = {
    "tile": tile,
    "stack": stack,
    "classify": classify,
    "rasterize": make_labels,
    "downsample": downsample,
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Path to YAML job config")
    parser.add_argument("--stage", choices=[*STAGES, "all"], required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    stages = STAGES if args.stage == "all" else {args.stage: STAGES[args.stage]}
    for name, run in stages.items():
        print(f"\n--- {name} ---", flush=True)
        run(section(config, name))


if __name__ == "__main__":
    main()
