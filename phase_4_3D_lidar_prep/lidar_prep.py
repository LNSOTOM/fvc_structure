#!/usr/bin/env python3
"""Prepare one previously ICP-registered LiDAR tile for structure analysis."""

import argparse
import json
import sys
from pathlib import Path

import laspy
import numpy as np
import rasterio
import yaml
from pyproj import CRS

PHASE_DIR = Path(__file__).resolve().parent
for directory in ("step_1_clipping", "step_2_csf_znorm", "step_3_structure_class"):
    sys.path.insert(0, str(PHASE_DIR / directory))

from maskclip import classify_single
from groundnorm import normalize_las_z, run_csf_ground_filter
from fvc3dstruct.config import DEFAULT_CONFIG
from fvc3dstruct.pipeline import process_tile


def load_job(path):
    with path.open(encoding="utf-8") as file:
        config = yaml.safe_load(file)
    if not isinstance(config, dict):
        raise ValueError(f"Job config must be a YAML mapping: {path}")
    return config


def mapping(job, key):
    value = job.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"Config section '{key}' must be a mapping")
    return value


def paths(job, key, names):
    settings = mapping(job, key)
    result = {}
    for name in names:
        value = settings.get(name)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"Missing path: {key}.{name}")
        result[name] = Path(value).expanduser()
    return result


def output_available(path):
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)


def horizontal_crs(crs):
    if crs is None:
        raise ValueError("LAS and rasters must have a CRS")
    crs = CRS.from_user_input(crs)
    if crs.is_compound:
        crs = crs.sub_crs_list[0]
    return crs


def check_inputs(source):
    with laspy.open(source["icp_las"]) as reader:
        header = reader.header
        if header.point_count == 0:
            raise ValueError("ICP LiDAR tile is empty")
        if header.version.minor != 4 or header.point_format.id < 6:
            raise ValueError("ICP LiDAR tile must be LAS 1.4 with point format 6 or newer")
        las_crs = horizontal_crs(header.parse_crs())
        if not las_crs.is_projected or any(axis.unit_name != "metre" for axis in las_crs.axis_info[:2]):
            raise ValueError("ICP LAS must use a projected CRS in metres")
        bounds = header.mins[:2], header.maxs[:2]
    with rasterio.open(source["predictor_raster"]) as predictor, rasterio.open(source["mask_raster"]) as mask:
        if not las_crs.equals(horizontal_crs(predictor.crs), ignore_axis_order=True):
            raise ValueError("ICP LAS and predictor raster CRS differ")
        if not las_crs.equals(horizontal_crs(mask.crs), ignore_axis_order=True):
            raise ValueError("ICP LAS and mask raster CRS differ; maskclip does not reproject")
        if mask.count != 1:
            raise ValueError("Structure mask must be single-band")
        values = np.unique(mask.read(1))
        valid = [64, 65, 66, 67, 68]
        if mask.nodata is not None:
            valid.append(mask.nodata)
        if not np.isin(values, valid).all() or not np.isin(values, [64, 65, 66, 67, 68]).any():
            raise ValueError(f"Mask must contain structure codes 64–68 and optional NoData: {values.tolist()}")
        if not (bounds[0][0] < mask.bounds.right and bounds[1][0] > mask.bounds.left
                and bounds[0][1] < mask.bounds.top and bounds[1][1] > mask.bounds.bottom):
            raise ValueError("ICP LAS and mask do not overlap")
        if not (predictor.bounds.left < mask.bounds.right and predictor.bounds.right > mask.bounds.left
                and predictor.bounds.bottom < mask.bounds.top and predictor.bounds.top > mask.bounds.bottom):
            raise ValueError("Predictor and mask rasters do not overlap")
        if predictor.transform != mask.transform or predictor.shape != mask.shape:
            print("WARNING: predictor and mask grids differ; inspect alignment and coverage before using results.")
        print(f"Inputs: {header.point_count} ICP LiDAR points; mask values {values.tolist()}")


def check_intermediate(path, source, *, normalized=False):
    with laspy.open(source) as original, laspy.open(path) as result:
        original_header, header = original.header, result.header
        if not horizontal_crs(original_header.parse_crs()).equals(
            horizontal_crs(header.parse_crs()), ignore_axis_order=True
        ):
            raise ValueError(f"Intermediate LAS CRS differs from ICP input: {path}")
        tolerance = np.maximum(original_header.scales, header.scales)
        if (header.point_count == 0
                or np.any(header.mins < original_header.mins - tolerance)
                or np.any(header.maxs > original_header.maxs + tolerance)):
            raise ValueError(f"Intermediate LAS is empty or outside ICP input bounds: {path}")
        if normalized and (header.point_count != original_header.point_count
                           or "Z_norm" not in header.point_format.dimension_names):
            raise ValueError(f"Normalized LAS must contain all input points and Z_norm: {path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--stage", choices=("all", "mask", "ground", "normalize", "fuse"), required=True)
    args = parser.parse_args()
    job = load_job(args.config)
    source = paths(job, "inputs", ("icp_las", "predictor_raster", "mask_raster"))
    output = paths(job, "outputs", ("masked_las", "ground_las", "normalized_las", "fused_las"))
    pipeline_json = output["ground_las"].with_suffix(".json")
    all_paths = [*source.values(), *output.values(), pipeline_json]
    if len({path.resolve() for path in all_paths}) != len(all_paths):
        raise ValueError("Input and output paths must all be distinct")
    for path in output.values():
        if path.suffix.lower() != ".las":
            raise ValueError(f"Output must be an uncompressed .las file: {path}")
    csf = mapping(job, "csf")
    normalization = mapping(job, "normalization")
    k = normalization.get("k")
    if not isinstance(k, int) or isinstance(k, bool) or k < 1:
        raise ValueError("normalization.k must be a positive integer")
    check_inputs(source)

    stages = ("mask", "ground", "normalize", "fuse") if args.stage == "all" else (args.stage,)
    targets = {"mask": output["masked_las"], "ground": output["ground_las"],
               "normalize": output["normalized_las"], "fuse": output["fused_las"]}
    for stage in stages:
        if targets[stage].exists():
            raise FileExistsError(f"Refusing to overwrite: {targets[stage]}")
        if stage == "ground" and pipeline_json.exists():
            raise FileExistsError(f"Refusing to overwrite: {pipeline_json}")

    for stage in stages:
        print(f"\n--- {stage} ---", flush=True)
        output_available(targets[stage])
        if stage == "mask":
            count = classify_single(source["icp_las"], source["predictor_raster"],
                                    source["mask_raster"], output["masked_las"])
            if count == 0:
                raise ValueError("Mask classification produced no points; check coverage and labels")
            print(f"Mask-classified points: {count}")
        elif stage == "ground":
            run_csf_ground_filter(
                source["icp_las"], output["ground_las"],
                resolution=csf["resolution"], iterations=csf["iterations"],
                threshold=csf["threshold"], rigidness=csf["rigidness"],
            )
        elif stage == "normalize":
            check_intermediate(output["ground_las"], source["icp_las"])
            with laspy.open(output["ground_las"]) as ground:
                if ground.header.point_count < k:
                    raise ValueError(f"Need at least {k} ground points for normalization")
            normalize_las_z(source["icp_las"], output["ground_las"],
                            output["normalized_las"], k=k)
        else:
            check_intermediate(output["normalized_las"], source["icp_las"], normalized=True)
            check_intermediate(output["masked_las"], source["icp_las"])
            print("Fusion settings:", json.dumps(DEFAULT_CONFIG, sort_keys=True))
            result = process_tile(output["normalized_las"], output["masked_las"],
                                  output["fused_las"])
            if not output["fused_las"].is_file():
                raise RuntimeError("Fusion did not write the expected LAS output")
            print(f"Fusion summary: {result}")


if __name__ == "__main__":
    main()
