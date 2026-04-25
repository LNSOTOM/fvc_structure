from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path


SUPPORTED_SUFFIXES = {".ply", ".las", ".laz"}
PDAL_READER_TYPES = {
    ".ply": "readers.ply",
    ".las": "readers.las",
    ".laz": "readers.las",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert local point-cloud scenes into production COPC LAZ files."
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        required=True,
        help="Root directory containing experiment subdirectories with .ply/.las/.laz scenes.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        required=True,
        help="Output root where COPC files will be written, preserving experiment folders.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing .copc.laz files.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the planned conversions without executing PDAL.",
    )
    parser.add_argument(
        "--a-srs",
        default="",
        help="Assign an output CRS to the COPC files, for example EPSG:7854.",
    )
    return parser.parse_args()


def require_pdal() -> str:
    pdal_path = shutil.which("pdal")
    if pdal_path is None:
        raise RuntimeError(
            "The 'pdal' CLI was not found. Install PDAL with COPC support before running this script."
        )
    return pdal_path


def discover_sources(input_root: Path) -> list[Path]:
    if not input_root.exists():
        raise FileNotFoundError(f"Input root does not exist: {input_root}")

    scene_paths = sorted(
        path
        for path in input_root.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES
    )
    if not scene_paths:
        raise RuntimeError(f"No supported scene files were found under {input_root}")
    return scene_paths


def build_output_path(scene_path: Path, input_root: Path, output_root: Path) -> Path:
    relative_path = scene_path.relative_to(input_root)
    return (output_root / relative_path).with_suffix(".copc.laz")


def build_pipeline(scene_path: Path, output_path: Path, a_srs: str) -> dict:
    suffix = scene_path.suffix.lower()
    writer = {
        "type": "writers.copc",
        "filename": str(output_path),
        "forward": "all",
    }
    if a_srs:
        writer["a_srs"] = a_srs

    return {
        "pipeline": [
            {
                "type": PDAL_READER_TYPES[suffix],
                "filename": str(scene_path),
            },
            writer,
        ]
    }


def run_pipeline(pdal_path: str, pipeline: dict) -> None:
    subprocess.run(
        [pdal_path, "pipeline", "--stdin"],
        input=json.dumps(pipeline).encode("utf-8"),
        check=True,
    )


def main() -> None:
    args = parse_args()
    pdal_path = require_pdal()
    scene_paths = discover_sources(args.input_root)

    for scene_path in scene_paths:
        output_path = build_output_path(scene_path, args.input_root, args.output_root)
        if output_path.exists() and not args.overwrite:
            print(f"skip  {output_path}")
            continue

        output_path.parent.mkdir(parents=True, exist_ok=True)
        pipeline = build_pipeline(scene_path, output_path, args.a_srs)
        print(f"build {scene_path} -> {output_path}")
        if not args.dry_run:
            run_pipeline(pdal_path, pipeline)


if __name__ == "__main__":
    main()
