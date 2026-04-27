from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path

try:
    import laspy  # type: ignore
except ModuleNotFoundError:  # pragma: no cover - runtime fallback
    laspy = None


COPC_SUFFIX = ".copc.laz"
SOURCE_METADATA_SUFFIXES = {".las", ".laz"}
EXCLUDED_SOURCE_DIMENSIONS = {"classification", "treeid"}
STANDARD_SOURCE_DIMENSIONS = {
    "x",
    "y",
    "z",
    "intensity",
    "returnnumber",
    "numberofreturns",
    "scandirectionflag",
    "edgeofflightline",
    "classification",
    "synthetic",
    "keypoint",
    "withheld",
    "overlap",
    "scananglerank",
    "scanchannel",
    "userdata",
    "pointsourceid",
    "gpstime",
    "red",
    "green",
    "blue",
}


def canonical_stem(path: Path) -> str:
    name = path.name
    if name.endswith(COPC_SUFFIX):
        return name[: -len(COPC_SUFFIX)]
    return path.stem


def detect_tile_id(name: str) -> str | None:
    match = re.search(r"(\d+)$", name)
    if match:
        return match.group(1)
    return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate STAC catalog files for production COPC assets."
    )
    parser.add_argument(
        "--copc-root",
        type=Path,
        required=True,
        help="Root directory containing COPC files under experiment subdirectories.",
    )
    parser.add_argument(
        "--stac-root",
        type=Path,
        required=True,
        help="Output directory where catalog.json, collections, items, and summaries are written.",
    )
    parser.add_argument(
        "--public-base-url",
        default="",
        help="Public base URL for the production bundle, e.g. https://data.example.com/fvc_structure-viewer. Only required when using absolute COPC hrefs.",
    )
    parser.add_argument(
        "--copc-public-prefix",
        default="copc",
        help=(
            "Public path segment under --public-base-url where COPC assets are served, "
            'for example copc or copc/3d. Used only with absolute asset hrefs.'
        ),
    )
    parser.add_argument(
        "--asset-href-mode",
        choices=("relative", "absolute"),
        default="relative",
        help="Write relative COPC asset hrefs by default so the final public domain can be chosen later without regenerating STAC.",
    )
    parser.add_argument(
        "--catalog-id",
        default="fvc-structure-production",
        help="Catalog identifier.",
    )
    parser.add_argument(
        "--catalog-title",
        default="FVC Structure Production Catalog",
        help="Catalog title.",
    )
    parser.add_argument(
        "--catalog-description",
        default="Production STAC catalog for FVC Structure COPC point-cloud scenes.",
        help="Catalog description.",
    )
    parser.add_argument(
        "--asset-group",
        default="",
        help=(
            "Optional public COPC group name used in absolute hrefs and metadata, "
            "for example medium or LOW. When omitted, the collection directory name is used."
        ),
    )
    parser.add_argument(
        "--metadata-root",
        type=Path,
        help=(
            "Optional root containing source LAS/LAZ tiles whose metadata should be copied into "
            "the generated STAC items and summaries."
        ),
    )
    parser.add_argument(
        "--asset-group-map",
        default="",
        help=(
            "Optional JSON object mapping collection ids to public COPC group names, "
            'for example {"medium":"medium","low":"low"}. Overrides --asset-group per collection.'
        ),
    )
    parser.add_argument(
        "--metadata-root-map",
        default="",
        help=(
            "Optional JSON object mapping collection ids to source LAS roots. "
            "Overrides --metadata-root per collection."
        ),
    )
    return parser.parse_args()


def infer_site_kind(name: str) -> str:
    lowered = name.lower()
    if "medium" in lowered or "_m_" in lowered:
        return "medium"
    if "low" in lowered or "_l_" in lowered:
        return "low"
    if "forest" in lowered:
        return "forest"
    return "unknown"


def discover_copc_files(copc_root: Path) -> dict[str, list[Path]]:
    if not copc_root.exists():
        raise FileNotFoundError(f"COPC root does not exist: {copc_root}")

    collections: dict[str, list[Path]] = {}
    for path in sorted(copc_root.rglob(f"*{COPC_SUFFIX}")):
        relative = path.relative_to(copc_root)
        if len(relative.parts) < 2:
            continue
        collection_id = relative.parts[0]
        collections.setdefault(collection_id, []).append(path)

    if not collections:
        raise RuntimeError(f"No {COPC_SUFFIX} files were found under {copc_root}")
    return collections


def read_header_metadata(copc_path: Path) -> dict:
    if laspy is not None:
        with laspy.open(copc_path) as reader:
            header = reader.header
            mins = [float(value) for value in header.mins]
            maxs = [float(value) for value in header.maxs]
            return {
                "point_count": int(header.point_count),
                "mins": mins,
                "maxs": maxs,
            }

    result = subprocess.run(
        ["pdal", "info", "--summary", str(copc_path)],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(result.stdout)
    summary = payload["summary"]
    bounds = summary["bounds"]
    mins = [float(bounds["minx"]), float(bounds["miny"]), float(bounds["minz"])]
    maxs = [float(bounds["maxx"]), float(bounds["maxy"]), float(bounds["maxz"])]
    return {
        "point_count": int(summary["num_points"]),
        "mins": mins,
        "maxs": maxs,
    }


def discover_source_metadata(metadata_root: Path) -> dict[str, dict[str, Path]]:
    if not metadata_root.exists():
        raise FileNotFoundError(f"Metadata root does not exist: {metadata_root}")

    by_name: dict[str, Path] = {}
    by_tile: dict[str, Path] = {}
    for path in sorted(metadata_root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in SOURCE_METADATA_SUFFIXES:
            continue
        stem = canonical_stem(path)
        by_name.setdefault(stem, path)

        tile_id = detect_tile_id(stem)
        if tile_id is not None:
            by_tile.setdefault(tile_id, path)

    return {"by_name": by_name, "by_tile": by_tile}


def resolve_source_metadata_path(item_id: str, metadata_index: dict[str, dict[str, Path]]) -> tuple[Path | None, str | None]:
    exact = metadata_index["by_name"].get(item_id)
    if exact is not None:
        return exact, "exact_name"

    tile_id = detect_tile_id(item_id)
    if tile_id is None:
        return None, None

    matched = metadata_index["by_tile"].get(tile_id)
    if matched is not None:
        return matched, "tile_id"
    return None, None


def read_source_metadata(source_path: Path) -> dict:
    if laspy is not None:
        with laspy.open(source_path) as reader:
            header = reader.header
            point_format = header.point_format
            crs = header.parse_crs()
            extra_dimensions = [
                dimension.name
                for dimension in point_format.extra_dimensions
                if dimension.name.lower() not in EXCLUDED_SOURCE_DIMENSIONS
            ]
            return {
                "source_file": source_path.name,
                "source_stem": canonical_stem(source_path),
                "source_point_count": int(header.point_count),
                "source_las_version": f"{header.version.major}.{header.version.minor}",
                "source_point_format": int(point_format.id),
                "source_srs": crs.to_string() if crs else None,
                "source_scales": [float(value) for value in header.scales],
                "source_offsets": [float(value) for value in header.offsets],
                "source_extra_dimensions": extra_dimensions,
                "source_vlr_count": len(header.vlrs),
                "source_evlr_count": len(header.evlrs) if header.evlrs is not None else 0,
            }

    metadata_result = subprocess.run(
        ["pdal", "info", "--metadata", str(source_path)],
        check=True,
        capture_output=True,
        text=True,
    )
    schema_result = subprocess.run(
        ["pdal", "info", "--schema", str(source_path)],
        check=True,
        capture_output=True,
        text=True,
    )
    metadata_payload = json.loads(metadata_result.stdout).get("metadata", {})
    schema_payload = json.loads(schema_result.stdout).get("schema", {})
    dimensions = schema_payload.get("dimensions", [])
    extra_dimensions = []
    for dimension in dimensions:
        name = dimension.get("name")
        if not isinstance(name, str):
            continue
        lowered = name.lower()
        if lowered in EXCLUDED_SOURCE_DIMENSIONS or lowered in STANDARD_SOURCE_DIMENSIONS:
            continue
        extra_dimensions.append(name)

    srs_payload = metadata_payload.get("srs")
    source_srs = None
    if isinstance(srs_payload, dict):
        source_srs = srs_payload.get("prettywkt") or srs_payload.get("compoundwkt") or srs_payload.get("horizontal")
    elif isinstance(srs_payload, str):
        source_srs = srs_payload

    vlr_count = len([key for key in metadata_payload if key.startswith("vlr_")])
    return {
        "source_file": source_path.name,
        "source_stem": canonical_stem(source_path),
        "source_point_count": int(metadata_payload.get("count", 0)),
        "source_las_version": f"{metadata_payload.get('major_version', 0)}.{metadata_payload.get('minor_version', 0)}",
        "source_point_format": int(metadata_payload.get("dataformat_id", 0)),
        "source_srs": source_srs,
        "source_scales": [
            float(metadata_payload.get("scale_x", 0.0)),
            float(metadata_payload.get("scale_y", 0.0)),
            float(metadata_payload.get("scale_z", 0.0)),
        ],
        "source_offsets": [
            float(metadata_payload.get("offset_x", 0.0)),
            float(metadata_payload.get("offset_y", 0.0)),
            float(metadata_payload.get("offset_z", 0.0)),
        ],
        "source_extra_dimensions": extra_dimensions,
        "source_vlr_count": vlr_count,
        "source_evlr_count": 0,
    }


def make_collection(collection_id: str, item_paths: list[Path]) -> dict:
    return {
        "stac_version": "1.0.0",
        "type": "Collection",
        "id": collection_id,
        "title": collection_id,
        "description": f"Production COPC collection for {collection_id}.",
        "license": "proprietary",
        "extent": {
            "spatial": {"bbox": [[0.0, 0.0, 0.0, 0.0]]},
            "temporal": {"interval": [[None, None]]},
        },
        "links": [],
    }


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def build_copc_href(
    collection_id: str,
    copc_path: Path,
    public_base_url: str,
    asset_href_mode: str,
    asset_group: str,
    copc_public_prefix: str,
) -> str:
    path_group = asset_group or collection_id
    if asset_href_mode == "absolute":
        if not public_base_url:
            raise ValueError("--public-base-url is required when --asset-href-mode=absolute")
        cleaned_prefix = copc_public_prefix.strip("/") or "copc"
        return f"{public_base_url}/{cleaned_prefix}/{path_group}/{copc_path.name}"
    return f"../../../copc/{collection_id}/{copc_path.name}"


def parse_json_mapping(raw_value: str, flag_name: str) -> dict[str, str]:
    if not raw_value.strip():
        return {}
    try:
        payload = json.loads(raw_value)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{flag_name} must be valid JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{flag_name} must be a JSON object")
    mapping: dict[str, str] = {}
    for key, value in payload.items():
        if not isinstance(key, str) or not isinstance(value, str):
            raise ValueError(f"{flag_name} keys and values must be strings")
        mapping[key] = value
    return mapping


def main() -> None:
    args = parse_args()
    public_base_url = args.public_base_url.rstrip("/")
    collections = discover_copc_files(args.copc_root)
    asset_group_map = parse_json_mapping(args.asset_group_map, "--asset-group-map")
    metadata_root_map_raw = parse_json_mapping(args.metadata_root_map, "--metadata-root-map")

    metadata_index_by_collection: dict[str, dict[str, dict[str, Path]] | None] = {}
    default_metadata_index = (
        discover_source_metadata(args.metadata_root.resolve())
        if args.metadata_root is not None
        else None
    )
    for collection_id in collections:
        override_root = metadata_root_map_raw.get(collection_id)
        if override_root:
            metadata_index_by_collection[collection_id] = discover_source_metadata(
                Path(override_root).expanduser().resolve()
            )
        else:
            metadata_index_by_collection[collection_id] = default_metadata_index

    catalog_links = []
    for collection_id, copc_paths in collections.items():
        collection_dir = args.stac_root / "collections" / collection_id
        item_dir = args.stac_root / "items" / collection_id
        summary_dir = args.stac_root / "summaries" / collection_id

        collection_doc = make_collection(collection_id, copc_paths)
        collection_doc["links"] = [
            {"rel": "self", "href": "collection.json", "type": "application/json"},
            {"rel": "root", "href": "../../catalog.json", "type": "application/json"},
        ]

        spatial_boxes = []
        for copc_path in copc_paths:
            item_id = copc_path.name[: -len(COPC_SUFFIX)]
            metadata = read_header_metadata(copc_path)
            mins = metadata["mins"]
            maxs = metadata["maxs"]
            spatial_boxes.append([mins[0], mins[1], maxs[0], maxs[1]])
            asset_group = asset_group_map.get(collection_id) or args.asset_group or collection_id

            source_metadata = None
            metadata_index = metadata_index_by_collection.get(collection_id)
            if metadata_index is not None:
                source_path, matched_by = resolve_source_metadata_path(item_id, metadata_index)
                if source_path is not None:
                    source_metadata = read_source_metadata(source_path)
                    source_metadata["source_match_mode"] = matched_by

            copc_href = build_copc_href(
                collection_id,
                copc_path,
                public_base_url,
                args.asset_href_mode,
                asset_group,
                args.copc_public_prefix,
            )
            item_payload = {
                "stac_version": "1.0.0",
                "type": "Feature",
                "id": item_id,
                "collection": collection_id,
                "bbox": [mins[0], mins[1], mins[2], maxs[0], maxs[1], maxs[2]],
                "geometry": {
                    "type": "Point",
                    "coordinates": [
                        (mins[0] + maxs[0]) / 2.0,
                        (mins[1] + maxs[1]) / 2.0,
                        (mins[2] + maxs[2]) / 2.0,
                    ],
                },
                "properties": {
                    "title": item_id,
                    "description": f"Production COPC scene for {item_id}.",
                    "site_kind": infer_site_kind(item_id + " " + collection_id),
                    "asset_group": asset_group,
                    "experiment_family": collection_id,
                    "point_count": metadata["point_count"],
                },
                "links": [
                    {"rel": "self", "href": f"{item_id}.json", "type": "application/json"},
                    {"rel": "root", "href": "../../catalog.json", "type": "application/json"},
                    {"rel": "parent", "href": f"../../collections/{collection_id}/collection.json", "type": "application/json"},
                ],
                "assets": {
                    "copc": {
                        "href": copc_href,
                        "type": "application/vnd.laszip+copc",
                        "title": "COPC point cloud",
                    },
                    "summary": {
                        "href": f"../../summaries/{collection_id}/{item_id}-summary.json",
                        "type": "application/json",
                        "title": "Summary metrics",
                    },
                },
            }
            if source_metadata is not None:
                item_payload["properties"].update(source_metadata)

            summary_payload = {
                "scene_id": item_id,
                "collection": collection_id,
                "point_count": metadata["point_count"],
                "copc_href": copc_href,
                "asset_group": asset_group,
                "bbox": item_payload["bbox"],
                "notes": "Production COPC scene generated from the local conversion pipeline.",
            }
            if source_metadata is not None:
                summary_payload["source_metadata"] = source_metadata

            write_json(item_dir / f"{item_id}.json", item_payload)
            write_json(summary_dir / f"{item_id}-summary.json", summary_payload)
            collection_doc["links"].append(
                {
                    "rel": "item",
                    "href": f"../../items/{collection_id}/{item_id}.json",
                    "type": "application/json",
                    "title": item_id,
                }
            )

        min_x = min(box[0] for box in spatial_boxes)
        min_y = min(box[1] for box in spatial_boxes)
        max_x = max(box[2] for box in spatial_boxes)
        max_y = max(box[3] for box in spatial_boxes)
        collection_doc["extent"]["spatial"]["bbox"] = [[min_x, min_y, max_x, max_y]]
        collection_doc["extent"]["temporal"]["interval"] = [[None, None]]

        write_json(collection_dir / "collection.json", collection_doc)
        catalog_links.append(
            {
                "rel": "child",
                "href": f"collections/{collection_id}/collection.json",
                "type": "application/json",
                "title": collection_id,
            }
        )

    catalog_payload = {
        "stac_version": "1.0.0",
        "type": "Catalog",
        "id": args.catalog_id,
        "title": args.catalog_title,
        "description": args.catalog_description,
        "links": [
            {"rel": "self", "href": "catalog.json", "type": "application/json"},
            *catalog_links,
        ],
    }
    write_json(args.stac_root / "catalog.json", catalog_payload)


if __name__ == "__main__":
    main()
