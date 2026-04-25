from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

try:
    import laspy  # type: ignore
except ModuleNotFoundError:  # pragma: no cover - runtime fallback
    laspy = None


COPC_SUFFIX = ".copc.laz"


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
) -> str:
    if asset_href_mode == "absolute":
        if not public_base_url:
            raise ValueError("--public-base-url is required when --asset-href-mode=absolute")
        return f"{public_base_url}/copc/{collection_id}/{copc_path.name}"
    return f"../../../copc/{collection_id}/{copc_path.name}"


def main() -> None:
    args = parse_args()
    public_base_url = args.public_base_url.rstrip("/")
    collections = discover_copc_files(args.copc_root)

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

            copc_href = build_copc_href(
                collection_id,
                copc_path,
                public_base_url,
                args.asset_href_mode,
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
            summary_payload = {
                "scene_id": item_id,
                "collection": collection_id,
                "point_count": metadata["point_count"],
                "copc_href": copc_href,
                "bbox": item_payload["bbox"],
                "notes": "Production COPC scene generated from the local conversion pipeline.",
            }

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
