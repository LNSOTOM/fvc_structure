"""Build the guide's sampled ForestFormer3D comparison data from saved PLYs."""

import argparse
import json
from pathlib import Path

import numpy as np
from plyfile import PlyData


SCENES = (
    ("medium-93", "medium", "93", "znorm_3Dlabels_m_L1_93.ply"),
    ("medium-192", "medium", "192", "znorm_3Dlabels_m_L1_192.ply"),
    ("low-43", "low", "43", "znorm_3Dlabels_l_L1_43.ply"),
    ("low-75", "low", "75", "znorm_3Dlabels_l_L1_75.ply"),
    ("forest-scion-31", "forest", "31", "SCION_SCION_plot_31_annotated_test.ply"),
    ("forest-scion-61", "forest", "61", "SCION_SCION_plot_61_annotated_test.ply"),
)
FIELDS = ("x", "y", "z", "semantic_gt", "semantic_pred", "instance_gt", "instance_pred")


def build_scene(path: Path, scene_id: str, site: str, tile: str, count: int) -> dict:
    vertices = PlyData.read(path)["vertex"].data
    missing = set(FIELDS) - set(vertices.dtype.names or ())
    if missing:
        raise ValueError(f"{path}: missing fields {sorted(missing)}")
    if not len(vertices):
        raise ValueError(f"{path}: no points")
    if not all(np.isfinite(vertices[field]).all() for field in ("x", "y", "z")):
        raise ValueError(f"{path}: non-finite coordinates")
    for field in ("semantic_gt", "semantic_pred"):
        if not np.isin(vertices[field], (0, 1, 2)).all():
            raise ValueError(f"{path}: unsupported {field} labels")

    sample_count = min(count, len(vertices))
    indices = np.random.default_rng(2026).choice(len(vertices), sample_count, replace=False)
    sample = vertices[indices]
    points = [
        [round(float(row["x"]), 3), round(float(row["y"]), 3), round(float(row["z"]), 3)]
        + [int(row[field]) for field in FIELDS[3:]]
        for row in sample
    ]
    return {
        "id": scene_id,
        "site": site,
        "tile": tile,
        "source": path.name,
        "total": len(vertices),
        "reference_instances": int(np.count_nonzero(np.unique(vertices["instance_gt"]) > 0)),
        "predicted_instances": int(np.count_nonzero(np.unique(vertices["instance_pred"]) >= 0)),
        "semantic_counts": [
            [int(np.count_nonzero(vertices[field] == label)) for label in range(3)]
            for field in ("semantic_gt", "semantic_pred")
        ],
        "points": points,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--medium-dir", type=Path, required=True)
    parser.add_argument("--low-dir", type=Path, required=True)
    parser.add_argument("--forest-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--points", type=int, default=60000)
    args = parser.parse_args()
    if args.points <= 0:
        parser.error("--points must be positive")

    directories = {"medium": args.medium_dir, "low": args.low_dir, "forest": args.forest_dir}
    scenes = [build_scene(directories[site] / filename, scene_id, site, tile, args.points)
              for scene_id, site, tile, filename in SCENES]
    with args.output.open("w", encoding="utf-8") as output:
        output.write("window.FVC_MODEL_COMPARISONS = ")
        json.dump(scenes, output, separators=(",", ":"), allow_nan=False)
        output.write(";\n")
    print(f"Wrote {len(scenes)} scenes to {args.output}")


if __name__ == "__main__":
    main()
