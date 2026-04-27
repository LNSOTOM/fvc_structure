from __future__ import annotations

import argparse
import copy
import json
import re
import shutil
import subprocess
import tempfile
from pathlib import Path


SUPPORTED_SUFFIXES = {".ply", ".las", ".laz"}
PDAL_READER_TYPES = {
    ".ply": "readers.ply",
    ".las": "readers.las",
    ".laz": "readers.las",
}
DEFAULT_STRUCTURE_ROOTS = (
    "/media/laura/8402326D023263F8/calperumResearch/site1_1_DD0001/lidar/clipped/3d_structure_mask_malleid/new",
    "/media/laura/8402326D023263F8/calperumResearch/site1_1_DD0001/lidar/clipped/3d_structure_mask_malleid",
    "/media/laura/8402326D023263F8/calperumResearch/site1_1_DD0001/lidar_l2/clipped/3d_structure_mask_malleid",
    "/media/laura/8402326D023263F8/calperumResearch/site2_1_DD0011/lidar/clipped/3d_structure_mask_malleid",
    "/media/laura/8402326D023263F8/calperumResearch/site2_1_DD0011/lidar_l2/clipped/3d_structure_mask_malleid",
)


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
    parser.add_argument(
        "--structure-root",
        type=Path,
        help="Optional root containing structure LAS/LAZ tiles used as the backbone for work_dirs PLY merges.",
    )
    parser.add_argument(
        "--forestformer-npy-root",
        type=Path,
        help="Optional root containing forestformer3d_npy sidecars (*_vert.npy, *_offsets.npy).",
    )
    parser.add_argument(
        "--alignment-tolerance",
        type=float,
        default=1e-3,
        help="Maximum absolute XYZ difference allowed when verifying row-aligned work_dirs-to-structure mapping.",
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


def should_merge_with_structure(scene_path: Path) -> bool:
    suffix = scene_path.suffix.lower()
    if suffix == ".ply":
        return scene_path.stem.startswith("znorm_3Dlabels")
    if suffix in {".las", ".laz"}:
        return scene_path.stem.startswith("znorm_3Dlabels") and scene_path.stem.endswith("_out")
    return False


def normalize_scene_stem(scene_stem: str) -> str:
    normalized = scene_stem[:-7] if scene_stem.endswith("_global") else scene_stem
    if normalized.endswith("_out"):
        normalized = normalized[:-4]
    return normalized


def extract_base_scene_stem(scene_stem: str) -> str | None:
    normalized = normalize_scene_stem(scene_stem)
    match = re.match(r"^znorm_3Dlabels_[a-z]_[A-Za-z0-9-]+_(\d+)$", normalized)
    if match:
        return f"znorm_3Dlabels_{match.group(1)}"
    match = re.match(r"^(znorm_3Dlabels_\d+)$", normalized)
    if match:
        return match.group(1)
    return None


def backbone_stem_candidates(scene_stem: str) -> list[str]:
    normalized = normalize_scene_stem(scene_stem)
    candidates: list[str] = []

    low_match = re.match(r"^(znorm_3Dlabels_l)_[A-Za-z0-9-]+_(\d+)$", normalized)
    if low_match:
        candidates.append(f"{low_match.group(1)}_{low_match.group(2)}")

    base_scene_stem = extract_base_scene_stem(scene_stem)
    if base_scene_stem is not None:
        candidates.append(base_scene_stem)

    unique: list[str] = []
    for candidate in candidates:
        if candidate not in unique:
            unique.append(candidate)
    return unique


def structure_search_roots(primary_root: Path | None) -> list[Path]:
    roots: list[Path] = []

    def add(path: Path | None) -> None:
        if path is None:
            return
        if path.exists() and path not in roots:
            roots.append(path)

    add(primary_root)
    if primary_root is not None and primary_root.name == "new":
        add(primary_root.parent)

    for raw_path in DEFAULT_STRUCTURE_ROOTS:
        add(Path(raw_path))

    return roots


def resolve_backbone_path(
    scene_stem: str,
    structure_root: Path | None,
    expected_point_count: int | None = None,
) -> Path:
    candidates: list[Path] = []
    for root in structure_search_roots(structure_root):
        for stem in backbone_stem_candidates(scene_stem):
            for suffix in (".las", ".laz"):
                candidate = root / f"{stem}{suffix}"
                if candidate.exists() and candidate not in candidates:
                    candidates.append(candidate)

    if expected_point_count is not None and candidates:
        import laspy

        for candidate in candidates:
            try:
                if len(laspy.read(str(candidate)).points) == expected_point_count:
                    return candidate
            except Exception:
                continue

    if candidates:
        return candidates[0]
    raise FileNotFoundError(
        f"Could not find a structure backbone for {scene_stem} under any known structure root"
    )


def resolve_npy_prefix(forestformer_npy_root: Path, base_scene_stem: str) -> Path:
    prefix = forestformer_npy_root / base_scene_stem
    for suffix in ("_vert.npy", "_offsets.npy"):
        candidate = prefix.parent / f"{prefix.name}{suffix}"
        if not candidate.exists():
            raise FileNotFoundError(f"Missing {candidate}")
    return prefix


def _clone_las(las_in, point_indices=None):
    import laspy
    import numpy as np

    try:
        header = las_in.header.copy()
    except Exception:
        header = laspy.LasHeader(
            version=str(las_in.header.version),
            point_format=las_in.header.point_format,
        )
        header.scales = las_in.header.scales
        header.offsets = las_in.header.offsets
        extra_dimensions = getattr(las_in.point_format, "extra_dimensions", [])
        for eb in extra_dimensions:
            try:
                header.add_extra_dim(eb)
            except Exception:
                pass

    las_out = laspy.LasData(header)
    points = las_in.points.copy() if hasattr(las_in.points, "copy") else las_in.points
    if point_indices is not None:
        points = points[np.asarray(point_indices, dtype=np.int64)]
    las_out.points = points

    try:
        las_out.header.global_encoding.value = las_in.header.global_encoding.value
    except Exception:
        pass

    try:
        las_out.header.vlrs = copy.deepcopy(las_in.header.vlrs)
    except Exception:
        pass

    try:
        if getattr(las_in, "evlrs", None) is not None:
            las_out.evlrs = copy.deepcopy(las_in.evlrs)
    except Exception:
        pass

    try:
        crs = las_in.header.parse_crs()
        if crs is not None:
            las_out.header.add_crs(crs)
    except Exception:
        pass

    return las_out


def _load_ply_vertex(scene_path: Path):
    from plyfile import PlyData

    return PlyData.read(str(scene_path))["vertex"].data


def _load_source_points(scene_path: Path):
    import laspy
    import numpy as np

    suffix = scene_path.suffix.lower()
    if suffix == ".ply":
        vertex = _load_ply_vertex(scene_path)
        xyz_keys = _xyz_from_vertex(vertex)
        arrays = {name: np.asarray(vertex[name]) for name in vertex.dtype.names}
        xyz = np.column_stack([arrays[key] for key in xyz_keys]).astype(np.float64)
        return xyz, arrays, set(xyz_keys)

    if suffix in {".las", ".laz"}:
        las = laspy.read(str(scene_path))
        arrays = {
            name: np.asarray(las[name])
            for name in las.point_format.dimension_names
        }
        xyz = np.column_stack([las.x, las.y, las.z]).astype(np.float64)
        return xyz, arrays, {"X", "Y", "Z", "x", "y", "z"}

    raise ValueError(f"Unsupported source suffix for merge: {scene_path.suffix}")


def _first_matching_array(arrays: dict[str, object], *candidates: str):
    lower_names = {name.lower(): name for name in arrays}
    for candidate in candidates:
        matched = lower_names.get(candidate.lower())
        if matched is not None:
            return arrays[matched]
    return None


def _assign_dimension(merged, existing_names: set[str], name: str, values) -> None:
    import laspy

    if name not in existing_names:
        merged.add_extra_dim(laspy.ExtraBytesParams(name=name, type=values.dtype))
        existing_names.add(name)
    merged[name] = values.astype(values.dtype, copy=False)


def _canonical_prediction_aliases(source_arrays: dict[str, object]) -> dict[str, object]:
    aliases: dict[str, object] = {}

    semantic_pred = _first_matching_array(
        source_arrays,
        "semantic_pred",
        "pred_semantic",
        "PredSemantic",
        "semantic_seg",
        "semantic",
    )
    if semantic_pred is not None:
        aliases["semantic_pred"] = semantic_pred

    instance_pred = _first_matching_array(
        source_arrays,
        "instance_pred",
        "PredInstance",
        "pred_instance",
        "predinstance",
    )
    if instance_pred is not None:
        aliases["instance_pred"] = instance_pred

    semantic_gt = _first_matching_array(
        source_arrays,
        "semantic_gt",
        "gt_semantic",
        "gt_semantic_segmentation",
    )
    if semantic_gt is not None:
        aliases["semantic_gt"] = semantic_gt

    instance_gt = _first_matching_array(
        source_arrays,
        "instance_gt",
        "treeID",
        "TreeID",
        "tree_id",
        "gt_tree_id",
    )
    if instance_gt is not None:
        aliases["instance_gt"] = instance_gt

    score = _first_matching_array(source_arrays, "score")
    if score is not None:
        aliases["score"] = score

    return aliases


def _xyz_from_vertex(vertex) -> tuple[str, str, str]:
    lower_names = {name.lower(): name for name in vertex.dtype.names}
    if {"x", "y", "z"}.issubset(lower_names):
        return lower_names["x"], lower_names["y"], lower_names["z"]
    names = vertex.dtype.names
    return names[0], names[1], names[2]


def merge_predictions_onto_structure(
    scene_path: Path,
    output_path: Path,
    structure_root: Path | None,
    forestformer_npy_root: Path | None,
    alignment_tolerance: float,
) -> Path:
    import laspy
    import numpy as np

    workdir_xyz, source_arrays, xyz_keys = _load_source_points(scene_path)
    backbone_expected_count = None
    matched_source_indices = None
    base_scene_stem = extract_base_scene_stem(scene_path.stem)

    used_sidecars = False
    if forestformer_npy_root is not None and base_scene_stem is not None:
        local_verts = None
        restored_xyz = None
        if base_scene_stem is not None:
            try:
                npy_prefix = resolve_npy_prefix(forestformer_npy_root, base_scene_stem)
                verts = np.load(str(npy_prefix.parent / f"{npy_prefix.name}_vert.npy")).astype(np.float64)
                offsets = np.load(str(npy_prefix.parent / f"{npy_prefix.name}_offsets.npy")).astype(np.float64).reshape(-1)[:3]
                local_verts = verts[:, :3]
                restored_xyz = local_verts + offsets
                backbone_expected_count = len(restored_xyz)
            except FileNotFoundError:
                pass

    backbone_path = resolve_backbone_path(
        scene_path.stem,
        structure_root,
        expected_point_count=backbone_expected_count,
    )
    backbone = laspy.read(str(backbone_path))
    backbone_xyz = np.column_stack([backbone.x, backbone.y, backbone.z]).astype(np.float64)

    if forestformer_npy_root is not None and base_scene_stem is not None:
        local_verts = None
        restored_xyz = None
        try:
            npy_prefix = resolve_npy_prefix(forestformer_npy_root, base_scene_stem)
            verts = np.load(str(npy_prefix.parent / f"{npy_prefix.name}_vert.npy")).astype(np.float64)
            offsets = np.load(str(npy_prefix.parent / f"{npy_prefix.name}_offsets.npy")).astype(np.float64).reshape(-1)[:3]
            local_verts = verts[:, :3]
            restored_xyz = local_verts + offsets
            if len(workdir_xyz) == len(backbone_xyz) == len(restored_xyz):
                local_diff = float(np.max(np.abs(workdir_xyz - verts[:, :3])))
                backbone_diff = float(np.max(np.abs(restored_xyz - backbone_xyz)))
                if local_diff <= alignment_tolerance and backbone_diff <= alignment_tolerance:
                    used_sidecars = True
        except FileNotFoundError:
            pass

        if (
            not used_sidecars
            and local_verts is not None
            and restored_xyz is not None
            and len(restored_xyz) == len(backbone_xyz)
            and scene_path.suffix.lower() in {".las", ".laz"}
        ):
            from scipy.spatial import cKDTree

            translated_local_verts = local_verts + (np.min(workdir_xyz, axis=0) - np.min(local_verts, axis=0))
            tree = cKDTree(workdir_xyz)
            distances, matched_source_indices = tree.query(translated_local_verts, k=1)
            match_tolerance = max(alignment_tolerance, 0.002)
            residual = float(np.max(distances))
            if residual <= match_tolerance:
                used_sidecars = True
            else:
                matched_source_indices = None

    if not used_sidecars:
        if len(workdir_xyz) != len(backbone_xyz):
            raise RuntimeError(
                f"Point-count mismatch for {scene_path.name}: workdir={len(workdir_xyz)} backbone={len(backbone_xyz)}"
            )
        deltas = backbone_xyz - workdir_xyz
        median_delta = np.median(deltas, axis=0)
        residual = float(np.max(np.abs(deltas - median_delta)))
        if residual > alignment_tolerance:
            raise RuntimeError(
                f"Alignment check failed for {scene_path.name}: direct backbone offset residual={residual}"
            )

    merged = _clone_las(backbone)
    merged_point_count = len(merged.points)
    existing_names = set(merged.point_format.dimension_names)
    for name, values in source_arrays.items():
        if name in xyz_keys:
            continue
        if matched_source_indices is not None and len(values) == len(workdir_xyz):
            values = np.asarray(values)[matched_source_indices]
        if len(values) != merged_point_count:
            raise RuntimeError(
                f"Dimension-length mismatch for {scene_path.name}:{name} source={len(values)} merged={merged_point_count}"
            )
        if name in existing_names:
            continue
        _assign_dimension(merged, existing_names, name, values)

    for name, values in _canonical_prediction_aliases(source_arrays).items():
        if matched_source_indices is not None and len(values) == len(workdir_xyz):
            values = np.asarray(values)[matched_source_indices]
        if len(values) != merged_point_count:
            raise RuntimeError(
                f"Alias-length mismatch for {scene_path.name}:{name} source={len(values)} merged={merged_point_count}"
            )
        _assign_dimension(merged, existing_names, name, values)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    merged.write(str(output_path))
    return output_path


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

    with tempfile.TemporaryDirectory(prefix="production_3d_merge_") as temp_dir_name:
        temp_dir = Path(temp_dir_name)

        for scene_path in scene_paths:
            output_path = build_output_path(scene_path, args.input_root, args.output_root)
            if output_path.exists() and not args.overwrite:
                print(f"skip  {output_path}")
                continue

            source_path = scene_path
            if (
                should_merge_with_structure(scene_path)
                and args.structure_root is not None
                and args.forestformer_npy_root is not None
            ):
                relative_path = scene_path.relative_to(args.input_root)
                merged_path = (temp_dir / relative_path).with_suffix(".las")
                print(f"merge {scene_path} -> {merged_path}")
                if not args.dry_run:
                    source_path = merge_predictions_onto_structure(
                        scene_path=scene_path,
                        output_path=merged_path,
                        structure_root=args.structure_root,
                        forestformer_npy_root=args.forestformer_npy_root,
                        alignment_tolerance=args.alignment_tolerance,
                    )
                else:
                    source_path = merged_path

            output_path.parent.mkdir(parents=True, exist_ok=True)
            pipeline = build_pipeline(source_path, output_path, args.a_srs)
            print(f"build {source_path} -> {output_path}")
            if not args.dry_run:
                run_pipeline(pdal_path, pipeline)


if __name__ == "__main__":
    main()
