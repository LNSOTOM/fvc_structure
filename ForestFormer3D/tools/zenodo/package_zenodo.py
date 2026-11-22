#!/usr/bin/env python3
"""Build the files for the Zenodo datasets and pre-trained models record.

Output (in ``--out``)::

    test_data_medium_L1.zip   rangeland UAS L1 PLY tiles, one zip per source folder,
    test_data_low_L1.zip      with the folder names the experiment runbook uses
                              (each tile is shipped once; experiments select
                              tiles through the split lists)
    meta_data.zip             train/val/test split lists
    model_<experiment>.zip    final-epoch checkpoint, config, training logs
    forestformer3d_predictions_rangeland.zip
                              ForestFormer3D test predictions behind Tables 7-10
                              (final-epoch checkpoints), configs, logs, metrics
    label_transfer_inputs_<site>.zip
                              per-tile inputs and output of the 2D-to-3D label
                              transfer (with --calperum-root)
    fvc_structure_code.zip    snapshot of the code repository (with --code-repo)
    segmentanytree_predictions_rangeland.zip
                              zero-shot SegmentAnyTree predictions, metrics, logs
                              (with --segmentanytree-root; forest runs with --sat-forest)
    README.md                 record description with file table and class mapping
    SHA256SUMS                checksums of every file above
    manifest.json             machine-readable summary

Steps:

1. Runs ``check_split_overlap.py`` on ``meta_data`` and stops on any error.
2. Checks that every scene named in a packaged list exists as ``<scene>.ply``
   in one of the data folders.
3. Checks that every checkpoint exists, then writes the zips.

Nothing is uploaded. Run with ``--dry-run`` first to see what would be packaged.

Example (inside the container)::

    python tools/zenodo/package_zenodo.py \
        --data-root /workspace/data/ForAINetV2 \
        --work-dirs /workspace/work_dirs \
        --out /workspace/zenodo_release \
        --dry-run
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import zipfile
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from check_split_overlap import check  # noqa: E402

# The release covers the experiments in the main manuscript, which use UAS L1
# only. UAS L2 (supplementary material) is not packaged.
DATA_DIRS = ["test_data_medium_L1", "test_data_low_L1"]

# (zip name, work_dir, checkpoint file, description)
EXPERIMENTS = [
    ("exp2A_joint_finetune", "exp2A_epoch3000_train", "epoch_3000_fix.pth",
     "Experiment 2A: fine-tuned from FOR-instanceV2 on medium + low L1 jointly"),
    ("exp2B_stage1_medium", "exp2B_stage1_medium_train", "epoch_3000_fix.pth",
     "Experiment 2B stage 1: fine-tuned from FOR-instanceV2 on medium L1"),
    ("exp2B_stage2_low", "exp2B_stage2_low_train", "epoch_3000_fix.pth",
     "Experiment 2B stage 2: initialised from 2B stage 1, fine-tuned on low L1"),
    ("exp3A_scratch_joint", "exp3A_scratch_train", "epoch_3000_fix.pth",
     "Experiment 3A: trained from scratch on medium + low L1 jointly"),
    ("exp3B_scratch_medium", "exp3B_scratch_medium_train", "epoch_3000.pth",
     "Experiment 3B: trained from scratch on medium L1 (checkpoint not spconv-converted)"),
    ("exp3C_scratch_low", "exp3C_scratch_low_train", "epoch_3000_fix.pth",
     "Experiment 3C: trained from scratch on low L1"),
]


# Per-tile inputs and output of the 2D-to-3D label transfer (Phases 3-4), by
# site: (folder in zip, glob relative to the site directory, description).
LABEL_TRANSFER = {
    "medium": ("site1_1_DD0001", [
        ("1_structure_annotations_shp", "inputs/masks/tiles_3072/raw/structure_class/annotations_shp/*",
         "2D growth-form annotation polygons"),
        ("2_structure_label_raster", "inputs/masks/tiles_3072/raw/structure_class/annotation_raster/*",
         "2D growth-form label rasters"),
        ("3_structure_labelled_image_points", "imagery/clipped/mask/mask_tiles_multispectral.*.las",
         "image-derived points labelled with growth-form classes 64-68"),
        ("4_lidar_ground_csf", "lidar/clipped/raster/ground_classification/csf/csf_tiles_multispectral.*",
         "CSF ground points (class 2) per tile, with the PDAL pipeline and parameters (.json)"),
        ("5_lidar_znorm", "lidar/clipped/raster/z_norm/csf/z_norm_csf_tiles_multispectral.*.las",
         "co-registered UAS L1 LiDAR with height above the CSF ground (Z_norm)"),
        ("6_fused_3d_labels", "lidar/clipped/3d_structure_mask_malleid/znorm_3Dlabels_*.las",
         "label-transfer output: LiDAR with growth-form class, Z_norm and treeID"),
    ]),
    "low": ("site2_1_DD0011", [
        ("1_structure_annotations_shp", "inputs/masks/tiles_3072/raw/structure_class/annotation_shp/*",
         "2D growth-form annotation polygons"),
        ("2_structure_label_raster", "inputs/masks/tiles_3072/raw/structure_class/annotation_raster/*",
         "2D growth-form label rasters"),
        ("3_structure_labelled_image_points", "imagery/clipped/mask/mask_tiles_multispectral.*.las",
         "image-derived points labelled with growth-form classes 64-68"),
        ("4_lidar_ground_csf", "lidar/clipped/raster/ground_classification/csf/csf_tiles_multispectral.*",
         "CSF ground points (class 2) per tile, with the PDAL pipeline and parameters (.json)"),
        ("5_lidar_znorm", "lidar/clipped/raster/z_norm/csf/z_norm_csf_tiles_multispectral.*.las",
         "co-registered UAS L1 LiDAR with height above the CSF ground (Z_norm)"),
        ("6_fused_3d_labels", "lidar/clipped/3d_structure_mask_malleid/znorm_3Dlabels_l_*.las",
         "label-transfer output: LiDAR with growth-form class, Z_norm and treeID"),
    ]),
}


def label_transfer_members(calperum_root: Path, site: str, problems: list[str]) -> list[tuple[Path, str]]:
    site_dir, parts = LABEL_TRANSFER[site]
    members = []
    for folder, pattern, _ in parts:
        files = sorted(f for f in (calperum_root / site_dir).glob(pattern) if f.is_file())
        n_tiles = len({f.name.split(".")[0] + "." + f.name.split(".")[1] for f in files}) \
            if folder.endswith(("shp", "csf")) else len(files)
        if n_tiles != 15:
            problems.append(f"label transfer {site}/{folder}: expected 15 tiles, found {n_tiles} ({pattern})")
        members += [(f, f"label_transfer_{site}/{folder}/{f.name}") for f in files]
    return members


# ForestFormer3D test runs behind the main-manuscript tables, all using the
# final-epoch checkpoint: (experiment, site, work_dir under --work-dirs, table).
FF3D_EVALS = [
    ("exp1A_zero_shot", "medium", "zero_shot_medium_L1_forest", "Table 7"),
    ("exp1A_zero_shot", "low", "zero_shot_low_L1_forest", "Table 7"),
    ("exp2A_joint_finetune", "medium", "exp2A_epoch3000_eval_medium_L1", "Table 8"),
    ("exp2A_joint_finetune", "low", "exp2A_epoch3000_eval_low_L1", "Table 8"),
    ("exp2B_stage1_medium", "medium", "exp2B_stage1_eval_medium_L1", "Table 9"),
    ("exp2B_stage2_low", "low", "exp2B_stage2_eval_low_L1", "Table 9"),
    ("exp3A_scratch_joint", "medium", "exp3A_scratch_eval_medium_L1", "Table 10"),
    ("exp3A_scratch_joint", "low", "exp3A_scratch_eval_low_L1", "Table 10"),
    ("exp3B_scratch_medium", "medium", "exp3B_scratch_eval_medium_L1", "Table 10"),
    ("exp3C_scratch_low", "low", "exp3C_epoch_3000_eval_low_L1", "Table 10"),
]

CLASS_MAPPING = """\
| Growth form | 2D label | Mask/LAS | PLY | Model ID | Paper class |
|---|---|---|---|---|---|
| Forb (FO) | 0 | 64 | 1 | 0 | ground |
| Woody debris (WD) | 1 | 65 | 2 | - | > 0.5 m reclassified as mallee; <= 0.5 m unlabelled |
| Grass hummock (GH) | 2 | 66 | 3 | 0 | ground |
| Shrub (SH) | 3 | 67 | 4 | 1 | wood |
| Mallee (MA) | 4 | 68 | 5 | 2 | leaf |

"Ground", "wood" and "leaf" are the shared forest-model categories used to
interpret rangeland growth forms; they do not mean the labels literally
describe bare soil, woody material or leaves. The PLY files keep the PLY IDs;
the ForestFormer3D exporter maps PLY 3/4/5 (and 1/4/5) to model IDs 0/1/2.
"""


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def human(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} GB"


def write_code_zip(repo: Path, ref: str, out: Path) -> dict:
    """Snapshot tracked files at ``ref`` with git archive; return provenance."""
    commit = subprocess.run(["git", "-C", str(repo), "rev-parse", f"{ref}^{{commit}}"],
                            check=True, capture_output=True, text=True).stdout.strip()
    tags = subprocess.run(["git", "-C", str(repo), "tag", "--points-at", commit],
                          check=True, capture_output=True, text=True).stdout.split()
    tmp = out.with_suffix(".zip.part")
    subprocess.run(["git", "-C", str(repo), "archive", "--format=zip", "--prefix=fvc_structure/",
                    "-o", str(tmp), commit], check=True)
    tmp.replace(out)
    return {"file": out.name, "commit": commit, "tags": tags,
            "repository": "https://github.com/LNSOTOM/fvc_structure"}


def write_zip(out: Path, members: list[tuple[Path, str]]) -> None:
    tmp = out.with_suffix(".zip.part")
    with zipfile.ZipFile(tmp, "w", allowZip64=True) as zf:
        for src, arc in members:
            method = zipfile.ZIP_STORED if src.suffix == ".pth" else zipfile.ZIP_DEFLATED
            zf.write(src, arc, compress_type=method)
    tmp.replace(out)


def run_files(work_dir: Path, prefix: str) -> list[tuple[Path, str]]:
    """Config dumped by mmengine and the logs of the latest run in ``work_dir``."""
    members = [(p, f"{prefix}/{p.name}") for p in sorted(work_dir.glob("*.py"))]
    runs = sorted(d for d in work_dir.iterdir() if d.is_dir() and d.name[:8].isdigit())
    if runs:
        run = runs[-1]
        for p in sorted(run.rglob("*")):
            if p.is_file() and (p.suffix in {".log", ".json", ".py"}):
                members.append((p, f"{prefix}/logs/{p.relative_to(run)}"))
    return members


def model_members(work_dirs: Path, wd: str, ckpt: Path, name: str) -> list[tuple[Path, str]]:
    """Final-epoch checkpoint with its training config and logs."""
    return [(ckpt, f"{name}/{ckpt.name}")] + run_files(work_dirs / wd, name)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-root", type=Path, required=True, help="ForAINetV2 data dir (has meta_data/ and test_data_*)")
    p.add_argument("--work-dirs", type=Path, required=True, help="Experiment work_dirs containing the checkpoints")
    p.add_argument("--out", type=Path, required=True, help="Output directory for the release files")
    p.add_argument("--data-dirs", nargs="+", default=DATA_DIRS, help="PLY folders under --data-root to package")
    p.add_argument("--skip-data", action="store_true", help="Package models only (lists need the data to be checked)")
    p.add_argument("--skip-models", action="store_true", help="Package data and lists only")
    p.add_argument("--segmentanytree-root", type=Path,
                   help="SegmentAnyTree output dir (has calperum/ and ForAINetV2/forest/) to package predictions")
    p.add_argument("--calperum-root", type=Path,
                   help="Root with site1_1_DD0001/ and site2_1_DD0011/ to package label-transfer inputs")
    p.add_argument("--code-repo", type=Path,
                   help="Git repository to snapshot into fvc_structure_code.zip (tracked files only)")
    p.add_argument("--code-ref", default="HEAD", help="Commit, branch or tag to snapshot (default: HEAD)")
    p.add_argument("--skip-ff3d-predictions", action="store_true",
                   help="Do not package the ForestFormer3D test predictions")
    p.add_argument("--sat-forest", action="store_true",
                   help="Also package SegmentAnyTree predictions on the FOR-instanceV2 forest test set")
    p.add_argument("--dry-run", action="store_true", help="Run all checks and print the plan; write nothing")
    args = p.parse_args()

    meta = args.data_root / "meta_data"
    problems: list[str] = []

    # 1. split overlap
    errors, _, lists = check(meta)
    if errors:
        print("\nStopping: fix the split overlaps above before packaging.")
        return 1

    # 2. data folders and scene coverage
    data_plan: list[tuple[str, list[tuple[Path, str]]]] = []
    available: set[str] = set()
    if not args.skip_data:
        for d in args.data_dirs:
            src = args.data_root / d
            if not src.is_dir():
                problems.append(f"data folder missing: {src}")
                continue
            files = sorted(f for f in src.rglob("*") if f.is_file())
            available |= {f.stem for f in files if f.suffix == ".ply"}
            data_plan.append((d, [(f, f"{d}/{f.relative_to(src)}") for f in files]))
        # Ship only lists whose scenes are all in the packaged data; the others
        # (forest lists, lists using pre-rename tile names) would not resolve.
        skipped = {}
        for name, ids in list(lists.items()):
            missing = sorted(set(ids) - available)
            if missing:
                skipped[name] = missing
                del lists[name]
        for name, missing in skipped.items():
            print(f"SKIP   {name}: {len(missing)} scene(s) not in packaged data, e.g. {missing[0]}")
        if not any(n.startswith("test_list_exp2_holdout") for n in lists):
            problems.append("no test_list_exp2_holdout_* list would be packaged")

    # 3. checkpoints
    model_plan: list[tuple[str, str, list[tuple[Path, str]]]] = []
    if not args.skip_models:
        for name, wd, ck, desc in EXPERIMENTS:
            ckpt = args.work_dirs / wd / ck
            if not ckpt.is_file():
                problems.append(f"checkpoint missing: {ckpt}")
                continue
            members = model_members(args.work_dirs, wd, ckpt, name)
            if not any(a.endswith(".py") and "/logs/" not in a for _, a in members):
                problems.append(f"{name}: no config (*.py) found in {args.work_dirs / wd}")
            model_plan.append((name, desc, members))

    # 4. ForestFormer3D test predictions (final-epoch checkpoints)
    pred_plan: list[tuple[str, str, list[tuple[Path, str]]]] = []
    if not args.skip_ff3d_predictions:
        members = []
        for exp, site, wd, _ in FF3D_EVALS:
            src = args.work_dirs / wd
            plys = sorted(src.glob("*.ply")) if src.is_dir() else []
            if not plys:
                problems.append(f"ForestFormer3D predictions missing: {src}")
                continue
            prefix = f"forestformer3d_rangeland/{exp}/{site}_L1"
            members += [(f, f"{prefix}/predictions/{f.name}") for f in plys]
            members += [(f, f"{prefix}/{f.name}") for f in sorted(src.glob("*.json"))]
            members += run_files(src, prefix)
        pred_plan.append(("forestformer3d_predictions_rangeland",
                          "ForestFormer3D test predictions and metrics, final-epoch checkpoints "
                          "(Experiments 1A, 2 and 3; Tables 7-10)", members))

    # 5. Label-transfer inputs and outputs per site
    lt_plan: list[tuple[str, list[tuple[Path, str]]]] = []
    if args.calperum_root:
        for site in LABEL_TRANSFER:
            lt_plan.append((site, label_transfer_members(args.calperum_root, site, problems)))

    # 6. SegmentAnyTree zero-shot predictions (no new weights: the model is Wielgosz et al.'s)
    if args.segmentanytree_root:
        groups = [("rangeland", "calperum", "rangeland UAS L1 tiles (medium and low)")]
        if args.sat_forest:
            groups.append(("forest", "ForAINetV2", "FOR-instanceV2 forest test set"))
        for tag, sub, desc in groups:
            base = args.segmentanytree_root / sub
            runs = sorted(d for d in base.iterdir() if d.is_dir() and "_L2_" not in d.name) if base.is_dir() else []
            if not runs:
                problems.append(f"SegmentAnyTree: no run folders in {base}")
                continue
            members = []
            for run in runs:
                n_laz = len(list((run / "results").glob("*.laz")))
                n_csv = len(list((run / "metrics").glob("*.csv")))
                if not n_laz or not n_csv:
                    problems.append(f"SegmentAnyTree {run.name}: {n_laz} result .laz, {n_csv} metrics .csv")
                members += [(f, f"segmentanytree_{tag}/{run.name}/{f.relative_to(run)}")
                            for f in sorted(run.rglob("*")) if f.is_file()]
            pred_plan.append((f"segmentanytree_predictions_{tag}", f"SegmentAnyTree zero-shot predictions and instance metrics: {desc}", members))

    print("\nPlan:")
    for d, members in data_plan:
        size = sum(s.stat().st_size for s, _ in members)
        n_ply = sum(1 for s, _ in members if s.suffix == ".ply")
        print(f"  {d}.zip  {n_ply} PLY tiles, {len(members)} files, {human(size)} uncompressed")
    if not args.skip_data:
        print(f"  meta_data.zip  {len(lists)} list files: {', '.join(lists)}")
    for name, _, members in model_plan:
        size = sum(s.stat().st_size for s, _ in members)
        print(f"  model_{name}.zip  {len(members)} files, {human(size)}")
        for _, arc in members:
            print(f"      {arc}")

    for zname, _, members in pred_plan:
        runs = sorted({"/".join(a.split("/")[1:3]) if a.startswith("forestformer3d") else a.split("/")[1]
                       for _, a in members})
        size = sum(s.stat().st_size for s, _ in members)
        print(f"  {zname}.zip  {len(runs)} runs, {len(members)} files, {human(size)}")
        for r in runs:
            print(f"      {r}")

    for site, members in lt_plan:
        print(f"  label_transfer_inputs_{site}.zip  {len(members)} files, "
              f"{human(sum(s.stat().st_size for s, _ in members))}")

    if problems:
        print("\nProblems:")
        for line in problems:
            print(f"  - {line}")
        print("\nStopping: fix the problems above (or narrow with --data-dirs/--skip-*).")
        return 1
    if args.dry_run:
        print("\nDry run: all checks passed, nothing written.")
        return 0

    # write
    args.out.mkdir(parents=True, exist_ok=True)
    manifest: dict = {"created": date.today().isoformat(), "data": [], "models": [], "predictions": [], "lists": {}}
    for d, members in data_plan:
        print(f"writing {d}.zip ...", flush=True)
        write_zip(args.out / f"{d}.zip", members)
        manifest["data"].append({"file": f"{d}.zip", "ply_tiles": sum(1 for s, _ in members if s.suffix == ".ply")})
    if not args.skip_data:
        write_zip(args.out / "meta_data.zip", [(meta / n, f"meta_data/{n}") for n in lists])
        manifest["lists"] = {n: len(v) for n, v in lists.items()}
    for name, desc, members in model_plan:
        print(f"writing model_{name}.zip ...", flush=True)
        write_zip(args.out / f"model_{name}.zip", members)
        manifest["models"].append({"file": f"model_{name}.zip", "description": desc,
                                   "checkpoint": members[0][1]})

    for zname, desc, members in pred_plan:
        print(f"writing {zname}.zip ...", flush=True)
        write_zip(args.out / f"{zname}.zip", members)
        manifest["predictions"].append({"file": f"{zname}.zip", "description": desc})

    manifest["label_transfer"] = []
    for site, members in lt_plan:
        print(f"writing label_transfer_inputs_{site}.zip ...", flush=True)
        write_zip(args.out / f"label_transfer_inputs_{site}.zip", members)
        manifest["label_transfer"].append({"file": f"label_transfer_inputs_{site}.zip", "site": site,
                                           "site_id": LABEL_TRANSFER[site][0]})

    if args.code_repo:
        print("writing fvc_structure_code.zip ...", flush=True)
        manifest["code"] = write_code_zip(args.code_repo, args.code_ref, args.out / "fvc_structure_code.zip")

    release = sorted(f for f in args.out.glob("*.zip"))
    sums = {f.name: sha256(f) for f in release}
    sizes = {f.name: f.stat().st_size for f in release}
    for section in ("data", "models", "predictions", "label_transfer"):
        for entry in manifest.get(section, []):
            entry["sha256"] = sums[entry["file"]]
            entry["bytes"] = sizes[entry["file"]]
    if "code" in manifest:
        manifest["code"]["sha256"] = sums[manifest["code"]["file"]]
        manifest["code"]["bytes"] = sizes[manifest["code"]["file"]]
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    (args.out / "README.md").write_text(render_readme(manifest, sizes), encoding="utf-8")
    sums["README.md"] = sha256(args.out / "README.md")
    sums["manifest.json"] = sha256(args.out / "manifest.json")
    (args.out / "SHA256SUMS").write_text("".join(f"{h}  {n}\n" for n, h in sorted(sums.items())), encoding="utf-8")

    print(f"\nDone: {len(release)} zip(s), README.md, manifest.json and SHA256SUMS in {args.out}")
    print(f"Verify with: (cd {args.out} && sha256sum -c SHA256SUMS)")
    return 0


def render_readme(manifest: dict, sizes: dict[str, int]) -> str:
    rows_data = "\n".join(f"| `{e['file']}` | {e['ply_tiles']} PLY tiles | {human(e['bytes'])} |"
                          for e in manifest["data"])
    rows_models = "\n".join(f"| `{e['file']}` | {e['description']} | {human(e['bytes'])} |"
                            for e in manifest["models"])
    rows_pred = "\n".join(f"| `{e['file']}` | {e['description']} | {human(e['bytes'])} |"
                          for e in manifest["predictions"])
    rows_ff3d = "\n".join(f"| {e} | {site} | `{wd}` | {t} |" for e, site, wd, t in FF3D_EVALS)
    pred_section = "" if not rows_pred else f"""
## Predictions: ForestFormer3D vs SegmentAnyTree on the rangeland sites

Saved test predictions for both models on the UAS L1 rangeland tiles, so the
main-manuscript results can be inspected and re-scored without a GPU.

| File | Contents | Size |
|---|---|---|
{rows_pred}

**ForestFormer3D** (`forestformer3d_predictions_rangeland.zip`): one folder per
experiment and site, `forestformer3d_rangeland/<experiment>/<site>_L1/`, with
predicted tiles (`predictions/*.ply`, tile-local coordinates), the evaluation
config, the evaluation log (final metrics on its last lines) and, where saved,
a metrics JSON. All runs use the final-epoch checkpoint. Experiment 1A covers
all 15 tiles per site; Experiments 2 and 3 cover the held-out test tiles.

| Experiment | Site | Source run | Paper |
|---|---|---|---|
{rows_ff3d}

**SegmentAnyTree** (`segmentanytree_predictions_*.zip`; Wielgosz et al., 2024,
https://doi.org/10.1016/j.rse.2024.114367) was applied zero-shot as an
instance-only reference (Experiment 1B); no SegmentAnyTree weights were trained
or changed, so none are included. Get the model from
https://github.com/SmartForest-no/SegmentAnyTree. Each run folder holds the
predicted tiles (`results/*_out.laz`, `PredInstance=0` is unassigned), instance
metrics at IoU 0.5 per file and per region (`metrics/*.csv`, with plots) and
inference logs (`runs/`). It covers all 15 tiles per site.

Compare the two models on Experiment 1 (both zero-shot, all 15 tiles); for
the fine-tuned and from-scratch ForestFormer3D models, compare on the held-out
test tiles only.
"""
    lt = manifest.get("label_transfer") or []
    lt_rows = "\n".join(f"| `{e['file']}` | {e['site']} ({e['site_id']}) | {human(e['bytes'])} |" for e in lt)
    lt_parts = "\n".join(f"| `{folder}/` | {desc} |" for folder, _, desc in LABEL_TRANSFER["medium"][1])
    lt_section = "" if not lt else f"""
## Label-transfer inputs (Phases 3-4)

The per-tile inputs and output of the 2D-to-3D label transfer for the same
15 + 15 tiles, so the label transfer can be rerun or checked tile by tile.
LAS files are in EPSG:7854 (GDA2020 / MGA zone 54). The 5-band predictor
tiles and full orthomosaics are in https://doi.org/10.5281/zenodo.15036860.

| File | Site | Size |
|---|---|---|
{lt_rows}

Each archive has one folder per stage, in processing order:

| Folder | Contents |
|---|---|
{lt_parts}

Growth-form classes are 64 forb, 65 woody debris, 66 grass hummock, 67 shrub
and 68 mallee (see Class mapping).

### From label transfer to model input

`6_fused_3d_labels` is converted to the PLY tiles in `test_data_*_L1.zip`
(Phase 5, `prepare_lidar.py`): same tiles, same points, with classes
remapped 64->1, 65->2, 66->3, 67->4, 68->5. The ForestFormer3D loader then
maps them to the model classes 0 ground, 1 wood, 2 leaf. The tile number
links the files:

| Site | Stage 6 (LAS, classes 64-68) | Released PLY (classes 1-5) |
|---|---|---|
| Medium | `label_transfer_medium/6_fused_3d_labels/znorm_3Dlabels_<tile>.las` | `test_data_medium_L1/znorm_3Dlabels_m_L1_<tile>.ply` |
| Low | `label_transfer_low/6_fused_3d_labels/znorm_3Dlabels_l_<tile>.las` | `test_data_low_L1/znorm_3Dlabels_l_L1_<tile>.ply` |

For example, medium tile 106: `znorm_3Dlabels_106.las` ->
`znorm_3Dlabels_m_L1_106.ply`.
"""
    code = manifest.get("code")
    code_licence_row = "" if not code else (
        "| `fvc_structure_code.zip` | MIT for original code; third-party folders under their own licences |\n")
    code_section = "" if not code else f"""
## Code

`{code['file']}` ({human(code['bytes'])}) is a snapshot of
{code['repository']} at commit `{code['commit'][:12]}`{' (tag ' + ', '.join(code['tags']) + ')' if code['tags'] else ''},
containing the full workflow: 2D-to-3D label transfer (phases 1-5), the
ForestFormer3D and SegmentAnyTree setups and the experiment runbook
(`ForestFormer3D/README_ForAINetV2_experiments.md`). Original code is MIT;
vendored third-party code keeps its own licence (see `LICENSE` in the zip):
ForestFormer3D CC BY-NC 4.0, SegmentAnyTree MIT/BSD, GeFolki GPL-3.0.
"""
    rows_lists = "\n".join(f"| `{n}` | {k} |" for n, k in manifest["lists"].items())
    return f"""\
# Rangeland 3D vegetation structure: labelled UAS LiDAR tiles and pre-trained ForestFormer3D models

Data and models accompanying the preprint *From 2D labels to 3D structure:
Scalable label transfer and benchmarking of 3D vegetation models in rangeland
ecosystems* (Sotomayor, Lucieer, Turner, Barthelemy and Kattenborn; EarthArXiv
preprint, version 2, https://doi.org/10.31223/X5948M).

Rangeland tiles come from two Calperum (South Australia) TERN sites:
a medium-density site (grass hummock, shrub, mallee) and a low-density site
(forb, shrub, sparse mallee). Each tile is 30.70 x 30.70 m.

Step-by-step workflow guide: https://lnsotom.github.io/fvc_structure/

Code, setup and the full experiment runbook:
https://github.com/LNSOTOM/fvc_structure
(see `ForestFormer3D/README_ForAINetV2_experiments.md`).

## Data

Labelled PLY tiles, one archive per source folder. Each tile is included once;
experiments select tiles through the split lists in `meta_data.zip`.
Unzip all archives into `data/ForAINetV2/` so the runbook paths work unchanged.

Tile names are `znorm_3Dlabels_<density>_L1_<plot>`, with density `m`
(medium) or `l` (low). All tiles are UAS LiDAR from a DJI Zenmuse L1 on a
Matrice 300 RTK, May 2022 (50 m AGL, 5 m/s); 15 tiles per site. This record
covers the experiments in the main manuscript, which use UAS L1 only.

| File | Contents | Size |
|---|---|---|
{rows_data}
| `meta_data.zip` | train/val/test split lists | {human(sizes.get('meta_data.zip', 0))} |

Split lists (scene count):

| List | Scenes |
|---|---|
{rows_lists}

`test_list_medium_L1.txt` lists every medium L1 tile, the pool that the
Experiment 2 and 3 train, validation and hold-out lists are drawn from; it is
not a held-out test set. `test_list_exp2_mediumL1_lowL1.txt` is deliberately
empty (training-only cache export).

## Experiments

| Experiment | Setting | Training data (L1) | Evaluation | Paper |
|---|---|---|---|---|
| 1A | ForestFormer3D zero-shot (forest-pretrained) | none | FOR-instanceV2 test split; all 15 L1 tiles per site | Table 7 |
| 1B | SegmentAnyTree zero-shot, instance metrics only | none | as 1A | Table 7 |
| 2A | Fine-tuned, medium + low jointly | `train/val_list_exp2_mediumL1_lowL1_fewshot.txt` (20/5) | `test_list_exp2_holdout_*` | Table 8 |
| 2B | Curriculum fine-tuning, medium then low | stage 1 `*_exp2B_mediumL1_only.txt` (10/3); stage 2 `*_exp2B_lowL1_only.txt` (10/2) | `test_list_exp2_holdout_*` | Table 9 |
| 3A | From scratch, medium + low jointly | as 2A | as 2A | Table 10 |
| 3B | From scratch, medium only | as 2B stage 1 | `test_list_exp2_holdout_medium_L1.txt` | Table 10 |
| 3C | From scratch, low only | `*_exp3C_lowL1_only.txt` (10/2) | `test_list_exp2_holdout_low_L1.txt` | Table 10 |

Per site the L1 split is 10/3/2 (medium) and 10/2/3 (low) train/val/test
tiles; Experiments 2 and 3 use the same partitions. Instances are matched at
IoU >= 0.5.

## Pre-trained models

Each archive holds the final-epoch (3000) model used for all results in the
main manuscript, with its training config and logs. `_fix` checkpoints are
converted for spconv compatibility (3B is unconverted).

| File | Model | Size |
|---|---|---|
{rows_models}

The forest-pretrained starting point for Experiment 2 is not included; download
`clean_forestformer.zip` from FOR-instanceV2 (https://doi.org/10.5281/zenodo.16742708).

{pred_section}{lt_section}{code_section}
## Class mapping

{CLASS_MAPPING}
## Related records

- Manual labels and UAS imagery: https://doi.org/10.5281/zenodo.15036860
- FOR-instanceV2 dataset and forest-pretrained model: https://doi.org/10.5281/zenodo.16742708

## Integrity

`SHA256SUMS` lists a SHA-256 checksum for every file. Check with
`sha256sum -c SHA256SUMS`.

## Licence and citation

| Files | Licence |
|---|---|
| `test_data_*.zip`, `meta_data.zip`, `label_transfer_inputs_*.zip` | CC BY 4.0 (https://creativecommons.org/licenses/by/4.0/) |
| `segmentanytree_predictions_*.zip` | CC BY 4.0 (https://creativecommons.org/licenses/by/4.0/) |
{code_licence_row}| `model_*.zip`, `forestformer3d_predictions_rangeland.zip` | CC BY-NC 4.0 (https://creativecommons.org/licenses/by-nc/4.0/) |

The models and their predictions derive from ForestFormer3D, which is licensed
CC BY-NC 4.0, so they are licensed for non-commercial use only.
The code is MIT-licensed; see https://github.com/LNSOTOM/fvc_structure.

LiDAR data were collected with field operations provided by TERN (Terrestrial
Ecosystem Research Network), Australia's land ecosystem observatory, enabled
by NCRIS. Please acknowledge TERN when using the data.

When using this record, please cite the preprint
(https://doi.org/10.31223/X5948M) and this record's DOI.
"""


if __name__ == "__main__":
    sys.exit(main())
