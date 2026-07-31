# Step 3: 3D Structure Classification (fvc3dstruct)

Tools for fusing **structure-class labels** (from co-registered structure/mask LiDAR) onto a **Z-normalized** LiDAR cloud, followed by mallee crown instance assignment and crown-centric refinement.

This folder contains the `fvc3dstruct` Python package used by the Step 3 scripts and notebooks.

## What it does

The main pipeline (see `fvc3dstruct/pipeline.py`) implements a multi-stage workflow:

1. **Data-driven height priors** per class (from structure ↔ Z_norm matching)
2. **Cartesian voxel label transfer** (baseline 3D fusion)
3. **Mallee crown detection** (watershed over mallee evidence)
4. **Per-point instance IDs (`treeID`)** for mallee crowns
5. **Crown-centric polar refinement** (r/θ bins around crown centroid + height priors)
6. **Height-based consolidation** + crown consistency rules (e.g., woody debris ↔ mallee)
7. **LAS outputs** (vegetation-only product + optional full fused product)

## Package layout

```
fvc3dstruct/
├── README.md
├── __init__.py
├── config.py          # DEFAULT_CONFIG
├── pipeline.py        # process_tile()
├── voxel_merge.py     # cartesian_voxel_merge()
├── watershed.py       # mallee crown grid + centroids
├── polar_refine.py    # refine_with_polar_coords()
├── treeid.py          # treeID assignment utilities
├── height_stats.py    # compute_class_height_thresholds(), auto_mallee_min_height()
└── io_las.py          # LAS writers and helpers
```

## Installation / importing

This repo does not currently expose Step 3 as a standalone pip package (no `pyproject.toml` in this step). In practice you have two common options:

### Option A: Run from this repo (recommended)
- Run notebooks and scripts from within this repository workspace.
- Python will typically resolve imports because the step directory is already on the path in the notebook/session.

### Option B: Add Step 3 to `PYTHONPATH`

```bash
export PYTHONPATH="/home/nvidia/projects/fvc_structure/phase_4_3D_lidar_prep/step_3_structure_class:$PYTHONPATH"
```

## Script usage

### Batch tiles

Use `process_folder()` to batch-process a directory of tiles.

Expected filename patterns (see `fvc3dstruct/pipeline.py`):

- Z_norm tiles: `z_norm_csf_tiles_multispectral.{tile_id}.las`
- Structure/mask tiles: `mask_tiles_multispectral.{tile_id}.las`

```python
from pathlib import Path
from fvc3dstruct.pipeline import process_folder

z_dir = Path("/path/to/z_norm/")
s_dir = Path("/path/to/mask/")
out_dir = Path("/path/to/out/")

process_folder(
  z_dir=z_dir,
  s_dir=s_dir,
  out_dir=out_dir,
  site_name="",  # optional prefix
  config={
    # optional overrides
    "voxel_size_xy": 0.10,
    "voxel_size_z": 0.10,
    "enable_radial_fill": False,
  },
)
```

### Single tile (Python)

```python
from fvc3dstruct.pipeline import process_tile

process_tile(
    z_path="/path/to/z_norm_tile.las",
    s_path="/path/to/mask_tiles_multispectral.tileid.las",
    out_path="/path/to/output_struct3d_tile.las",
    config={
        # overrides (optional)
        "voxel_size_xy": 0.10,
        "voxel_size_z": 0.10,
        "r_bin": 0.10,
        "theta_bin_deg": 10,
        "min_conf": 0.30,
        "min_flip_ratio": 1.50,
    },
)
```

`process_tile()` now returns a summary dictionary with run metrics, including:

- Stage-2 labeled-point percentage
- non-zero structure voxel count
- Stage-4 label-change percentage
- final vegetation-retention percentage
- final non-zero treeID count

For analysis-only runs, set `"write_outputs": False` in the config to skip LAS writing.

### Voxel-size sensitivity sweep for a supplementary figure

Use the helper below to compare the active Step-3 pipeline across voxel sizes such as
5, 10, 15, 20, and 25 cm. The script writes:

- `voxel_size_sweep_per_tile.csv`
- `voxel_size_sweep_average.csv`
- `supplementary_voxel_size_sensitivity.png`

By default it runs in summary-only mode and does not write LAS outputs.

```bash
python phase_4_3D_lidar_prep/step_3_structure_class/compare_voxel_sizes.py \
  --z-dir /path/to/z_norm \
  --s-dir /path/to/structure \
  --output-dir /path/to/voxel_sweep \
  --sizes 0.05,0.10,0.15,0.20,0.25
```

Add `--write-las` if you also want the per-run LAS products saved under size-specific folders.

### Validation figure for the 2D to 3D label transfer workflow

Use the helper below to create a four-panel figure showing:

- the original 2D structure mask,
- the initial 2D to 3D label projection,
- the refined 3D labels after voxel and polar refinement,
- and a vertical HAG view of the refined cloud.

```bash
python phase_4_3D_lidar_prep/step_3_structure_class/make_label_transfer_validation_figure.py \
  --mask /media/laura/8402326D023263F8/calperumResearch/site1_1_DD0001/imagery/clipped/mask/mask_tiles_multispectral.109.las \
  --projected-las /media/laura/8402326D023263F8/calperumResearch/site1_1_DD0001/lidar/clipped/3d_structure_mask_malleid/mask_tiles_multispectral.109.las \
  --refined-las /media/laura/8402326D023263F8/calperumResearch/site1_1_DD0001/lidar/clipped/3d_structure_mask_malleid/znorm_3Dlabels_109.las \
  --output //media/laura/8402326D023263F8/calperumResearch/site1_1_DD0001/lidar/clipped/validation/label_transfer_validation_tile109.png
```

### Batch validation summaries for Tables 5 and 6

Use the script below to compute validation metrics across every matched tile in a
site. It writes both per-tile tables and across-tile mean tables for:

- Table 5: overall agreement and class-wise IoU between projected and refined labels
- Table 6: class-wise height plausibility of the refined labels using `Z_norm`

```bash
python phase_4_3D_lidar_prep/step_3_structure_class/summarize_label_transfer_validation.py \
  --site-root /media/laura/8402326D023263F8/calperumResearch/site1_1_DD0001 \
  --site-condition Medium
```

Default directories inferred from `--site-root`:

- masks: `inputs/masks/tiles_3072/raw/structure_class/annotation_raster/downsample_5cm_nearest`
- projected LAS: `lidar/clipped/mask/coreg/icp`
- refined LAS: `lidar/clipped/3d_structure_mask_malleid`
- outputs: `lidar/clipped/validation`

Outputs written to the validation folder:

- `table5_label_agreement_per_tile.csv`
- `table5_label_agreement_average.csv`
- `table6_height_plausibility_per_tile.csv`
- `table6_height_plausibility_average.csv`
- `treeid_instance_summary_per_tile.csv`
- `treeid_instance_summary_average.csv`
- matching `.md` files for quick manuscript copy/paste

Notes:

- Tile matching is based on the trailing numeric tile ID in each filename.
- Table 5 metrics are computed on the common occupied mask cells for each tile.
- Average tables are simple across-tile means, not point-weighted aggregates.
- The treeID summary table reports Step 4 watershed crown counts and Step 9 final non-zero `treeID` statistics so the last gallery row is reproducible in table form.
- The treeID summary also includes per-tile crown-instance validation metrics, including crown-count delta, crown-retention percentage, and mean points per watershed crown / final treeID.
- If your manuscript uses different numbering, you can rename the exported files without changing the metric logic.

## Notebooks

- `notebooks/01_run_pipeline_site01_workflow.ipynb`
  - End-to-end workflow for a representative site
  - Includes QC plots and 2D/3D before/after visualizations (notebook cell 28)

## Outputs

The pipeline writes a LAS product at `out_path`.

Key points:

- Outputs are written as **LAS 1.4 / Point Format 6 (PF6)**.
- Structure labels are carried in the LAS **`classification`** field using the mask codes **64–68**.
- Height-above-ground is written in an ExtraBytes dimension **`Z_norm`**.
- Mallee instance IDs are written in an ExtraBytes dimension **`treeID`** (Option A: only canopy mallee points get a non-zero `treeID`).

Depending on config, it can also write stage outputs and/or a “full fused” LAS (same points as the input Z_norm tile) with additional dimensions.

Common fields:
- `Z_norm` (height above ground)
- per-point fused `classification`
- optional `treeID` for mallee crown instances

Common output names when using `process_folder()`:

- Veg-only product: `{site_name_}tile{tile_id}_struct_polar.las`
- Optional full fused product: `{site_name_}tile{tile_id}_struct_polar_full_fused.las`

## Configuration

Defaults live in `fvc3dstruct/config.py` as `DEFAULT_CONFIG`. Typical knobs:
- `voxel_size_xy`, `voxel_size_z` (voxel transfer resolution)
- `r_bin`, `theta_bin_deg` (polar refinement bins)
- `min_conf`, `min_flip_ratio` (update/flip thresholds)
- watershed parameters (`min_tree_dist`, `smoothing_sigma`) or set to `"auto"`

Output controls:

- `write_full_fused` (default `False`): also write the full fused LAS (same points as input Z_norm)
- `save_stage_outputs` + `stage_out_dir`: write intermediate stage snapshots for debugging/QC
- `write_veg_mask_npy` (default `False`): write a boolean vegetation mask aligned to the full Z_norm LAS

## Dependencies

At runtime this step mainly relies on:
- `laspy` (LAS I/O)
- `numpy`
- `scipy` (spatial / interpolation utilities used in supporting modules)
- `matplotlib` (notebook QC plots)

Environment management in this project is commonly done via `micromamba` (see the repo-level environment files).
