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
