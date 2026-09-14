# Step 3: 3D Structure Classification (fvc3dstruct)

Tools for fusing **structure-class labels** (from co-registered structure/mask LiDAR) onto a **Z-normalized** LiDAR cloud, followed by mallee crown instance assignment and crown-centric refinement.

This is the **workflow guide** for the step: what to run, in what order, with what inputs and
outputs.

> For the **Python API** — function signatures, every `DEFAULT_CONFIG` key, and the `process_tile()`
> return fields — see [`fvc3dstruct/README.md`](fvc3dstruct/README.md).

## What it does

The main pipeline (see `fvc3dstruct/pipeline.py`) implements a multi-stage workflow:

1. **Data-driven height priors** per class (from structure ↔ Z_norm matching)
2. **Cartesian voxel label transfer** (baseline 3D fusion)
3. **Mallee crown detection** (watershed over mallee evidence)
4. **Per-point instance IDs (`treeID`)** for mallee crowns
5. **Crown-centric polar refinement** (r/θ bins around crown centroid + height priors)
6. **Height-based consolidation** + crown consistency rules (e.g., woody debris ↔ mallee)
7. **LAS outputs** (vegetation-only product + optional full fused product)

## What's in this step

```
step_3_structure_class/
├── fvc3dstruct/         The package — all pipeline logic  (see its own README)
├── main_run.py          Batch entry point
├── compare_voxel_sizes.py                   Voxel-size sensitivity sweep (site 1, pipeline counters only)
├── parameter_sensitivity.py                 One-at-a-time sweep of every tuned parameter, both sites
├── plot_parameter_sensitivity.py            Supplementary figures + tables from that sweep
├── plot_voxel_transfer.py                   The voxel transfer drawn on one tile window
├── make_label_transfer_validation_figure.py Four-panel validation figure
├── summarize_label_transfer_validation.py   Batch validation tables
├── test_pipeline.sh     Smoke test over main_run.py
└── notebooks/           Interactive workflows and analyses
```

| Script | Produces | Used by |
|---|---|---|
| `main_run.py` | Fused LAS tiles for a site | `test_pipeline.sh` |
| `compare_voxel_sizes.py` | Sweep CSVs + sensitivity figure | notebooks 04, 05 |
| `parameter_sensitivity.py` | Per-tile / summary CSVs for voxel size, matching radius, height-prior weight, `min_conf`, `min_flip_ratio` | `plot_parameter_sensitivity.py` |
| `plot_parameter_sensitivity.py` | Supplementary figures + the tables in [`README_parameter_sensitivity.md`](README_parameter_sensitivity.md) | notebook 04 |
| `plot_voxel_transfer.py` | Plan-view drawing of labels spreading past the 2D footprint at each voxel size | notebook 04 |
| `make_label_transfer_validation_figure.py` | Validation figure (PNG) | notebooks 02, 03 |
| `summarize_label_transfer_validation.py` | Tables 5 and 6 (CSV + MD) | — |

> **`main_run.py` needs editing before it runs.** Its input paths are hardcoded at lines 14–19
> (`/home/nvidia/...`, `/data/calperumResearch/...`) and point at a different machine. Change them
> to your own, or call `process_folder()` directly as shown below.

Legacy pre-`fvc3dstruct` scripts (the `merge_znorm_*` variants and friends) live on the `dev`
branch; their logic is now inside the package.

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
- Stage-4 mean label confidence and low-confidence-point percentage
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

This sweep reports how much the pipeline's own counters move, which cannot pick one voxel size
over another. The reviewer-facing comparison — every tuned parameter, both sites, scored against
the 2D labels — is `parameter_sensitivity.py`, written up in
[`README_parameter_sensitivity.md`](README_parameter_sensitivity.md).

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

| Notebook | What it covers |
|---|---|
| `01_run_pipeline_site01*.ipynb` | End-to-end workflow for site 01 (plain / `_improved` / `_workflow` / `_workflow_L2` variants); QC plots and 2D/3D before-after views |
| `01_run_pipeline_site11*.ipynb` | Same for site 11 |
| `02_label_transfer_validation_tile109.ipynb` | Single-tile validation of the 2D→3D transfer |
| `03_label_transfer_validation_gallery_tile109.ipynb` | Multi-panel validation gallery |
| `04_voxel_size_sensitivity_and_label_confidence.ipynb` | Voxel size and parameter sensitivity for the reviewer comments on 3.2.2: renders the tables and figures behind [`README_parameter_sensitivity.md`](README_parameter_sensitivity.md) and draws the voxels on tile 106 |
| `05_shrub_class_transition_matrix.ipynb` | Shrub class transition/confusion analysis |

Helper modules beside the notebooks (`flow.py`, `maskCoreg.py`, `maskCoreg_low.py`,
`test_label_transfer.py`, `test_label_transfer_low.py`) support these workflows.

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

Defaults live in `fvc3dstruct/config.py` as `DEFAULT_CONFIG`; pass a partial dict to override.
The knobs you are most likely to change from a script or notebook:

- `voxel_size_xy`, `voxel_size_z` — voxel transfer resolution
- `min_conf`, `min_flip_ratio` — label update/flip thresholds
- `write_outputs: False` — analysis-only runs, no LAS written

**Every key, with defaults and meaning, is tabulated in
[`fvc3dstruct/README.md`](fvc3dstruct/README.md#default_config).**

## Dependencies

At runtime this step mainly relies on:
- `laspy` (LAS I/O)
- `numpy`
- `scipy` (spatial / interpolation utilities used in supporting modules)
- `matplotlib` (notebook QC plots)

Environment management in this project is commonly done via `micromamba` (see the repo-level environment files).
