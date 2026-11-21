# Reproducible Co-registration Handoff (Phase 2)

Minimal file structure to reproduce the GeFolki LiDAR–imagery co-registration workflow.

## Deliverable Structure

```text
phase_2_lidar_coregistration/
├── README_phase2.md
├── REPRODUCIBILITY.md
├── configs/
│   └── coregistration_job.example.yml
├── gefolki/
│   ├── python/                        # Vendored GeFolki library (ONERA)
│   ├── python_myscripts/              # Config-driven processing scripts
│   │   ├── phase2_config.py           # Shared YAML config loader
│   │   ├── gefolki_preprocess.py
│   │   ├── gefolki_flow_l1.py
│   │   ├── gefolki_flow_l2.py
│   │   ├── align_lidar_with_flow_l1.py
│   │   └── align_lidar_with_flow_l2.py
│   └── notebooks/                     # Interactive versions + validation
└── step_0_checks/                     # CRS validation utilities
```

## What Each File Does

- `configs/coregistration_job.example.yml`: copy-and-edit job template describing one site + one LiDAR pass (paths and GeFolki parameters).
- `gefolki/python_myscripts/phase2_config.py`: resolves the job config (`--config` flag, `FVC_PHASE2_CONFIG` env var, or `configs/coregistration_job.local.yml`) and puts the vendored GeFolki library on `sys.path`.
- `gefolki_preprocess.py`: resamples the LiDAR DSM onto the imagery DSM grid (identical shape/CRS/transform).
- `gefolki_flow_l1.py` / `gefolki_flow_l2.py`: compute the GeFolki (EFolki) optical flow between the imagery DSM (base) and the LiDAR DSM (moving); the L2 variant additionally normalises the DSMs and writes a residual map.
- `align_lidar_with_flow_l1.py` / `align_lidar_with_flow_l2.py`: smooth the flow field and apply it chunk-wise to the full-density LAS point cloud.
- `step_0_checks/`: CRS inspection/reprojection helpers to run before co-registration.

## Inputs Required From A New User

Phase 2 starts from Phase 1 outputs. A new user must provide:

1. The imagery (RGB) DSM GeoTIFF for the site (base/reference grid).
2. The LiDAR DSM GeoTIFF from Phase 1 for the acquisition to align.
3. The original full-density georeferenced `.las`/`.laz` for that acquisition.
4. Writable output locations for the resampled DSM, flow fields (`.npy`) and the co-registered LAS.

All rasters must share the same CRS (check with `step_0_checks/crs.py` first).

## Environment Setup

From the repository root:

```bash
uv sync
```

The Phase 2 Python dependencies are included in the shared `pyproject.toml`.

## Configure A Co-registration Job

```bash
cp phase_2_lidar_coregistration/configs/coregistration_job.example.yml \
   phase_2_lidar_coregistration/configs/coregistration_job.local.yml
```

Edit the three sections of `coregistration_job.local.yml`:

- `preprocess`: `lidar_dsm`, `imagery_dsm`, `output`
- `flow`: `base_image`, `moving_image`, EFolki parameters, `flow_u_output`, `flow_v_output`
- `align`: `las_input`, `reference_grid`, `flow_u`, `flow_v`, `correction_direction`, `gaussian_sigma`, optional `las_output`

Parameter guidance:

- L1 flow: `radius: [16, 8, 4]`, `levels: 6` (up to ~64 px displacement); for larger displacements use `radius: [32, 16, 8, 4]`, `levels: 7`.
- L2 flow: `radius: [8, 4, 2]`, `levels: 6`.
- `correction_direction: -1` inverts the flow; flip to `1` if visual checks show the correction is backwards.
- `gaussian_sigma`: increase (3.0–4.0) if the corrected cloud shows holes, decrease (1.0) if the correction looks damped.

## Reproduce The Co-registration Run

Run the three steps in order from the repository root (one config describes one
site + one LiDAR pass; use the `_l1` or `_l2` script matching the pass):

```bash
cd phase_2_lidar_coregistration/gefolki/python_myscripts

# 1. Resample the LiDAR DSM onto the imagery DSM grid
uv run python gefolki_preprocess.py --config ../../configs/coregistration_job.local.yml

# 2. Compute GeFolki optical flow (choose the variant for the pass)
uv run python gefolki_flow_l2.py --config ../../configs/coregistration_job.local.yml

# 3. Apply the flow to the point cloud
uv run python align_lidar_with_flow_l2.py --config ../../configs/coregistration_job.local.yml
```

Alternatively set the config once:

```bash
export FVC_PHASE2_CONFIG=/absolute/path/to/coregistration_job.local.yml
```

The scripts are cell-based (`#%%`) and can also be run interactively (VS Code /
Jupytext); the notebook versions live in `gefolki/notebooks/`.

## Expected Outputs

- Resampled LiDAR DSM on the imagery grid: `preprocess.output`.
- Flow fields: `flow.flow_u_output`, `flow.flow_v_output` (`.npy`).
- Registered LiDAR DSM: `<moving_image stem>_coreg.tif` next to the moving image.
- L2 only: residual map `<moving_image stem>_coreg_residual.tif`.
- Co-registered point cloud: `align.las_output`, or `<las_input stem>_coreg.las` by default.

The alignment scripts write only points inside the flow grid's interpolation
extent; points outside it are omitted. Inspect the output extent and point count.
Validate alignment with `gefolki/notebooks/stats_align_lidar_with_flow.py.ipynb`.
Large rasters, flow fields and LAS files should not be committed to the repository.
