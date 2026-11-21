# Phase 4: 3D LiDAR preparation handoff

## Input and scope

Provide a **previously ICP-registered LiDAR tile**, matching predictor raster
and Phase 3 structure mask. Verify registration and tile identity before running.
ICP is not implemented here; Phase 2 GeFolki is not a substitute. The ICP
output must be LAS 1.4, point format 6 or newer, in a projected CRS in metres.

The mask must be single-band, use classes 64–68 with an optional declared
NoData value, overlap the LiDAR tile, and share its horizontal CRS. No
reprojection or resampling is performed. The predictor is checked for CRS and
overlap; the existing `maskclip.classify_single` function currently uses the
mask raster, not its predictor argument, for cropping and classification.
The runner warns when the grids differ, but overlap and CRS checks cannot
prove that labels, imagery and LiDAR are accurately aligned. Inspect their
origins, resolution, extents and visual overlays before using the results.

## Environment

The Phase 1–2 `uv` environment is not sufficient. For example, create a
separate conda environment with PDAL's executable and Python bindings:

```text
conda create -n fvc-phase4 -c conda-forge python=3.10 pdal python-pdal laspy rasterio pyproj numpy scipy scikit-image pyyaml dask distributed
conda activate fvc-phase4
python -c "import pdal, laspy, rasterio, pyproj, scipy, skimage, yaml, dask, distributed; print('Phase 4 imports OK')"
pdal --version
```

Confirm that the PDAL installation includes `filters.csf` and
`filters.colorization` (for example, `pdal --drivers`). Record the environment
export, input versions and config with each run. Use Windows paths with
Windows Python, and Linux/WSL paths with Linux/WSL Python. Quote Windows
paths with single quotes in YAML.

## Run one tile

From the repository root, copy the example and edit all input and output paths:

```text
cp phase_4_3D_lidar_prep/configs/lidar_prep_job.example.yml phase_4_3D_lidar_prep/configs/lidar_prep_job.local.yml
```

Use distinct, new `.las` output paths. The CSF stage also writes a JSON
pipeline alongside the ground LAS. Existing outputs are not overwritten.
Keep data outside the repository.

```text
python phase_4_3D_lidar_prep/lidar_prep.py --config phase_4_3D_lidar_prep/configs/lidar_prep_job.local.yml --stage all
```

Alternatively, run the same command four times, replacing `all` in order with
`mask`, `ground`, `normalize` and `fuse`. Use the same config and verify each
intermediate belongs to the selected ICP tile; basic metadata checks cannot
establish provenance. If a stage fails, inspect and remove or rename only
its incomplete outputs before retrying.

1. `mask`: crop to the mask and transfer labels 64–68 to LAS classification.
2. `ground`: extract ground points with CSF (0.5 m resolution, 500 iterations,
   0.01 threshold and rigidness 2 by default).
3. `normalize`: calculate non-negative `Z_norm` using inverse-distance
   weighting of the nearest three ground points by default. The underlying
   `r_max` option is currently unused.
4. `fuse`: use the repository's `fvc3dstruct` default voxel, crown and
   refinement settings, printed at runtime, to write a vegetation-only LAS.

The outputs are mask-classified LAS, ground-only LAS and its PDAL pipeline
JSON, full-tile `Z_norm` LAS, and vegetation-only fused LAS with `Z_norm`,
`treeID` and nonzero structure classes. The fusion writer retains existing
LAS 1.4 formats 6 and later, so the point format is not necessarily 6.
The default does not write the optional full fused product.

## Validation and limitations

Inspect CRS, grids, alignment, point counts, class distributions, heights and
tree IDs for every tile. A run that finishes is not scientific validation.
For DD0001 tile 109, a local run using its historical ICP LAS and saved mask
completed all four stages: 1,593,979 input points, 331,887 mask-classified
points, 500,092 ground points and 399,912 fused vegetation points. Its
predictor and mask grids differed, so this is a workflow check, not proof of
alignment or equivalence to a historical fused product. Other sites and tiles
have not been established as equivalent by this check.
