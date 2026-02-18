
# Phase 5: 3D CNN Training Prep (ForestFormer / SegmentAnyTree)

This phase converts the Phase 4 structure-classified LAS tiles into model-ready inputs (primarily **ForestFormer3D PLY**), and provides notebooks to:

- preprocess LAS → PLY
- export ForestFormer-compatible tensors (NPY) when needed by a downstream model
- reconstruct predicted outputs back to LAS for QC/visualization (CloudCompare, etc.)
- visualize SegmentAnyTree instance outputs from PLY

## Inputs (from Phase 4)

Typical input LAS tiles come from Phase 4 Step 3 (fusion) and contain:

- `classification` (LAS class codes): **64–68** (plus 0 background)
- ExtraBytes: `Z_norm` (height above ground)
- ExtraBytes: `treeID` (mallee instance id; 0 for non-mallee / non-canopy)

## Method / Key decisions

1. **Class remap for training**
	- LAS codes 64–68 are remapped to compact class IDs 0–4.
	- LAS 0/background becomes `ignore_index` (default `-100`).

2. **PLY is written in local tile coordinates (precision-safe)**
	- PLY coordinates are written as a local frame: `xyz_local = xyz_global - xyz_min`.
	- This avoids float quantization issues when writing large-magnitude UTM coordinates to PLY.

3. **Offsets are persisted for exact reconstruction**
	- For each `{scan}.ply` written in local coords, a sidecar `{scan}_offset.npy` is saved (when enabled).
	- Global coordinates can be reconstructed as: `xyz_global = xyz_local + offset`.

## Contents

```
phase_5_3d_cnn_training/
├── lidar_preprocessing/                 # LAS → PLY preprocessing
│   ├── config.py                        # PreprocessingConfig (toggles + mapping)
│   ├── preprocessing.py                 # process_las(), batch_process()
│   └── utils.py
├── forestformer_preprocessing/          # Helpers expected by ForestFormer exports
│   ├── plyutils.py                      # read_ply() compatible module
│   ├── load_forainetv2_data.py
│   └── batch_load_ForAINetV2_data.py
└── notebooks/
	 ├── 01_lidar_preprocessing_batchLoader.ipynb
	 ├── 02_lidar_preprocessing_forestformer.ipynb
	 └── 03_lidarModelling_back_to_las.ipynb
```

## Usage

### Notebook workflow (3 steps)

The notebooks are intended to be used in this order:

1. **LAS → PLY**: `notebooks/01_lidar_preprocessing_batchLoader.ipynb`
2. **PLY → NPY** (ForestFormer export, when needed): `notebooks/02_lidar_preprocessing_forestformer.ipynb`
3. **NPY and/or PLY → LAS** (QC/visualization + global reconstruction via offsets): `notebooks/03_lidarModelling_back_to_las.ipynb`

### A) LAS → PLY (recommended, PLY-only)

Use the notebook:

- `notebooks/01_lidar_preprocessing_batchLoader.ipynb`

or run from Python:

```python
from lidar_preprocessing import PreprocessingConfig, batch_process

cfg = PreprocessingConfig(
	 input_dir="/path/to/phase4/struct_polar_tiles",
	 out_ply_dir="/path/to/preprocessed/forestformer3d_ply",
	 ply_use_local_coords=True,     # precision-safe
	 save_ply_offset=True,          # write {stem}_offset.npy
)

batch_process(cfg, file_pattern="*.las")
```

Outputs:

- `{stem}.ply` with columns: `x y z semantic_seg treeID z_norm intensity`
- `{stem}_offset.npy` (when enabled)
- `ply_list.txt`

### B) ForestFormer export + reconstruction QC

- `notebooks/02_lidar_preprocessing_forestformer.ipynb`
	- exports the preprocessed PLY into ForestFormer-style NPY artifacts (when required by a downstream model)
- `notebooks/03_lidarModelling_back_to_las.ipynb`
	- reconstructs PLY→LAS and/or NPY→LAS for visualization
  - applies `{stem}_offset.npy` when present to recover global coordinates

### C) SegmentAnyTree instance visualization (PLY)

In `notebooks/03_lidarModelling_back_to_las.ipynb`, the final section plots SegmentAnyTree instance IDs from a `.ply` output (instance-only coloring). It auto-detects a likely instance column name (e.g. `treeID`, `instance_id`, `ins_label`, …) and prints available columns if none are found.

## Configuration

Main configuration is in `lidar_preprocessing/config.py` (`PreprocessingConfig`). Important knobs:

- `ply_use_local_coords`: keep `True` unless you have a strong reason not to
- `save_ply_offset`: keep `True` if you ever need global-coordinate reconstruction
- `class_remap`: LAS class code → class ID mapping

## Dependencies

Runtime dependencies used in this phase include:

- `laspy`, `numpy`, `pandas`
- `pyntcloud` (PLY read/write)
- `matplotlib` (notebook QC/plots)

Some notebooks/scripts also rely on packages used earlier in the project (PDAL, rasterio, etc.), but Phase 5 preprocessing itself is primarily Python + laspy + pyntcloud.

