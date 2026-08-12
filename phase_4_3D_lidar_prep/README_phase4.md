# Phase 4: 3D LiDAR Preparation

Tools for processing LiDAR point clouds to generate structure-classified 3D data for CNN training.

## Method

Co-registered LiDAR point clouds (Phase 2) were processed to create 3D training datasets by:
1. Clipping point clouds to match 2D orthomosaic tiles (3072×3072 px grid)
2. Classifying ground points (CSF or SMRF) and computing height-above-ground as `Z_norm`
3. Fusing 3D LiDAR geometry with 2D structure mask labels via voxel transfer + crown-centric polar refinement

The structure fusion pipeline combines Z-normalized LiDAR with 2D mask labels using:
- **Cartesian voxel merge**: Majority-vote label transfer in 3D voxel grid
- **Mallee crown detection + instance IDs**: Watershed-based crown detection + per-point `treeID` assignment
- **Polar refinement**: Crown-centric analysis with data-driven height priors for mallee refinement
- **Height-based validation**: Data-driven height ranges ensure label consistency

All final products are written as **LAS 1.4 / Point Format 6 (PF6)** and preserve scale/offset.

## Contents

```
phase_4_3D_lidar_prep/
├── step_1_clipping/
│   ├── rasterclip/                           # Clip LAS to raster tile bounds (Python package)
│   ├── maskclip/                             # Apply mask raster -> LAS Classification (Python package)
│   ├── notebooks/                            # Clipping + mask workflows
│   └── pyproject.toml
├── step_2_csf_znorm/
│   ├── groundnorm/                           # Ground filtering + Z_norm (Python package + CLI)
│   ├── notebooks/
│   │   └── 01_ground_classification_site01.ipynb
│   └── pyproject.toml
└── step_3_structure_class/
        ├── fvc3dstruct/                          # Structure fusion pipeline package
        ├── structure_fusion_3d.py                # Full reference implementation (script)
        ├── notebooks/                            # End-to-end fusion workflows
        └── test_pipeline.sh
```

## Usage

### Step 1: Clip LiDAR to Tiles
Use `rasterclip` to clip a large LAS to raster tile bounds, then `maskclip` to transfer 2D mask values into the LAS `Classification` field.

- Interactive workflows live under `step_1_clipping/notebooks/`.
- For details and APIs, see `step_1_clipping/README.md`.

### Step 2: Ground Classification & Height Normalization
This step writes a `Z_norm` ExtraBytes dimension (height above ground) into the LAS.

```bash
# Examples (see step_2_csf_znorm/README.md for full CLI)
groundnorm csf input.las ground_csf.las
groundnorm normalize input.las ground_csf.las z_norm_csf.las --k 3
```

### Step 3: 3D Structure Classification
Fuses mask-derived structure labels onto the Z-normalized point cloud and assigns mallee `treeID` instances.

- Interactive workflows live under `step_3_structure_class/notebooks/`.
- The maintained pipeline implementation is `step_3_structure_class/fvc3dstruct/pipeline.py` (see `step_3_structure_class/README.md`).

Minimal batch example (expects the default filename patterns used in `fvc3dstruct.pipeline.process_folder()`):

```python
from pathlib import Path
from fvc3dstruct.pipeline import process_folder

z_dir = Path("/path/to/z_norm/")    # contains: z_norm_csf_tiles_multispectral.*.las
s_dir = Path("/path/to/mask/")      # contains: mask_tiles_multispectral.*.las
out_dir = Path("/path/to/out/")

process_folder(z_dir=z_dir, s_dir=s_dir, out_dir=out_dir, site_name="")
```

## Ground Classification (CSF / SMRF)

This project supports two PDAL-based ground filters:

- **CSF (Cloth Simulation Filter)** defaults: `resolution=0.5`, `iterations=500`, `threshold=0.01`, `rigidness=2`
- **SMRF (Simple Morphological Filter)** defaults: `slope=0.15`, `window=3.5`, `threshold=0.25`, `scalar=1.35`

**Output:** Ground-only LAS + normalized LAS with `Z_norm` (HAG - Height Above Ground)

## Structure Fusion Pipeline

**Inputs:**
- Z-normalized LiDAR tiles (geometry + `Z_norm`)
- Structure/mask LAS tiles with labels stored in LAS `classification` (64–68)

**Fusion Strategy:**
1. **Voxel merge** (default 0.10m): Transfer structure labels to Z_norm points via majority vote in a shared 3D grid
2. **Crown detection + `treeID`**: Detect mallee crowns (class 68 evidence) and assign per-point instance IDs
3. **Polar refinement**: Crown-centric refinement in $(r,\theta,z)$ bins with data-driven height priors
4. **Height-based consolidation**: Post-rules to enforce crown consistency and remove implausible low vegetation labels

**Structure labels (LAS `classification` codes):**
- **64**: Forb
- **65**: Woody debris
- **66**: Hummock
- **67**: Shrub
- **68**: Mallee
- **0**: Unmatched/background

## Dependencies

```bash
conda install -c conda-forge pdal laspy rasterio numpy scipy scikit-image
pip install dask[distributed] tqdm
```

## Outputs

- **Clipped LAS tiles**: LiDAR subsets matching raster/mask tile extents
- **Mask-classified structure LAS**: LAS `classification` populated from mask raster values (64–68)
- **Ground-only LAS**: Ground points only (CSF or SMRF)
- **Z-normalized LAS**: Adds `Z_norm` ExtraBytes (height above ground)
- **Fused vegetation-only LAS (training product)**:
    - LAS 1.4 PF6
    - non-zero structure classes (64–68)
    - `Z_norm` and `treeID` ExtraBytes
    - optional “full fused” product when enabled in config

Common Step 3 output naming (see `fvc3dstruct.pipeline.process_folder()`):

- Veg-only: `{site_name_}tile{tile_id}_struct_polar.las`
- Optional full fused: `{site_name_}tile{tile_id}_struct_polar_full_fused.las`
