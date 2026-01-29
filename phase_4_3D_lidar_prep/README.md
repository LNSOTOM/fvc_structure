# Phase 4: 3D LiDAR Preparation

Tools for processing LiDAR point clouds to generate structure-classified 3D data for CNN training.

## Method

Co-registered LiDAR point clouds (Phase 2) were processed to create 3D training datasets by:
1. Clipping point clouds to match 2D orthomosaic tiles (3072×3072 px grid)
2. Classifying ground points using Cloth Simulation Filter (CSF) and normalizing heights
3. Fusing 3D LiDAR geometry with 2D structure labels through voxel-based and polar coordinate merging

The structure fusion pipeline combines Z-normalized LiDAR with 2D mask labels using:
- **Cartesian voxel merge**: Majority-vote label transfer in 3D voxel grid
- **Polar refinement**: Crown-centric analysis with height priors for mallee detection
- **Height-based validation**: Data-driven height ranges ensure label consistency

## Contents

```
phase_4_3D_lidar_prep/
├── step_1_clipping/
│   ├── clipping_toMask_coreg.py              # Clip LiDAR to 2D mask tiles (co-registered)
│   ├── clipping_toMask_imagery.py            # Clip LiDAR to imagery extent
│   ├── clipping_toRaster_onefile.py          # Clip single file to raster bounds
│   ├── clipping_toRaster_daskMultiples.py    # Batch clipping with Dask parallelization
│   ├── notebooks/
│   │   ├── 01_clip_site01.ipynb              # Site 1 clipping workflow
│   │   ├── 01_clip_site11.ipynb              # Site 11 clipping workflow
│   │   ├── 02_mask_site01.ipynb              # Mask-based clipping (Site 1)
│   │   ├── 03_icp_regCheck.ipynb             # ICP registration verification
│   │   └── 04_maskCoreg_site01.ipynb         # Co-registered mask clipping
│   └── pyproject.toml
├── step_2_csf_znorm/
│   ├── cloth_simulation_filter_3.py          # CSF ground classification pipeline
│   ├── run_csf_ground_filter_1.py            # CSF filter wrapper
│   ├── normalise_las_z_2.py                  # Height normalization (Z - ground)
│   ├── notebooks/
│   │   └── 01_ground_classification_site01.ipynb
│   └── pyproject.toml
└── step_3_structure_class/
    ├── structure_fusion_3d.py                # Main fusion pipeline (voxel + polar)
    ├── merge_znorm_struct_to_maskClasses_polar.py           # Polar coordinate merging
    ├── merge_znorm_struct_to_maskClasses_polar_crown.py     # Crown-based classification
    ├── merge_znorm_struct_to_maskClasses_polar_malle_id.py  # Mallee individual detection
    ├── merge_znorm_struct_to_maskClasses_voxel.py           # Voxel-based merging
    ├── main_run.py                           # Batch processing script
    ├── notebooks/
    │   ├── 01_run_pipeline_site01.ipynb      # Complete pipeline (Site 1)
    │   └── 01_run_pipeline_site11.ipynb      # Complete pipeline (Site 11)
    └── pyproject.toml
```

## Usage

### Step 1: Clip LiDAR to Tiles
```bash
# Interactive (recommended)
jupyter notebook step_1_clipping/notebooks/02_mask_site01.ipynb

# Or run script
python step_1_clipping/clipping_toMask_coreg.py
```

### Step 2: Ground Classification & Height Normalization
```bash
# Interactive
jupyter notebook step_2_csf_znorm/notebooks/01_ground_classification_site01.ipynb

# Or run script
python step_2_csf_znorm/cloth_simulation_filter_3.py
```

### Step 3: 3D Structure Classification
```bash
# Interactive (recommended)
jupyter notebook step_3_structure_class/notebooks/01_run_pipeline_site01.ipynb

# Or run batch processing
python step_3_structure_class/main_run.py
```

## Ground Classification (CSF)

**Cloth Simulation Filter** parameters:
- Cloth resolution: 0.5m
- Rigidness: 3 (steep terrain)
- Classification threshold: 0.3m
- Iterations: 500

**Output:** Ground-classified LAS with normalized heights (HAG - Height Above Ground)

## Structure Fusion Pipeline

**Inputs:**
- Z-normalized LiDAR (geometry + HAG)
- 2D structure labels (IDs 0-4 from Phase 3)

**Fusion Strategy:**
1. **Voxel merge** (0.05m grid): Transfer 2D labels to 3D points via majority vote
2. **Polar refinement** (r, θ, z): Crown-centric analysis with CHM watershed for mallee
3. **Height validation**: Apply data-driven height ranges per class

**Structure IDs:**
- **0**: Forbs (ground-level)
- **1**: Woody debris/stem
- **2**: Grass
- **3**: Shrub
- **4**: Mallee

## Dependencies

```bash
conda install -c conda-forge pdal laspy rasterio numpy scipy
pip install CSF tqdm dask
```

## Outputs

- **Clipped LAS tiles**: LiDAR subsets matching 2D grid (10m×10m)
- **Ground-classified LAS**: Points labeled as ground/non-ground with HAG
- **Structure-classified LAS**: 3D points with structure IDs (0-4) from 2D fusion
- **Processing logs**: Classification statistics, height distributions, crown counts
