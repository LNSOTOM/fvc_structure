# Phase 2: LiDAR Co-registration

Tools for co-registering multi-temporal LiDAR point clouds using optical flow (GeFolki algorithm).

## Method

LiDAR point clouds from different acquisition dates were co-registered using the **GeFolki optical flow algorithm**. DSMs generated from each LiDAR acquisition (Phase 1) were used as input to compute displacement fields between temporal pairs. The computed flow fields were then applied to shift LiDAR points, achieving precise spatial alignment between acquisitions.

The workflow was inspired by: https://github.com/RS-Haynes/efolki_coreg/blob/main/efolki_coreg_example.ipynb (loads hyperspectral images in ENVI format, extracts matching wavelengths, then calculates optical flow)

**Workflow:**
1. Preprocess DSMs (histogram matching, filtering)
2. Compute optical flow between temporal DSM pairs
3. Apply displacement field to LiDAR point clouds
4. Validate alignment accuracy

## Contents

```
phase_2_lidar_coregistration/
├── gefolki/
│   ├── python_myscripts/              # Main processing scripts
│   │   ├── gefolki_preprocess.py      # DSM preprocessing
│   │   ├── gefolki_flow_l1.py         # Flow computation (L1 sensor)
│   │   ├── gefolki_flow_l2.py         # Flow computation (L2 sensor)
│   │   ├── align_lidar_with_flow_l1.py # Apply flow to points (L1)
│   │   └── align_lidar_with_flow_l2.py # Apply flow to points (L2)
│   ├── notebooks/                     # Interactive workflows
│   │   ├── 1_gefolki_preprocess.ipynb
│   │   ├── 2_gefolki_flow_l1-site01.ipynb
│   │   ├── 2_gefolki_flow_l1-site11.ipynb
│   │   ├── 2_gefolki_flow_l2.ipynb
│   │   ├── 3_align_lidar_with_flow_l2.ipynb
│   │   └── stats_align_lidar_with_flow.py.ipynb # Statistical validation
│   ├── python/                        # GeFolki core library (Python)
│   ├── matlab/                        # GeFolki core library (MATLAB)
│   └── datasets/                      # Example datasets for testing
└── step_0_checks/                     # CRS validation tools
```

## Usage

### Notebooks (Recommended)
```bash
# Step 1: Preprocess DSMs
jupyter notebook gefolki/notebooks/1_gefolki_preprocess.ipynb

# Step 2: Compute optical flow
jupyter notebook gefolki/notebooks/2_gefolki_flow_l1-site01.ipynb

# Step 3: Apply flow to LiDAR
jupyter notebook gefolki/notebooks/3_align_lidar_with_flow_l2.ipynb

# Step 4: Statistical validation (optional)
jupyter notebook gefolki/stats_align_lidar_with_flow.py.ipynb
```

### Scripts
```bash
# Edit paths in scripts, then run sequentially:
python gefolki/python_myscripts/gefolki_preprocess.py
python gefolki/python_myscripts/gefolki_flow_l1.py
python gefolki/python_myscripts/align_lidar_with_flow_l1.py
```

## Dependencies

```bash
conda install -c conda-forge numpy scipy rasterio laspy gdal matplotlib
pip install scikit-image
```

## GeFolki Algorithm

GeFolki (Geoscience Extended Flow Optical Lucas-Kanade Iterative) is an optical flow method designed for geoscience applications. It computes dense displacement fields between image pairs using:
- Lucas-Kanade iterative approach
- Multi-scale pyramid processing
- Robust correlation metrics

**References:**
- Original implementation: `gefolki/python/` (adapted from ONERA)
- Documentation: `gefolki/manual_gefolki_english.pdf`
- Additional implementation: https://github.com/RS-Haynes/efolki_coreg
