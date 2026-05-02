# Phase 1: DSM Generation

Tools for creating Digital Surface Models (DSMs) from LiDAR point clouds.

## Method

A **Digital Surface Model (DSM) GeoTIFF** was generated from LiDAR point clouds using a PDAL pipeline. The `writers.gdal` stage was configured with `output_type: "max"` to ensure each pixel represented the **highest elevation point** within its grid cell. This output was used for LiDAR point cloud co-registration with RGB dense clouds.

**Key Parameters:**
- Resolution: 0.03 m (3 cm)
- Output Type: Maximum elevation per cell
- Format: GeoTIFF with CRS preservation

## Contents

```
phase_1_image_processing/
├── dsm_lidar.py                       # DSM generation script (PDAL)
└── notebooks/
    ├── 01_create_dsm_from_las.ipynb   # Interactive DSM creation
    └── 02_visualiseStats_las_dsm.ipynb # QA/QC visualization
```

## Usage

### Script
```bash
python dsm_lidar.py  # Edit paths and parameters in script
```

### Notebook
```bash
jupyter notebook notebooks/01_create_dsm_from_las.ipynb
```

## Dependencies

```bash
conda install -c conda-forge pdal python-pdal gdal laspy rasterio matplotlib
pip install open3d  # Optional for 3D visualization
```

## Supported Sensors
- DJI Zenmuse L1 (2022)
- DJI L2 (2024)
- Riegl VUX-120
