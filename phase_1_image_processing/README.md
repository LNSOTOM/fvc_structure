# Phase 1: Image Processing and DSM Generation

This directory contains tools for creating Digital Surface Models (DSMs) from LiDAR point clouds and performing quality control checks.

## Method

### DSM Generation

A **Digital Surface Model (DSM) GeoTIFF** was generated from the LiDAR point cloud using a PDAL pipeline. The `writers.gdal` stage was configured with the `output_type: "max"` to ensure that each pixel in the resulting DSM represented the **highest elevation point** within its corresponding grid cell. This output was used for LiDAR point cloud co-registration with RGB dense clouds.

**Key Parameters:**
- **Resolution**: 0.03 m (3 cm) grid cells
- **Output Type**: Maximum elevation per cell
- **Format**: GeoTIFF with coordinate reference system preservation
- **NoData Handling**: Proper masking of empty cells

### Quality Control

Statistical and visual comparisons between the original point cloud and generated DSM ensure data quality and identify potential information loss during rasterization.

## Directory Structure

```
phase_1_image_processing/
├── README.md                          # This file
├── dsm_lidar.py                       # Main DSM generation script
└── notebooks/
    ├── 01_create_dsm_from_las.ipynb   # Interactive DSM creation workflow
    └── 02_visualiseStats_las_dsm.ipynb # QA/QC visualization and comparison
```

## Scripts

#### `dsm_lidar.py`
Primary script for DSM generation from LAS/LAZ files using PDAL.

**Usage:**
```bash
python dsm_lidar.py
```

**Features:**
- Batch processing of LiDAR files
- Configurable resolution and output type
- PDAL pipeline execution
- Automatic CRS preservation
- Multi-sensor support (DJI L1, L2, Riegl VUX-120)

**Configuration:**
- Edit file paths in the script for your data location
- Set `resolution` parameter (default: 0.03 m)
- Choose `output_type`: "max", "min", "mean", or "idw"

## Notebooks

#### `01_create_dsm_from_las.ipynb`
Interactive notebook for DSM creation with step-by-step documentation.

**Features:**
- Visual pipeline configuration
- Real-time PDAL execution
- Immediate result visualization
- Parameter tuning and testing

#### `02_visualiseStats_las_dsm.ipynb`
Quality control and validation notebook.

**Features:**
- Point cloud statistics and visualization (2D/3D)
- DSM raster analysis and hillshade generation
- Comparative analysis (point cloud vs DSM)
- Spatial overlap verification
- Statistical summary export

**Outputs:**
- Multiple visualization plots
- Elevation histograms
- Comparison metrics
- Summary text files

## Data Requirements

### Input Data
- **LAS/LAZ files**: Raw LiDAR point clouds
  - Formats: LAS 1.2-1.4, LAZ compressed
  - Required fields: X, Y, Z coordinates
  - Optional: Classification, Intensity, RGB

### Output Data
- **DSM GeoTIFF**: Gridded elevation rasters
- **Statistics**: Text summaries and comparison reports
- **Visualizations**: PNG plots and screenshots

## Supported Sensors

The workflow supports data from multiple LiDAR sensors:

1. **DJI Zenmuse L1** (2022 campaign)
   - High-density terrestrial scanning
   - Trajectory: `DJI_20220519102349_0001_Zenmuse-L1-mission_sbet.out`

2. **DJI L2** (2024 campaign)
   - Enhanced sensor specifications
   - Trajectory: `DJI_20240312130506_0002_L_sbet.out`

3. **Riegl VUX-120** (Reference data)
   - Professional-grade airborne LiDAR

## Dependencies

### Required Python Packages
```bash
# Core processing
pdal
laspy
rasterio
numpy

# Visualization
matplotlib
open3d (optional, for 3D visualization)
```

### System Requirements
- **PDAL**: Point Data Abstraction Library (C++ with Python bindings)
- **GDAL**: Geospatial Data Abstraction Library

### Installation
```bash
# Using conda (recommended)
conda install -c conda-forge pdal python-pdal gdal laspy rasterio

# Visualization tools
conda install -c conda-forge matplotlib
pip install open3d  # Optional for 3D point cloud visualization
```

## Workflow

### Step 1: DSM Generation
```bash
# Edit dsm_lidar.py to set input/output paths
python dsm_lidar.py
```

Or use the interactive notebook:
```bash
jupyter notebook notebooks/01_create_dsm_from_las.ipynb
```

### Step 2: Quality Control
```bash
# Run QA/QC notebook
jupyter notebook notebooks/02_visualiseStats_las_dsm.ipynb
```

**Check for:**
- Elevation range consistency
- Spatial coverage alignment
- NoData pixel distribution
- Point density vs pixel resolution trade-offs

### Step 3: Validation
Review generated statistics:
- Compare min/max/mean elevations
- Verify spatial overlap with RGB imagery
- Assess information loss from gridding

## Outputs

### DSM Files
- Location: `output/` or sensor-specific directories
- Format: GeoTIFF (.tif)
- Naming convention: `{site}_{sensor}_dsm_{resolution}.tif`
- Example: `SASMDD0011_l2_dsm_03.tif`

### Statistics Files
- Location: `output/`
- Format: Plain text (.txt)
- Content: Point cloud vs DSM comparison metrics

### Visualizations
- 2D plots: Top view, side profiles, histograms
- 3D screenshots: Multiple viewing angles
- Hillshade renderings: For terrain analysis

## Notes

### Resolution Selection
- **0.03 m (3 cm)**: High-resolution for detailed canopy structure
- **0.05 m**: Balanced detail and file size
- **0.10 m**: Coarser for landscape-scale analysis

### Output Type Selection
- **max**: Captures canopy tops (used for co-registration)
- **min**: Ground surface detection
- **mean**: Average elevation per cell
- **idw**: Inverse distance weighted interpolation

### Known Issues
- Large point clouds may require memory management
- Open3D 3D visualization requires X11 display (use screenshot mode on SSH)
- Processing time scales with point density and output resolution

## References

- PDAL Documentation: https://pdal.io/
- GDAL Raster API: https://gdal.org/api/raster_c_api.html
- LAStools: https://rapidlasso.com/lastools/
