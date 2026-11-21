# Phase 1: DSM Generation

Create a digital surface model (DSM) from a georeferenced `.las` or `.laz` point cloud. Upstream sensor
conversion, trajectory processing and georeferencing (HySpex/Ouster/APX) are handled in the
[remote_sensing-analytics DSM package](https://github.com/UQ-Drones-CRP/remote_sensing-analytics/blob/main/sensors_data/hyspec/dsm_package/dsm/README_phase1.md).

## Method

PDAL `writers.gdal` uses `output_type: "max"` to write the highest elevation in each grid cell. The DSM supports LiDAR co-registration with RGB dense clouds.

**Key Parameters:**
- Resolution: 0.03 m (3 cm)
- Output Type: Maximum elevation per cell
- Format: GeoTIFF with CRS preservation

## Contents

```
phase_1_image_processing/
├── configs/
│   └── dsm_job.example.yml            # Reusable input/output path config
├── dsm_lidar.py                       # DSM generation script (PDAL)
└── notebooks/
    ├── 01_create_dsm_from_las.ipynb   # Interactive DSM creation
    └── 02_visualiseStats_las_dsm.ipynb # QA/QC visualization
```

## Usage

For a reproducible handoff structure and exact setup commands, see [REPRODUCIBILITY.md](REPRODUCIBILITY.md).

### Reusable script

Put paths and metadata in a config file so the Python script stays reusable:

```yaml
job:
    las_input: /path/to/georeferenced_input.las
    dsm_output: /path/to/dsm/georeferenced_input_dsm_03.tif
    site_id: DD0001
    product_level: LIDAR_DSM
    sensor: Ouster

resolution: 0.03
output_type: max
nodata: -9999.0
compression: LZW
stac: true
datetime: "2024-01-31T00:00:00Z"
```

Run it with:

```bash
python3 phase_1_image_processing/dsm_lidar.py --config phase_1_image_processing/configs/dsm_job.example.yml
```

You can still override processing settings from the command line:

```bash
python3 phase_1_image_processing/dsm_lidar.py --config phase_1_image_processing/configs/dsm_job.example.yml --resolution 0.10
```

Or pass arbitrary georeferenced LAS/LAZ inputs without a config file:

```bash
python3 phase_1_image_processing/dsm_lidar.py \
    --las-input /path/to/georeferenced_ouster.las \
    --dsm-output /path/to/dsm/georeferenced_ouster_dsm_03.tif \
    --resolution 0.03 \
    --output-type max
```

Inspect predefined configs or estimate cost before writing a DSM:

```bash
python3 phase_1_image_processing/dsm_lidar.py --list-sites
python3 phase_1_image_processing/dsm_lidar.py --config phase_1_image_processing/configs/dsm_job.example.yml --estimate-only
```

Run a predefined site config:

```bash
python3 phase_1_image_processing/dsm_lidar.py --site DD0011_L2
```

### Python API

Use the same YAML config from notebooks or other scripts:

```python
from dsm_lidar import job_from_config, create_dsm_from_job

job = job_from_config("phase_1_image_processing/configs/dsm_job.example.yml")

create_dsm_from_job(job, resolution=0.03, output_type="max")
```

### Notebook
```bash
jupyter notebook notebooks/01_create_dsm_from_las.ipynb
```

## Dependencies

### uv environment

For the DSM workflow, use the project `pyproject.toml` with `uv`:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Then install the Python environment:

```bash
uv sync
```

Run the reusable script inside that environment:

```bash
uv run python phase_1_image_processing/dsm_lidar.py \
    --config phase_1_image_processing/configs/dsm_job.example.yml \
    --estimate-only
```

Register the notebook kernel:

```bash
uv run python -m ipykernel install --user --name fvc-structure-dsm --display-name "fvc-structure-dsm"
```

`uv` installs the Python packages used by the script and notebook. The `pdal` command-line executable must also be available on `PATH`. If it is not already installed, use conda/mamba for the geospatial system packages:

```bash
conda install -c conda-forge pdal python-pdal gdal laspy rasterio matplotlib
```

Optional for 3D visualization:

```bash
uv add open3d
```

## Supported Sensors
- DJI Zenmuse L1 (2022)
- DJI L2 (2024)
- Riegl VUX-120
