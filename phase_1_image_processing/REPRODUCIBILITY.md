# Reproducible DSM Script Handoff

This is the minimal file structure to deliver the reusable LAS/LAZ-to-DSM workflow.

## Deliverable Structure

```text
fvc_structure/
├── pyproject.toml
└── phase_1_image_processing/
    ├── README_phase1.md
    ├── REPRODUCIBILITY.md
    ├── dsm_lidar.py
    ├── configs/
    │   └── dsm_job.example.yml
    └── notebooks/
        └── 01_create_dsm_from_las.ipynb
```

## What Each File Does

- `pyproject.toml`: Python dependencies for `uv`.
- `phase_1_image_processing/dsm_lidar.py`: reusable PDAL DSM generation script.
- `phase_1_image_processing/configs/dsm_job.example.yml`: copy-and-edit config template for input/output paths and metadata.
- `phase_1_image_processing/notebooks/01_create_dsm_from_las.ipynb`: interactive example with automatic resolution selection.
- `phase_1_image_processing/README_phase1.md`: user-facing workflow notes.
- `phase_1_image_processing/REPRODUCIBILITY.md`: this handoff checklist.

## Inputs Required From A New User

The script starts after direct georeferencing is complete. A new user must provide:

1. A georeferenced `.las` or `.laz` file.
2. A writable DSM output directory.
3. Site metadata for the config file, such as `site_id`, `product_level`, `sensor`, and acquisition `datetime`.

Upstream conversion and georeferencing of raw sensor data (HySpex/Ouster/APX) is handled in the
[remote_sensing-analytics DSM package](https://github.com/UQ-Drones-CRP/remote_sensing-analytics/blob/main/sensors_data/hyspec/dsm_package/dsm/README_phase1.md)
and is out of scope here.

## Environment Setup

Install `uv` if needed:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Create the Python environment from the repository root:

```bash
uv sync
```

Install the external PDAL executable if it is not already available on `PATH`:

```bash
conda install -c conda-forge pdal python-pdal gdal laspy rasterio matplotlib
```

Check that PDAL is visible:

```bash
pdal --version
```

## Configure A DSM Job

Copy the example config and edit paths:

```bash
cp phase_1_image_processing/configs/dsm_job.example.yml phase_1_image_processing/configs/dsm_job.local.yml
```

Required edits in `dsm_job.local.yml`:

```yaml
job:
  las_input: /absolute/path/to/georeferenced_input.las
  dsm_output: /absolute/path/to/output/dsm_03cm.tif
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

Keep `datetime` quoted so YAML treats it as a string.

## Reproduce The DSM Run

Estimate before writing a DSM:

```bash
uv run python phase_1_image_processing/dsm_lidar.py \
    --config phase_1_image_processing/configs/dsm_job.local.yml \
    --estimate-only
```

Create the DSM:

```bash
uv run python phase_1_image_processing/dsm_lidar.py \
    --config phase_1_image_processing/configs/dsm_job.local.yml
```

Override resolution without editing the config:

```bash
uv run python phase_1_image_processing/dsm_lidar.py \
    --config phase_1_image_processing/configs/dsm_job.local.yml \
    --resolution 0.05
```

## Notebook Reproduction

The `batch` section in the example config contains two DD0011 LAS/DSM pairs.
Edit those paths in the local config before running the notebook's sequential
batch cell. The command-line script processes the single `job` section only;
it does not run the batch.

Register the `uv` environment as a Jupyter kernel:

```bash
uv run python -m ipykernel install --user \
    --name fvc-structure-dsm \
    --display-name "fvc-structure-dsm"
```

Open `phase_1_image_processing/notebooks/01_create_dsm_from_las.ipynb`, select the `fvc-structure-dsm` kernel, and update the config or LAS path cells for the local dataset.

## Expected Outputs

The main DSM run writes:

- DSM GeoTIFF: the configured `dsm_output` path.
- Optional STAC sidecar: `<dsm_output>.stac-item.json` when `stac: true` and metadata is present.

Large DSM GeoTIFFs should not be committed to the repository.