# groundnorm — Ground Classification and Z Normalization

Python package for LAS point cloud ground filtering and height-above-ground (HAG) normalization using CSF and SMRF algorithms.

## Features

- **CSF (Cloth Simulation Filter)**: Terrain extraction using cloth simulation
- **SMRF (Simple Morphological Filter)**: Morphological ground filtering
- **Z Normalization**: Height-above-ground (HAG) computation via kNN interpolation
- **Batch Processing**: Parallel processing of multiple LAS tiles
- **CLI & Python API**: Use from command line or import in scripts/notebooks

## Installation

```bash
# Install in development mode
pip install -e .

# Or install from directory
pip install /path/to/step_2_csf_znorm
```

## Command-Line Usage

### Single File Processing

```bash
# CSF ground filter
groundnorm csf input.las ground_csf.las

# SMRF ground filter
groundnorm smrf input.las ground_smrf.las --slope 0.15 --window 3.5

# Normalize Z values
groundnorm normalize input.las ground.las normalized.las --k 3
```

### Batch Processing

```bash
# Process all tiles with CSF (with normalization)
groundnorm batch-csf /data/tiles /data/output --normalize --workers 4

# Process all tiles with SMRF (ground only)
groundnorm batch-smrf /data/tiles /data/output --workers 8
```

## Python API Usage

```python
from groundnorm import run_csf_ground_filter, run_smrf_ground_filter, normalize_las_z
from pathlib import Path

# Single file CSF
ground_las = run_csf_ground_filter(
    input_las="tile.las",
    output_las="ground_csf.las",
    resolution=0.5,
    iterations=500,
    threshold=0.01,
    rigidness=2,
)

# Single file SMRF
ground_las = run_smrf_ground_filter(
    input_las="tile.las",
    output_las="ground_smrf.las",
    slope=0.15,
    window=3.5,
    threshold=0.25,
)

# Normalize Z
normalize_las_z(
    input_las="tile.las",
    ground_las=ground_las,
    output_las="normalized.las",
    k=3,
    extra_dim_name="Z_norm",
)
```

### Batch Processing

```python
from groundnorm.dask_runner import process_batch_csf, process_batch_smrf
from pathlib import Path

# CSF batch
process_batch_csf(
    input_dir=Path("/data/tiles"),
    output_base=Path("/data/output"),
    pattern="tiles_*.las",
    normalize=True,
    n_workers=4,
)

# SMRF batch
process_batch_smrf(
    input_dir=Path("/data/tiles"),
    output_base=Path("/data/output"),
    pattern="tiles_*.las",
    normalize=True,
    n_workers=8,
)
```

## Output Structure

```
output_base/
├── ground_classification/
│   ├── csf/
│   │   ├── csf_tile1.las
│   │   └── csf_tile2.las
│   └── smrf/
│       ├── smrf_tile1.las
│       └── smrf_tile2.las
└── z_norm/
    ├── csf/
    │   ├── z_norm_csf_tile1.las
    │   └── z_norm_csf_tile2.las
    └── smrf/
        ├── z_norm_smrf_tile1.las
        └── z_norm_smrf_tile2.las
```

## Parameters

### CSF Parameters
- `resolution`: Cloth grid resolution (default: 0.5)
- `iterations`: Maximum iterations (default: 500)
- `threshold`: Classification threshold (default: 0.01)
- `rigidness`: 1=steep, 2=relief, 3=flat terrain (default: 2)

### SMRF Parameters
- `slope`: Slope parameter (default: 0.15)
- `window`: Window size (default: 3.5)
- `threshold`: Elevation threshold (default: 0.25)
- `scalar`: Scalar parameter (default: 1.35)

### Normalization Parameters
- `k`: Number of nearest neighbors (default: 3)
- `extra_dim_name`: Name of HAG dimension (default: "Z_norm")

## Requirements

- Python ≥ 3.10
- PDAL (with CSF and SMRF filters)
- numpy
- scipy
- laspy ≥ 2.0
- tqdm

## License

MIT
