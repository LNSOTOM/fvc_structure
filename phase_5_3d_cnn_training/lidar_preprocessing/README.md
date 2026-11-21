# LiDAR Preprocessing Package

Unified preprocessing pipeline for Calperum LiDAR tiles, producing outputs compatible with both MINSU3D and ForestFormer3D frameworks.

## Features

- **Dual Output Format**: Generates both NPZ (MINSU3D) and PLY (ForestFormer3D) from the same preprocessing
- **Configurable**: Easy-to-use configuration dataclass
- **Validated**: Built-in validation and comparison utilities
- **Reproducible**: Ensures identical preprocessing for fair model comparisons

## Installation

```bash
# Install required dependencies
pip install laspy numpy pandas pyntcloud
```

## Quick Start

### As a Package

```python
from lidar_preprocessing import PreprocessingConfig, batch_process

# Create configuration
config = PreprocessingConfig(
    input_dir="/path/to/las/files",
    out_npz_dir="/path/to/output/npz",
    out_ply_dir="/path/to/output/ply",
    voxel_size=0.10,
)

# Process all files
npz_files, ply_files, errors = batch_process(config)
```

### As a Notebook

Open and run `notebooks/lidar_preprocessing_pipeline.ipynb` for an interactive workflow with visualizations.

For the historical notebook's 1–5 mapping and the checked three-class HSM
workflow, see [Phase 5 reproducibility](../REPRODUCIBILITY.md). The package
defaults below use a different 0–4 class mapping.

## Output Formats

### NPZ Format (MINSU3D)

Each NPZ file contains:
- `coords`: (N, 3) int32 - Voxel indices
- `feats`: (N, 5) float32 - Features [x_local, y_local, z_local, z_norm, intensity]
- `semantic_label`: (N,) int32 - Class labels (0..K-1, or -100 for ignore)
- `instance_label`: (N,) int32 - Tree instance IDs
- `offset`: (3,) float64 - XYZ offset for coordinate reconstruction

### PLY Format (ForestFormer3D)

Each PLY file contains point-level attributes:
- `x, y, z`: Local coordinates by default; optional global coordinates
- `semantic_seg`: Class labels (0..K-1)
- `treeID`: Tree instance IDs
- `z_norm`: Height above ground
- `intensity`: Normalized intensity [0, 1]

**Note**: PLY files exclude points with `semantic_label == IGNORE_INDEX`
and default local coordinates require the `{stem}_offset.npy` sidecar for
global reconstruction.

## Configuration

```python
@dataclass
class PreprocessingConfig:
    input_dir: Path  # Directory with LAS files
    out_npz_dir: Path  # Output directory for NPZ files
    out_ply_dir: Path  # Output directory for PLY files
    voxel_size: float = 0.10  # Voxel size in meters
    ignore_index: int = -100  # PyTorch ignore index
    class_remap: Dict[int, int]  # LAS code -> model class mapping
```

Default class mapping:
- 64 → 0 (Forb)
- 65 → 1 (Woody debris)
- 66 → 2 (Hummock)
- 67 → 3 (Shrub)
- 68 → 4 (Mallee)
- 0 → -100 (Background/Unmatched)

## Utilities

### Validation

```python
from lidar_preprocessing.utils import compare_npz_ply

# Verify NPZ and PLY consistency
is_valid = compare_npz_ply(npz_path, ply_path)
```

### Analysis

```python
from lidar_preprocessing.utils import (
    load_npz_summary,
    load_ply_summary,
    dataset_statistics,
    npz_to_dataframe
)

# Get summary statistics
npz_stats = load_npz_summary(npz_path)
ply_stats = load_ply_summary(ply_path)

# Dataset-wide statistics
stats = dataset_statistics(npz_list_path)

# Convert to DataFrame for analysis
df = npz_to_dataframe(npz_path)
```

## Directory Structure

```
lidar_preprocessing/
├── __init__.py          # Package exports
├── config.py            # Configuration dataclass
├── preprocessing.py     # Core preprocessing functions
├── utils.py             # Validation and analysis utilities
└── README.md           # This file

notebooks/
└── lidar_preprocessing_pipeline.ipynb  # Interactive workflow
```

## Example Workflow

1. **Configure** preprocessing parameters
2. **Process** LAS files → NPZ + PLY
3. **Validate** output consistency
4. **Analyze** class distributions and statistics
5. **Train** models using identical preprocessed data

## Notes

- Intensity is automatically normalized based on bit depth (8-bit, 16-bit, or custom)
- Z_norm is extracted from LAS ExtraBytes if available, otherwise computed as z - z_min
- Tree IDs must be present in LAS files as `treeID` ExtraBytes dimension
- Points with unknown class (0) are assigned to IGNORE_INDEX

## License

Part of the FVC Structure project.
