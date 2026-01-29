# fvc3dstruct Package Improvements

## Overview

The `fvc3dstruct` package has been **completely refactored** to match the comprehensive `merge_znorm_struct_to_maskClasses_polar_malle_id.py` pipeline while maintaining a clean, modular structure.

## Key Improvements

### 1. **Complete Pipeline Implementation** (`pipeline.py`)
- ✅ **7-stage comprehensive pipeline**:
  1. Data-driven height threshold computation
  2. Cartesian voxel merge (majority voting)
  3. Watershed crown detection with fallback logic
  4. Crown-centric polar refinement
  5. Height sanity rules (merge woody debris → mallee)
  6. TreeID assignment (mallee canopy only)
  7. LAS output with proper format conversion

### 2. **Enhanced Voxel Merge** (`voxel_merge.py`)
- ✅ Proper 3D voxel grid with separate XY and Z resolution
- ✅ Efficient voxel-to-class mapping with Counter-based majority voting
- ✅ Detailed progress reporting (voxel count, label transfer rate)
- ✅ Matches merge_znorm implementation exactly

### 3. **Robust Watershed Crown Detection** (`watershed.py`)
- ✅ Hybrid blob + peak detection for better crown segmentation
- ✅ **Smart fallback logic**: uses Z_norm points when structure LAS lacks mallee
- ✅ Progressive threshold relaxation (1.0m → 0.8m → 0.6m → 0.4m → 0.3m → 0.1m)
- ✅ Blob-based marker enhancement to avoid missing isolated crowns
- ✅ Returns crown labels, centroids, and grid origin for downstream use

### 4. **Crown-Centric Polar Refinement** (`polar_refine.py`) **[NEW]**
- ✅ Calculate polar coordinates (r, θ) around each crown centroid
- ✅ Discretize into radial and angular bins (configurable resolution)
- ✅ Build class probability distributions from structure LAS per bin
- ✅ Apply soft height priors based on data-driven ranges
- ✅ Update Z_norm labels with confidence thresholds:
  - Fill empty labels if confidence ≥ `min_conf`
  - Flip existing labels if new confidence ≥ `min_flip_ratio` × old confidence
- ✅ Detailed update statistics

### 5. **Data-Driven Height Statistics** (`height_stats.py`)
- ✅ Already comprehensive (no changes needed)
- ✅ NN-based matching between structure and Z_norm clouds
- ✅ Percentile-based threshold estimation (15th–85th by default)
- ✅ Robust outlier removal (clip to 5th–95th percentile first)
- ✅ Auto mallee minimum height computation

### 6. **LAS I/O Utilities** (`io_las.py`)
- ✅ Added `ensure_las14_pf6()` for proper format conversion
- ✅ Preserves XYZ coordinates, standard dimensions, and extra dimensions
- ✅ Handles scale/offset correctly
- ✅ Safe path handling with automatic directory creation

### 7. **Configuration** (`config.py`)
- ✅ Updated to match merge_znorm defaults:
  - `voxel_size_xy`: 0.10 m
  - `voxel_size_z`: 0.10 m
  - `r_bin`: 0.10 m (polar radial bins)
  - `theta_bin_deg`: 10° (polar angular bins)
  - `min_conf`: 0.3 (minimum confidence to update labels)
  - `min_flip_ratio`: 1.5 (ratio to flip existing labels)
  - `enable_radial_fill`: False (optional radial gap filling)

## Module Structure

```
fvc3dstruct/
├── __init__.py           # Package exports
├── config.py             # Default configuration
├── pipeline.py           # Main processing pipeline
├── voxel_merge.py        # Cartesian voxel merge
├── watershed.py          # Crown detection via watershed
├── polar_refine.py       # Crown-centric polar refinement [NEW]
├── height_stats.py       # Data-driven height thresholds
└── io_las.py             # LAS I/O utilities
```

## Usage

### Simple Script Usage
```python
from fvc3dstruct import process_tile

process_tile(
    z_path="path/to/z_norm.las",
    s_path="path/to/structure.las",
    out_path="path/to/output.las",
    config={
        "voxel_size_xy": 0.10,
        "r_bin": 0.15,
        # ... other overrides
    }
)
```

### Advanced Usage with Individual Components
```python
from fvc3dstruct import (
    cartesian_voxel_merge,
    get_mallee_centroids_watershed,
    refine_with_polar_coords,
    compute_class_height_thresholds,
)

# Step 1: Compute height ranges
height_ranges = compute_class_height_thresholds(las_s, las_z)

# Step 2: Voxel merge
classification = cartesian_voxel_merge(las_z, las_s)

# Step 3: Detect crowns
labels, centroids, min_x, min_y = get_mallee_centroids_watershed(
    las_s,
    fallback_points=(las_z.x, las_z.y, zn_arr),
    fallback_min_height=1.0,
)

# Step 4: Polar refinement
classification = refine_with_polar_coords(
    las_z, las_s, classification, zn_arr,
    labels, centroids, min_x, min_y, 0.10,
    height_ranges,
)
```

## Environment Setup

The package now uses **micromamba** for environment management:

```bash
# Activate environment
micromamba activate fvc_structure

# Run pipeline
python main_run.py
```

Or use the test script:
```bash
./test_pipeline.sh
```

## Key Differences from Original merge_znorm Script

### Simplified:
- ❌ Removed optional radial gap filling (can be re-added if needed)
- ❌ Removed staged output saving (focus on final output)
- ❌ Removed debug plotting (keep notebooks for visualization)
- ❌ Removed folder-level batch processing (handled by main_run.py)

### Preserved:
- ✅ All core pipeline stages (voxel merge → watershed → polar → height rules)
- ✅ Data-driven height threshold estimation
- ✅ Robust fallback logic for crown detection
- ✅ Soft height priors in polar refinement
- ✅ TreeID assignment for mallee canopy only
- ✅ LAS 1.4 format 6 output with proper extra dimensions

### Enhanced:
- ✅ Better code organization (modular, testable)
- ✅ Clearer function signatures and documentation
- ✅ Progress reporting at each stage
- ✅ Easier to customize via config dict
- ✅ Simpler to integrate into other workflows

## Testing

The notebook `01_run_pipeline.ipynb` has been updated to:
1. Use the new `process_tile()` function
2. Work with micromamba environment
3. Demonstrate all pipeline stages with visualizations

## Migration Notes

**Old code:**
```python
merge_znorm_structure_one_file(
    z_norm_las=z_path,
    structure_las=s_path,
    output_las=out_full,
    output_las_no0=out_no0,  # Removed
    config=config,
)
```

**New code:**
```python
process_tile(
    z_path=z_path,
    s_path=s_path,
    out_path=out_full,
    config=config,
)
```

## Performance

The improved pipeline maintains the same computational complexity as the original:
- **Voxel merge**: O(N log N) for sorting + O(N) for mapping
- **Watershed**: O(W×H) for grid operations + O(K log K) for peak detection
- **Polar refinement**: O(N) for coordinate transform + O(B) for bin statistics
- **Overall**: Dominated by point cloud size N and grid resolution

## Future Enhancements

Potential additions (not yet implemented):
- [ ] Optional radial gap filling (Stage 3 from merge_znorm)
- [ ] Staged output saving (Cartesian, Polar, Radial snapshots)
- [ ] Debug visualization exports
- [ ] Multi-tile batch processing helper
- [ ] Parallel processing for large datasets
- [ ] Quality metrics and validation reports
