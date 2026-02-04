# Step 1: LiDAR Clipping

Tools for clipping LiDAR point clouds to raster tile extents and applying mask-based classification.

## rasterclip - Clip LAS to raster tiles

Clips large LAS files to match raster tile extents with CRS reprojection.

**Features:**
- Pre-crops in LAS native CRS using transformed raster bounds
- Reprojects output to raster CRS
- Parallel processing with Dask
- Progress tracking and signature logging

**Python API:**
```python
from rasterclip import clip_single, clip_many, clip_many_dask

# Single tile
pts = clip_single(input_las, tile_raster, output_las)

# Batch (sequential)
outs = clip_many(input_las, tiles_dir, out_dir, pattern="*.tif")

# Batch (Dask parallel)
outs = clip_many_dask(
    input_las, tiles_dir, out_dir,
    pattern="tiles_multispectral*.tif",
    skip_existing=True,
    write_signature=True
)
```

**Notebooks:**
- `01_clip_site01.ipynb` – Site 1 (DD0001) raster clipping workflow
- `01_clip_site11.ipynb` – Site 11 (DD0011) raster clipping workflow

---

## maskclip - Mask-based LAS classification

Applies structure classification to clipped LAS tiles using 2D mask rasters.

**Features:**
- Reprojects LAS to raster CRS and crops to bounds
- Transfers mask pixel values (64-68) to LAS `Classification` field
- Filters points by desired structure classes
- Parallel Dask processing with progress tracking

**Python API:**
```python
from maskclip import classify_single, classify_many_dask

outs = classify_many_dask(
    las_dir, raster_dir, mask_dir, out_dir,
    pattern="tiles_multispectral*.las",
    skip_existing=True,
    write_signature=True
)
```

**Notebooks:**
- `02_mask_site01.ipynb` – Site 1 mask classification with histograms
- `02_mask_site11.ipynb` – Site 11 mask classification with histograms
- `03_icp_regCheck.ipynb` – ICP registration verification
- `04_maskCoreg_site01.ipynb` – Co-registered mask clipping workflow

---

## Installation

```bash
cd to folder: (fvc_structure) nvidia@brev-rvbt83sj0:~/projects/fvc_structure/phase_4_3D_lidar_prep/step_1_clipping
pip install -e . or
pip install -e /media/laura/laura_usb/code/fvc_structure/phase_4_3D_lidar_prep/step_1_clipping
```

## Dependencies

See [pyproject.toml](pyproject.toml) for full list:
- pdal, laspy, rasterio
- dask[distributed], tqdm
- shapely, pyproj

Python API
```
from rasterclip import clip_single, clip_many, clip_many_dask

# Dask with signature logging
outs = clip_many_dask(
	input_las=..., aoi_raster_dir=..., out_dir=...,
	pattern="tiles_multispectral*.tif", skip_existing=True, write_signature=True
)
```
