rasterclip — LAS clipping to raster tiles

Overview
maskclip — Mask-based LAS classification

Overview
- Reprojects LAS to predictor raster CRS and crops to raster bounds.
- Colorizes points using structure mask raster into `Classification` and filters desired classes.
- Parallel processing with Dask, progress bar, and signature file.

CLI (via module)
```
python -m maskclip.cli /path/las_dir /path/raster_dir /path/mask_dir /path/out_dir \
  --pattern "tiles_multispectral*.las" --no-skip --no-signature
```

Python API
```
from maskclip import classify_single, classify_many_dask

outs = classify_many_dask(
    las_dir=..., raster_dir=..., mask_dir=..., out_dir=...,
    pattern="tiles_multispectral*.las", skip_existing=True, write_signature=True,
)
```

Notebooks
- `notebooks/02_mask_site01.ipynb` – Site 1 config, Dask run, signature, and classification histogram.
- `notebooks/02_mask_site11.ipynb` – Site 11 config, Dask run, signature, and classification histogram.


```
python -m pip install -e /home/nvidia/projects/fvc_structure/phase_4_3D_lidar_prep/step_1_clipping
```
Usage
- Single tile:
```
```
- Batch (sequential):
```
rasterclip many /path/input.las /path/tiles_dir /path/out_dir --pattern "tiles_multispectral*.tif"
```
- Batch (Dask):
```
rasterclip dask /path/input.las /path/tiles_dir /path/out_dir \
	--pattern "tiles_multispectral*.tif" \
	--no-skip            # process even if outputs exist \
	--no-signature       # skip writing clip_signature.txt
```

Python API
```
from rasterclip import clip_single, clip_many, clip_many_dask

# Dask with signature logging
outs = clip_many_dask(
	input_las=..., aoi_raster_dir=..., out_dir=...,
	pattern="tiles_multispectral*.tif", skip_existing=True, write_signature=True
)
```
