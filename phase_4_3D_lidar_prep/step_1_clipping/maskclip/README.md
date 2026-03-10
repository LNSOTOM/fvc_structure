Mask-based LAS classification (maskclip)

Overview
- Reprojects LAS to raster CRS, crops to raster bounds.
- Colorizes using mask raster into Classification, filters desired classes, writes LAS.
- Dask parallel runner with progress and signature logging.

CLI
- `maskclip /path/las_dir /path/raster_dir /path/mask_dir /path/out_dir --pattern "tiles_multispectral*.las"`

Python API
- `from maskclip import classify_single, classify_many_dask`
