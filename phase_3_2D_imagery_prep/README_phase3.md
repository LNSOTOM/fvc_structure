# Phase 3: 2D Imagery Preparation

Tools for processing orthomosaic imagery into structured tiles with vegetation classification labels.

## Method

Orthomosaic imagery from UAS acquisitions was processed into gridded tiles with corresponding vegetation structure labels. The workflow involves:
1. Clipping orthomosaics into regular grid tiles (30.70 × 30.70 m/pixel)
2. Extracting and stacking multispectral bands
3. Converting FVC (Fractional Vegetation Cover) annotations to structure-based classification
4. Rasterizing vector labels to match tile resolution

## Contents

```
phase_3_2D_imagery_prep/
├── step_1_orthomosaic_to_tiling/
│   ├── 1_1_clip_shp_gridAnnotation_MULTISPECTRALsampling.py  # Clip rasters by annotated grid
│   ├── 1_2_imageProcessing_stackedBandsRaster.py             # Stack multispectral bands
│   └── notebooks/
│       └── 01_clip_orthomosaic_by_grid.ipynb                 # Interactive tiling workflow
├── step_2_structure_2D_labels/
│   ├── 2_1_structure_2D_labels.py          # Convert FVC to structure classes
│   ├── 2_2_structure_2D_rasterLabels.py    # Rasterize vector labels
│   ├── 2_3_downsampling_labels.py          # Downsample labels (integer)
│   ├── 2_4_dowsampling_labels_float.py     # Downsample labels (float)
│   └── logs/                               # Classification processing logs
└── documentation/                          # Additional reference materials
```

## Usage

### Step 1: Orthomosaic Tiling
```bash
# Interactive workflow (recommended)
jupyter notebook step_1_orthomosaic_to_tiling/notebooks/01_clip_orthomosaic_by_grid.ipynb

# Or run scripts directly
python step_1_orthomosaic_to_tiling/1_1_clip_shp_gridAnnotation_MULTISPECTRALsampling.py
python step_1_orthomosaic_to_tiling/1_2_imageProcessing_stackedBandsRaster.py
```

### Step 2: Structure Classification
```bash
# Process in sequence
python step_2_structure_2D_labels/2_1_structure_2D_labels.py
python step_2_structure_2D_labels/2_2_structure_2D_rasterLabels.py
python step_2_structure_2D_labels/2_3_downsampling_labels.py
```

## Structure Classification Scheme

**Structure IDs:**
- **ID 0**: Forbs
- **ID 1**: Woody debris/stem (dead branches, standing stems, CWD)
- **ID 2**: Grass (hummock grass, senescent grass)
- **ID 3**: Shrubs (chenopod shrubs, all shrub types)
- **ID 4**: Mallee

**Excluded classes:** `bare_ground`, `shadow`

## Dependencies

```bash
conda install -c conda-forge gdal rasterio geopandas
```

## Outputs

- **Tiled rasters**: 30.72 × 30.72 m/pixel multispectral tiles (5 cm resolution)
- **Vector labels**: Structure-classified shapefiles with `structure_id` attribute
- **Raster labels**: Binary masks and multi-class rasters matching tile dimensions
- **Processing logs**: Classification statistics and polygon counts

## Notes

Related research context lives in the `second-brain` Obsidian vault, notably `wiki/concepts/label-quality-assessment.md`, `wiki/concepts/2d-to-3d-label-transfer.md`, and `wiki/concepts/fractional-vegetation-cover-fvc-mapping.md`.
