# Phase 3: 2D Imagery Preparation

Tools for processing orthomosaic imagery into structured tiles with vegetation classification labels.

## Method

Orthomosaic imagery from UAS acquisitions was processed into gridded tiles with corresponding vegetation structure labels. The workflow involves:
1. Clipping orthomosaics into regular non-overlapping grid tiles (30.72 × 30.72 m)
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
├── step_3_label_quality/
│   ├── label_qa.py                         # Quality assessment of the 2D reference labels
│   ├── run_label_qa.sh                     # Path wrapper for one site
│   ├── make_label_qa_table.py              # The assessment as LaTeX tables
│   ├── label_quality_tables.tex            # Generated tables, for the manuscript
│   ├── plot_centroids.py                   # Centroid/footprint figure
│   ├── label_centroids.{png,pdf,tex}       # Generated figure, medium site
│   ├── label_centroids_low.{png,pdf,tex}   # Generated figure, low site
│   └── notebooks/
│       └── 01_label_quality_assessment.ipynb
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

### Step 3: Reference-label quality assessment

```bash
python step_3_label_quality/label_qa.py --help
```

Writes a per-tile summary and a per-polygon issue log covering annotation coverage and class
balance, digitising artefacts and polygon overlap, per-class IoU through rasterisation,
through the 1 cm to 5 cm downsample and onto the predictor grid, sensitivity to a 5 cm
boundary shift, and per-polygon top-of-canopy height. The summary closes with an `Average`
and a `Median` row; `notebooks/01_label_quality_assessment.ipynb` runs both sites and renders
the **mean / median IoU** at each step plus a summary paragraph generated from the CSVs.

Per-feature centroid displacement (RMSE and MAE, in metres) and per-feature IoU follow the
co-registration assessment of [Haynes et al.
(2025)](https://www.mdpi.com/2504-446X/9/2/132), separating how far a labelled feature moved
from how much of it survived. Measured here, the labels hold their position to within a third
of a cell while losing feature shape, which is what decides the repair.

See `step_3_label_quality/README.md` for what each measure means — including the 5 cm grid
offset it found, which inflates the delivered 5 cm labels by roughly 10%, and the stale
low-site shapefiles that collapse one class's vector-to-raster IoU to 0.014.

No repeat annotation exists, so inter-annotator agreement is not measured or claimed; the
`Delineation IoU` columns bound how much a different outline could change instead.

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

- **Tiled rasters**: 30.72 × 30.72 m five-band multispectral tiles at each orthomosaic's
  native resolution — 5 cm at site1_1_DD0001 (614 × 614 px), 6 cm at site2_1_DD0011 (512 × 512 px)
- **Vector labels**: Structure-classified shapefiles with `structure_id` attribute
- **Raster labels**: multi-class rasters at 1 cm (3072 × 3072 px), downsampled to 5 cm
  (614 × 614 px) for model input
- **Processing logs**: Classification statistics and polygon counts

## Notes

Related research context lives in the `second-brain` Obsidian vault, notably `wiki/concepts/label-quality-assessment.md`, `wiki/concepts/2d-to-3d-label-transfer.md`, and `wiki/concepts/fractional-vegetation-cover-fvc-mapping.md`.
