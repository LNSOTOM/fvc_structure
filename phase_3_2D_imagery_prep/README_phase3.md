# Phase 3: 2D Imagery Preparation

Tools for processing orthomosaic imagery into structured tiles with vegetation classification labels.

For inputs, environment checks, historical settings and known reproducibility limitations, see
the [Phase 3 handoff](REPRODUCIBILITY.md). The config-driven
`imagery_prep.py` runs the preparation steps; the original scripts remain
site-specific interactive examples.

## Method

Orthomosaic imagery from UAS acquisitions was processed into gridded tiles with corresponding vegetation structure labels. The workflow involves:
1. Clipping orthomosaics into regular non-overlapping grid tiles (30.72 × 30.72 m)
2. Extracting and stacking multispectral bands
3. Converting FVC (Fractional Vegetation Cover) annotations to structure-based classification
4. Rasterizing vector labels on a 1 cm reference grid
5. Downsampling labels to 5 cm and remapping classes
6. Assessing label quality and predictor-grid alignment separately

## Contents

```
phase_3_2D_imagery_prep/
├── imagery_prep.py                         # Config-driven preparation stages
├── phase3_config.py                         # YAML config loader
├── configs/
│   └── imagery_prep_job.example.yml         # Copy-and-edit job template
├── REPRODUCIBILITY.md                       # Setup and handoff guide
├── tests/
│   └── test_imagery_prep.py                 # Preparation workflow checks
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
│       └── 01_label_quality_assessment.ipynb  # Runs both sites; tables, summary, figures
└── documentation/                          # Additional reference materials
```

## Usage

### Config-driven preparation

After setting up the environment and editing the example config as described in
the [handoff](REPRODUCIBILITY.md), run all stages from the repository root:

```bash
python phase_3_2D_imagery_prep/imagery_prep.py --config phase_3_2D_imagery_prep/configs/imagery_prep_job.local.yml --stage all
```

This does not pause between stages. To inspect each stage before continuing,
replace `all` with `tile`, `stack`, `classify`, `rasterize` and `downsample`
in turn. Review the log and run label QA before using the results.

### Original interactive workflow
```bash
# Interactive workflow (recommended)
jupyter notebook step_1_orthomosaic_to_tiling/notebooks/01_clip_orthomosaic_by_grid.ipynb

# The scripts contain site-specific paths; configure and run the relevant cells interactively.
```

### Step 2: Structure Classification
```bash
# Configure paths and run the relevant cells interactively in sequence.
# See REPRODUCIBILITY.md before running: these scripts have multiple example cells.
```

### Step 3: Reference-label quality assessment

```bash
./step_3_label_quality/run_label_qa.sh site1_1_DD0001 Medium annotations_shp
./step_3_label_quality/run_label_qa.sh site2_1_DD0011 Low     annotation_shp
python step_3_label_quality/make_label_qa_table.py --summary ... --summary ... \
    --figure step_3_label_quality/label_centroids.tex \
    --figure step_3_label_quality/label_centroids_low.tex \
    --output step_3_label_quality/label_quality_tables.tex
```

`label_qa.py` follows a label from the drawn polygon to the array a model reads, in four checks:

| Check | Measures |
|---|---|
| 1. Annotation QA | invalid geometries, cross-class overlap, polygons below the minimum mapping unit |
| 2. Polygon → 1 cm raster | class IoU |
| 3. 1 cm → 5 cm label grid | class IoU, polygon IoU, centroid RMSE |
| 4. Label grid → predictor grid | class IoU, polygon IoU, centroid RMSE, grid offset |
| 5. ±5 cm boundary perturbation | delineation IoU |

It writes a per-tile summary, closing with `Average` and `Median` rows, and a per-polygon issue
log. `notebooks/01_label_quality_assessment.ipynb` runs both sites and renders a QA table in
the order above and a per-class IoU table, both as **mean / median**, a summary paragraph
generated from the CSVs, and the centroid figure for each site with a guide to reading it.
`make_label_qa_table.py` writes the same two tables as LaTeX for the manuscript.

Polygon IoU and centroid RMSE follow the co-registration assessment of [Haynes et al.
(2025)](https://www.mdpi.com/2504-446X/9/2/132), comparing each feature's footprint with its
footprint on the preceding grid. Centroids move less than a quarter of a cell at every step at
both sites while feature shape changes, so the labels are in the right place and the wrong shape.

`plot_centroids.py` draws that per feature: one example polygon per growth form on the 1 cm RGB
and the multispectral predictor, with its 5 cm label and predictor footprints, their centroids,
and the centroid shift (mm) for `label 1 cm → label 5 cm` and `label 5 cm → predictor`.

LiDAR height is not part of the assessment: the scheme classifies growth form, which implies no
stature, so a height comparison could describe the classes but not grade the labels.

See `step_3_label_quality/README.md` for what each measure means, how to read the figures, and
the findings to settle before reporting: the 1–4 cm 5 cm grid offset on 13 of 15 medium-site
tiles, the 6 cm predictor pixels against 5 cm labels at the low site, and the stale low-site
shapefiles that collapse Woody debris polygon → 1 cm IoU to 0.014.

No repeat annotation exists, so inter-annotator agreement is not measured or claimed; the
`Delineation IoU` columns bound how much a different outline could change instead.

## Structure Classification Scheme

**Structure IDs:**
- **ID 0**: Forbs
- **ID 1**: Woody debris/stem (dead branches, standing stems, CWD)
- **ID 2**: Grass (hummock grass, senescent grass)
- **ID 3**: Shrubs (chenopod shrubs, all shrub types)
- **ID 4**: Mallee

**Excluded classes:** `bare_ground`, `shadow`, `be`, `unknowm` (spelling in the script).
Other unrecognized values are removed when a type column is present; see the handoff.

## Dependencies

```bash
conda install -c conda-forge gdal rasterio geopandas
```

## Outputs

- **Tiled rasters**: 30.72 × 30.72 m five-band multispectral tiles at each orthomosaic's
  native resolution — 5 cm at site1_1_DD0001 (614 × 614 px), 6 cm at site2_1_DD0011 (512 × 512 px)
- **Vector labels**: Structure-classified shapefiles with `struct_id` attribute
- **Raster labels**: multi-class rasters at 1 cm (3072 × 3072 px), downsampled to 5 cm
  (614 × 614 px) for model input
- **Processing logs**: Classification statistics and polygon counts

## Notes

Related research context lives in the `second-brain` Obsidian vault, notably `wiki/concepts/label-quality-assessment.md`, `wiki/concepts/2d-to-3d-label-transfer.md`, and `wiki/concepts/fractional-vegetation-cover-fvc-mapping.md`.
