# Step 3: Quality assessment of the 2D reference labels

The 2D structure labels built in `step_2_structure_2D_labels` are the reference for every
downstream experiment. This step quantifies their quality so the number can be reported
rather than assumed.

| Script | Needs | Answers |
|---|---|---|
| `label_qa.py` | only the delivered artefacts | How internally consistent, and how faithfully preserved, are the labels? |
| `compare_reannotation.py` | a repeat delineation of a tile subset | How reproducible is the annotation itself? |

`label_qa.py` can be run today. `compare_reannotation.py` needs a second digitisation
first — see [Producing the repeat annotation](#producing-the-repeat-annotation).
`notebooks/01_label_quality_assessment.ipynb` drives both and explains each number.

## `label_qa.py`

Writes three files into `--output-dir`:

| File | Contents |
|---|---|
| `<site>_qa_summary.csv` | one row per tile, plus an Average row |
| `<site>_qa_issues.csv` | one row per individual problem, so a flagged polygon can be opened in QGIS |
| `<site>_qa_report.md` | the numbers written out as the paragraph you would send a reviewer |

```bash
python label_qa.py \
    --site site1_1_DD0001 --site-condition Medium \
    --shp-dir         /path/.../structure_class/annotations_shp \
    --raster-dir      /path/.../structure_class/annotation_raster \
    --downsampled-dir /path/.../annotation_raster/downsample_5cm_nearest \
    --las-dir         /path/site1_1_DD0001/lidar/clipped/raster/z_norm/csf \
    --output-dir      /path/site1_1_DD0001/lidar/clipped/validation/label_quality
```

Or `./run_label_qa.sh <site> <condition> <shp-subdir>`, which fills in the layout on `D:` and
falls back to the `/tmp/wsl_drive_d` mirror when the drive is not mounted in WSL:

```bash
./run_label_qa.sh site1_1_DD0001 Medium annotations_shp   # medium condition
./run_label_qa.sh site2_1_DD0011 Low     annotation_shp   # low condition
```

The two are **different sites**, named as they are throughout phases 2, 4 and 5.

Only `--site`, `--shp-dir` and `--output-dir` are required; any measure whose inputs are
absent is left blank rather than failing. `--tiles 26,109` restricts the run. Tiles are
matched on the trailing numeric ID, as in phase 4. Note the shapefile directory is named
`annotations_shp` for DD0001 and `annotation_shp` for DD0011.

### What the summary measures

**Annotation.** Labelled area as a share of the tile, per-class area share, and the
digitising artefacts: invalid rings, duplicates, sub-MMU slivers, and polygons of different
classes that overlap. Sparse polygon annotation is not wall-to-wall, so the labelled
percentage states how much of each tile actually carries a reference value. Cross-class
overlap area is a direct, annotator-independent measure of label ambiguity: a pixel under
two polygons of different classes has no single reference value, and which one it receives
is decided by shapefile row order.

**Preservation, as IoU.** The labels pass through two lossy transformations before a model
sees them, and each gets one number per class:

| Column | Compares | Measures |
|---|---|---|
| `IoU raster <class>` | polygons rasterised exactly (centroid rule) vs the delivered 1 cm raster | the cost of `all_touched=True`, which dilates every polygon by up to one 1 cm ring, plus the cost of resolving overlaps by draw order |
| `IoU 5 cm <class>` | majority class of each 5 cm cell's twenty-five 1 cm sub-pixels vs the delivered 5 cm raster | how much the downsample changed the reference the models actually consume |

Nodata is counted as a category in the majority rule. Taking the majority over labelled
sub-pixels only would call a cell "woody debris" on the strength of a single 1 cm return in
twenty-five, which no downsampling rule would deliver, and would depress the IoU of every
sparse class for no reason.

Two columns, `5 cm grid offset (cm)` and `Labelled area change at 5 cm (%)`, sit beside the
IoU as its diagnosis — see [the 5 cm grid offset](#the-5-cm-grid-offset) below.

**Plausibility.** Per polygon, the P95 of `Z_norm` among the LiDAR returns inside it — its
top-of-canopy height — against the range its class implies. LiDAR height was never used to
make the 2D labels, so this is a semi-independent check that the classes mean what the
scheme says.

This is deliberately a *per-polygon top-of-canopy* statistic. A 2D mask is a crown
*footprint*, so every return beneath the plant — ground, litter, understorey — falls inside
it; taking the median over the whole column measures the ground, not the plant, and reports
every tall class as implausible by construction. Polygons with fewer than `--min-points`
(default 10) returns are excluded, since a P95 over a handful of points measures where the
returns landed.

### Issue types in `<site>_qa_issues.csv`

`invalid_geometry`, `empty_geometry`, `duplicate_geometry`, `small_area`,
`overlap_same_class`, `overlap_different_class`, `unmapped_class`, `height_implausible`,
`too_few_lidar_points`, `grid_misaligned_5cm`, `missing_raster`, `missing_downsampled`,
`missing_las`.

## `compare_reannotation.py`

| Table | Metric |
|---|---|
| R1 agreement | OA, Cohen's kappa, per-class IoU and F1, reported twice: over all labelled cells, and with a 5-cell buffer around class boundaries excluded |
| R2 area | per class, polygon count and area in each version, and the symmetric % difference |
| R3 confusion | pooled cell-level confusion matrix between the two versions |

The two scopes matter: the all-cells number mixes *delineation* disagreement (exactly where
a crown edge falls) with *classification* disagreement (what the plant is). The
interior-only number isolates the second, which is the one that limits achievable model
accuracy. Reporting both pre-empts the objection that a mediocre OA is just edge noise.

```bash
python compare_reannotation.py \
    --shp-dir-a  /path/.../annotations_shp \
    --shp-dir-b  /path/.../annotations_shp_reannotation \
    --raster-dir /path/.../annotation_raster \
    --site-condition Medium \
    --output-dir /path/site1_1_DD0001/lidar/clipped/validation/label_quality
```

Rasterisation uses `all_touched=True` to match step 2, so the agreement describes the labels
as the experiments actually consume them. `--no-all-touched` gives the geometry-only
comparison.

## Producing the repeat annotation

`label_qa.py` characterises the labels but cannot measure annotation *reproducibility* — for
that a second delineation is needed. A defensible and affordable design:

1. Draw **3 tiles per site** at random, stratified so each site's dominant classes appear.
   Record the seed.
2. Re-digitise them from the same 1 cm orthomosaic, blind to the originals — a second
   annotator if one is available, otherwise the same annotator after a gap long enough that
   the first delineation is not recalled. Same class scheme, same QGIS setup.
3. Save to a parallel directory with identical filenames, run `compare_reannotation.py`, and
   report the average row of R1 plus the R3 confusion matrix.

If any field data exist for these sites, a third comparison against field-identified plants
would upgrade this from *reproducibility* to *accuracy*. Reproducibility alone is still the
standard answer to this reviewer question and is normally sufficient.

## Findings to settle before reporting

### The 5 cm grid offset

**The delivered 5 cm labels are dilated relative to the 1 cm delineation.** Measured on the
medium site: 13 of 15 tiles have a 5 cm grid sitting 1–4 cm off the 1 cm grid, and those
tiles carry **+9.6% to +14.1% more labelled area** than their 1 cm raster implies. The
inflation is concentrated in the thinnest classes — on tile 121, woody debris covers 2.5×
the cell-equivalents it should, grass 1.4×, mallee 1.08×. IoU 5 cm falls to 0.36–0.77 on
those tiles and stays at 0.99 on the two aligned ones.

The mechanism: a 3072 px tile at 1 cm is 30.72 m, and 30.72 / 0.05 = 614.4 is not an
integer, so `gdalwarp -tr 0.05` in `2_3_downsampling_labels.py` cannot land on the 1 cm
grid — it snaps to 614 cells and shifts the origin. Because that same call passes
`-srcnodata 255`, nearest resampling onto the shifted grid takes the nearest *valid* source
pixel in preference to a nodata one, so sparse classes are dilated rather than sampled.
Verified: 99.8% of the extra labelled cells have a 1 cm pixel of that exact class inside
their own 5×5 block, so this is neighbour dilation, not different source material.

The models consume the 5 cm labels, so of everything here this has the most direct bearing
on the reported results. The fix is to make the 5 cm grid a whole multiple of the 1 cm grid
before warping — either pad the 1 cm tile to 3075 px, or pass `-tap` and drop `-srcnodata`
so nodata decimates like any other value.

### The `litter` class differs between sites

**The `litter` class is handled inconsistently.** In `2_1_structure_2D_labels.py` the mapping
line `'litter': 1` is commented out (line 14). The low-site labels were produced while it was
active, the medium-site labels after it was disabled, so the two runs did not use the same
definition of ID 1.

Counted from the committed logs, **the consequence is one-sided**: the low site contained
just **1** litter polygon, so folding it in changed almost nothing there, while the medium
site contained **752**, all dropped. ID 1 therefore means "woody debris, litter excluded" at
the medium site and "woody debris plus litter" at the low site — but only the medium site
lost material to the choice. Either re-run one site so both use the same mapping, or state
the exclusion explicitly. This is a decision about label semantics, so it has deliberately
not been changed here.

### The low-site shapefiles do not match the low-site rasters

**`EXCLUDE_CLASSES` contains the typo `'unknowm'`** (line 31), so `'unknown'` never matches
the exclusion list. The consequence is larger than a miscounted class, and it differs by
site, so check the delivered files rather than the logs.

The medium-site shapefiles match their log and contain no `unknown`. The low-site shapefiles
on `D:` are dated 2025-10-01, predating the committed 2025-10-09 log run, and **retain 13
`unknown` polygons carrying `struct_id` 1 (Woody debris)** — one in each of 13 of the 15
tiles. These are not stray slivers: they average **119 m², 12.6% of a tile**, 1550 m² in
total, against just **10.4 m² of genuine woody-debris content** at that site. They are the
annotator's catch-all background polygon.

The delivered 1 cm rasters were built *after* those polygons were removed, so the rasters are
correct and `IoU raster Woody debris` collapses to **0.014** — that is the shapefiles and the
rasters disagreeing, not an `all_touched` effect. `label_qa.py` raises
`raster_disagrees_with_shapefile` whenever a class scores below 0.50.

The practical risk: **re-running `2_2_structure_2D_rasterLabels.py` over the shapefiles
currently on disk would inject a false Woody-debris region covering an eighth of each
affected tile.** Fix the typo and re-export the low-site shapefiles before any step of the
pipeline is re-run from them.

### Class coverage differs sharply between sites

The medium site has IDs 1–4 and no forbs; the low site has IDs 0, 1 and 3 and no grass or
mallee. Only classes 1 and 3 are common to both — and class 1 is the one affected by both
issues above. Any class with 0 polygons in a site leaves every per-class column blank for
that site.

## Measured so far

Both sites, 15 tiles each, from `label_qa.py`.

| | Medium (DD0001) | Low (DD0011) |
|---|---|---|
| Labelled area | 20.2% of tile | 35.1% of tile |
| Polygons per tile | 408, median 0.076 m² | 413, median 0.278 m² |
| Invalid geometries per tile | 7.6, all repairable | 0.8, all repairable |
| Sub-MMU slivers | 1.38% | 0.68% |
| Overlapping pairs per tile | 13.3 | 147.8 |
| Cross-class overlap | 0.021% of labelled area | 0.161% of labelled area |
| IoU vector → 1 cm | 0.853–0.984 | 0.956–0.980, **Woody debris 0.014** |
| IoU 1 cm → 5 cm | 0.678–0.949 | 0.991–0.999 |
| 5 cm grid offset | 1–4 cm on 13 of 15 tiles | 0 cm on all tiles |

Both preservation problems are site-specific and independent: the medium site's 5 cm labels
are dilated, and the low site's shapefiles disagree with its rasters. Neither shows up in a
count of invalid geometries, which is why the IoU columns exist.

### Top-of-canopy height, both sites

| Class · site | Expected (m) | Median P95 (m) | Polygons outside |
|---|---|---|---|
| Forb · Low | 0.00–0.40 | 0.030 | 1% |
| Grass · Medium | 0.00–0.80 | 0.077 | 3% |
| Woody debris · Medium | 0.05–5.00 | 0.265 | 24% |
| Woody debris · Low | 0.05–5.00 | 0.265 | 55% |
| **Shrub · Medium** | 0.40–3.00 | **0.111** | **90%** |
| **Shrub · Low** | 0.40–3.00 | **0.297** | **81%** |
| Mallee · Medium | 1.50–8.00 | 1.418 | 55% |

**Shrub sits below its range at both sites.** The two sites hold different shrub communities —
chenopod at the low site, mixed shrub at the medium — measured from separate LiDAR
acquisitions, and both land far below 0.40 m. Forb and Grass sit comfortably inside their
ranges, so this is not a systematic `Z_norm` offset affecting every class. That points at the
`Shrub` range inherited from `fvc3dstruct.height_stats.DEFAULT_HEIGHT_RANGES` rather than at
the delineation; revise the range before treating 80–90% of shrub polygons as mislabelled.

**Check the LiDAR belongs to the site before trusting these columns.** DD0011's LiDAR directory
held a byte-for-byte copy of DD0001's tiles until it was restaged, and `label_qa.py` reported
`missing_las` for all 15 tiles rather than wrong heights only because the two sites' tile IDs
happen not to overlap (DD0001 is 16–205, DD0011 is 25–79). Section 9.5 of the notebook
re-checks the match on every run.

The mix-up was helped along by the data directory being named `site1_1_DD0011`, giving both
sites a `site1_1_*` prefix so that anything globbing it swept up both. It has been renamed to
`site2_1_DD0011` to match phases 2, 4 and 5.

## Dependencies

`geopandas`, `rasterio`, `shapely`, `numpy`, `scipy` (for `compare_reannotation.py`), and
`laspy` for the height columns only. All present in the `fvc_structure` environment.
