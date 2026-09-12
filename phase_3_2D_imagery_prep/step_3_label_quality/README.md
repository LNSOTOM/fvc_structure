# Step 3: Quality assessment of the 2D reference labels

The 2D structure labels built in `step_2_structure_2D_labels` are the reference for every
downstream experiment. This step quantifies their quality so the number can be reported
rather than assumed.

| Script | Needs | Answers |
|---|---|---|
| `label_qa.py` | only the delivered artefacts | How internally consistent, how faithfully preserved, how well registered to the imagery, and how exposed to boundary placement are the labels? |

`label_qa.py` needs nothing but the delivered files, and it carries the answer to the
reviewer. **No repeat annotation has been made and none is claimed**; `label_qa.py`'s
`Delineation IoU` columns are what stands in for one, bounding how much a different outline
could change without anyone drawing one.
`notebooks/01_label_quality_assessment.ipynb` drives the whole thing and explains each
number, and its section 11 states what remains unquantified.

## The tiling scheme the labels live on

Each site's orthomosaic was cut by a fixed grid of **30.72 × 30.72 m** cells
(`grid_annotation_3072.shp` in step 1), and the 15 cells per site flagged `annotated = 1`
were carried forward — a scattered subset of the grid, not a contiguous block. The grid is a
**partition, not a moving window**: tile origins are exactly one tile apart and no two tiles
share a pixel, so nothing is labelled twice and no pair of tiles can leak between a training
and a test split through an overlap. Labels are delivered at **1 cm**, 3072 × 3072 px per
tile, in EPSG:7854, and downsampled to **5 cm** (614 × 614) for the models — 3072 × 0.01 /
0.05 = 614.4, the non-integer behind [the 5 cm grid offset](#the-5-cm-grid-offset). The
predictor imagery is a five-band stack at each orthomosaic's native ground sample distance,
and that distance differs between the sites: **5 cm** at the medium site (614 × 614 px)
against **6 cm** at the low site (512 × 512 px), so one image pixel covers exactly one 5 cm
label cell at the medium site but 1.2 of them at the low site. Section 10 of the notebook
measures all of this from the delivered files rather than restating it.

## `label_qa.py`

Writes two files into `--output-dir`:

| File | Contents |
|---|---|
| `<site>_qa_summary.csv` | one row per tile, plus an Average row |
| `<site>_qa_issues.csv` | one row per individual problem, so a flagged polygon can be opened in QGIS |

The answer to the reviewer is assembled from those two CSVs in
[`notebooks/01_label_quality_assessment.ipynb`](notebooks/01_label_quality_assessment.ipynb),
which is the deliverable; the script deliberately does not generate a second prose copy of the
same numbers.

```bash
python label_qa.py \
    --site site1_1_DD0001 --site-condition Medium \
    --shp-dir         /path/.../structure_class/annotations_shp \
    --raster-dir      /path/.../structure_class/annotation_raster \
    --downsampled-dir /path/.../annotation_raster/downsample_5cm_nearest \
    --image-dir       /path/.../predictors/tiles_3072/multispec/res_05/stacked \
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
`annotations_shp` for DD0001 and `annotation_shp` for DD0011, and the predictor stack sits
under `predictors/tiles_3072/multispec/res_05/stacked` at DD0001 but directly under
`predictors/tiles_3072/stacked` at DD0011 — `run_label_qa.sh` takes whichever exists.

**Class coding.** `2_2_structure_2D_rasterLabels.py` writes the 1 cm rasters as `struct_id`
(0–4, nodata 255), then `2_3_downsampling_labels.py` remaps the 5 cm rasters through
`{0:64, 1:65, 2:66, 3:67, 4:68}` to LAS classification codes. Every label raster is read here
through `remap_class_codes()`, which detects the 64–68 codes and reverses them, so either
coding can be passed to either flag.

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

**Registration against the imagery.** A model reads a label and a predictor pixel at the same
row and column, so the two grids have to interlock. They are written by different steps — the
labels by `2_3_downsampling_labels.py`, the predictors by the stacking step — and nothing
between here and training checks that the results agree. A label set can be flawless as
annotation and still be the wrong reference if it is read against pixels it does not cover.

| Column | Measures |
|---|---|
| `Imagery GSD (cm)` | resolution of the predictor stack the models actually read |
| `Label cells per imagery pixel` | the GSD ratio. 1.0 means one label per pixel; anything else means the label tensor is resampled again before training, and that resampling is not measured here |
| `Imagery grid offset (cm)` | separation of the two origins modulo one label cell, so 0 means the grids interlock and half a cell is the worst case |
| `IoU imagery grid <class>` | the 1 cm labels put on the imagery grid correctly, by majority of the sub-pixels each pixel covers, against the same labels put there the way a dataloader does it, by nearest neighbour on the delivered 5 cm raster. 1.000 means the grid can be taken at face value |

The two sites fail this differently. DD0001's predictors are 5 cm like its labels but their
origin sits 2 cm off the label grid, so every label is read against a pixel it only partly
covers. DD0011's predictors are **6 cm** against 5 cm labels — same origin, but 1.2 label
cells per pixel, so the two tensors cannot be indexed together at all without resampling one
of them again.

**Delineation sensitivity.** Whether a second annotator would draw the same outline cannot
be answered without a second annotator. How much it would matter can. Each class is eroded
and dilated by `--boundary` metres (default 0.05, one cell at the resolution the models
consume) and the IoU between the two is reported as `Delineation IoU <class>` — the agreement
two delineations disagreeing by that much *everywhere* would still reach. It is a floor, not
an estimate: a class scoring high cannot be moved much by where the line went, whoever drew
it. `Edge share <class> (%)` is the same quantity as a proportion of the class, and
`Labelled area near a class boundary (%)` counts only the cells within reach of a *different*
class, i.e. where a shifted line changes which class a pixel reports rather than only whether
it is labelled at all.

**Independence.** Per polygon, the P95 of `Z_norm` among the LiDAR returns inside it — its
top-of-canopy height. LiDAR height was never used to make the 2D labels, so how far the
classes separate on it is evidence they track something measurable rather than annotator
habit. `Height AUC <class>` is the probability that a random polygon of the class stands
taller than a random polygon of any other class at that site: 0.5 is indistinguishable, and
distance from 0.5 in either direction is separation — below it the class usually stands
shorter.

This describes the classes, it does not grade them. The scheme classifies **growth form** —
woodiness, branching, stem persistence — which implies no stature, so a class near 0.5 is not
mislabelled and **no expected height range is asserted or scored against**. An earlier version
scored each class against `fvc3dstruct.height_stats.DEFAULT_HEIGHT_RANGES`, which belongs to
the phase-4 3D height-strata classification; that comparison and its `Height outliers` columns
have been removed.

The height is deliberately a *per-polygon top-of-canopy* statistic. A 2D mask is a crown
*footprint*, so every return beneath the plant — ground, litter, understorey — falls inside
it; taking the median over the whole column measures the ground, not the plant. Polygons with
fewer than `--min-points` (default 10) returns are excluded, since a P95 over a handful of
points measures where the returns landed.

### Issue types in `<site>_qa_issues.csv`

`invalid_geometry`, `empty_geometry`, `duplicate_geometry`, `small_area`,
`overlap_same_class`, `overlap_different_class`, `unmapped_class`,
`boundary_dominated_class`, `too_few_lidar_points`, `grid_misaligned_5cm`,
`imagery_gsd_differs_from_labels`, `imagery_grid_offset`, `imagery_grid_disagreement`,
`missing_raster`, `missing_downsampled`, `missing_imagery`, `missing_las`.

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

Descriptive. Growth form implies no stature, so there is no expected range and nothing here
is a verdict. `AUC` is the probability a random polygon of the class stands taller than a
random polygon of any other class at that site.

| Class · site | Median P95 (m) |
|---|---|
| Forb · Low | 0.030 |
| Grass · Medium | 0.077 |
| Woody debris · Medium | 0.265 |
| Woody debris · Low | 0.265 |
| Shrub · Medium | 0.111 |
| Shrub · Low | 0.297 |
| Mallee · Medium | 1.418 |

Shrub is short at both sites — chenopod at the low site, mixed shrub at the medium, measured
from separate LiDAR acquisitions. Scored against the phase-4 `Shrub` range of 0.40–3.00 m
that read as 90% and 81% of shrub polygons "outside range", which invites the conclusion that
the shrub labels are unreliable. They are not: the range describes a **height stratification**
and these labels are **growth form**. The comparison has been removed rather than reported
with a caveat. Run the notebook for the current `Height AUC` values, which say the useful
thing instead — whether the classes differ from each other on a measurement the annotator
never saw.

**Check the LiDAR belongs to the site before trusting these columns.** DD0011's LiDAR directory
held a byte-for-byte copy of DD0001's tiles until it was restaged, and `label_qa.py` reported
`missing_las` for all 15 tiles rather than wrong heights only because the two sites' tile IDs
happen not to overlap (DD0001 is 16–205, DD0011 is 25–79). Section 9.5 of the notebook
re-checks the match on every run.

The mix-up was helped along by the data directory being named `site1_1_DD0011`, giving both
sites a `site1_1_*` prefix so that anything globbing it swept up both. It has been renamed to
`site2_1_DD0011` to match phases 2, 4 and 5.

## Dependencies

`geopandas`, `rasterio`, `shapely`, `numpy`, `pandas`, `scipy` (the delineation columns), and
`laspy` for the height columns only. All present in the `fvc_structure` environment.
