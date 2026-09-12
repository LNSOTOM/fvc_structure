# Step 3: Quality assessment of the 2D reference labels

The 2D structure labels built in `step_2_structure_2D_labels` are the reference for every
downstream experiment. This step quantifies their quality so the number can be reported
rather than assumed.

| Script | Needs | Answers |
|---|---|---|
| `label_qa.py` | only the delivered artefacts | How internally consistent, how faithfully preserved, how well registered to the imagery, and how exposed to boundary placement are the labels? |
| `make_label_qa_table.py` | the summary CSVs | the same answer as two LaTeX tables, for the manuscript |
| `plot_centroids.py` | the delivered artefacts | the centroid figure: what each grid does to a feature's shape and to its centre |

`label_qa.py` needs nothing but the delivered files, and it carries the answer to the
reviewer. **No repeat annotation has been made and none is claimed**; `label_qa.py`'s
`Delineation IoU` columns are what stands in for one, bounding how much a different outline
could change without anyone drawing one.
[`notebooks/01_label_quality_assessment.ipynb`](notebooks/01_label_quality_assessment.ipynb)
runs both sites and renders the answer: four tables — class IoU at each step, the same per
class, tile context, and per-feature displacement — all as **mean / median**, plus a summary
paragraph generated from the CSVs, the centroid figure for each site, and a measurement of how
far the label and predictor footprints actually diverge. No number in it is typed by hand. Everything below explains
what those numbers mean and what remains unquantified.

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
label cell at the medium site but 1.2 of them at the low site. `label_qa.py` measures all of
this from the delivered files rather than restating it.

## `label_qa.py`

Writes two files into `--output-dir`:

| File | Contents |
|---|---|
| `<site>_qa_summary.csv` | one row per tile, plus an `Average` and a `Median` row |
| `label_quality_tables.tex` | the manuscript tables, and the figures when `--figure` is passed |
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

**Read the mean against the median.** Every IoU is computed per tile and per class, and the
CSV closes with two across-tile rows, `Average` and `Median`. The notebook reports the pair
as `mean / median`, pooled over tiles and classes. They agree when the labels are uniformly
good or uniformly bad, and separate when a *few* tiles or a *single* class fail on their
own — which is what happens at both sites. Pooled vector-to-raster IoU at the low site drops
its mean far below its median because Woody debris alone collapses to 0.014 while every
other class clears 0.95; a mean quoted without its median would read as labels that are
mediocre everywhere, which is the wrong diagnosis and the wrong repair. The per-class table
in the notebook is there to name the class responsible whenever the two diverge.

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

The two sites fail this differently. DD0001's predictors are 5 cm like its labels, and the
two grids interlock on **13 of 15 tiles**; the 2 cm offset appears on tiles 26 and 109 only.
Those are exactly the two tiles whose 5 cm label grid carries **no** offset from the 1 cm
grid, and the other thirteen are exactly those whose label grid does — the imagery followed
the label grid's drift on the thirteen and did not on the two, so the two offsets are
complementary rather than independent. Treat them as one problem with one cause: the
non-integer 614.4 snap described below. DD0011's predictors are **6 cm** against 5 cm labels
— same origin, but 1.2 label cells per pixel, so the two tensors cannot be indexed together
at all without resampling one of them again.

**Per feature: where it went, and how much survived.** Every IoU above is pooled over a whole
class, which cannot distinguish a class that shifted from a class that was eroded — both read
as the same lost overlap. [Haynes et al. (2025)](https://www.mdpi.com/2504-446X/9/2/132),
co-registering UAS pushbroom imaging spectroscopy to RGB, separate the two by digitising the
same features in each image and reporting **both** the RMSE and MAE of the Euclidean distance
between feature centroids (their equations 2 and 3) and the **mean and median IoU** of the
features themselves (their Table S5). `assess_feature_displacement()` measures the same pair
here, per polygon, against each grid the pipeline writes the label to:

| Column | Measures |
|---|---|
| `Centroid RMSE <stage> (m)`, `Centroid MAE <stage> (m)` | how far the feature's centre moved, in metres. RMSE is outlier-sensitive and MAE is not, so a gap between them means a few polygons moved far rather than all of them moving a little |
| `Polygon IoU mean <stage>`, `Polygon IoU median <stage>` | how much of the individual feature survived, mean and median over polygons as in Table S5 |

`<stage>` is `1 cm` (the delivered raster's `all_touched=True` convention), `5 cm` (the label
grid the models consume) and `imagery` (the predictor grid). Each polygon is rasterised in its
own window, so a feature keeps its identity where polygons overlap — the low site averages 148
overlapping pairs per tile, and a whole-tile rasterisation would lose every one of them to
draw order.

**One difference from Haynes et al. must be stated.** They compare two *independent
delineations* of the same feature, so their numbers include the annotator. There is one
delineation here, so what is compared is that delineation against itself after each grid has
quantised it. These columns bound the positional cost of the **grids**, not of the annotator —
the same restriction that applies to `Delineation IoU`.

What they show is that the two failures are not the same failure. Centroids barely move
(RMSE 0.016 m and 0.010 m on the predictor grid, inside a third of one cell) while per-feature
IoU falls to 0.809 and 0.858. The labels are in the right *place* and the wrong *shape*:
features are eroded and dilated by quantisation, not shifted off their ground. That decides
the repair, because a shift is correctable by a warp and a shape change is not.

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
`boundary_dominated_class`, `feature_displacement`, `too_few_lidar_points`,
`grid_misaligned_5cm`,
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
| Class IoU, vector → 1 cm | 0.924 / 0.934 | 0.679 / 0.957, **Woody debris 0.014** |
| Class IoU, 1 cm → 5 cm | 0.806 / 0.804 | 0.996 / 0.997 |
| Class IoU, label → predictor grid | 0.784 / 0.782 | 0.930 / 0.941 |
| Delineation IoU (±5 cm floor) | 0.321 / 0.254 | 0.508 / 0.547 |
| Centroid RMSE / MAE, predictor grid | 0.016 / 0.009 m | 0.010 / 0.007 m |
| Polygon IoU, predictor grid | 0.809 / 0.808 | 0.858 / 0.881 |
| 5 cm grid offset | 1–4 cm on 13 of 15 tiles | 0 cm on all tiles |
| Imagery grid offset | 2 cm on 2 of 15 tiles | 0 cm, but 6 cm pixels on 5 cm labels |

Class IoU figures are `mean / median` pooled over tiles and classes; centroid and polygon
figures are per feature, averaged across tiles. Read the two halves of each pair together —
the low site's vector → 1 cm mean of 0.679 against a median of 0.957 is one class failing
alone, not labels that are poor throughout.

Both preservation problems are site-specific and independent: the medium site's 5 cm labels
are dilated, and the low site's shapefiles disagree with its rasters. Neither shows up in a
count of invalid geometries, which is why the IoU columns exist. The per-feature columns then
show that neither problem is a *displacement* — centroids hold to within a third of a cell at
both sites, so what the grids cost is feature shape, not feature position.

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
happen not to overlap (DD0001 is 16–205, DD0011 is 25–79). Check the tile IDs resolve to the
site's own LiDAR before reading the height columns.

The mix-up was helped along by the data directory being named `site1_1_DD0011`, giving both
sites a `site1_1_*` prefix so that anything globbing it swept up both. It has been renamed to
`site2_1_DD0011` to match phases 2, 4 and 5.

## `make_label_qa_table.py`

The manuscript tables, generated from the same CSVs the notebook reads so that no number is
transcribed by hand:

```bash
python make_label_qa_table.py \
    --summary /path/site1_1_DD0001/.../site1_1_DD0001_qa_summary.csv \
    --summary /path/site2_1_DD0011/.../site2_1_DD0011_qa_summary.csv \
    --output  label_quality_tables.tex
```

`--output -` prints to stdout instead. `--summary` is repeated once per site and sets the
column order. The committed [`label_quality_tables.tex`](label_quality_tables.tex) is the
output of that command on the current delivered files, and holds two tables:

| Table | Contents |
|---|---|
| `tab:label_quality` | the assessment itself: one row per metric, `Mean` and `Median` columns under each site, ruled into annotation, class IoU, per-feature geometry, and the two bounds |
| `tab:label_quality_class` | per-class IoU at each step, which is where a class that fails alone becomes visible |

Only quantities that genuinely have both get two cells. A count or a share spans the pair, and
a quantity that is only ever a median (polygon area) leaves the mean cell as `--` rather than
printing one number twice under two headings. Tile counts live in the caption, not in a row.

Both need `\usepackage{booktabs}`. The second table exists because the first cannot show
the low site's Woody debris collapsing to 0.014 — pooled with the other classes it reads only
as a mean below its median, and the reader needs to see which class is responsible.

Pass `--figure` once per figure `.tex` written by `plot_centroids.py` to carry them into the
same file, so tables and figures arrive together and every caption still has one author:

```bash
python make_label_qa_table.py --summary ... --summary ... \
    --figure label_centroids.tex --figure label_centroids_low.tex \
    --output label_quality_tables.tex
```

That output needs `\usepackage{booktabs}` and `\usepackage{graphicx}`. Regenerate after any
re-run of `label_qa.py`; the tables carry no numbers of their own.

### Error propagation from 1 cm to 5 cm

**5 cm is the working resolution** — it is what the models read and what the 3D work uses — so
the second block of the table is about that one step. `Class IoU, 1 cm → 5 cm` is the class-level
cost, `Polygon IoU, polygon → 5 cm` the per-feature cost, `Centroid RMSE at 5 cm` how far a
feature's centre moves, and `Labelled area change at 5 cm (%)` how much area the downsample
invents. The medium site pays **+10.5% labelled area** and drops to 0.806 class IoU because
13 of its 15 tiles land off the 1 cm grid; the low site loses essentially nothing (0.996).

Everything is scored on the 5 cm grid for the same reason. A disagreement finer than one 5 cm
cell is not something a model or the 3D pipeline can express, so measuring it on the 1 cm grid
would report a precision nothing acts on. That choice moves the numbers: the label-vs-predictor
comparison scores 0.788 on the 5 cm grid against 0.748 on the 1 cm grid for medium tile 26.

## `plot_centroids.py`

The per-feature columns say centroids barely move while per-feature IoU falls. That is a claim
about geometry, and it is easier to check by eye than to take on trust:

```bash
python plot_centroids.py --site site1_1_DD0001 --tile 26 \
    --shp-dir ... --raster-dir ... --downsampled-dir ... --image-dir ... \
    --output label_centroids
```

Writes `<output>.png`, `.pdf` and `.tex`. One column per growth form present in the tile, and
two rows: the **1 cm RGB the polygons were digitised on** (`--rgb-dir`) over the **predictor
stack the models read** (`--image-dir`), so the figure is the annotator's view above the
model's view of the same feature. Each panel carries three outlines and their centroids:

| Overlay | Drawn as | What it is |
|---|---|---|
| drawn polygon | translucent white **fill** | the shapefile geometry, exactly as digitised |
| label cells covering it | solid amber **outline** | the cells the **5 cm label raster** assigns to that feature — the reference a model is trained against |
| predictor pixels covering it | cyan **hatch** with a dashed edge | the pixels the **predictor stack** assigns to it — the image pixels a model actually reads for that feature |

The three use different *kinds* of mark, not three colours of the same mark, and that is the
whole point. The grid footprints hug the polygon, so as three plain outlines they sat on the
same path and whichever drew last hid the rest — in particular the label and predictor
outlines coincide wherever the grids interlock, which is most of the medium site, so one of
them vanished entirely. A fill gives the drawn geometry an area for the staircases to read
against, and hatching the predictor makes it legible *through* the amber outline rather than
on top of it. Hatch density is deliberately sparse (`//` at 1.1 pt) so the feature stays
visible underneath; `////` hides it.

The last two are the same feature seen through the two grids that have to agree for a model to
pair a label with a pixel. Where those grids interlock the two outlines coincide; where they do
not, the amber and cyan outlines separate and the gap is the registration error, per feature.
The centroid shift printed under each column is how far the feature's centre moves between the
drawn geometry and each grid, in millimetres.

Type is 16 pt and the shift is printed once per column, under the bottom row, since it depends
on the feature and the grids rather than on which backdrop is shown. `--width-per-column`
(default 0.25) is written into the generated `\includegraphics`, so a four-class figure comes
out at `\textwidth` and a two-class one at `0.5\textwidth` and the panels are the same size on
the page in both.

The multispectral row defaults to `--rgb-bands 5,3,1`, a false-colour composite with
near-infrared in red, which separates vegetation from soil far more strongly than true colour;
the band triple is printed on the row label and stated in the caption so the figure is never
silently false-colour. Overlays are drawn white/amber/cyan with a dark stroke, because no
single line colour reads on both pale soil and vegetation glowing red at 5,3,1.

```bash
# medium site
./plot_centroids.py --site site1_1_DD0001 --tile 26 ... --output label_centroids
# low site: predictors sit directly under tiles_3072, and are 6 cm not 5 cm
./plot_centroids.py --site site2_1_DD0011 --tile 25 ... --output label_centroids_low
```

**`--exclude-structure` defaults to `unknown`, and must stay that way for the low site.** Its
shapefiles retain one `unknown` MultiPolygon per tile carrying `struct_id` 1 — on tile 25 that
is **127.09 m², 13.5% of the tile, in 1875 parts**, and it is the *only* class-1 polygon there.
Drawn as a growth form it fills the panel with scattered blobs and reports a meaningless
180 mm centroid "shift", since the centre of 1875 scattered parts is not a feature location.
It is the annotator's bare-earth catch-all, admitted by the `'unknowm'` typo described above.
Excluding it leaves the low site with Forb and Shrub, which is the honest set.

### Why the label and predictor outlines coincide

The notebook's last section measures it rather than arguing it, per polygon over all 30 tiles:

| Site | Grids | Tiles | Polygons | Footprint IoU (mean / median) | Identical |
|---|---|---|---|---|---|
| Medium | same origin | 13 | 5,311 | **1.000 / 1.000** | 99% |
| Medium | origins offset | 2 | 692 | 0.788 / 0.804 | 0% |
| Low | 6 cm pixels on 5 cm cells | 15 | 6,187 | 0.854 / 0.877 | 1% |

`label_qa.py` measures this per polygon into `Label vs predictor IoU mean`, `… median` and
`Label vs predictor identical (%)`; the notebook groups those columns rather than recomputing
the comparison, so there is only one implementation to keep right.

On 13 of 15 medium tiles the two grids share an origin and the footprints are **identical** —
the amber and cyan outlines are the same outline, and no styling can separate what is not
distinct. The two offset tiles disagree on about a fifth of their combined area, yet still look
alike: a 2 cm shift on a 5 cm grid changes which cells are picked without moving the silhouette,
and the eye reads the silhouette while the IoU reads the cells.

The two grid failures are reported separately because a pooled mean would hide the difference
between them. A **shifted origin** spares most tiles entirely and ruins a couple. A **different
pixel size** costs every feature on every tile a little, which is the low site: 6 cm pixels can
never align with 5 cm cells, so only 1% of 6,174 polygons come out identical.

**Choose the tile deliberately.** On the 13 medium-site tiles where the label and predictor
grids interlock, the two footprints coincide and the figure shows only the 5 cm quantisation.
Tiles 26 and 109 are the two where the grids are offset, so the footprints separate — that is
why tile 26 is the committed example. If the divergence still reads as too subtle, the honest
lever is `--zoom` rather than styling: at `--zoom 0.4` individual 5 cm cells are large enough
to see the two staircases part. The generated caption detects which case it is drawing
and says so, rather than letting an offset tile stand in for the general case.

## Dependencies

`geopandas`, `rasterio`, `shapely`, `numpy`, `pandas`, `scipy` (the delineation columns),
`matplotlib` (`plot_centroids.py` only), and `laspy` for the height columns only. All present
in the `fvc_structure` environment.
