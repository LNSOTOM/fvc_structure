# Step 3: Quality assessment of the 2D reference labels

The 2D structure labels built in `step_2_structure_2D_labels` are the reference for every
downstream experiment. This step quantifies their quality so the number can be reported
rather than assumed.

| Script | Needs | Answers |
|---|---|---|
| `label_qa.py` | only the delivered artefacts | How internally consistent, how faithfully preserved, how well registered to the imagery, and how exposed to boundary placement are the labels? |
| `make_label_qa_table.py` | the summary CSVs | the same answer as two LaTeX tables, for the manuscript |
| `plot_centroids.py` | the delivered artefacts | the centroid figure: what each grid does to a feature's shape and to its centre |

`label_qa.py` needs nothing but the delivered files. **No repeat annotation has been made and none is claimed**; `label_qa.py`'s
`Delineation IoU` columns are what stands in for one, bounding how much a different outline
could change without anyone drawing one.
[`notebooks/01_label_quality_assessment.ipynb`](notebooks/01_label_quality_assessment.ipynb)
runs both sites and renders the answer: a QA table following the four checks and a per-class
IoU table, both as **mean / median**, a summary paragraph generated from the CSVs, the centroid
figure for each site with a guide to reading it, and a measurement of how far the label and
predictor footprints diverge. No number in it is typed by hand. Everything below explains what
those numbers mean and what remains unquantified.

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

The results are assembled from those two CSVs in
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

Four checks, following a label from the drawn polygon to the array a model reads:

```
Shapefiles
    │
    ├── 1. Annotation QA
    │     ├── invalid geometry
    │     ├── cross-class overlap
    │     └── below MMU
    │
    ├── 2. Polygon → 1 cm raster
    │     └── class IoU
    │
    ├── 3. 1 cm → 5 cm label grid
    │     ├── class IoU
    │     ├── polygon IoU
    │     └── centroid RMSE
    │
    ├── 4. Label grid → predictor grid
    │     ├── class IoU
    │     ├── polygon IoU
    │     ├── centroid RMSE
    │     └── grid offset
    │
    └── 5. ±5 cm boundary perturbation
          └── delineation IoU
```

| Check | Columns | Measures |
|---|---|---|
| 1. Annotation QA | `Invalid geometries`, `Cross-class overlap (%)`, `Below MMU`, `Below MMU (%)` | digitising artefacts. Cross-class overlap is the share of labelled area under two polygons of different classes — pixels with no single reference value, resolved by shapefile row order. The MMU defaults to four 5 cm cells (`--mmu`) |
| 2. Polygon → 1 cm | `IoU 1 cm <class>` | polygons rasterised exactly (centroid rule) against the delivered 1 cm raster: the cost of `all_touched=True` plus resolving overlaps by draw order |
| 3. 1 cm → 5 cm | `IoU 5 cm <class>`, `Polygon IoU mean/median 5 cm`, `Centroid RMSE 5 cm (m)` | the majority class of each 5 cm cell's 1 cm sub-pixels against the delivered 5 cm raster; per feature, the 5 cm footprint against the 1 cm footprint |
| 4. Label → predictor | `IoU predictor <class>`, `Polygon IoU mean/median predictor`, `Centroid RMSE predictor (m)`, `Predictor grid offset (cm)`, `Predictor GSD (cm)` | the 1 cm labels put on the predictor grid by majority against the delivered 5 cm labels read there by nearest neighbour, the way a dataloader reads them; per feature, the predictor footprint against the 5 cm label footprint, scored on the 5 cm grid; the origin separation modulo one label cell. GSD is carried alongside the offset because an offset only means "interlock" when the cell sizes match |
| 5. Boundary perturbation | `Delineation IoU <class>` | each class eroded and dilated by `--boundary` (default 0.05 m); the IoU of the two is the agreement two delineations disagreeing by that much *everywhere* would still reach. A floor, not an estimate |

Nodata is counted as a category in the majority rule. Taking the majority over labelled
sub-pixels only would call a cell "woody debris" on the strength of a single 1 cm pixel in
twenty-five, which no downsampling rule would deliver, and would depress the IoU of every
sparse class for no reason.

**Per feature, after [Haynes et al. (2025)](https://www.mdpi.com/2504-446X/9/2/132).** Class
IoU is pooled over a whole class, so it cannot distinguish a class that shifted from one that
was eroded. Each polygon is rasterised in its own window on each grid, and its footprint is
compared with its footprint on the grid before it: centroid RMSE (their equation 2) says how
far it moved, polygon IoU how much of it survived. Haynes et al. compare two *independent
delineations*; there is one here, so these columns bound the cost of the **grids**, not of the
annotator — the same restriction that applies to `Delineation IoU`.

**Read the mean against the median.** Every IoU is computed per tile and per class, and the
CSV closes with two across-tile rows, `Average` and `Median`. The notebook reports the pair
as `mean / median`, pooled over tiles and classes. They separate when a *few* tiles or a
*single* class fail on their own — which is what happens at both sites. At the low site,
Woody debris alone collapses to 0.014 at the polygon → 1 cm step while every other class
clears 0.95; a mean quoted without its median would read as labels that are mediocre
everywhere, which is the wrong diagnosis and the wrong repair.

**The two sites fail the predictor check differently.** DD0001's predictors are 5 cm like its
labels, and the two grids interlock on **13 of 15 tiles**; the 2 cm offset appears on tiles 26
and 109 only. DD0011's predictors are **6 cm** against 5 cm labels — same origin, but the two
tensors cannot be indexed together at all without resampling one of them again.

LiDAR height is no longer part of the assessment: the scheme classifies growth form, which
implies no stature, so a height comparison could describe the classes but not grade the labels.

### Issue types in `<site>_qa_issues.csv`

`invalid_geometry`, `cross_class_overlap`, `below_mmu`, `raster_disagrees_with_shapefile`,
`feature_displacement`, `predictor_gsd_differs_from_labels`, `predictor_grid_offset`,
`predictor_grid_disagreement`, `boundary_dominated_class`,
`missing_raster`, `missing_downsampled`, `missing_imagery`.

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
correct and `IoU 1 cm Woody debris` collapses to **0.014** — that is the shapefiles and the
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
| **1. Annotation QA** | | |
| Invalid geometries per tile | 7.6 | 0.8 |
| Cross-class overlap | 0.021% of labelled area | 0.161% of labelled area |
| Polygons below MMU | 1.38% | 0.68% |
| **2. Polygon → 1 cm** | | |
| Class IoU | 0.924 / 0.934 | 0.679 / 0.957, **Woody debris 0.014** |
| **3. 1 cm → 5 cm** | | |
| Class IoU | 0.806 / 0.804 | 0.996 / 0.997 |
| Polygon IoU | 0.809 / 0.803 | 0.878 / 0.899 |
| Centroid RMSE | 0.012 m | 0.007 m |
| **4. Label → predictor grid** | | |
| Class IoU | 0.784 / 0.782 | 0.930 / 0.941 |
| Polygon IoU | 0.972 / 1.000 | 0.859 / 0.892 |
| Centroid RMSE | 0.002 m | 0.012 m |
| Grid offset | 2 cm on 2 of 15 tiles | 0 cm, but 6 cm pixels on 5 cm labels |
| **5. ±5 cm boundary perturbation** | | |
| Delineation IoU | 0.321 / 0.254 | 0.508 / 0.547 |

Class IoU figures are `mean / median` pooled over tiles and classes; polygon IoU is the mean of
per-tile means / median of per-tile medians, and centroid RMSE is averaged across tiles. Read the two halves of each pair together —
the low site's vector → 1 cm mean of 0.679 against a median of 0.957 is one class failing
alone, not labels that are poor throughout.

Both preservation problems are site-specific and independent: the medium site's 5 cm labels
are dilated, and the low site's shapefiles disagree with its rasters. Neither shows up in a
count of invalid geometries, which is why the IoU columns exist. The per-feature columns then
show that neither problem is a *displacement* — centroids hold to within a quarter of a cell at
every step at both sites, so what the grids cost is feature shape, not feature position.

The low site's data directory was once named `site1_1_DD0011`, giving both sites a `site1_1_*`
prefix so that anything globbing it swept up both. It has been renamed to `site2_1_DD0011` to
match phases 2, 4 and 5.

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
| `tab:label_quality` | the assessment itself: one row per metric, `Mean` and `Median` columns under each site, ruled into the checks: annotation QA, polygon → 1 cm, 1 cm → 5 cm, label → predictor grid, and the boundary perturbation |
| `tab:label_quality_class` | per-class IoU at each step, which is where a class that fails alone becomes visible |

Only quantities that genuinely have both get two cells; a count or a share spans the pair.
Tile counts live in the caption, not in a row.

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
the third block of the table is about that one step. `Class IoU, 1 cm → 5 cm` is the class-level
cost, `Polygon IoU, 1 cm → 5 cm` the per-feature cost, and `Centroid RMSE, 1 cm → 5 cm` how far a
feature's centre moves. The medium site drops to 0.806 class IoU because 13 of its 15 tiles land
off the 1 cm grid; the low site loses essentially nothing (0.996).

The label → predictor comparison is scored on the 5 cm grid for the same reason. A disagreement finer than one 5 cm
cell is not something a model or the 3D pipeline can express, so measuring it on the 1 cm grid
would report a precision nothing acts on. That choice moves the numbers: the label-vs-predictor
comparison scores 0.788 on the 5 cm grid against 0.748 on the 1 cm grid for medium tile 26.

## `plot_centroids.py`

The per-feature columns say centroids barely move while per-feature IoU falls. That is a claim
about geometry, and it is easier to check by eye than to take on trust:

```bash
# medium site
python plot_centroids.py --site site1_1_DD0001 --tile 26 \
    --shp-dir ... --raster-dir ... --downsampled-dir ... --image-dir ... --rgb-dir ... \
    --output label_centroids
# low site: predictors are 6 cm, not 5 cm
python plot_centroids.py --site site2_1_DD0011 --tile 25 ... --output label_centroids_low
```

Writes `<output>.png`, `.pdf` and `.tex` (a figure environment with a generated caption). The
notebook redraws both when `REDRAW = True` and follows them with a *How to read the figures*
section.

### How to read the figure

**Layout.** One column per growth form present in the tile, two rows:

| Row | Label | Backdrop |
|---|---|---|
| top | `RGB imagery, 1 cm` | the 1 cm RGB tile (`--rgb-dir`) |
| bottom | `Multispectral predictor, 5 cm` (6 cm at the low site) | the predictor stack (`--image-dir`), as a false-colour composite of `--rgb-bands 5,3,1`, near-infrared in red, so vegetation appears red. The band triple is stated in the caption rather than on the image |

**Column headings**, e.g. `Woody debris / 0.25 m² / n = 42, median 0.09 m²`: the class, the area
of the example polygon drawn, and the number and median area of that class's polygons in the
tile. The example is the polygon nearest the `--quantile` (default 0.95) of class area whose
window fits inside the tile — large enough to read at 5 cm, and deliberately not typical, which
is why the median is printed beside it.

**Legend**, at the bottom:

| Entry | Drawn as | What it is |
|---|---|---|
| Reference polygon | translucent white **fill** | the shapefile geometry, exactly as digitised |
| Reference centroid | ○ | its centre |
| 5 cm label footprint | solid amber **outline** | the cells the 5 cm label raster assigns to the feature |
| Label centroid | × | their centre |
| Predictor footprint | cyan **hatch** with a dashed edge | the pixels the predictor raster assigns to the feature |
| Predictor centroid | + | their centre |

**Centroid shift (mm)**, under each column, compares each raster with the one before it — the
same step-to-step comparison as `label_qa.py`'s centroid RMSE columns:

| Line | Meaning |
|---|---|
| `label 1 cm → label 5 cm` | centre of the feature's 1 cm footprint to centre of its 5 cm label footprint: what downsampling moves |
| `label 5 cm → predictor` | centre of the 5 cm label footprint to centre of the predictor footprint: how far the label and the pixels it is paired with disagree |

The 1 cm footprint is not drawn — at this zoom it is indistinguishable from the polygon — so the
first shift is effectively ○ to ×, and the second × to +. The figure shows one polygon per class;
the tables summarise every polygon on every tile.

### Design choices

The three overlays use different *kinds* of mark, not three colours of the same mark. The grid
footprints hug the polygon, so as three plain outlines they sat on the same path and whichever
drew last hid the rest. A fill gives the drawn geometry an area for the staircases to read
against, and hatching the predictor keeps it legible *through* the amber outline. Hatch density
is deliberately sparse (`//` at 1.1 pt) so the feature stays visible underneath. Every overlay
carries a dark stroke, because no single line colour reads on both pale soil and red vegetation.

Layout: headings and shift labels are split over several short lines, and the shift text is set
at 90% of the 16 pt base, so neighbouring columns cannot collide on a four-class figure; the row
gap is set explicitly, since `tight_layout`'s `h_pad` is lost on wide figures; and the legend
hangs just below the lowest shift label rather than at a fixed height. `--width-per-column`
(default 0.25) is written into the generated `\includegraphics`, so a four-class figure comes out
at `\textwidth` and a two-class one at `0.5\textwidth`, with panels the same size on the page.

**`--exclude-structure` defaults to `unknown`, and must stay that way for the low site.** Its
shapefiles retain one `unknown` MultiPolygon per tile carrying `struct_id` 1 — on tile 25 that
is **127.09 m², 13.5% of the tile, in 1875 parts**, and it is the *only* class-1 polygon there.
Drawn as a growth form it fills the panel with scattered blobs and reports a meaningless
centroid shift, since the centre of 1875 scattered parts is not a feature location. It is the
annotator's bare-earth catch-all, admitted by the `'unknowm'` typo described above. Excluding it
leaves the low site with Forb and Shrub.

### Why the label and predictor footprints coincide on most tiles

The notebook's last table measures it per polygon over all 30 tiles:

| Site | Grids | Tiles | Polygons | Footprint IoU (mean / median) |
|---|---|---|---|---|
| Medium | same origin | 13 | 5,425 | **1.000 / 1.000** |
| Medium | origins offset | 2 | 692 | 0.788 / 0.804 |
| Low | 6 cm pixels on 5 cm cells | 15 | 6,199 | 0.854 / 0.877 |

`label_qa.py` measures this per polygon into `Polygon IoU mean predictor` and `… median`; the
notebook groups those columns rather than recomputing the comparison.

On 13 of 15 medium tiles the two grids share an origin and a 5 cm cell, so the footprints are
**identical** and the `label 5 cm → predictor` shift is zero. A **shifted origin** spares most
tiles entirely and costs a couple a great deal; a **different pixel size** costs every feature on
every tile a little, which is the low site, where 6 cm pixels can never align with 5 cm cells.

**Choose the tile deliberately.** Tiles 26 and 109 are the only medium-site tiles where the
grids are offset, so the footprints separate — that is why tile 26 is the committed example, and
why its caption must say it is not typical. If the divergence reads as too subtle, the lever is
`--zoom` rather than styling: at `--zoom 0.4` individual 5 cm cells are large enough to see the
two staircases part.

## Dependencies

`geopandas`, `rasterio`, `shapely`, `numpy`, `pandas`, `scipy` (the delineation columns),
and `matplotlib` (`plot_centroids.py` only). All present
in the `fvc_structure` environment.
