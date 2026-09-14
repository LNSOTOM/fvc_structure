# Parameter sensitivity of the 2D → 3D label transfer

This answers three review comments on subsection 3.2.2, all asking how the parameters of the
3D label-fusion pipeline were chosen:

> 1. The choice of 10 cm voxel size appears empirical. Could the authors provide a comparison of
>    different voxel sizes (e.g., 5, 10, 15, 20, 25 cm) and include the results as a
>    supplementary figure?
> 2. Please explain why the 0.30 matching radius was selected.
> 3. It should be explained why various parameters such as voxel size (10 cm), weighting
>    (1, 0.1), confidence threshold, and flip ratio were selected at their current values.

Each parameter was varied on its own across a range while the others stayed at their defaults,
on all 30 tiles (15 per site): 660 pipeline runs. Every run was scored against the 2D labels
it was built from. The tables below are copied from
[`assets/parameter_sensitivity/parameter_sensitivity_tables.md`](assets/parameter_sensitivity/parameter_sensitivity_tables.md),
which `plot_parameter_sensitivity.py` generates from the per-tile CSV. Rerun both scripts
rather than editing a number here.
[`notebooks/04_voxel_size_sensitivity_and_label_confidence.ipynb`](notebooks/04_voxel_size_sensitivity_and_label_confidence.ipynb)
renders the same tables and figures from that CSV, and draws the voxels on a real tile.

## The answer in brief

| Parameter | Default | Stage | What it decides | Effect on the final labels across the range tested |
|---|---|---|---|---|
| Voxel size | 10 cm | 2 | how far a 2D label extends to neighbouring lidar returns | a steady trade-off with no optimum: from 5 to 25 cm the labels lying outside their 2D footprint rise from 3.5% to 32.5% (Medium) and from 5.7% to 36.2% (Low) |
| Matching radius | 0.30 m | 1 | how far a labelled point may look for its height above ground | none: every labelled point finds its height at 0.000 m, and 0.01–1.00 m give identical labels |
| Height-prior weight | 1 / 0.1 | 4 | how strongly a class outside its height range is penalised | none at Medium; at most 0.13% of points on any Low tile |
| `min_conf` | 0.3 | 4 | whether an unlabelled point inside a crown is filled | none at Medium; at most 0.002% on any Low tile |
| `min_flip_ratio` | 1.5 | 4 | whether an existing label is overturned | none at Medium; at most 0.016% on any Low tile |

Voxel size is the only parameter that changes more than 0.13% of the final labels on any
tile. The Stage-1 radius is never reached, and the three Stage-4 parameters act inside mallee
crowns, which Stage 5 then converts wholly to mallee. [Why the Stage-4 parameters barely reach the final labels](#why-the-stage-4-parameters-barely-reach-the-final-labels)
explains that last point; it is also the main thing to settle before replying (see
[Findings to settle](#findings-to-settle-before-responding)).

The same summary, with the largest change each parameter makes on any tile measured rather than
described, is the **answer in brief** table in
[`assets/parameter_sensitivity/parameter_sensitivity_tables.md`](assets/parameter_sensitivity/parameter_sensitivity_tables.md),
and the first table of notebook 04.

## What the pipeline is given

Most of the answers follow from the data rather than from the parameter values.
`parameter_sensitivity.py` measures each of the facts below on every tile rather than
assuming it.

**The labelled cloud is the lidar cloud itself.** `notebooks/maskCoreg*.py` builds the
structure cloud with PDAL `filters.colorization`: each lidar return takes the class of the
5 cm label cell beneath it, and returns on unlabelled cells are dropped. The structure cloud is
therefore a subset of the Z_norm cloud: the same returns at the same coordinates, carrying the
delivered 5 cm labels.

| Check | Medium | Low |
|---|---:|---:|
| Tiles | 15 | 15 |
| Structure points with an exact Z_norm twin, lowest tile (%) | 100.000 | 100.000 |
| Largest XY match distance, any tile (m) | 0.000 | 0.000 |
| Structure points beyond the 0.30 m radius, any tile (%) | 0.000 | 0.000 |
| Heights taken from another return in the same column, range over tiles (%) | 8.3–19.1 | 4.3–10.5 |
| Largest class-range shift that causes, any tile (m) | 0.017 | 0.003 |
| Lidar points on labelled 5 cm cells missing from the structure cloud, all tiles | 362 (0.005%) | 0 (0.000%) |
| Plan-view mIoU of the 5 cm labels themselves (mean / median) | 0.925 / 0.953 | 0.884 / 0.899 |
| 5 cm labels lying outside the 1 cm footprint (mean / median, %) | 8.4 / 9.0 | 2.5 / 2.5 |

The 362 missing points are all on Medium tile 26, almost all of them in the 2.5 cm strip along
the label raster's eastern edge, where the structure cloud stops short. Even the 2D labels themselves fall short of an mIoU of 1 in
plan view. Many 5 cm cells hold no lidar return, so they can never be labelled from the points.
That mIoU is the ceiling the 3D labels are measured against below.

Two consequences run through every section:

- **Stage 1 always finds a height at 0 mm.** Its radius is never reached (section 2).
- **Stage 2 cannot invent a label.** It can only extend a label to returns that share a voxel
  with a labelled return, or settle a voxel holding more than one class (section 1).

## How the sensitivity was measured

| Parameter | Config key | Stage | Default | Values tested |
|---|---|---|---:|---|
| Voxel size | `voxel_size_xy` = `voxel_size_z` | 2 | 0.10 m | 0.05, 0.10, 0.15, 0.20, 0.25 m |
| Matching radius | `height_match_radius` | 1 | 0.30 m | 0.01, 0.05, 0.10, 0.30, 1.00 m |
| Height-prior weight, out of range | `height_prior_out_of_range_weight` | 4 | 0.1 | 1, 0.5, 0.1, 0.01, 0 |
| Confidence threshold | `min_conf` | 4 | 0.3 | 0.2, 0.3, 0.4, 0.5, 0.6, 0.8 |
| Flip ratio | `min_flip_ratio` | 4 | 1.5 | 1, 1.25, 1.5, 2, 3 |

Each run is scored per lidar point against the **5 cm input labels**, the raster the structure
cloud was cut from. Measured against them, the labels the pipeline was handed are perfect by
construction, so any departure is the pipeline's own doing:

| Measure | Meaning |
|---|---|
| Outside footprint (%) | share of labelled points lying on cells the 2D labels leave unlabelled: how far labels have spread past their footprint |
| Footprint recall (%) | share of points on labelled cells that carry their cell's class |
| mIoU vs 5 cm input | plan view: the 3D labels rasterised top-down onto the 5 cm grid, each cell taking the majority class of its labelled points, then IoU per class against the 2D labels, averaged over the classes present |
| Changed from default (%) | share of all lidar points whose label differs from the default run's. Needs no reference |

The per-tile CSV also scores every run against the annotator's **1 cm** raster, with the
same measures. Those columns add the 1 cm → 5 cm downsampling loss that
[phase 3 step 3](../../phase_3_2D_imagery_prep/step_3_label_quality/README.md) measured, which
lowers every score but changes none of the trends below. Tables report `mean / median` across a
site's tiles; figures draw the median with the inter-quartile range shaded.

**What this cannot show.** There is no independent 3D reference for these sites. Every measure
here is either fidelity to the 2D labels or stability against the default. A label extended to
a neighbouring return can be scored as leaving its footprint, but not as right or wrong.

## Why the Stage-4 parameters barely reach the final labels

Stage 3 detects mallee crowns, and Stage 4 revises labels only inside them. Stage 5 then holds
a vote in every crown: if more than half of its points are mallee, all of them become mallee.

| At the defaults | Medium | Low |
|---|---:|---:|
| Tiles with a detected crown | 15 of 15 | 4 of 15 |
| Crowns holding lidar points, put to the Stage-5 vote, all tiles | 619 | 8 |
| … of which more than half mallee, so turned wholly mallee | 619 | 0 |
| Mallee share of the least-mallee crown, any tile (%) | 77.0 | 0.0 |
| Lidar points inside crowns (mean / median, %) | 22.2 / 22.8 | 0.7 / 0.0 |
| Stage-4 labels changed (mean / median, % of points) | 0.551 / 0.556 | 0.051 / 0.000 |
| Stage-4 mean confidence, tiles with crowns (mean) | 0.9985 | 0.9908 |
| Points turned mallee by the vote (mean / median, %) | 0.32 / 0.31 | 0.00 / 0.00 |

At the Medium site every crown reached the vote at least 77% mallee. So every point inside a
crown ends as mallee whatever Stage 4 decided, and no tested setting changed a final label
there.

The Low site has no mallee in its 2D labels. On 4 of its 15 tiles the watershed's height
fallback still finds crowns, 8 in all, and none of them is mallee-majority. Stage-4 decisions
survive there, but they touch few points:

| Parameter | Values tested | Medium: worst tile (%) | Low: worst tile (%) |
|---|---|---:|---:|
| Matching radius (m) | 0.01, 0.05, 0.1, 0.3, 1 | 0.0000 | 0.0000 |
| Out-of-range weight | 1, 0.5, 0.1, 0.01, 0 | 0.0000 | 0.1288 |
| min_conf | 0.2, 0.3, 0.4, 0.5, 0.6, 0.8 | 0.0000 | 0.0018 |
| min_flip_ratio | 1, 1.25, 1.5, 2, 3 | 0.0000 | 0.0163 |

*Share of all lidar points whose final label differs from the default run's, on the worst tile
for any value tested.*

![Sensitivity to the matching radius and the Stage-4 parameters](assets/parameter_sensitivity/supplementary_parameter_sensitivity.png)

**Supplementary figure, draft caption.** Sensitivity of the label-fusion workflow to (a, e) the
Stage-1 height matching radius and to the Stage-4 (b, f) out-of-range height-prior weight,
(c, g) confidence threshold and (d, h) flip ratio. Blue: Medium site; orange: Low site. Grey
lines mark the default values. Top row: share of lidar points whose label Stage 4 changes,
median across 15 tiles with the inter-quartile range shaded. Bottom row: share of points whose
final label differs from the default configuration, on the worst tile. At the Low site crowns
were detected on 4 of 15 tiles, so its Stage-4 median is zero.

## 1. Voxel size, 10 cm (comments 1 and 3)

**What it does.** Stage 2 (`voxel_merge.cartesian_voxel_merge`) bins both clouds into cubes of
side *v* and gives every lidar point the majority class of the labelled points in its cube.
The labelled points are themselves lidar points, so the transfer can do only two things:

- extend a label to unlabelled returns in the same cube, at most one voxel past the edge of
  the 2D footprint;
- settle cubes that hold more than one class, by majority.

![Voxel-size sensitivity](assets/parameter_sensitivity/supplementary_voxel_size_sensitivity.png)

**Supplementary figure, draft caption.** Sensitivity of the 2D-to-3D label transfer to voxel
size at the Medium (blue) and Low (orange) sites. Lines are medians across the 15 tiles of each
site, with the inter-quartile range shaded. The grey line marks the 10 cm default.
(a) Share of lidar points labelled by the Cartesian voxel transfer (Stage 2). (b) Share of those
labels lying on cells the 2D labels leave unlabelled. (c) Share of points on labelled 2D cells
that keep their cell's class. (d) Labelled points per occupied voxel. (e) Plan-view agreement
between the transferred labels and the 2D labels: mean IoU over classes on the 5 cm grid.
(f) Share of lidar points whose final label differs from the 10 cm result.

**What it looks like on a tile.** `plot_voxel_transfer.py` draws the same effect in a 2 m
window of Medium tile 106. Every lidar return is seen from above, shown blue when its label lies
inside its 2D footprint, orange when the transfer spread a label past it, and grey when
unlabelled. The voxel grid and the 2D footprint outline are drawn on top.

![Voxel transfer on tile 106](assets/parameter_sensitivity/voxel_transfer_tile106.png)

As the voxels grow, the orange labels spread past the outline in voxel-sized blocks, and each
sits in a voxel that also holds a labelled return. Read it with two cautions:

- **The grid squares are voxel columns.** A return takes a label only from labelled returns in
  the same voxel, which also means at the same height, so a grey return can share a square with
  blue ones.
- **The window is chosen, not typical.** It holds the most spread labels at 25 cm among windows
  a quarter to three quarters labelled, so its percentages run above the site means below.

**The transfer (Stage 2).**

| Site | Voxel size (cm) | Labelled by transfer (%) | Outside footprint (%) | Footprint recall (%) | mIoU vs 5 cm input | Points per voxel | Single-point voxels (%) | Mixed-class voxels (%) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Medium | 5 | 27.2 / 26.1 | 3.5 / 3.1 | 99.97 / 99.99 | 0.787 / 0.786 | 1.0 / 1.0 | 67.2 / 69.3 | 0.04 / 0.02 |
| Medium | **10** | 30.0 / 28.9 | 12.5 / 11.8 | 99.89 / 99.94 | 0.648 / 0.646 | 2.3 / 2.0 | 33.9 / 35.2 | 0.24 / 0.14 |
| Medium | 15 | 33.0 / 32.2 | 20.6 / 19.9 | 99.80 / 99.87 | 0.556 / 0.556 | 4.4 / 4.0 | 20.4 / 20.7 | 0.59 / 0.39 |
| Medium | 20 | 35.9 / 35.3 | 27.0 / 26.9 | 99.69 / 99.80 | 0.495 / 0.480 | 7.6 / 7.0 | 14.7 / 15.1 | 1.04 / 0.78 |
| Medium | 25 | 38.8 / 38.5 | 32.5 / 32.3 | 99.57 / 99.67 | 0.446 / 0.433 | 11.6 / 11.0 | 11.6 / 11.7 | 1.64 / 1.29 |
| Low | 5 | 24.4 / 23.5 | 5.7 / 5.6 | 99.88 / 99.90 | 0.787 / 0.808 | 1.5 / 2.0 | 48.3 / 46.7 | 0.20 / 0.17 |
| Low | **10** | 27.4 / 26.4 | 16.0 / 15.9 | 99.68 / 99.68 | 0.692 / 0.708 | 5.0 / 5.0 | 17.6 / 16.8 | 0.92 / 0.84 |
| Low | 15 | 30.4 / 28.8 | 24.2 / 23.8 | 99.45 / 99.44 | 0.624 / 0.633 | 10.5 / 10.0 | 11.0 / 11.3 | 1.85 / 1.82 |
| Low | 20 | 33.3 / 31.2 | 30.8 / 31.0 | 99.21 / 99.14 | 0.572 / 0.572 | 17.2 / 16.0 | 8.4 / 8.3 | 2.80 / 2.74 |
| Low | 25 | 36.1 / 33.4 | 36.2 / 36.0 | 98.92 / 98.91 | 0.526 / 0.534 | 25.6 / 22.0 | 7.1 / 6.9 | 3.81 / 3.79 |

**The final labels (after Stages 3–5).**

| Site | Voxel size (cm) | Labelled (%) | Outside footprint (%) | mIoU vs 5 cm input | mIoU vs 1 cm | Changed from 10 cm (%) |
|---|---:|---:|---:|---:|---:|---:|
| Medium | 5 | 28.8 / 28.2 | 14.4 / 15.3 | 0.551 / 0.539 | 0.460 / 0.466 | 1.44 / 1.34 |
| Medium | **10** | 30.3 / 29.9 | 18.4 / 18.3 | 0.472 / 0.478 | 0.402 / 0.400 | 0.00 / 0.00 |
| Medium | 15 | 32.2 / 32.0 | 23.4 / 23.6 | 0.413 / 0.412 | 0.357 / 0.359 | 2.75 / 2.68 |
| Medium | 20 | 34.2 / 34.2 | 27.9 / 27.9 | 0.374 / 0.380 | 0.325 / 0.327 | 4.04 / 4.10 |
| Medium | 25 | 36.4 / 36.8 | 32.2 / 32.5 | 0.340 / 0.346 | 0.299 / 0.297 | 6.64 / 6.77 |
| Low | 5 | 24.4 / 23.5 | 6.0 / 5.8 | 0.713 / 0.796 | 0.712 / 0.795 | 3.09 / 2.64 |
| Low | **10** | 27.4 / 26.6 | 16.2 / 16.1 | 0.632 / 0.700 | 0.631 / 0.700 | 0.00 / 0.00 |
| Low | 15 | 30.4 / 29.0 | 24.3 / 23.8 | 0.572 / 0.623 | 0.571 / 0.622 | 4.49 / 4.17 |
| Low | 20 | 33.2 / 31.3 | 30.9 / 31.0 | 0.523 / 0.560 | 0.523 / 0.559 | 6.09 / 5.84 |
| Low | 25 | 36.1 / 33.5 | 36.2 / 36.0 | 0.482 / 0.502 | 0.482 / 0.502 | 9.62 / 8.81 |

The final mIoU sits lower than the Stage-2 mIoU at every voxel size, and the gap is not a
voxel effect. Stage 5 deletes woody debris below 0.5 m and merges the rest into mallee, so the
final woody-debris IoU is 0 at both sites. It also promotes tall unlabelled returns to mallee
and, at the Medium site, turns whole crowns to mallee. The figure therefore plots the Stage-2
agreement in (e), and panel (f) carries the final labels.

**Reading.** Voxel size trades fidelity to the 2D footprint against extension beyond it, and
it does so steadily. No voxel size scores best on both counts.

- **Inside the footprints almost nothing changes.** On every tile and at every size, at least
  98.82% (Medium) and 97.73% (Low) of the points on labelled cells keep their class.
- **What changes is how far labels travel past the footprints.** From 5 to 25 cm the share of
  transferred labels outside their footprint rises from 3.5% to 32.5% (Medium) and from 5.7% to
  36.2% (Low). The plan-view mIoU against the 2D labels falls from 0.787 to 0.446 and from 0.787
  to 0.526.
- **At 5 cm the transfer barely transfers.** A voxel is the size of one label cell and holds a
  median of one (Medium) or one to two (Low) labelled points; 67% of Medium voxels hold exactly
  one. The transfer is close to reading the 2D raster onto the points.
- **At 10 cm a label reaches at most one voxel past its footprint.** 12.5% (Medium) and 16.0%
  (Low) of transferred labels lie outside their footprint, and on average fewer than 1% of
  voxels mix classes (0.24% and 0.92%).
- **Above 10 cm voxels approach feature size.** Each further 5 cm step adds another 5–8
  percentage points of labels outside their footprint, and mixed-class voxels reach 1.64% and
  3.81% at 25 cm. The median polygon covers 0.076 m² at the Medium site, roughly 28 cm across
  (phase 3, `label_quality_manuscript.tex`), so a 25 cm voxel is about the width of a typical
  feature.
- **The final labels move modestly around 10 cm.** Relative to 10 cm, they differ for 1.4%
  (Medium) and 3.1% (Low) of points at 5 cm, and for 2.8% and 4.5% at 15 cm.

Against the 2D labels, 5 cm always scores best. That is expected, because the 2D labels are the
input and every extension counts against them. Whether an extended label is *right* cannot be
scored without 3D reference data. The sweep therefore documents a trade-off; it does not locate
an optimum at 10 cm (see [Findings to settle](#findings-to-settle-before-responding)).

**Draft reply to comment 1.**

> We thank the reviewer for this suggestion and have added the comparison as Supplementary
> Figure S#. The complete workflow was run with voxel sizes of 5, 10, 15, 20 and 25 cm on all
> 30 reference tiles (15 per site). Because the labelled points are the lidar returns
> themselves, each carrying the class of the 5 cm label cell beneath it, voxel size has almost
> no effect on returns inside the labelled footprints: on every tile and at every size, at least
> 98.8% (Medium site) and 97.7% (Low site) of them kept their 2D class. What voxel size controls is how far
> labels are extended to neighbouring returns. The share of transferred labels lying outside
> their 2D footprint rose from 3.5% at 5 cm to 12.5% at 10 cm and 32.5% at 25 cm at the Medium
> site (5.7%, 16.0% and 36.2% at the Low site). Plan-view agreement with the 2D labels decreased
> accordingly (mIoU 0.79, 0.65 and 0.45 at the Medium site; 0.79, 0.69 and 0.53 at the Low
> site). At 5 cm a voxel is the size of a single label cell and holds a median of one or two
> labelled returns, so the transfer adds little beyond the 2D raster; above 10 cm, voxels approach the
> size of the mapped features (median polygon area 0.08 m² at the Medium site) and mix classes
> more often. We therefore retained 10 cm, which extends each label by at most one voxel while
> keeping mixed-class voxels below 1% on average at both sites. Relative to 10 cm, final labels differed
> for 1.4% (Medium) and 3.1% (Low) of points at 5 cm, and for 2.8% and 4.5% at 15 cm.

The sentence beginning "We therefore retained 10 cm" states a rationale, not a measured
optimum. Confirm it reflects the reasoning behind the choice before using it (see
[Findings to settle](#findings-to-settle-before-responding)).

## 2. Matching radius, 0.30 m (comment 2)

**What it does.** Stage 1 (`height_stats.compute_class_height_thresholds`) gives each labelled
point the height above ground (`Z_norm`) of its XY-nearest Z_norm point, provided one lies
within the radius. It then takes each class's 15th–85th percentile of those heights as that
class's height range. The ranges feed the Stage-4 height prior and the minimum mallee crown
height.

**Result.** On all 30 tiles every labelled point has an exact twin in the Z_norm cloud, so the
lookup finds its height at 0.000 m and no point ever lies beyond the radius (inputs table). The
sweep confirms the radius has no effect. With radii of 0.01, 0.05, 0.10, 0.30 and 1.00 m, the
Stage-4 output and the final labels are identical to the default on every tile: 0.0000% of
points changed. The radius is a guard the data never reaches.

**A detail the review did not ask about.** The lookup matches in XY only. Where several returns
share a 1 cm XY position, it can take the height of a different return in the same column. That
happens for 8.3–19.1% of labelled points per tile at Medium and 4.3–10.5% at Low. Those returns
fall on the same 5 cm cell and carry the same label, so the class height ranges move by at most
0.017 m (Medium) and 0.003 m (Low). Reading `Z_norm` by exact coordinates would remove both this
approximation and the radius. It is not needed for the reply.

**Draft reply to comment 2.**

> The 0.30 m radius bounds the nearest-neighbour search that assigns a height above ground to
> each labelled point when the class height ranges are estimated (Stage 1). In our data the
> labelled points are the lidar returns themselves, each carrying the class of the 5 cm label
> cell beneath it, so every labelled point has an exact counterpart in the height-normalised
> point cloud and was matched at a distance of 0.00 m on all 30 tiles. The radius is therefore
> never reached, and its value does not affect the results. Repeating the complete workflow
> with radii of 0.01, 0.05, 0.10, 0.30 and 1.00 m produced identical labels on every tile
> (Supplementary Figure S#).

## 3. Height-prior weights, 1 and 0.1 (comment 3)

**What they do.** Stage 4 (`polar_refine.refine_with_polar_coords`) divides each detected
mallee crown into polar bins, 10 cm radially by 10°. It considers every lidar point in a bin
that holds at least three labelled points. Each class's share of the bin is multiplied by 1 if
the point's height lies inside that class's Stage-1 range, and by *w* = 0.1 if it does not.
The products are then renormalised.

Renormalising cancels any factor common to both weights, so (1, 0.1) is one parameter, the
ratio *w*. A class whose height range excludes the point needs ten times the bin share of a
class whose range includes it. The ends of the range each break something:

- *w* = 1 switches the prior off.
- *w* = 0 makes it a veto. That would forbid each class for the 30% of its own points that a
  15th–85th percentile range excludes by construction.

**Result.** Medium site, `mean / median` across tiles:

| Out-of-range weight | 0 | 0.01 | **0.1** | 0.5 | 1 |
|---|---:|---:|---:|---:|---:|
| Stage-4 fills (% of points) | 0.288 / 0.263 | 0.490 / 0.453 | **0.490 / 0.453** | 0.490 / 0.453 | 0.490 / 0.453 |
| Stage-4 flips (% of points) | 0.066 / 0.036 | 0.075 / 0.042 | **0.060 / 0.036** | 0.046 / 0.029 | 0.043 / 0.029 |
| Final labels changed (%) | 0.000 / 0.000 | 0.000 / 0.000 | **0.000 / 0.000** | 0.000 / 0.000 | 0.000 / 0.000 |

Within Stage 4 the weight matters mostly at its ends:

- **A veto (*w* = 0) removes 41% of fills** (0.490% → 0.288% of points). A bin whose classes
  all lie out of range for a point then yields no probability at all, so the point is skipped.
- **Switching the prior off (*w* = 1) lowers flips** from 0.060% to 0.043% of points.

None of this reaches the final labels at the Medium site. On the worst Low tile the weight
changes 0.13% of points at *w* = 0 and 0.05% at *w* = 1.

**Draft reply, weights (comment 3).**

> The two weights define a soft height prior in the crown-centric refinement (Stage 4). A
> class's share of a polar bin is kept in full when the point lies within that class's
> 15th–85th percentile height range and multiplied by 0.1 when it does not. Because the weighted
> shares are renormalised, only the ratio of the two weights matters: a class outside its height
> range must be ten times as frequent in the bin as a class inside it to be assigned. The weight
> is not set to zero because each range excludes, by construction, 30% of that class's own
> points, and a hard veto would forbid those labels. In our tests it removed 41% of the labels
> Stage 4 assigns to previously unlabelled points. A weight of 1 would disable the prior.
> Varying the weight from 0 to 1 changed no final label at the Medium site and at most 0.13% of
> points on any Low-site tile (Supplementary Figure S#).

## 4. Confidence threshold, `min_conf` 0.3 (comment 3)

**What it does.** `min_conf` decides only whether Stage 4 fills a point that has no label: the
point takes its bin's most probable class if that probability is at least `min_conf`. It is
never applied to changing an existing label; that is the flip ratio's job.

The most probable of *k* classes always has probability at least 1/*k*. So in a bin holding
up to three classes the fill always passes at 0.3. The threshold can refuse a fill only where
four or five classes are close to tied.

**Result.** Medium site, `mean / median` across tiles:

| `min_conf` | 0.2 | **0.3** | 0.4 | 0.5 | 0.6 | 0.8 |
|---|---:|---:|---:|---:|---:|---:|
| Stage-4 fills (% of points) | 0.490 / 0.453 | **0.490 / 0.453** | 0.490 / 0.453 | 0.490 / 0.453 | 0.490 / 0.453 | 0.488 / 0.453 |
| Evaluated points below the threshold (%) | 0.000 / 0.000 | **0.000 / 0.000** | 0.000 / 0.000 | 0.000 / 0.000 | 0.092 / 0.062 | 0.302 / 0.194 |
| Final labels changed (%) | 0.000 / 0.000 | **0.000 / 0.000** | 0.000 / 0.000 | 0.000 / 0.000 | 0.000 / 0.000 | 0.000 / 0.000 |

The threshold does not bind at the default. At 0.3 or below no evaluated point on any tile falls
under it; at 0.4 and 0.5 at most 0.0011% do, on one Medium tile, and fills stay at 0.490% of
points. Stage-4 probabilities cluster near 1 (mean 0.9985), because nearly every polar bin
inside a crown is dominated by a single class. Points start falling below the threshold at 0.6
(up to 0.32% of evaluated points on the worst Medium tile) and 0.8 (up to 1.05%), and even at
0.8 fills drop by just 0.002 percentage points. The final labels are unchanged at the Medium
site and change by at most 0.002% of points on any Low tile.

**Draft reply, confidence threshold (comment 3).**

> The confidence threshold governs only whether Stage 4 assigns a class to a point that
> received no label from the voxel transfer. The most probable class in the point's polar bin
> is assigned if its height-weighted probability is at least 0.3. The most probable of *k*
> classes always has a probability of at least 1/*k*, so a threshold of 0.3 accepts any bin
> containing up to three classes and rejects only near-ties among four or five. In practice the
> threshold was not binding: Stage-4 probabilities averaged 0.998, no evaluated point on any
> tile fell below 0.3 and at most 0.001% fell below 0.5, and thresholds from 0.2 to 0.8 changed
> no final label at the Medium site and at most 0.002% of points on any Low-site tile
> (Supplementary Figure S#).

## 5. Flip ratio, `min_flip_ratio` 1.5 (comment 3)

**What it does.** Stage 4 replaces an existing label when the most probable class's
probability exceeds *r* times that of the current class. In a bin holding two classes, the
challenger therefore needs more than *r*/(1 + *r*) of the bin's height-weighted probability:

| *r* | 1 | 1.25 | **1.5** | 2 | 3 |
|---|---:|---:|---:|---:|---:|
| Challenger's share needed to flip | 50% | 56% | **60%** | 67% | 75% |

At 1.5, a label taken from the 2D annotation is overturned only by a 60:40 majority, not a
bare one. A current class that is absent from the bin has probability 0 and is replaced
whatever *r* is.

**Result.** Medium site, `mean / median` across tiles:

| `min_flip_ratio` | 1 | 1.25 | **1.5** | 2 | 3 |
|---|---:|---:|---:|---:|---:|
| Stage-4 flips (% of points) | 0.070 / 0.040 | 0.064 / 0.037 | **0.060 / 0.036** | 0.054 / 0.033 | 0.047 / 0.029 |
| Final labels changed (%) | 0.000 / 0.000 | 0.000 / 0.000 | **0.000 / 0.000** | 0.000 / 0.000 | 0.000 / 0.000 |

Flips fall steadily as *r* rises, and fills are unaffected (0.490% at every value). The final
labels are unchanged at the Medium site and change by at most 0.016% of points on any Low tile.

**Draft reply, flip ratio (comment 3).**

> The flip ratio sets how much stronger the evidence for a new class must be before Stage 4
> overturns a label transferred from the 2D annotation: the new class's probability must exceed
> 1.5 times that of the current class. In a bin containing two classes this corresponds to a
> 60:40 majority rather than a bare majority, so labels are not reversed on near-ties. Varying
> the ratio from 1 to 3 reduced the share of overturned labels from 0.070% to 0.047% of points,
> changed no final label at the Medium site, and changed at most 0.016% of points on any
> Low-site tile (Supplementary Figure S#).

**Draft closing sentence for the three Stage-4 replies.**

> At the Medium site the final labels are insensitive to all three parameters because the
> subsequent crown-consistency step (Stage 5) assigns every point of a crown that is more than
> half mallee to mallee, and all 619 crowns were at least 77% mallee. The parameters therefore
> shape the intermediate Stage-4 labels (about 0.55% of points at the default values) but not
> the delivered labels.

## Findings to settle before responding

### Stage 4 does not reach the final labels at the Medium site

All 619 Medium crowns were at least 77% mallee when Stage 5 voted, so the vote made every one of
them wholly mallee. That overwrote every decision Stage 4 had made inside them, and none of its
three parameters changed a single final label on any tile.

This is a strong answer to comment 3, because the values cannot matter. It also means the
crown-centric refinement described in 3.2.2 has no effect on the labels delivered for that site.
The draft closing sentence says so, because without it the replies would read as a robust
refinement rather than an overwritten one. A reviewer may then reasonably ask why the stage is
there. Before replying, decide whether to:

- keep the stage and state its role plainly, as the closing sentence does;
- restrict the Stage-5 vote so it no longer overrides Stage 4, which would change the results;
  or
- describe the two stages together as one crown-consistency step.

That is a methodological decision, and nothing in the pipeline has been changed to pre-empt it.

### No voxel size agrees with the 2D labels better than 5 cm

Every agreement measure is best at 5 cm and declines steadily, at both sites. That is expected,
since the 2D labels are the input and any extension counts against them. It does mean the sweep
documents a trade-off rather than an optimum at 10 cm.

The draft reply grounds 10 cm in that trade-off: the smallest voxel that extends labels beyond
copying single cells, below the median feature size, with fewer than 1% of voxels mixing
classes on average. Confirm that matches the reasoning behind the choice.

There may be a stronger reason. In `phase_5_3d_cnn_training`, `make_voxel_npz_id.py` voxelises at
10 cm with the comment "Matches your Cartesian grid size from Step 1", and
`preprocessing_lidar_modelMINSU3D.py` also uses 10 cm. If the 3D model reported in the paper
consumes 10 cm voxels, labels built on the same grid give every training voxel a single label,
which is a better argument than anything the sweep can show. The ForestFormer3D configs in this
repository use 0.2 m and 0.05 m, so check which model the manuscript reports.

Switching to 5 cm instead would change 1.4% (Medium) and 3.1% (Low) of final labels, and
everything downstream of them.

### The Low site has crowns and mallee labels without mallee in its 2D labels

On 4 of the 15 Low tiles the watershed's height fallback finds 8 crowns, and Stage 4 runs inside
them. On 3 tiles Stage 5 promotes unlabelled returns at or above 1.3 m to mallee, up to 0.24% of a tile's points
(`Stage-5 promoted points (%)` in the per-tile CSV). Check this before describing mallee
processing as specific to the Medium site.

## Reproduce

```bash
cd phase_4_3D_lidar_prep/step_3_structure_class

# 30 tiles x 22 configurations; the recorded run took about 25 min with 16 workers
python parameter_sensitivity.py --workers 16 --output-dir assets/parameter_sensitivity

# figures + tables, from the per-tile CSV
python plot_parameter_sensitivity.py \
    --csv assets/parameter_sensitivity/parameter_sensitivity_per_tile.csv

# the voxels drawn on one tile (reads the lidar tiles; notebook 04 runs this too)
python plot_voxel_transfer.py --site site1_1_DD0001 --tile 106 \
    --output assets/parameter_sensitivity/voxel_transfer_tile106
```

`--data-root` defaults to `/mnt/d/paper2/data/calperumResearch`. Every run re-reads its two LAS
files, and over the WSL mount of `D:` workers spent much of their time waiting on reads, so the
recorded run used a local copy passed as `--data-root`. Only these files are needed, about
2.1 GB, at the same relative paths:

- `lidar/clipped/raster/z_norm/csf`
- the structure clouds: `lidar/clipped/mask/coreg/icp_top` for DD0001, `…/coreg/icp` for DD0011
- the `annotation_raster` directory and its `downsample_5cm_nearest` subdirectory

`--parameters`, `--sites` and `--tiles` restrict a run.

| Output (`assets/parameter_sensitivity/`) | Contents |
|---|---|
| `parameter_sensitivity_per_tile.csv` | one row per site, tile, parameter and value, plus one `input` row per tile holding the checks on what the pipeline is given |
| `parameter_sensitivity_summary.csv` | mean, median and standard deviation across tiles |
| `supplementary_voxel_size_sensitivity.png` / `.pdf` | the voxel-size figure (comment 1) |
| `supplementary_parameter_sensitivity.png` / `.pdf` | the other four parameters |
| `parameter_sensitivity_tables.md` | the tables and extremes this README quotes |
| `voxel_transfer_tile106.png` / `.pdf` | labels spreading past the 2D footprint as voxels grow, on one tile window |
| `tile_106_3d_voxel_sweep.png` | tile 106 divided into voxels at each size, in 3D (geometry only; drawn by notebook 04) |
