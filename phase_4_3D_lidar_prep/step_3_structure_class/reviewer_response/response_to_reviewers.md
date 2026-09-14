# Response to reviewers — error propagation and parameter choices

Four comments, four replies. Each reply is written to be pasted into the response letter; the
evidence behind it is in [`04_reviewer_responses.ipynb`](04_reviewer_responses.ipynb), and every
number comes from
[`assets/parameter_sensitivity/reviewer_response_tables.md`](../assets/parameter_sensitivity/reviewer_response_tables.md),
which `figures.py` generates. **No number in this document is typed by hand.** If the
measurements are rerun, regenerate the tables and check the replies against them.

| Comment | Subject | Evidence |
|---|---|---|
| 1 | 6.1: quantify the error introduced at each step | Figure S1, Table S1 |
| 2 | 3.2.2: compare voxel sizes of 5–25 cm | Figure S2, Figure S3 |
| 3 | Why the 0.30 m matching radius | Figure S4(a, e) |
| 4 | Why the voxel size, weights, confidence threshold and flip ratio | Figure S4, Table S2 |

Two facts run through all four replies, and both are measured on every tile rather than assumed:

- **The labelled points are the lidar returns themselves.** The structure cloud is cut from the
  height-normalised cloud with PDAL `filters.colorization`: each return takes the class of the
  5 cm label cell beneath it. It is an exact coordinate subset of that cloud — on all 30 tiles,
  100% of its points have an exact twin, matched at 0.000 m.
- **There is no independent 3D reference for these sites.** Every measure reported here is
  either fidelity to the 2D annotation or stability against the default configuration. A label
  extended to a neighbouring return can be scored as leaving its 2D footprint, but not as right
  or wrong.

---

## Comment 1 — "In Section 6.1, could the authors explicitly quantify or at least qualitatively characterise the error introduced at each step of the pipeline?"

> We thank the reviewer: this is a fair request and the revised Section 6.1 now carries an
> error budget rather than a general statement of limitations. We have added
> **Supplementary Figure S1** and **Supplementary Table S1**, and summarise them in the text.
>
> The difficulty in quantifying error step by step is that the steps are not measured in the
> same units — a boundary displacement is in metres, a downsampling loss in cells, a voxel
> transfer in points. We therefore put every intermediate product of the pipeline on one
> common scale: each is rasterised top-down onto the 5 cm grid, taking the majority class of
> the labelled points in each cell, and scored as plan-view mean IoU against **the annotator's
> 1 cm raster**, the one reference that predates every step. This gives a single descending
> ladder whose steps can be compared with each other (Figure S1):
>
> | Step | Medium site | Low site |
> |---|---:|---:|
> | The annotator's 1 cm delineation (reference) | 1.000 | 1.000 |
> | after downsampling to the 5 cm label grid | 0.806 (−0.194) | 0.996 (−0.004) |
> | after transfer onto the labelled lidar returns | 0.694 (−0.112) | 0.881 (−0.115) |
> | after the Stage-2 voxel transfer at 10 cm | 0.531 (−0.162) | 0.691 (−0.190) |
> | after the Stage 3–5 crown steps (delivered labels) | 0.402 (−0.129) | 0.631 (−0.060) |
>
> The relative contribution of each component therefore differs strongly between the two
> sites. At the medium site the four steps account for 32%, 19%, 27% and 22% of a total loss
> of 0.598; at the low site, 1%, 31%, 51% and 16% of a total of 0.369. The single largest
> component at the medium site is **not** in the 3D pipeline at all but in the 2D → 5 cm
> downsampling, where a 3072 px tile at 1 cm does not divide into 5 cm cells and 13 of 15
> tiles land on a shifted grid; the low site, whose labels happen to align, loses almost
> nothing there and is dominated instead by the voxel transfer.
>
> Characterising each step in its own terms, and quantifying it where a measurement exists:
>
> - **Delineation.** The dominant uncertainty, and the only one we cannot measure directly:
>   no repeat annotation was made, and none is claimed. As a bound, eroding and dilating each
>   class by ±5 cm gives a delineation IoU of 0.321 (medium) and 0.508 (low) — the agreement
>   two delineations disagreeing by that much everywhere would still reach. Cross-class overlap
>   in the delivered polygons is 0.02% and 0.16% of labelled area.
> - **Polygon → 1 cm raster.** Rasterisation and the resolution of overlaps by draw order:
>   class IoU 0.924 (medium) and 0.679 (low); the low-site mean is one class failing alone
>   (median 0.957).
> - **1 cm → 5 cm label grid.** The step above. Feature centroids move by only 0.012 m and
>   0.007 m, so what this costs is feature *shape*, not position.
> - **5 cm labels → lidar returns.** Two distinct errors. Labelled cells holding no return
>   cannot be labelled at all, which sets a ceiling of 0.925 (medium) and 0.884 (low) on
>   agreement with the 5 cm raster before the pipeline runs; separately, only 362 points
>   (0.005%) and 0 points on labelled cells are missing from the structure cloud.
> - **Stage 1, height lookup.** Negligible: the nearest-neighbour search is in XY only, which
>   shifts a class height range by at most 0.017 m (medium) and 0.003 m (low).
> - **Stage 2, voxel transfer.** The transfer extends 12.5% (medium) and 16.0% (low) of the
>   labels it assigns past their 2D footprint, and loses at most 1.2% and 2.3% of the points
>   inside a footprint on the worst tile. Fewer than 1% of occupied voxels mix classes.
> - **Stages 3–5, crown steps.** These change little in the delivered labels beyond one
>   deliberate act: Stage 5 removes woody debris, which is not a growth form. That single
>   removal accounts for **80% of the medium site's Stage 3–5 loss and 98% of the low site's**
>   (Table S1). Excluding it, the crown steps cost 0.036 mIoU at the medium site, through
>   mallee, and essentially nothing at the low site.
>
> Three qualifications belong with this budget, and are stated in 6.1.
>
> First, the ladder reports **successive measurements against one reference, not independent
> error terms**: the differences show where agreement is lost, but they are not separable
> quantities that multiply or sum in a formal error model.
>
> Second, a loss against the 2D annotation is **not the same as an error**. The annotation is
> the input as well as the reference, so any extension of a label beyond its plan-view outline
> is scored against the pipeline even where it recovers a feature's true extent; and the
> delineation bound above shows the reference itself carries substantial uncertainty. The
> largest single component at the medium site — the 5 cm grid offset — is by contrast an
> unambiguous, fixable defect.
>
> Third, one error source appears nowhere in the IoU ladder. Because the labels are transferred
> by a plan-view raster lookup with no height gate, every return in the air column above a
> labelled cell inherits that cell's class. Where a ground class lies beneath mallee canopy,
> canopy returns are written with the ground class. The signature is an upper tail in the class
> height distribution: medium-site Shrub runs 0.65 m at p95 but 2.15 m at p99, with 1.6% of its
> returns above 1 m. Woody debris carries a much heavier tail (11.9% medium, 40.5% low) but is
> removed by Stage 5 and does not reach the delivered labels. We now state this as a known
> limitation of plan-view label transfer; a height-gated lookup would remove it and is the
> clearest improvement available to the method.

**Supplementary Figure S1, draft caption.** Error budget of the label pipeline at the medium
(left) and low (right) sites. Each bar is the loss in plan-view agreement introduced by one
step, measured as mean IoU over classes on the 5 cm grid against the annotator's 1 cm raster,
averaged over the 15 tiles of each site. Bars are annotated with the loss and its share of the
total. Colours group the steps into the 2D annotation chain (brown), the move onto lidar returns
(yellow), the Cartesian voxel transfer (blue) and the crown detection, refinement and vote
(green). Levels above each marker are the running agreement.

**Supplementary Table S1.** Every step in its own units, and the class-by-class attribution of
the Stage 3–5 loss. Generated as sections 1 of `reviewer_response_tables.md`.

---

## Comment 2 — "In subsection 3.2.2, the choice of 10 cm voxel size appears empirical. Could the authors provide a comparison of different voxel sizes (e.g. 5, 10, 15, 20, 25 cm) and include the results as a supplementary figure?"

> We have added this comparison as **Supplementary Figure S2**. The complete workflow was run
> at 5, 10, 15, 20 and 25 cm on all 30 reference tiles (15 per site).
>
> We should first clarify what this parameter is, since 3.2.2 was not explicit. The voxel size
> governs the **2D → 3D label transfer** used to build the training labels; it is not the input
> resolution of the segmentation models, which consume the labelled points directly and apply
> their own internal voxelisation. It controls how far a class annotated in 2D is propagated to
> neighbouring lidar returns.
>
> Because the labelled points are the lidar returns themselves, each already carrying the class
> of the 5 cm cell beneath it, voxel size has almost no effect on returns *inside* the labelled
> footprints: at every size and on every tile, at least 98.8% (medium) and 97.7% (low) kept
> their 2D class. What it controls is extension beyond those footprints, and agreement with the
> 2D labels therefore falls monotonically as the voxel grows (plan-view mIoU 0.79 → 0.65 → 0.45
> at the medium site for 5, 10 and 25 cm; 0.79 → 0.69 → 0.53 at the low site). We have revised
> 3.2.2 to state plainly that this is a **monotonic trade-off in which the comparison does not
> identify an optimum**: the 2D annotation is both the input and the reference, so every
> extension is scored as disagreement, and no independent 3D reference exists for these sites.
>
> We therefore justify 10 cm on **survey point density**, which makes the choice a property of
> the data rather than a tuned value. The merge can only act where a voxel holds more than one
> labelled return, so the relevant scale is the mean spacing of the labelled returns: 4.5 cm at
> the medium site and 5.6 cm at the low site (493 and 321 labelled returns m⁻²). A 5 cm voxel is
> about one spacing wide — narrower than the spacing at the low site — and accordingly holds a
> median of one labelled return, with 67% (medium) and 48% (low) of voxels holding exactly one;
> the merge then reduces to projecting the 2D raster onto the points. At 10 cm the voxel is
> roughly two spacings, the median rises to 2.3 and 5.0 returns, single-occupancy falls to 34%
> and 18%, labels still reach at most one voxel beyond their footprint, and fewer than 1% of
> voxels mix classes (0.24% and 0.92%). Above 10 cm voxels approach the size of the mapped
> features (median polygon ≈28 cm across) and mixed-class voxels grow steadily, to 1.6% and 3.8%
> at 25 cm. Expressed as a rule, **the voxel should be about twice the mean spacing of the
> labelled returns**, which scales as the inverse square root of survey density and so transfers
> to surveys of other densities.
>
> Two limitations belong with this. First, that spacing is a plan-view average, and return
> density varies strongly with height (**Supplementary Figure S3**): above 3 m the medium site
> carries only ~8 labelled returns m⁻², so in the upper canopy a 10 cm voxel usually holds a
> single labelled return and the transfer is correspondingly weaker for tall vegetation. The
> chosen size satisfies the two-spacing criterion through the understorey, where the great
> majority of labelled returns lie, but not throughout the profile; a height-varying voxel would
> follow the density more closely and is a natural extension of the method. Second, the
> additional returns labelled by enlarging the voxel lie almost entirely (99–100%) on cells the
> 2D annotation leaves unlabelled, and without 3D reference data we cannot determine whether
> these represent spurious spread or genuine recovery of a feature's extent beyond its plan-view
> outline. Relative to 10 cm, the final labels differ for 1.4% (medium) and 3.1% (low) of points
> at 5 cm, and for 2.7% and 4.5% at 15 cm.

**Supplementary Figure S2, draft caption.** Sensitivity of the 2D-to-3D label transfer to voxel
size at the medium (blue) and low (orange) sites. Lines are medians across the 15 tiles of each
site, with the inter-quartile range shaded; the grey line marks the 10 cm default. (a) Share of
lidar points labelled by the Cartesian voxel transfer (Stage 2). (b) Share of those labels lying
on cells the 2D labels leave unlabelled. (c) Share of points on labelled 2D cells that keep
their cell's class. (d) Labelled points per occupied voxel. (e) Plan-view agreement between the
transferred labels and the 2D labels: mean IoU over classes on the 5 cm grid. (f) Share of lidar
points whose final label differs from the 10 cm result.

**Supplementary Figure S3, draft caption.** Vertical distribution of lidar returns, summed over
the 15 tiles of each site, in 10 cm height bins on a logarithmic count axis. (a) All returns.
(b) Returns carrying a 2D label. (c) Labelled returns by growth form at the medium site. The
labelled-return density that sets the useful voxel size falls by more than an order of magnitude
between the understorey and the upper canopy.

---

## Comment 3 — "Please explain why the 0.30 matching radius was selected."

> The 0.30 m radius bounds the nearest-neighbour search that assigns a height above ground to
> each labelled point when the class height ranges are estimated (Stage 1). Those ranges feed
> the Stage-4 height prior and the minimum crown height.
>
> In our data the labelled points **are** the lidar returns, so every one has an exact
> counterpart in the height-normalised cloud and was matched at a distance of **0.00 m on all 30
> tiles**; no point on any tile lay beyond the radius. Its value therefore cannot affect the
> result, and we confirmed this by repeating the complete workflow with radii of 0.01, 0.05,
> 0.10, 0.30 and 1.00 m: the labels were **identical on every tile** (0.0000% of points changed;
> **Supplementary Figure S4**, panels a and e).
>
> The radius is thus a guard against unmatched points that this dataset never triggers, and we
> have restated it in 3.2.2 as such rather than as a tuned value. We note for completeness that
> the lookup matches in XY only, so where several returns share a 1 cm XY position it may take
> the height of another return in the same column; those returns lie on the same 5 cm cell and
> carry the same label, and the resulting shift in any class height range is at most 0.017 m
> (medium) and 0.003 m (low).

---

## Comment 4 — "It should be explained why various parameters such as voxel size (10 cm), weighting (1, 0.1), confidence threshold and flip ratio were selected at their current values."

> Each parameter was varied across its range with the others held at their defaults, on all 30
> tiles, and every run scored against the 2D labels it was built from (**Supplementary Figure
> S4**, **Supplementary Table S2**). We take them in turn.
>
> **Voxel size (10 cm)** — as in our reply to the preceding comment, with Supplementary
> Figure S2. It is the only one of the four that materially changes the delivered labels.
>
> **Height-prior weights (1 and 0.1)** — these form a *soft* height prior in the crown-centric
> refinement (Stage 4). A class's share of a polar bin is kept in full when the point lies
> within that class's 15th–85th percentile height range, and multiplied by 0.1 when it does not.
> Because the weighted shares are renormalised, **only the ratio of the two matters**: a class
> outside its height range must be ten times as frequent in the bin as one inside it to be
> assigned. The ends of the range each break something, which is what fixes the value between
> them. A weight of 1 disables the prior. A weight of 0 turns it into a veto, which would forbid
> each class for the 30% of its own points that a 15th–85th percentile range excludes by
> construction; in our tests it removed 41% of the labels Stage 4 assigns to previously
> unlabelled points (0.471% → 0.278% of points). Varying the weight from 0 to 1 changed no final
> label at the medium site and at most 0.13% of points on any low-site tile.
>
> **Confidence threshold (`min_conf` = 0.3)** — this governs only whether Stage 4 assigns a
> class to a point the voxel transfer left unlabelled; it is never applied to changing an
> existing label. The most probable of *k* classes always has probability at least 1/*k*, so a
> threshold of 0.3 accepts any bin containing up to three classes and rejects only near-ties
> among four or five. In practice it was not binding: Stage-4 probabilities averaged 0.999,
> no evaluated point on any tile fell below 0.3, and at most 0.003% fell below 0.5. Thresholds
> from 0.2 to 0.8 changed no final label at the medium site and at most 0.002% of points on any
> low-site tile.
>
> **Flip ratio (`min_flip_ratio` = 1.5)** — this sets how much stronger the evidence for a new
> class must be before Stage 4 overturns a label inherited from the 2D annotation: the new
> class's probability must exceed 1.5 times that of the current class. In a bin containing two
> classes this corresponds to a 60:40 majority rather than a bare one, so labels are not
> reversed on near-ties; the annotation is treated as the default and displaced only on clear
> evidence. Varying the ratio from 1 to 3 moved the share of overturned labels from 0.069% to
> 0.047% of points, changed no final label at the medium site, and changed at most 0.016% of
> points on any low-site tile.
>
> **Why the last three barely matter, which we now state in the text.** Stages 3–5 act only
> inside detected mallee crowns, and Stage 5 ends by holding a vote in each crown: where more
> than half its points are mallee, all of them become mallee. At the medium site 815 of the 816
> crowns put to that vote were mallee-majority, so the vote overwrote essentially every decision
> Stage 4 had made, and **no tested value of any of these three parameters changed a single
> final label on any medium-site tile**. They shape the intermediate Stage-4 labels — about
> 0.53% of points at the default settings — but not the delivered ones. At the low site, whose
> 2D labels contain no mallee, the watershed still finds 8 crowns on 4 of 15 tiles and those
> Stage-4 decisions do survive, which is why the residual effects quoted above are non-zero
> there. Across all four parameters, all 30 tiles and every value tested, the largest change to
> the final labels on any tile was 0.13% of points.

**Supplementary Figure S4, draft caption.** Sensitivity of the label-fusion workflow to (a, e)
the Stage-1 height matching radius and to the Stage-4 (b, f) out-of-range height-prior weight,
(c, g) confidence threshold and (d, h) flip ratio. Blue: medium site; orange: low site. Grey
lines mark the default values. Top row: share of lidar points whose label Stage 4 changes,
median across 15 tiles with the inter-quartile range shaded. Bottom row: share of points whose
final label differs from the default configuration, on the worst tile. At the low site crowns
were detected on 4 of 15 tiles, so its Stage-4 median is zero.

**Supplementary Table S2.** The parameters, their defaults, the values tested and the largest
change each made to the final labels on any tile. Generated as section 4 of
`reviewer_response_tables.md`.

---

## Decisions to settle before submitting

These are methodological choices the measurements expose but cannot make. Nothing in the
pipeline has been changed to pre-empt them.

1. **Stage 4 does not reach the delivered labels at the medium site.** 815 of 816 crowns were
   mallee-majority, so the Stage-5 vote overwrote the refinement inside them. The reply above
   states this plainly, because without it the answer would read as a robust refinement rather
   than an overwritten one — and a reviewer may reasonably then ask why the stage exists.
   Decide whether to keep the stage and state its role, restrict the Stage-5 vote so it no
   longer overrides Stage 4 (which changes the results), or describe the two as a single
   crown-consistency step.
2. **The 10 cm justification rests on point density, not on agreement.** Every agreement measure
   is best at 5 cm. If the 3D model reported in the manuscript itself voxelises at 10 cm — as
   `phase_5_3d_cnn_training/make_voxel_npz_id.py` and `preprocessing_lidar_modelMINSU3D.py` do,
   while the ForestFormer3D configs here use 0.2 m and 0.05 m — then matching the label grid to
   the model grid is a stronger argument than anything the sweep shows. Confirm which model the
   manuscript reports.
3. **The low site produces mallee labels although its 2D labels contain none.** On 3 tiles
   Stage 5 promotes unlabelled returns at or above 1.3 m to mallee. Check this before
   describing mallee processing as specific to the medium site.
4. **The 5 cm grid offset is the largest single error component at the medium site** and it is a
   fixable defect, not an inherent cost: `gdalwarp -tr 0.05` cannot land on the 1 cm grid because
   3072 × 0.01 / 0.05 = 614.4. Decide whether to report it as a limitation or fix it and re-run,
   which would change everything downstream.
