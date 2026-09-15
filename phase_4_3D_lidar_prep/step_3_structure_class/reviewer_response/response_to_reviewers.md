# Response to reviewers: sensitivity analysis and error propagation

This document contains the four replies supported by
[`04_reviewer_responses.ipynb`](04_reviewer_responses.ipynb). The complete workflow was evaluated
on 30 tiles (15 per site). All values are generated from the sensitivity-analysis CSVs by
`figures.py`.

| Comment | Analysis | Evidence |
|---|---|---|
| 1 | Sensitivity to voxel sizes from 5 to 25 cm | Figure S2 |
| 2 | Error propagation through the pipeline | Figure S1 and Table S1 |
| 3 | Sensitivity to the 0.30 m matching radius | Figure S4(a, e) |
| 4 | Impact of the selected parameter values | Figure S4 and Table S2 |

The reported metrics measure fidelity to the 2D annotation or change from the default
configuration. They do not measure absolute 3D accuracy because independent 3D reference labels
are unavailable for these sites.

---

## Comment 2: error propagation

> *In Section 6.1, could the authors explicitly quantify or at least qualitatively characterise
> the error introduced at each step of the pipeline?*

**Response.** We added an error-propagation analysis as Supplementary Figure S1 and Table S1.
Because the pipeline stages produce outputs in different units, every intermediate product was
rasterised onto the 5 cm grid and compared with the annotator's 1 cm raster using plan-view mean
intersection-over-union (mIoU).

| Pipeline level | Medium site | Low site |
|---|---:|---:|
| Annotator's 1 cm raster | 1.000 | 1.000 |
| After 1 cm to 5 cm downsampling | 0.806 (-0.194) | 0.996 (-0.004) |
| After transfer to labelled lidar returns | 0.694 (-0.112) | 0.881 (-0.115) |
| After 10 cm voxel transfer | 0.531 (-0.162) | 0.691 (-0.190) |
| After crown processing | 0.402 (-0.129) | 0.631 (-0.060) |

At the medium site, the four steps contributed 32%, 19%, 27%, and 22% of the total mIoU loss of
0.598. At the low site, they contributed 1%, 31%, 51%, and 16% of the total loss of 0.369.
Therefore, the dominant source differed between sites: downsampling was largest at the medium
site, whereas voxel transfer was largest at the low site.

Table S1 also characterises each stage using a stage-specific metric:

- polygon-to-raster and grid conversion: class IoU and centroid displacement;
- transfer to lidar returns: plan-view mIoU and missing labelled returns;
- height matching: maximum class-height-range displacement;
- voxel transfer: footprint extension, footprint recall, and mixed-class voxels; and
- crown processing: changed labels, promoted points, and class removal.

These are successive losses against one common reference, not statistically independent error
terms. A reduction in agreement with the 2D annotation also does not necessarily imply incorrect
3D geometry because the annotation is both an input and the available reference.

**Supplementary Figure S1 caption.** Error propagation through the label-generation pipeline at
the medium and low sites. Each bar is the change in plan-view mIoU against the annotator's 1 cm
raster, averaged over 15 tiles. Percentages show each stage's contribution to the total loss.

---

## Comment 1: voxel-size sensitivity

> *In subsection 3.2.2, the choice of 10 cm voxel size appears empirical. Could the authors
> provide a comparison of 5, 10, 15, 20, and 25 cm voxel sizes?*

**Response.** We added Supplementary Figure S2 and repeated the complete workflow using voxel
sizes of 5, 10, 15, 20, and 25 cm on all 30 tiles.

Increasing voxel size increased the proportion of labelled lidar points but also propagated more
labels outside the 2D footprint. Consequently, plan-view agreement decreased monotonically. At
the medium site, median mIoU was 0.786, 0.646, 0.556, 0.480, and 0.433 from 5 to 25 cm. At the low
site, the corresponding values were 0.808, 0.708, 0.633, 0.572, and 0.534. The comparison therefore
shows a trade-off rather than an accuracy optimum.

The 10 cm value was selected using labelled-return spacing. Mean spacing was 4.51 cm at the medium
site and 5.58 cm at the low site. A 5 cm voxel was approximately one return spacing wide and often
contained only one labelled return, leaving little information to merge. A 10 cm voxel was about
twice the mean spacing, increased point co-occupancy, and kept mixed-class voxels below 1% (0.24%
and 0.92%). At larger sizes, mixed-class voxels increased steadily, reaching 1.64% and 3.81% at
25 cm. Thus, 10 cm is a survey-density-based compromise rather than a value selected by optimizing
against the same labels used as input.

**Supplementary Figure S2 caption.** Sensitivity of the 2D-to-3D label transfer to voxel sizes of
5, 10, 15, 20, and 25 cm. Lines show site medians across 15 tiles and shading shows the
interquartile range. Panels report labelled points, labels outside the 2D footprint, footprint
recall, labelled points per voxel, plan-view mIoU, and final-label change relative to 10 cm.

---

## Comment 3: matching-radius sensitivity

> *Please explain why the 0.30 m matching radius was selected.*

**Response.** The radius is the maximum XY search distance used to match labelled points to the
height-normalised cloud during Stage 1. It is a conservative guard against unmatched points, not
a tuned spatial scale.

All labelled points had exact counterparts in the height-normalised cloud and matched at 0.000 m
on all 30 tiles. No point exceeded the 0.30 m radius. Repeating the workflow with radii of 0.01,
0.05, 0.10, 0.30, and 1.00 m produced identical final labels on every tile. The maximum shift in
any derived class-height range from XY-only matching was 0.017 m at the medium site and 0.003 m at
the low site. Therefore, the selected radius had no impact on the reported results.

---

## Comment 4: impact of selected parameters

> *It should be explained why voxel size, weighting, confidence threshold, and flip ratio were
> selected at their current values.*

**Response.** We performed a one-at-a-time sensitivity analysis on all 30 tiles, varying each
parameter while holding the others at their defaults.

| Parameter | Default | Values tested | Largest final-label change: medium | Largest final-label change: low |
|---|---:|---|---:|---:|
| Voxel size | 10 cm | 5, 10, 15, 20, 25 cm | 9.24% | 16.28% |
| Matching radius | 0.30 m | 0.01, 0.05, 0.10, 0.30, 1.00 m | 0.00% | 0.00% |
| Out-of-range height weight | 0.1 | 1, 0.5, 0.1, 0.01, 0 | 0.00% | 0.13% |
| Confidence threshold | 0.3 | 0.2, 0.3, 0.4, 0.5, 0.6, 0.8 | 0.00% | 0.002% |
| Flip ratio | 1.5 | 1, 1.25, 1.5, 2, 3 | 0.00% | 0.016% |

- **Voxel size (10 cm):** selected as approximately twice the mean labelled-return spacing. It was
  the only parameter with a material effect on the final labels.
- **Height-prior weights (1 and 0.1):** retain full weight for evidence inside a class's height
  range and apply a tenfold penalty outside it. A weight of 1 would disable the prior, whereas 0
  would make it a hard exclusion.
- **Confidence threshold (0.3):** accepts a clear winner among up to three candidate classes while
  rejecting near-ties among four or five. Mean Stage-4 confidence was 0.999 at the medium site and
  0.991 at the low site, so the threshold did not bind in these data.
- **Flip ratio (1.5):** requires a 60:40 majority in a two-class case before replacing a label
  inherited from the 2D annotation, preventing changes based on near-ties.

The last three parameters changed at most 0.13% of final labels on any tile. This indicates low
sensitivity around the selected defaults; it does not imply that those defaults were numerically
optimized.

**Supplementary Figure S4 caption.** One-at-a-time sensitivity to matching radius, out-of-range
height weight, confidence threshold, and flip ratio. The top row shows Stage-4 label changes; the
bottom row shows final-label changes relative to the default configuration on the worst tile.
Grey lines mark the selected default values.

---

## Reproducibility

```bash
cd phase_4_3D_lidar_prep/step_3_structure_class/reviewer_response
python sensitivity.py all --workers 16
python figures.py
```