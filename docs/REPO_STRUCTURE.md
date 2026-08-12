# fvc_structure — Repository Structure & Reuse Map

Proposal to bring `fvc_structure` in line with the layout convention used in
[`fvc_composition`](https://github.com/LNSOTOM/fvc_composition), so the pipeline is navigable by
phase and reusable by people who did not write it.

Status: **proposal — not yet applied.** Nothing has been moved.

---

## 1. Where the repo is today

| Measure | Value | Why it matters |
|---|---:|---|
| Python files (own code) | 163 | — |
| Files with hardcoded absolute paths | **75 (46%)** | Cannot run on any machine but the author's |
| Versioned / scratch duplicates (`_v0`, `_v1`, ` copy`, `test/`, `testing/`) | **54** | A newcomer cannot tell which file is current |
| Config modules | 2 (`fvc3dstruct/config.py`, `lidar_preprocessing/config.py`) | No single place to point at data |
| READMEs | 15, inconsistent | No predictable per-stage contract |
| Vendored upstreams at root | 2 (`ForestFormer3D/`, `SegmentAnyTree/`) | Third-party code is indistinguishable from ours |
| Declared deps | `requirements.txt` (46 bytes) + `environment.yml` | Two sources of truth, one near-empty |

The pipeline logic itself is sound and already phase-ordered. The barrier to reuse is **not** the
science — it is path hardcoding, unclear file provenance, and no shared entry-point convention.

### The one pattern already worth copying

`phase_4_3D_lidar_prep/step_3_structure_class/fvc3dstruct/` is the only stage packaged properly:

```
fvc3dstruct/
  __init__.py     config.py     io_las.py      pipeline.py
  polar_refine.py voxel_merge.py watershed.py  treeid.py  height_stats.py
```

Config separated from I/O, from algorithm, from orchestration. **This is the target shape for every
stage**, and the migration below generalises it rather than inventing something new.

---

## 2. Proposed structure

Phase numbers are preserved 1:1 with the current `phase_N_*` directories, so existing mental models
and all documentation references still hold.

```
configs/
  paths.yaml                  Data roots, model weights, output locations (per-site)
  preprocessing.yaml          Clipping, CSF/z-norm, tiling parameters
  structure_class.yaml        Voxel sizes, polar-refinement, min_conf thresholds
  models.yaml                 Training/inference config pointers

src/
  01_image_processing/        DSM generation, voxel thinning, raycloud conversion
  02_coregistration/          GeFolki optical-flow LiDAR-imagery alignment, CRS checks
  03_imagery_2d_prep/         Orthomosaic clipping, tiling, 2D structure labels, downsampling
  04_lidar_3d_prep/
    01_clipping/              Raster + mask clipping to tiles
    02_csf_znorm/             Ground filtering and height normalisation
    03_structure_class/       Voxel/polar fusion -> structure classes  (fvc3dstruct)
  05_model_training/
    01_datasets/              ForAINetV2 / MinSU3D / benchmark converters, voxel NPZ
    02_forestformer3d/        Config + runbook wrappers around the vendored upstream
    03_segmentanytree/        Config + runbook wrappers around the vendored upstream
  06_evaluation/              Clustering, occupancy metrics, qualitative viewer, comparison
  utils/                      paths.py, io_las.py, crs.py, manifests.py, logging

external/
  ForestFormer3D/             Vendored upstream (unmodified)
  SegmentAnyTree/             Vendored upstream (unmodified)

notebooks/                    Mirrors src/ phase numbering
archive/                      Superseded versions, kept out of the working tree
docs/                         This file, workflow overviews, viewer app
environment.yml               Single source of truth for the environment
```

### Why not collapse phases 1–4 into `01_preprocessing/`

`fvc_composition` uses coarse buckets (`01_preprocessing`, `02_dino_distil`, `03_models`,
`04_evaluation`) because its preprocessing is one stage deep. Here, phases 1–4 are four genuinely
distinct stages with their own inputs, outputs and failure modes — collapsing them would bury
`04_lidar_3d_prep`'s three sub-steps three levels down and lose the phase numbering that every
existing README already cites. The numbered top level keeps the convention while matching this
project's actual depth.

---

## 3. Migration map

Use `git mv` throughout so history follows the files.

| Current | Proposed | Notes |
|---|---|---|
| `phase_1_image_processing/dsm_lidar.py` | `src/01_image_processing/dsm_lidar.py` | |
| `phase_1_image_processing/testing/*` | `archive/01_image_processing/` | 5 scratch scripts |
| `phase_2_lidar_coregistration/gefolki/python/` | `external/gefolki/` | Upstream GeFolki library |
| `phase_2_lidar_coregistration/gefolki/python_myscripts/` | `src/02_coregistration/` | **Current** versions |
| `phase_2_lidar_coregistration/gefolki/*.py` (root) | *delete* | **Verified byte-identical** to `python_myscripts/` (all 5 files) — safe to drop |
| `phase_2_lidar_coregistration/gefolki/testing/` | `archive/02_coregistration/` | 11 `_v0`/`_v1` files |
| `phase_2_lidar_coregistration/step_0_checks/crs.py` | `src/utils/crs.py` | Shared — used by several phases |
| `phase_3_2D_imagery_prep/step_1_orthomosaic_to_tiling/` | `src/03_imagery_2d_prep/01_tiling/` | |
| `phase_3_2D_imagery_prep/step_2_structure_2D_labels/` | `src/03_imagery_2d_prep/02_labels/` | |
| `phase_4_3D_lidar_prep/step_1_clipping/{maskclip,rasterclip}` | `src/04_lidar_3d_prep/01_clipping/` | |
| `phase_4_3D_lidar_prep/step_1_clipping/test/` | `archive/04_lidar_3d_prep/` | 4 versioned files |
| `phase_4_3D_lidar_prep/step_2_csf_znorm/groundnorm/` | `src/04_lidar_3d_prep/02_csf_znorm/` | |
| `phase_4_3D_lidar_prep/step_2_csf_znorm/test/` | `archive/04_lidar_3d_prep/` | 4 versioned files |
| `phase_4_3D_lidar_prep/step_3_structure_class/fvc3dstruct/` | `src/04_lidar_3d_prep/03_structure_class/fvc3dstruct/` | **Keep as-is — reference package** |
| `phase_4_3D_lidar_prep/step_3_structure_class/merge_znorm_*.py` (13 variants) | see §4 | Needs consolidation, not a move |
| `phase_5_3d_cnn_training/lidar_preprocessing/` | `src/05_model_training/01_datasets/` | Already has `config.py` |
| `phase_5_3d_cnn_training/forestformer_preprocessing/` | `src/05_model_training/02_forestformer3d/` | |
| `phase_5_3d_cnn_training/make_voxel_npz*.py`, `read_*.py` | `src/05_model_training/01_datasets/` | |
| `phase_5_3d_cnn_training/test/` | `archive/05_model_training/` | |
| `phase_cluster/*.py` | `src/06_evaluation/clustering/` | |
| `tools/qualitative_results_app.py`, `docs/*.html` | `src/06_evaluation/viewer/` | Keep `Procfile`/`wsgi.py` beside it |
| `ForestFormer3D/`, `SegmentAnyTree/` | `external/` | Unmodified vendored upstreams |
| `phase_*/notebooks/` | `notebooks/0N_*/` | Mirror phase numbering |

---

## 4. The `merge_znorm_*` cluster

`step_3_structure_class/` contains **13 near-identical** `merge_znorm_*` scripts differing by
variant (`polar`, `polar_crown`, `polar_vegid`, `polar_malle_id`, `polar_watershed`,
`voxel_density`, `onefile`, …), two of which are literally ` copy` files. This is the single least
approachable part of the repo for a newcomer.

They should not be moved as-is. The `fvc3dstruct` package already contains the shared machinery
(`voxel_merge`, `polar_refine`, `watershed`, `treeid`), so the variants collapse into **one entry
point with config-selected strategy**:

```bash
python -m fvc3dstruct.merge --config configs/structure_class.yaml --variant polar_vegid
```

Each current script becomes a named preset in `configs/structure_class.yaml`. That is a
consolidation task in its own right — recommended, but separable from the directory move.

---

## 5. What actually unblocks reuse

Directory moves alone change nothing for a new user. These three do:

**5.1 `configs/paths.yaml` + `src/utils/paths.py`.** Replaces the 75 files with hardcoded roots.

```yaml
sites:
  site1_1_DD0001:
    lidar_raw:  ${data_root}/site1_1_DD0001/lidar/raw
    z_norm:     ${data_root}/site1_1_DD0001/lidar/clipped/raster/z_norm/csf
    mask_coreg: ${data_root}/site1_1_DD0001/lidar/clipped/mask/coreg/icp_top
data_root: /home/uqlsoto1/data
outputs:    ./outputs
```

Migrate opportunistically: a script gets converted when it is next touched. A hardcoded path that
still works is not urgent; a hardcoded path in a script someone else needs is blocking.

**5.2 A uniform per-stage README.** Every `src/0N_*/` gets the same five headings — *Purpose*,
*Inputs*, *Outputs*, *Run*, *Config*. `step_3_structure_class/README.md` is the closest existing
example. Predictability matters more than depth: a reader who has read one stage's README should
know where to look in every other.

**5.3 One CLI convention.** Currently entry points are `main_run.py`, `compare_voxel_sizes.py`,
bare scripts, and notebooks — no shared pattern. Standardise on `argparse` +
`--config configs/<stage>.yaml`, with notebooks importing the same functions rather than
reimplementing them (`04_voxel_size_sensitivity_and_label_confidence.ipynb` already does this
correctly by importing `compare_voxel_sizes`).

---

## 6. Suggested order

Each step is independently useful; stopping after any of them leaves the repo better than before.

1. **`archive/` the 54 superseded files.** Highest clarity-per-effort — halves the apparent surface
   area. Pure `git mv`, no code changes, nothing breaks.
2. **Move vendored upstreams to `external/`.** Makes "ours vs theirs" obvious at a glance.
3. **Create `src/` and move phases in.** Mechanical; update imports as they break.
4. **Add `configs/paths.yaml` + `utils/paths.py`.** Convert scripts opportunistically thereafter.
5. **Normalise the READMEs** to the five-heading template.
6. **Consolidate `merge_znorm_*`** into the config-driven entry point (§4).

Steps 1–3 are reversible and low-risk. Steps 4–6 change behaviour and want testing per stage.

---

## 7. Open questions

- **`archive/` or delete?** Archiving keeps the tree clean while preserving recovery; the files are
  in git history regardless, so deletion is defensible if you are confident they are dead.
- **Is `SegmentAnyTree/` still active?** If it has been superseded by ForestFormer3D, it could move
  to `archive/` rather than `external/`.
- **Site coverage in `paths.yaml`** — how many sites beyond `site1_1_DD0001` need entries?
