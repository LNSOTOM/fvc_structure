# `fvc3dstruct` — Package API Reference

Python library implementing the Step-3 structure-fusion pipeline. This document covers the
**importable API**: modules, functions, config keys, and return values.

> For the **workflow** — CLI scripts, notebooks, expected inputs and outputs — see
> [`../README_phase4_step3.md`](../README_phase4_step3.md). This file deliberately does not repeat it.

---

## Public API

Everything in `__all__`, importable directly from `fvc3dstruct`:

| Export | Module | Purpose |
|---|---|---|
| `process_tile(z_path, s_path, out_path, config=None)` | `pipeline` | Run all stages on one tile pair; returns a metrics dict |
| `cartesian_voxel_merge(las_z, las_s, voxel_xy=0.10, voxel_z=0.10, return_stats=False)` | `voxel_merge` | Stage 2 — majority-vote label transfer on a Cartesian grid |
| `get_mallee_centroids_watershed(...)` | `watershed` | Stage 3 — crown detection; returns label grid + centroids |
| `refine_with_polar_coords(...)` | `polar_refine` | Stage 4 — crown-centric refinement; returns `(classification, confidence)` |
| `compute_class_height_thresholds(...)` | `height_stats` | Stage 1 — data-driven per-class height ranges |
| `summarise_height_ranges(thr)` | `height_stats` | Height thresholds as a `pandas.DataFrame` |
| `auto_mallee_min_height(thr, fallback=1.0)` | `height_stats` | Derive mallee crown minimum height |
| `print_height_report(thr)` | `height_stats` | Console summary of derived thresholds |
| `DEFAULT_CONFIG` | `config` | Baseline config dict (see below) |
| `read_las(path)` / `write_las(las_obj, out_path)` | `io_las` | LAS I/O helpers |
| `ensure_las14_pf6(las_in)` | `io_las` | Coerce to LAS 1.4 / Point Format 6 |

`process_folder()` is **not** in `__all__` — import it explicitly:

```python
from fvc3dstruct.pipeline import process_folder
```

## Modules

| Module | Lines | Responsibility |
|---|---:|---|
| `pipeline.py` | 448 | Stage orchestration — `process_tile()`, `process_folder()` |
| `watershed.py` | 307 | Crown detection, `auto_watershed_params()`, `plot_mallee_debug()` |
| `polar_refine.py` | 230 | Polar coords, bin indices, `soft_height_prior()`, refinement |
| `io_las.py` | 205 | LAS readers/writers, stage snapshots, ExtraBytes fields |
| `height_stats.py` | 200 | Class height statistics (`match_structure_heights()` then `class_height_ranges()`), `CLASS_NAMES` |
| `voxel_merge.py` | 89 | Cartesian voxel majority vote |
| `config.py` | 44 | `DEFAULT_CONFIG` |
| `treeid.py` | 27 | `assign_treeID_optionA()` — canopy-mallee-only instance IDs |

## `process_tile()` return value

A metrics dict, one entry per pipeline stage. These are the fields the sweep and validation
scripts aggregate:

| Key | Meaning |
|---|---|
| `stage2_labeled_points` / `stage2_labeled_pct` | Points labelled by the voxel transfer |
| `stage2_nonzero_structure_voxels` | Occupied structure voxels at the chosen resolution |
| `stage3_watershed_crowns` | Crowns detected by watershed |
| `stage3_points_in_crowns` | Points falling inside detected crowns |
| `stage4_changed_points` / `stage4_changed_pct` | Labels revised by polar refinement |
| `stage4_mean_confidence` | Mean winning-class probability over evaluated points |
| `stage4_low_confidence_pct` | Share of evaluated points below `min_conf` |
| `stage5_promoted_points` | Unlabelled points promoted by height |
| `stage5_forced_points` | Points forced to mallee inside crowns |
| `stage5_crowns_voted` / `stage5_crowns_mallee_majority` | Crowns put to the majority vote / crowns more than half mallee, which the vote turns wholly mallee |
| `stage5_min_crown_mallee_pct` | Mallee share of the least-mallee crown at the vote (NaN without crowns) |
| `stage5_majority_reclassified_points` | Points reassigned by crown majority |
| `final_veg_points` / `final_veg_pct` | Vegetation retention in the final product |
| `final_treeids` | Distinct non-zero `treeID` values |

**Note on `stage4_mean_confidence`:** it is computed from structure-cloud polar bins and the
point's own `Z_norm`, so it is *independent of `voxel_size_xy`/`voxel_size_z` by construction*.
Identical values across a voxel-size sweep are expected, not a bug — and it is therefore not
evidence that voxel choice is costless for final label quality.

Set `"return_classification_arrays": True` to also get the full per-point Stage-2 baseline,
Stage-4 refined and final classification arrays (`classification_baseline`,
`classification_refined`, `classification_final`), for confusion/transition analyses.

## `DEFAULT_CONFIG`

Defined in `config.py`. Pass a partial dict to `process_tile()`/`process_folder()` to override.

**Fusion geometry**

| Key | Default | Meaning |
|---|---|---|
| `voxel_size_xy` / `voxel_size_z` | `0.10` | Stage-2 voxel resolution (m) |
| `ground_thresh` | `0.15` | Ground exclusion height (m) |
| `height_match_radius` | `0.30` | Stage-1 XY radius within which a structure point takes its nearest Z_norm point's height (m) |
| `r_bin` | `0.10` | Polar radial bin width (m) |
| `theta_bin_deg` | `10.0` | Polar angular bin width (degrees) |

**Label update thresholds**

| Key | Default | Meaning |
|---|---|---|
| `min_conf` | `0.3` | Minimum confidence to fill an empty label |
| `min_flip_ratio` | `1.5` | Ratio required to flip an existing label |
| `height_prior_out_of_range_weight` | `0.1` | Stage-4 height-prior weight for a class outside its 15th–85th percentile `Z_norm` range (in range = 1.0) |
| `max_fill_height` | `2.0` | Max height for radial fill (m) |
| `enable_radial_fill` | `False` | Enable radial gap filling |

**Stage-5 consolidation**

| Key | Default | Meaning |
|---|---|---|
| `merge_65_to_68_znorm` | `0.5` | Woody-debris → mallee merge height (m) |
| `promote_unlabelled_znorm` | `1.3` | Height above which unlabelled points are promoted (m) |

**Crown detection**

| Key | Default | Meaning |
|---|---|---|
| `min_tree_dist` | `1.5` | Minimum distance between crown peaks (m) |
| `smoothing_sigma` | `1.0` | Gaussian smoothing applied to the CHM |
| `mallee_min_znorm` | `None` | Override auto-derived crown minimum height (m) |
| `mallee_force_min_znorm` | `None` | Force in-crown points above this height to mallee (m) |

**Output control**

| Key | Default | Meaning |
|---|---|---|
| `write_outputs` | `True` | Write LAS products; `False` for analysis-only runs |
| `write_full_fused` | `False` | Also write the full fused LAS (all input points) |
| `write_veg_mask_npy` | `False` | Write a boolean vegetation mask aligned to the Z_norm LAS |
| `save_stage_outputs` / `stage_out_dir` | `False` / `None` | Write per-stage snapshots for QC |
| `collapse_treeid_to_one` | `False` | Collapse non-zero `treeID` to `1` on export |
| `return_classification_arrays` | `False` | Include full label arrays in the return dict |

## Importing

No `pyproject.toml` — the package is used from the repo, not installed. Either run from the step
directory, or put it on the path:

```python
import sys
sys.path.insert(0, ".../phase_4_3D_lidar_prep/step_3_structure_class")
from fvc3dstruct.pipeline import process_tile
```

Version: `fvc3dstruct.__version__` (`1.0.0`).

## Runtime dependencies

`laspy` (LAS I/O) · `numpy` · `scipy` (spatial/interpolation) · `pandas` (height summaries) ·
`matplotlib` (only for `watershed.plot_mallee_debug()`)
