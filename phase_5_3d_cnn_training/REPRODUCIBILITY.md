# Phase 5: LiDAR-to-PLY handoff

## Scope and input

Prepare **one** Phase 4 LAS tile as PLY. This does not train a model or build
the full dataset and splits. Input must contain `classification`, `Z_norm`
and `treeID`; the runner fails if these are missing.

## Class conventions

The historical batch notebook explicitly sets PLY IDs 1–5, although the
`PreprocessingConfig` package default uses 0–4. The new single-tile runner
uses the notebook's mapping in both profiles:

| Paper class | 2D ID | LAS classification | PLY `semantic_seg` | Paper model interpretation | HSM exporter model ID |
|---|---:|---:|---:|---|---|
| Forb (FO) | 0 | 64 | 1 | Ground | Rejected |
| Woody debris (WD) | 1 | 65 | 2 | See fusion note below | Rejected |
| Grass hummock (GH) | 2 | 66 | 3 | Ground | 0 |
| Shrub (SH) | 3 | 67 | 4 | Wood | 1 |
| Mallee (MA) | 4 | 68 | 5 | Leaf | 2 |
| Background | — | 0 | Excluded, count reported | Not a growth-form class | Excluded |

The preprint's Sections 3.2 and 4.1 retain native growth-form classes during
2D-to-3D transfer and use ground/wood/leaf for the shared experimental
formulation. This is a structural correspondence for modelling, not literal
bare-ground, wood-material or leaf-material annotation. In its fusion rules,
class 65 points above 0.5 m are reclassified as mallee and class 65 at or below 0.5 m
points become unlabelled; the paper does not specify woody debris as a
separate model class. Phase 5 PLY conversion retains the historical PLY IDs
above; it does not apply the paper's final model interpretation.

Set `class_profile` to `historical_five_class` to retain classes 64–68, or
`three_class_hsm` to stop if a tile contains 64 or 65. Unknown classes are
rejected in either profile. The three-class option does not silently assign or
drop forb or woody debris; decide their treatment scientifically before using
such tiles. Other historical package APIs may silently ignore unknown classes.

For PLYs containing IDs 3/4/5, the current
[ForestFormer3D exporter](../ForestFormer3D/data/ForAINetV2/load_forainetv2_data.py)
maps 3/4/5 to 0/1/2. The
[medium HSM config](../ForestFormer3D/configs/oneformer3d_qs_radius16_qp300_2many_medium_hsm.py)
names these hummock/shrub/mallee; the base forest config names them
ground/wood/leaf. The paper associates these corresponding categories, and
also associates forb with ground. The strict HSM profile rejects forb and
does not by itself reproduce the complete paper formulation.

The exporter maps an exact PLY set 1/4/5 to 0/1/2, but leaves a mixed set
1/3/4/5 unchanged, and leaves a set containing only 1 unchanged. Thus
`historical_five_class` is not a general paper-compatible three-class
export. Some evaluation configs, including
[medium-to-GWL](../ForestFormer3D/configs/oneformer3d_qs_radius16_qp300_2many_medium_to_gwl.py)
and [four-class evaluation](../ForestFormer3D/configs/oneformer3d_qs_radius16_qp300_2many_gwl_eval4gt.py),
map shrub to ground, contrary to the paper's common formulation. No code or
config mapping has been changed here. Inspect actual exported IDs and any
subsequent transforms for each tile; do not assume three numeric IDs alone
establish matching semantics. Confirm the selected dataset, exporter,
checkpoint, label metadata and instance policy for the experiment. In particular, the current
ForestFormer exporter treats semantic ID 0 as background for instances and
sets its instance IDs to 0; other points retain their supplied IDs. The older Phase 5
`forestformer_preprocessing` exporter uses a different `semantic_seg - 1`
conversion; do not interchange the two without validating labels.

## Environment

Use a separate environment. For example:

```text
conda create -n fvc-phase5 -c conda-forge python=3.10 pip
conda activate fvc-phase5
python -m pip install laspy numpy "pandas<3" pyntcloud pyyaml
python -c "import laspy, numpy, pandas, pyntcloud, yaml; print('Phase 5 imports OK')"
```

These are not a locked cross-platform environment. Record your environment and
versions with the run. Use Windows paths with Windows Python and Linux/WSL
paths with Linux/WSL Python; quote Windows paths with single quotes in YAML.
PyntCloud 0.3.1 emitted pandas indexing deprecation warnings during validation
with pandas 2.3.3; the example avoids assuming compatibility with pandas 3.
ForestFormer training has additional dependencies and GPU requirements; see
its [setup](../ForestFormer3D/readme.md) and
[experiment runbook](../ForestFormer3D/README_ForAINetV2_experiments.md).

## Prepare a tile

From the repository root, copy the example and edit all values:

```text
cp phase_5_3d_cnn_training/configs/lidar_to_ply.example.yml phase_5_3d_cnn_training/configs/lidar_to_ply.local.yml
```

Select the class profile and specify a Phase 4 LAS and writable output
directory. The runner refuses to overwrite an existing PLY or offset.

```text
python phase_5_3d_cnn_training/prepare_lidar.py --config phase_5_3d_cnn_training/configs/lidar_to_ply.local.yml
```

The PLY contains `x`, `y`, `z`, `semantic_seg`, `treeID`, `z_norm` and
normalized `intensity`. Coordinates are local to the minimum XYZ of the
**input LAS**; `{stem}_offset.npy` holds that global minimum. Retain the
sidecar and original LAS/CRS metadata: PLY itself does not store a CRS.
Recover global coordinates with `xyz_global = xyz_local + offset`. The
runner checks point counts, classes, tree IDs, heights and reconstruction to
0.1 mm after export. `treeID=0` is unassigned, not an individual tree.

The ForestFormer exporter may recenter PLY coordinates again. Keep its
additional offset separately: reconstruction requires both the PLY offset
and the exporter's recentering offset, unless an export procedure explicitly
records their sum. Check the reconstructed coordinates against the original
LAS before visualization or analysis.

## Validation and model preparation

For DD0001 tile 109, the three-class run wrote 394,473 points; every PLY
column and the offset matched the saved historical PLY exactly. The current
ForestFormer exporter produced 20,158 hummock (0), 3,622 shrub (1), and
370,693 mallee (2) labels on that tile. This checks one tile and the exporter,
not training, evaluation quality, or other sites.

Inspect the PLY fields, class and instance counts, coordinate offsets, CRS and
alignment for each input. A successful conversion does not establish that the
annotations are correct or that all classes have training examples. For model
dataset creation, split selection, training and evaluation, follow the
[ForestFormer3D runbook](../ForestFormer3D/README_ForAINetV2_experiments.md)
and verify the actual exported labels and instance handling before training.
The single-tile command does not create NPZ, NPY training arrays, split lists or
model predictions.

## Rebuild the guide's comparison examples

The [guide](https://lnsotom.github.io/fvc_structure/#model-comparison-heading) displays sampled
historical ForestFormer3D Experiment 1A and 2B results, not Phase 5 PLY inputs. The
medium tiles 93 and 192 are from `exp2B_stage1_eval_medium_L1`, evaluated
with the stage-1 `epoch_3000_fix.pth`. Low tiles 43 and 75 are from
`exp2B_stage2_eval_low_L1`, evaluated with the stage-2 `epoch_3000_fix.pth`.
The forest scenes `SCION_SCION_plot_31_annotated_test.ply` and
`SCION_SCION_plot_61_annotated_test.ply` are from
`zero_shot_forest`, evaluated with the forest-trained
`clean_forestformer/epoch_3000_fix.pth` (Experiment 1A).
These are saved prediction PLYs containing `semantic_gt`, `semantic_pred`,
`instance_gt` and `instance_pred`; the full original evaluation files and
logs are needed to reproduce metrics.

In an environment with NumPy and `plyfile` (included in the ForestFormer3D
Docker environment), replace the three example directories with the saved
evaluation directories and run from the repository root:

```text
python tools/build_model_examples.py --medium-dir PATH_TO_STAGE1_MEDIUM_EVAL --low-dir PATH_TO_STAGE2_LOW_EVAL --forest-dir PATH_TO_ZERO_SHOT_FOREST_EVAL --output docs/figures/model-comparison-data.js
```

The script selects up to 60,000 points per scene with seed 2026, rounds XYZ
to 1 mm and retains their reference and predicted labels. The density control
uses prefixes of this sample, so reference and prediction always show the same
points. Counts in the guide
are calculated from the full PLY. The sample is solely for visualization; it
must not be used to calculate experiment metrics. Instance ID 0 in the
reference is unassigned, whereas prediction ID -1 is unassigned and
prediction ID 0 is a valid instance. Keep the source files, offsets, CRS,
checkpoint and label mappings separately. Instance counts are unique assigned
IDs in the saved PLY, not necessarily the post-processed counts in the paper.

The comparison generator's focused tests also require `plyfile` and NumPy:

```text
python -m unittest discover -s tools/tests -p test_model_comparison_examples.py
```
