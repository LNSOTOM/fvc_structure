# Phase 5: LiDAR-to-PLY handoff

## Scope and input

This handoff prepares **one** Phase 4 structure-classified LAS tile as PLY.
It does not train a model, create train/validation splits, or build the full
ForestFormer3D dataset. Input must contain LAS `classification`, `Z_norm`
and `treeID`; the runner fails rather than substituting approximate heights or
zero IDs when those dimensions are missing.

## Class conventions

The historical batch notebook explicitly sets PLY IDs 1–5, although the
`PreprocessingConfig` package default uses 0–4. The new single-tile runner
uses the notebook's mapping in both profiles:

| LAS classification | Vegetation | PLY `semantic_seg` | Three-class HSM exporter label |
|---|---|---|---|
| 64 | Forb | 1 | Rejected by the HSM profile |
| 65 | Woody debris | 2 | Rejected by the HSM profile |
| 66 | Hummock | 3 | 0 |
| 67 | Shrub | 4 | 1 |
| 68 | Mallee | 5 | 2 |
| 0 | Background | Excluded, count reported | Excluded |

Set `class_profile` to `historical_five_class` to retain classes 64–68, or
`three_class_hsm` to stop if a tile contains 64 or 65. Unknown classes are
rejected in either profile. The three-class option does not silently assign or
drop forb or woody debris; decide their treatment scientifically before using
such tiles. Other historical package APIs may silently ignore unknown classes.

For PLYs containing only IDs 3–5, the current
[ForestFormer3D exporter](../ForestFormer3D/data/ForAINetV2/load_forainetv2_data.py)
maps 3/4/5 to 0/1/2. The
[medium HSM config](../ForestFormer3D/configs/oneformer3d_qs_radius16_qp300_2many_medium_hsm.py)
names these hummock/shrub/mallee. This is **not** the base forest model's
ground/wood/leaf taxonomy; three numeric IDs alone do not make those meanings
interchangeable. Confirm the selected dataset, exporter, checkpoint, label
metadata and instance policy for the experiment. In particular, the current
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
