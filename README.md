# fvcStructureRangeland3D

UAS LiDAR and imagery are prepared as structure-labelled 3D tiles for
ForestFormer3D and SegmentAnyTree.

See the [research workflow guide](https://lnsotom.github.io/fvc_structure/) for practical Phase 1–5 instructions.

## Citation

If you use this code, please cite the paper that describes it:

> Sotomayor, L.N., Lucieer, A., Turner, D., Barthelemy, J., Kattenborn, T. (2026).
> From 2D labels to 3D structure: Scalable label transfer and benchmarking of 3D
> vegetation models in rangeland ecosystems. *EarthArXiv* (preprint).
> https://doi.org/10.31223/X5948M

GitHub's **Cite this repository** button gives the same reference (see [CITATION.cff](CITATION.cff)).

## Data and models

Labelled 3D tiles, label-transfer data, split lists, the six pre-trained
ForestFormer3D models and test predictions are on Zenodo:
[doi:10.5281/zenodo.24431573](https://doi.org/10.5281/zenodo.24431573).
UAS imagery and FVC labels: [doi:10.5281/zenodo.15036860](https://doi.org/10.5281/zenodo.15036860).

```
LiDAR + UAS imagery → DSM → co-registration → 2D tiles + labels
                          → structure-classified LAS → model inputs → instance + semantic predictions
```

## Python API (`fvc3d`)

The `fvc3d` package turns a label raster and a LiDAR point cloud into a
labelled 3D point cloud in one call. It runs the Phase 4 label-transfer method.

```bash
uv sync            # or: pip install -e .
```

```python
from fvc3d import transfer_labels

result = transfer_labels(
    labels="growth_forms.tif",        # codes 64–68; class_map="paper" for the published 0–4 rasters
    point_cloud="lidar.laz",          # same CRS, already co-registered to the imagery
    output="labelled_vegetation.laz",
    validate_alignment=True,          # raises AlignmentError if CRS, overlap or offset fail
)
print(result.summary())
```

**Inputs.** The raster holds forb (64), woody debris (65), hummock (66),
shrub (67) and mallee (68). For the published rasters on Zenodo, which use
0–4, pass `class_map="paper"`; for other codes pass a dict such as
`{1: "shrub", 2: "mallee"}`. The point cloud needs height above ground: a
`Z_norm` dimension, ground points classified as 2, or a ground-only file
passed as `ground=`.

**Your own classes.** The paper's five growth forms are the default. For
other classes, copy [configs/classes.yaml](configs/classes.yaml), edit the
codes, names, height ranges and crown class, and pass
`classes="my_classes.yaml"`. `class_map` maps your raster values to those
class names. `classes="FOR-instanceV2"` runs the paper's method and writes
FOR-instanceV2's ground (1), wood (2) and leaf (3) classes, the classes
ForestFormer3D is trained on.

**Output.** The LAS/LAZ file holds the labelled points, with `classification`,
`Z_norm` and a `treeID` for each mallee crown. Pass `keep="all"` to keep
every point.

**Alignment check.** `fvc3d` does not co-register data; see Phase 2 for that.
The check needs the same projected CRS in metres and overlapping data. It also
measures the horizontal offset between the LiDAR canopy (Z_norm ≥ 1 m) and
the mallee labels, and fails above `max_shift` (default 0.50 m, i.e. 10 pixels at 5 cm).

**Settings.** Pass a `config` dict or a YAML file
([configs/transfer_labels.yaml](configs/transfer_labels.yaml)). The defaults
are the paper's, including `"auto"` crown detection, the setting behind the
published outputs.

| Module | What it does |
|---|---|
| `fvc3d.io` | Read label rasters and LAS/LAZ; write labelled LAS/LAZ |
| `fvc3d.registration` | `check_alignment()`: CRS, overlap and canopy offset |
| `fvc3d.label_transfer` | Label points from the raster, compute `Z_norm`, voxel majority vote |
| `fvc3d.refinement` | Class height ranges, watershed crowns, polar refinement, consolidation |
| `fvc3d.quality_control` | The paper's diagnostics: voxel-level label agreement, height plausibility of refined classes, crown-instance retention |
| `fvc3d.pipeline` | `transfer_labels()` for files; `fuse_labels()` for arrays in memory |

See [examples/transfer_labels.py](examples/transfer_labels.py) for one tile or
a folder of tiles.

`fvc3d` gives the same labels as `fvc3dstruct` (the code behind the paper),
with one fix. In `fvc3dstruct`, points beyond the crown-detection grid can
take an edge crown's ID, and the crown vote can then turn them into mallee.
`fvc3d` gives them no crown. On tile 109 this affects 274 ground points. Set
`config={"legacy_crown_clamp": True}` to reproduce `fvc3dstruct` exactly.

## Repository layout

```
fvc_structure/
├── src/fvc3d/                       # Python API: transfer_labels() and its stages
├── configs/ examples/               # fvc3d settings and usage examples
├── phase_1_image_processing/        # DSM generation from LAS (PDAL)
├── phase_2_lidar_coregistration/    # gefolki/ (optical flow), step_0_checks/ (CRS)
├── phase_3_2D_imagery_prep/         # step_1 tiling, step_2 2D structure labels
├── phase_4_3D_lidar_prep/           # step_1 clipping, step_2 csf_znorm, step_3 structure_class
├── phase_5_3d_cnn_training/         # lidar_preprocessing/, forestformer_preprocessing/
├── ForestFormer3D/                  # Vendored model + ForAINetV2 docs
├── SegmentAnyTree/                  # Vendored model + inference docs
├── tools/ + docs/                   # Qualitative results viewer app and HTML
└── environment.yml, requirements.txt, Procfile, wsgi.py
```

## Pipeline phases

Follow the phase links for methods and the handoffs for reproducible commands.
Phase 4 requires an ICP-registered tile; Phase 2 GeFolki is not ICP.

| Phase | Output | Handoff |
|---|---|---|
| [1 — DSM generation](phase_1_image_processing/README_phase1.md) | Maximum-elevation DSM GeoTIFF | [Phase 1](phase_1_image_processing/REPRODUCIBILITY.md) |
| [2 — Co-registration](phase_2_lidar_coregistration/README_phase2.md) | GeFolki-aligned LiDAR | [Phase 2](phase_2_lidar_coregistration/REPRODUCIBILITY.md) |
| [3 — 2D imagery prep](phase_3_2D_imagery_prep/README_phase3.md) | Imagery tiles and label rasters | [Phase 3](phase_3_2D_imagery_prep/REPRODUCIBILITY.md) |
| [4 — 3D LiDAR prep](phase_4_3D_lidar_prep/README_phase4.md) | Structure-classified LAS with `Z_norm` and `treeID` | [Phase 4](phase_4_3D_lidar_prep/REPRODUCIBILITY.md) |
| [5 — Model input prep](phase_5_3d_cnn_training/README.md) | Labelled PLY and coordinate offset | [Phase 5](phase_5_3d_cnn_training/REPRODUCIBILITY.md) |

Step and package docs:

- Phase 2 — [GeFolki library and citations](phase_2_lidar_coregistration/gefolki/README.md)
- Phase 3 — [structure classification of 2D labels](phase_3_2D_imagery_prep/step_2_structure_2D_labels/README.md)
- Phase 4 — [step 1 clipping](phase_4_3D_lidar_prep/step_1_clipping/README_phase4_step1.md) · [step 2 CSF + `Z_norm`](phase_4_3D_lidar_prep/step_2_csf_znorm/README_phase4_step2.md) · [step 3 workflow](phase_4_3D_lidar_prep/step_3_structure_class/README_phase4_step3.md) · [`fvc3dstruct` API](phase_4_3D_lidar_prep/step_3_structure_class/fvc3dstruct/README.md)
- Phase 5 — [`lidar_preprocessing` package](phase_5_3d_cnn_training/lidar_preprocessing/README.md)


## Models

Run on the Phase 5 outputs.

**ForestFormer3D (ForAINetV2)** — unified, end-to-end framework for precise individual tree
(instance) and semantic segmentation.

- [Environment setup / Docker](ForestFormer3D/readme.md)
- [Quick setup](ForestFormer3D/README_quick_setup.md) — enter container → data prep → train → test
- [ForAINetV2 experiment runbook](ForestFormer3D/README_ForAINetV2_experiments.md)

**SegmentAnyTree (ForAINetV2 / FOR-instanceV2)**

- [Environment setup / Docker](SegmentAnyTree/README.md) — also on [GitHub](https://github.com/LNSOTOM/fvc_structure/tree/main/SegmentAnyTree)
- [Multi-GPU Docker inference](SegmentAnyTree/README_ForAINetV2_multi_gpu_inference.md)

## Licence

The original code in this repository is released under the [MIT License](LICENSE).
Third-party code keeps its own licence: [ForestFormer3D](ForestFormer3D/LICENSE)
(CC BY-NC 4.0), [SegmentAnyTree](SegmentAnyTree/LICENSE) (MIT, with BSD-licensed
parts) and [GeFolki](phase_2_lidar_coregistration/gefolki/LICENSE) (GPL-3.0).
Data and models are licensed separately in their Zenodo records.
