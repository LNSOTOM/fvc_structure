# fvc_structure

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

## Repository layout

```
fvc_structure/
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

<details>
<summary>Maintainers: enable the guide's clone counter</summary>

The counter shows GitHub clones over the last 14 days, not all-time clones.
Create a fine-grained token restricted to this repository with **Administration: read**
and **Contents: read and write**, and save it as the `REPO_TRAFFIC_TOKEN`
[Actions secret](https://docs.github.com/en/actions/security-for-github-actions/security-guides/using-secrets-in-github-actions).
Its owner must be able to push to the default branch, including under branch protection.
Run **Actions → Update clone statistics → Run workflow** once. The daily workflow commits
the public count to the default branch; the write permission lets that commit trigger Pages.
Keep the token secret. If it expires or a run fails, the last count and timestamp remain.

</details>
