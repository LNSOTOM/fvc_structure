# fvc_structure

Fractional vegetation cover / 3D structure pipeline: UAS LiDAR and imagery are processed
phase by phase into structure-classified 3D tiles, which then feed the instance and semantic
segmentation models (ForestFormer3D, SegmentAnyTree).

See the [research workflow guide](docs/index.html) for practical Phase 1–5 instructions.

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

Phases run in order; each consumes the previous phase's outputs. Every phase README documents
its method, contents and usage.

| Phase | Does | Output |
|---|---|---|
| [1 — DSM generation](phase_1_image_processing/README_phase1.md) | PDAL pipeline over LiDAR (`writers.gdal`, max elevation) | DSM GeoTIFF, 0.03 m |
| [2 — Co-registration](phase_2_lidar_coregistration/README_phase2.md) | GeFolki optical flow on DSM pairs, applied to the point clouds | Aligned multi-temporal LiDAR |
| [3 — 2D imagery prep](phase_3_2D_imagery_prep/README_phase3.md) | Ortho tiling, band stacking, FVC → structure labels, rasterization | Tiles + label rasters |
| [4 — 3D LiDAR prep](phase_4_3D_lidar_prep/README_phase4.md) | Clipping, ground classification, 2D→3D label fusion | LAS 1.4 point format 6 or later (`classification` 64–68, `Z_norm`, `treeID`) |
| [5 — Training prep](phase_5_3d_cnn_training/README.md) | LAS → model-ready tensors, predictions back to LAS | PLY / NPZ model inputs |

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
