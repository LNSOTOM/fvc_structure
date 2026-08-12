# fvc_structure

Fractional vegetation cover / 3D structure pipeline: UAV LiDAR and imagery are processed
phase by phase into structure-classified 3D tiles, which then feed the instance-segmentation
models (ForestFormer3D, SegmentAnyTree).

## Overview

Data flow through the repository:

```
LiDAR point clouds ──▶ Phase 1  DSM (GeoTIFF, 0.03 m)
                             │
UAV orthomosaics ────────────┼──▶ Phase 2  co-registered LiDAR (optical flow)
                             │                    │
                             └──▶ Phase 3  tiles + 2D structure labels (raster)
                                                  │
                                   Phase 4  ──────┴──▶ structure-classified LAS 1.4 PF6
                                                       (classification 64–68, Z_norm, treeID)
                                                              │
                                   Phase 5  ─────────────────┴──▶ PLY / NPZ model inputs
                                                                        │
                                        ForestFormer3D · SegmentAnyTree ┘  ──▶ instance predictions
```

Contents:

```
fvc_structure/
├── phase_1_image_processing/        # DSM generation from LAS (PDAL)
│   ├── dsm_lidar.py
│   └── notebooks/
├── phase_2_lidar_coregistration/    # Multi-temporal alignment (GeFolki optical flow)
│   ├── gefolki/                     # Upstream library + project scripts
│   └── step_0_checks/               # CRS / sanity checks
├── phase_3_2D_imagery_prep/         # Orthomosaic tiling + 2D structure labels
│   ├── step_1_orthomosaic_to_tiling/
│   ├── step_2_structure_2D_labels/
│   └── documentation/
├── phase_4_3D_lidar_prep/           # 3D structure classification
│   ├── step_1_clipping/             # rasterclip + maskclip packages
│   ├── step_2_csf_znorm/            # groundnorm package (CSF/SMRF + Z_norm)
│   └── step_3_structure_class/      # fvc3dstruct package (fusion, crowns, polar refinement)
├── phase_5_3d_cnn_training/         # LAS -> model-ready PLY/NPZ + reconstruction
│   ├── lidar_preprocessing/         # MINSU3D (NPZ) + ForestFormer3D (PLY)
│   ├── forestformer_preprocessing/
│   └── notebooks/
├── ForestFormer3D/                  # Vendored model + ForAINetV2 experiment docs
├── SegmentAnyTree/                  # Vendored model + multi-GPU inference docs
├── tools/                           # Qualitative results viewer app
├── docs/                            # Repo structure notes + viewer HTML
├── environment.yml / requirements.txt
└── Procfile / wsgi.py               # Deployment for the results viewer
```

## Pipeline phases

Phases run in order — each one consumes the previous phase's outputs. Every phase README
documents its method, contents and usage.

### Phase 1 — DSM generation

[phase_1_image_processing/README_phase1.md](phase_1_image_processing/README_phase1.md)

Digital Surface Models from LiDAR point clouds via a PDAL pipeline (`writers.gdal`,
`output_type: max`, 0.03 m). These DSMs are the input to co-registration.

### Phase 2 — LiDAR–RGB co-registration

[phase_2_lidar_coregistration/README_phase2.md](phase_2_lidar_coregistration/README_phase2.md)

Multi-temporal LiDAR alignment using the GeFolki optical flow algorithm on Phase 1 DSMs:
preprocess DSMs → compute flow between temporal pairs → apply the displacement field to the
point clouds → validate alignment.

- Upstream GeFolki library and citations: [phase_2_lidar_coregistration/gefolki/README.md](phase_2_lidar_coregistration/gefolki/README.md)

### Phase 3 — 2D imagery preparation

[phase_3_2D_imagery_prep/README_phase3.md](phase_3_2D_imagery_prep/README_phase3.md)

Orthomosaics clipped into a regular grid of tiles, multispectral bands stacked, and FVC
annotations converted into structure classes and rasterized to tile resolution.

- Step 2 — structure classification of 2D labels: [phase_3_2D_imagery_prep/step_2_structure_2D_labels/README.md](phase_3_2D_imagery_prep/step_2_structure_2D_labels/README.md)

### Phase 4 — 3D LiDAR preparation

[phase_4_3D_lidar_prep/README_phase4.md](phase_4_3D_lidar_prep/README_phase4.md)

Co-registered point clouds clipped to the 2D tiles, ground-classified and height-normalized,
then fused with the 2D structure masks (voxel transfer + crown-centric polar refinement) to
produce structure-classified LAS 1.4 PF6 tiles.

- Step 1 — clipping to raster tiles and mask classification: [step_1_clipping/README_phase4_step1.md](phase_4_3D_lidar_prep/step_1_clipping/README_phase4_step1.md)
- Step 2 — ground classification (CSF/SMRF) and `Z_norm`: [step_2_csf_znorm/README_phase4_step2.md](phase_4_3D_lidar_prep/step_2_csf_znorm/README_phase4_step2.md)
- Step 3 — structure fusion workflow guide: [step_3_structure_class/README_phase4_step3.md](phase_4_3D_lidar_prep/step_3_structure_class/README_phase4_step3.md)
- Step 3 — `fvc3dstruct` package API reference: [step_3_structure_class/fvc3dstruct/README.md](phase_4_3D_lidar_prep/step_3_structure_class/fvc3dstruct/README.md)

### Phase 5 — 3D CNN training preparation

[phase_5_3d_cnn_training/README.md](phase_5_3d_cnn_training/README.md)

Phase 4 LAS tiles converted into model-ready inputs (ForestFormer3D PLY in local tile
coordinates, NPZ/NPY tensors), plus reconstruction of predictions back to LAS for QC.

- `lidar_preprocessing` package (NPZ for MINSU3D + PLY for ForestFormer3D): [phase_5_3d_cnn_training/lidar_preprocessing/README.md](phase_5_3d_cnn_training/lidar_preprocessing/README.md)

## Models

The sections below cover running the segmentation models on the Phase 5 outputs.

## SegmentAnyTree (ForAINetV2 / FOR-instanceV2)

- Multi-GPU Docker inference guide: [SegmentAnyTree/README_ForAINetV2_multi_gpu_inference.md](SegmentAnyTree/README_ForAINetV2_multi_gpu_inference.md)

## ForestFormer3D (ForAINetV2)

- Environment setup / Docker guide: [ForestFormer3D/readme.md](ForestFormer3D/readme.md)
- Quick setup (enter container → data prep → train → test): [ForestFormer3D/README_quick_setup.md](ForestFormer3D/README_quick_setup.md)
- ForAINetV2 experiment runbook: [ForestFormer3D/README_ForAINetV2_experiments.md](ForestFormer3D/README_ForAINetV2_experiments.md)

