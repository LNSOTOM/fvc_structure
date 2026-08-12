# fvc_structure

Fractional vegetation cover / 3D structure pipeline: UAV LiDAR and imagery are processed
phase by phase into structure-classified 3D tiles, which then feed the instance-segmentation
models (ForestFormer3D, SegmentAnyTree).

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
- ForAINetV2 experiment runbook: [ForestFormer3D/README_ForAINetV2_experiments.md](ForestFormer3D/README_ForAINetV2_experiments.md)

## Quickstart (ForestFormer3D Docker)

This is a minimal, copy-paste workflow showing:
- how to enter the running container
- data preparation (ForAINetV2)
- training
- testing (your own checkpoint or the provided checkpoint)

### 0) Enter the running container (host)

Example host session:

```bash
nvidia@brev-rvbt83sj0:~/projects$ cd /home/nvidia/projects/fvc_structure/ForestFormer3D

# Enter the running container (use sudo if your Docker requires it)
nvidia@brev-rvbt83sj0:~/projects/fvc_structure/ForestFormer3D$ sudo docker exec -it forestformer3d-container /bin/bash
```


Inside the container you should be in `/workspace` (the ForestFormer3D repo root):

```bash
root@<container-id>:/workspace# pwd
/workspace
```

### 1) Data preparation (inside container)

Ensure these folders exist under `/workspace/data/ForAINetV2`:
- `meta_data`
- `train_val_data` (put your train/val `.ply` here)
- `test_data` (put your test `.ply` here)

Then run preprocessing:

```bash
root@<container-id>:/workspace# cd data/ForAINetV2
root@<container-id>:/workspace/data/ForAINetV2# python batch_load_ForAINetV2_data.py

# Creates/updates:
#   data/ForAINetV2/forainetv2_instance_data
#   - train.npy
#   - val.npy
#   - test.npy

root@<container-id>:/workspace/data/ForAINetV2# cd /workspace
root@<container-id>:/workspace# python tools/create_data_forainetv2.py forainetv2

# Creates/updates (paths under data/ForAINetV2):
#   points/
#   instance_mask/
#   semantic_mask/
#   forainetv2_oneformer3d_infos_{train,val,test}.pkl
```

### 2) Train (inside container)

Multi-GPU training example (3 GPUs exposed):

```bash
root@<container-id>:/workspace# export PYTHONPATH=/workspace

root@<container-id>:/workspace# CUDA_VISIBLE_DEVICES=0,1,2 \
	bash tools/dist_train.sh \
	configs/oneformer3d_qs_radius16_qp300_2many.py \
	3 \
	--work-dir work_dirs/train_forestformer_3gpu
```

### 3) Test your own trained checkpoint (inside container)

If you hit spconv incompatibilities, fix the checkpoint once:

```bash
root@<container-id>:/workspace# python tools/fix_spconv_checkpoint.py \
	--in-path work_dirs/train_forestformer_3gpu/epoch_10.pth \
	--out-path work_dirs/train_forestformer_3gpu/epoch_10_fix.pth
```

Run single-GPU test:

```bash
root@<container-id>:/workspace# CUDA_VISIBLE_DEVICES=0 \
	python tools/test.py \
	configs/oneformer3d_qs_radius16_qp300_2many.py \
	work_dirs/train_forestformer_3gpu/epoch_10_fix.pth
```

#### Output directory for predictions (no code edits)

Prediction outputs are controlled by `model.test_cfg.output_dir` (see configs).
If you want a specific output folder, prefer changing it in the config or overriding via `--cfg-options`.

Example override:

```bash
root@<container-id>:/workspace# CUDA_VISIBLE_DEVICES=0 \
	python tools/test.py \
	configs/oneformer3d_qs_radius16_qp300_2many.py \
	work_dirs/train_forestformer_3gpu/epoch_10_fix.pth \
	--cfg-options model.test_cfg.output_dir=work_dirs/test_results
```

### 4) Test the provided pre-trained checkpoint (inside container)

Single-GPU:

```bash
root@<container-id>:/workspace# CUDA_VISIBLE_DEVICES=0 \
	python tools/test.py \
	configs/oneformer3d_qs_radius16_qp300_2many.py \
	work_dirs/clean_forestformer/epoch_3000_fix.pth
```

Multi-GPU distributed test:

```bash
root@<container-id>:/workspace# CUDA_VISIBLE_DEVICES=0,1,2,3,4,6,7 \
	bash tools/dist_test.sh \
	configs/oneformer3d_qs_radius16_qp300_2many.py \
	work_dirs/clean_forestformer/epoch_3000_fix.pth \
	7
```

### 5) Useful utilities (metrics + plots)

Summarize validation metrics from logs:

```bash
root@<container-id>:/workspace# python3 /workspace/tools/print_val_metrics_summary.py \
	--log-glob "/workspace/work_dirs/exp2B_stage2_low_train_from_ep2400/*/*.log" \
	--epoch 3000 \
	--top-k 5
```

Plot train/val curves:

```bash
root@<container-id>:/workspace# python3 /workspace/tools/plot_train_val_curves.py \
	--log-glob "/workspace/work_dirs/exp2B_stage2_low_train_from_ep2400/*/*.log" \
	--out-dir "/workspace/work_dirs/exp2B_stage2_low_train_from_ep2400/plots"
```