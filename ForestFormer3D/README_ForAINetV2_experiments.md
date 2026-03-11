# ForestFormer3D — ForAINetV2 experiments (baseline + calperum)

This README collects the **exact commands** used to run ForestFormer3D zero-shot inference on:
- **Baseline ForAINetV2**
- **Calperum**: `low_L1`, `low_L2`, `medium_L1`, `medium_L2`

These commands are intended to be run **inside the ForestFormer3D Docker container**, where the repo is mounted at `/workspace`.

## Paper experiment mapping

All commands in this file are for:

**Experiment 1: Zero-shot transfer performance of ForestFormer3D from forest training data to unseen forest and semi-arid rangeland ecosystem test sets**

## Notes (important)

- In bash, a line-continuation backslash `\` **must be the last character on the line** (no trailing spaces/tabs after it).
- `CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,7` exposes **7 GPUs** to the process, so `--gpus 7` is consistent.
- The preprocessed instance data directory referenced below already exists in this repo at:
  - `/workspace/data/ForAINetV2/forainetv2_instance_data_forest`

## Common settings

All experiments below use:
- `FORAINETV2_PROTOCOL=forest`
- `FORAINETV2_INSTANCE_DATA_DIR=forainetv2_instance_data_forest`
- `FORAINETV2_BIN_BASE_DIR=forainetv2_instance_data_forest`
- Config: `/workspace/configs/oneformer3d_qs_radius16_qp300_2many.py`
- Checkpoint: `/workspace/work_dirs/clean_forestformer/epoch_3000_fix.pth`

## Experiment 1.1 — Baseline ForAINetV2 (forest)

```bash
FORAINETV2_PROTOCOL=forest \
FORAINETV2_INSTANCE_DATA_DIR=forainetv2_instance_data_forest \
FORAINETV2_BIN_BASE_DIR=forainetv2_instance_data_forest \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,7 \
python /workspace/tools/zero_shot_infer_split.py \
  --split forest \
  --scene-list /workspace/data/ForAINetV2/meta_data/test_list.txt \
  --ply-dir /workspace/data/ForAINetV2/test_data \
  --extra-tag forainetv2_baseline_forest \
  --config /workspace/configs/oneformer3d_qs_radius16_qp300_2many.py \
  --checkpoint /workspace/work_dirs/clean_forestformer/epoch_3000_fix.pth \
  --work-dir /workspace/work_dirs/zero_shot_baseline_forest \
  --dist --gpus 7 \
  --overwrite-pkl
```

## Experiment 1.2 — Calperum low_L1 (forest)

```bash
FORAINETV2_PROTOCOL=forest \
FORAINETV2_INSTANCE_DATA_DIR=forainetv2_instance_data_forest \
FORAINETV2_BIN_BASE_DIR=forainetv2_instance_data_forest \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,7 \
python /workspace/tools/zero_shot_infer_split.py \
  --split forest \
  --scene-list /workspace/data/ForAINetV2/meta_data/test_list_low_L1.txt \
  --ply-dir /workspace/data/ForAINetV2/test_data_low_L1 \
  --extra-tag forainetv2_low_L1_forest \
  --config /workspace/configs/oneformer3d_qs_radius16_qp300_2many.py \
  --checkpoint /workspace/work_dirs/clean_forestformer/epoch_3000_fix.pth \
  --work-dir /workspace/work_dirs/zero_shot_low_L1_forest \
  --dist --gpus 7 \
  --overwrite-pkl
```

## Experiment 1.3 — Calperum low_L2 (forest)

```bash
FORAINETV2_PROTOCOL=forest \
FORAINETV2_INSTANCE_DATA_DIR=forainetv2_instance_data_forest \
FORAINETV2_BIN_BASE_DIR=forainetv2_instance_data_forest \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,7 \
python /workspace/tools/zero_shot_infer_split.py \
  --split forest \
  --scene-list /workspace/data/ForAINetV2/meta_data/test_list_low_L2.txt \
  --ply-dir /workspace/data/ForAINetV2/test_data_low_L2 \
  --extra-tag forainetv2_low_L2_forest \
  --config /workspace/configs/oneformer3d_qs_radius16_qp300_2many.py \
  --checkpoint /workspace/work_dirs/clean_forestformer/epoch_3000_fix.pth \
  --work-dir /workspace/work_dirs/zero_shot_low_L2_forest \
  --dist --gpus 7 \
  --overwrite-pkl
```

## Experiment 1.4 — Calperum medium_L1 (forest)

```bash
FORAINETV2_PROTOCOL=forest \
FORAINETV2_INSTANCE_DATA_DIR=forainetv2_instance_data_forest \
FORAINETV2_BIN_BASE_DIR=forainetv2_instance_data_forest \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,7 \
python /workspace/tools/zero_shot_infer_split.py \
  --split forest \
  --scene-list /workspace/data/ForAINetV2/meta_data/test_list_medium_L1.txt \
  --ply-dir /workspace/data/ForAINetV2/test_data_medium_L1 \
  --extra-tag forainetv2_medium_L1_forest \
  --config /workspace/configs/oneformer3d_qs_radius16_qp300_2many.py \
  --checkpoint /workspace/work_dirs/clean_forestformer/epoch_3000_fix.pth \
  --work-dir /workspace/work_dirs/zero_shot_medium_L1_forest \
  --dist --gpus 7 \
  --overwrite-pkl
```

## Experiment 1.5 — Calperum medium_L2 (forest)

```bash
FORAINETV2_PROTOCOL=forest \
FORAINETV2_INSTANCE_DATA_DIR=forainetv2_instance_data_forest \
FORAINETV2_BIN_BASE_DIR=forainetv2_instance_data_forest \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,7 \
python /workspace/tools/zero_shot_infer_split.py \
  --split forest \
  --scene-list /workspace/data/ForAINetV2/meta_data/test_list_medium_L2.txt \
  --ply-dir /workspace/data/ForAINetV2/test_data_medium_L2 \
  --extra-tag forainetv2_medium_L2_forest \
  --config /workspace/configs/oneformer3d_qs_radius16_qp300_2many.py \
  --checkpoint /workspace/work_dirs/clean_forestformer/epoch_3000_fix.pth \
  --work-dir /workspace/work_dirs/zero_shot_medium_L2_forest \
  --dist --gpus 7 \
  --overwrite-pkl
```
