# ForestFormer3D - ForAINetV2 experiments (baseline + calperum)

This README collects the **exact commands** used to run ForestFormer3D:
- **Experiment 1**: zero-shot inference (baseline ForAINetV2 + Calperum splits)
- **Experiment 2**: few-shot fine-tuning (Variant A — joint fine-tuning)

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

# Experiment 1: Zero-shot transfer performance of ForestFormer3D from forest training data to unseen forest and semi-arid rangeland ecosystem test sets
## Experiment 1.1 — Baseline ForAINetV2 (Forest domain (in-domain))

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

## Experiment 1.2 — Calperum low_L1 (Cross-domain transfer)

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

## Experiment 1.3 — Calperum low_L2 (Cross-domain transfer)

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

## Experiment 1.4 — Calperum medium_L1 (Cross-domain transfer)

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

## Experiment 1.5 — Calperum medium_L2 (Cross-domain transfer)

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

# Experiment 2: Few-shot fine-tuning (Variant A — Joint fine-tuning)

## Variant A — Joint fine-tuning (medium_L1 + low_L1 only)

The model is adapted simultaneously using samples from **both** rangeland sites:
- Calperum `medium_L1`
- Calperum `low_L1`

Not used in Experiment 2:
- `medium_L2`
- `low_L2`

### Common settings (Experiment 2)

All Experiment 2 steps below assume these env vars are set consistently:
- `FORAINETV2_PROTOCOL=forest`
- `FORAINETV2_INSTANCE_DATA_DIR=forainetv2_instance_data_exp2A`
- `FORAINETV2_BIN_BASE_DIR=forainetv2_instance_data_exp2A`

### Step 1 — Generate exp2 lists (python snippet)

If you already have the three files below, you can skip this step.

This writes:
- `train_list_exp2_mediumL1_lowL1_fewshot.txt`
- `val_list_exp2_mediumL1_lowL1_fewshot.txt`
- `test_list_exp2_mediumL1_lowL1.txt` (kept empty to skip test export in Step 3)

```bash
python - <<'PY'
from pathlib import Path
import random

meta = Path('/workspace/data/ForAINetV2/meta_data')
meta.mkdir(parents=True, exist_ok=True)

def read_ids(p: Path) -> list[str]:
    ids = [ln.strip() for ln in p.read_text(encoding='utf-8').splitlines() if ln.strip()]
    return ids

# Source pools (L1 only)
medium = read_ids(meta / 'test_list_medium_L1.txt')
low = read_ids(meta / 'test_list_low_L1.txt')

pool = medium + low
random.seed(0)
random.shuffle(pool)

# Few-shot split sizes (edit if needed)
n_train = 20
n_val = 10

train_ids = pool[:n_train]
val_ids = pool[n_train:n_train + n_val]

(meta / 'train_list_exp2_mediumL1_lowL1_fewshot.txt').write_text('\n'.join(train_ids) + '\n', encoding='utf-8')
(meta / 'val_list_exp2_mediumL1_lowL1_fewshot.txt').write_text('\n'.join(val_ids) + '\n', encoding='utf-8')

# Keep this empty on purpose for training-only cache export
(meta / 'test_list_exp2_mediumL1_lowL1.txt').write_text('', encoding='utf-8')

print('Wrote:')
print('- train_list_exp2_mediumL1_lowL1_fewshot.txt', len(train_ids))
print('- val_list_exp2_mediumL1_lowL1_fewshot.txt', len(val_ids))
print('- test_list_exp2_mediumL1_lowL1.txt (empty)')
PY
```

### Step 2 — Stage the chosen train+val PLYs into one folder

This creates a unified `train_val_data_exp2A/` matching the new train/val lists.

```bash
python /workspace/tools/make_ply_subset_dir.py \
  --src-dirs \
    /workspace/data/ForAINetV2/test_data_medium_L1 \
    /workspace/data/ForAINetV2/test_data_low_L1 \
  --scene-list /workspace/data/ForAINetV2/meta_data/train_list_exp2_mediumL1_lowL1_fewshot.txt \
  --scene-list /workspace/data/ForAINetV2/meta_data/val_list_exp2_mediumL1_lowL1_fewshot.txt \
  --out-dir /workspace/data/ForAINetV2/train_val_data_exp2A \
  --mode symlink \
  --overwrite
```

### Step 3 — Export instance-data cache for exp2A (creates npys)

This writes into:
- `/workspace/data/ForAINetV2/forainetv2_instance_data_exp2A/`

Note: we keep `test_list_exp2_mediumL1_lowL1.txt` empty, so this call effectively exports **train/val only**.

```bash
cd /workspace/data/ForAINetV2

FORAINETV2_PROTOCOL=forest \
python batch_load_ForAINetV2_data.py \
  --output_folder ./forainetv2_instance_data_exp2A \
  --train_forainetv2_dir ./train_val_data_exp2A \
  --train_scan_names_file ./meta_data/train_list_exp2_mediumL1_lowL1_fewshot.txt \
  --val_scan_names_file ./meta_data/val_list_exp2_mediumL1_lowL1_fewshot.txt \
  --test_forainetv2_dir ./test_data \
  --test_scan_names_file ./meta_data/test_list_exp2_mediumL1_lowL1.txt
```

### Step 4 — Create the train/val PKLs from your explicit lists

These list files already exist:
- `/workspace/data/ForAINetV2/meta_data/train_list_exp2_mediumL1_lowL1_fewshot.txt`
- `/workspace/data/ForAINetV2/meta_data/val_list_exp2_mediumL1_lowL1_fewshot.txt`

Generate training PKLs (the fine-tune config expects `*_train_L1.pkl` and `*_val_L1.pkl`):

```bash
FORAINETV2_PROTOCOL=forest \
FORAINETV2_INSTANCE_DATA_DIR=forainetv2_instance_data_exp2A \
FORAINETV2_BIN_BASE_DIR=forainetv2_instance_data_exp2A \
python /workspace/tools/create_forainetv2_train_val_from_lists.py \
  --root-path /workspace/data/ForAINetV2 \
  --out-dir /workspace/data/ForAINetV2 \
  --train-list /workspace/data/ForAINetV2/meta_data/train_list_exp2_mediumL1_lowL1_fewshot.txt \
  --val-list /workspace/data/ForAINetV2/meta_data/val_list_exp2_mediumL1_lowL1_fewshot.txt \
  --overwrite
```

### Step 5 — Run distributed fine-tuning

```bash
FORAINETV2_PROTOCOL=forest \
FORAINETV2_INSTANCE_DATA_DIR=forainetv2_instance_data_exp2A \
FORAINETV2_BIN_BASE_DIR=forainetv2_instance_data_exp2A \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,7 \
bash /workspace/tools/dist_train.sh \
  /workspace/configs/exp2_finetune_l1_eval_gt.py \
  7 \
  --work-dir /workspace/work_dirs/exp2_variantA_joint_ft
```

Optional: if you prefer to explicitly pin the PKLs and data-prefix paths via `--cfg-options` (as in your runs), use:

```bash
FORAINETV2_PROTOCOL=forest \
FORAINETV2_INSTANCE_DATA_DIR=forainetv2_instance_data_exp2A \
FORAINETV2_BIN_BASE_DIR=forainetv2_instance_data_exp2A \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,7 \
bash /workspace/tools/dist_train.sh /workspace/configs/exp2_finetune_l1_eval_gt.py 7 \
  --work-dir /workspace/work_dirs/exp2A_harmonized_fewshot \
  --cfg-options \
    model.prepare_epoch=0 \
    train_dataloader.dataset.ann_file=/workspace/data/ForAINetV2/forainetv2_oneformer3d_infos_train_L1.pkl \
    val_dataloader.dataset.ann_file=/workspace/data/ForAINetV2/forainetv2_oneformer3d_infos_val_L1.pkl \
    train_dataloader.dataset.data_prefix.pts=forainetv2_instance_data_exp2A/points \
    train_dataloader.dataset.data_prefix.pts_instance_mask=forainetv2_instance_data_exp2A/instance_mask \
    train_dataloader.dataset.data_prefix.pts_semantic_mask=forainetv2_instance_data_exp2A/semantic_mask \
    val_dataloader.dataset.data_prefix.pts=forainetv2_instance_data_exp2A/points \
    val_dataloader.dataset.data_prefix.pts_instance_mask=forainetv2_instance_data_exp2A/instance_mask \
    val_dataloader.dataset.data_prefix.pts_semantic_mask=forainetv2_instance_data_exp2A/semantic_mask
```

Note: the repo contains `train_list_exp2_mediumL1_lowL1_fewshot.txt` / `val_list_exp2_mediumL1_lowL1_fewshot.txt` (L1 only).
The older names `*_mediumL1L2lowL1_*` do not exist here and would re-introduce `medium_L2`.

### Step 6 — Evaluation after fine-tuning (medium_L1 + low_L1 test sets)

Run `tools/zero_shot_infer_split.py` but point `--checkpoint` to your fine-tuned checkpoint.
Keep `FORAINETV2_INSTANCE_DATA_DIR=forainetv2_instance_data_exp2A` so test exports land in the same cache.

Medium L1 test:

```bash
FORAINETV2_PROTOCOL=forest \
FORAINETV2_INSTANCE_DATA_DIR=forainetv2_instance_data_exp2A \
FORAINETV2_BIN_BASE_DIR=forainetv2_instance_data_exp2A \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,7 \
python /workspace/tools/zero_shot_infer_split.py \
  --split forest \
  --scene-list /workspace/data/ForAINetV2/meta_data/test_list_medium_L1.txt \
  --ply-dir /workspace/data/ForAINetV2/test_data_medium_L1 \
  --extra-tag exp2_variantA_ft_medium_L1 \
  --config /workspace/configs/oneformer3d_qs_radius16_qp300_2many.py \
  --checkpoint /workspace/work_dirs/exp2_variantA_joint_ft/epoch_XXXX.pth \
  --work-dir /workspace/work_dirs/exp2_variantA_eval_medium_L1 \
  --dist --gpus 7 \
  --overwrite-pkl
```

Low L1 test:

```bash
FORAINETV2_PROTOCOL=forest \
FORAINETV2_INSTANCE_DATA_DIR=forainetv2_instance_data_exp2A \
FORAINETV2_BIN_BASE_DIR=forainetv2_instance_data_exp2A \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,7 \
python /workspace/tools/zero_shot_infer_split.py \
  --split forest \
  --scene-list /workspace/data/ForAINetV2/meta_data/test_list_low_L1.txt \
  --ply-dir /workspace/data/ForAINetV2/test_data_low_L1 \
  --extra-tag exp2_variantA_ft_low_L1 \
  --config /workspace/configs/oneformer3d_qs_radius16_qp300_2many.py \
  --checkpoint /workspace/work_dirs/exp2_variantA_joint_ft/epoch_XXXX.pth \
  --work-dir /workspace/work_dirs/exp2_variantA_eval_low_L1 \
  --dist --gpus 7 \
  --overwrite-pkl
```
