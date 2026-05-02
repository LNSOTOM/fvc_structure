# ForestFormer3D - ForAINetV2 experiments (baseline + calperum)

This README collects the **exact commands** used to run ForestFormer3D:
- **Experiment 1**: zero-shot inference (baseline ForAINetV2 + Calperum splits)
- **Experiment 2**: few-shot fine-tuning (Variant A joint + Variant B curriculum)

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

Helper scripts used by Experiment 2 (included in this repo):
- `/workspace/tools/make_ply_subset_dir.py` (stages/symlinks the selected PLYs into one `train_val_data_exp2A/` folder)
- `/workspace/tools/create_forainetv2_train_val_from_lists.py` (creates `*_train_L1.pkl` and `*_val_L1.pkl` from explicit scene-id lists)

### Strict held-out test split (recommended)

The total Calperum L1 pool here is **30 tiles** = **15 medium_L1** + **15 low_L1**.

To get a strictly held-out test set, we use:
- Train: **20 tiles**
- Val: **5 tiles**
- Held-out Test: **5 tiles**

Stratified by site (so both sites appear in all splits):
- `medium_L1`: 10 train / 3 val / 2 test
- `low_L1`: 10 train / 2 val / 3 test

### Common settings (Experiment 2)

All Experiment 2 steps below use:
- `FORAINETV2_PROTOCOL=forest`
- `FORAINETV2_INSTANCE_DATA_DIR=forainetv2_instance_data_exp2A`
- `FORAINETV2_BIN_BASE_DIR=forainetv2_instance_data_exp2A`
- Fine-tune config: `/workspace/configs/exp2_finetune_l1_eval_gt.py`
- Base config: `/workspace/configs/oneformer3d_qs_radius16_qp300_2many.py`
- Initialization checkpoint: `/workspace/work_dirs/clean_forestformer/epoch_3000_fix.pth`

### Step 1 — Generate exp2 lists (python snippet)

If you already have the three files below, you can skip this step.

This writes:
- `train_list_exp2_mediumL1_lowL1_fewshot.txt`
- `val_list_exp2_mediumL1_lowL1_fewshot.txt`
- `test_list_exp2_holdout_medium_L1.txt`
- `test_list_exp2_holdout_low_L1.txt`
- `test_list_exp2_holdout_mediumL1_lowL1.txt`
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

assert len(medium) == 15, f"Expected 15 medium_L1 tiles, got {len(medium)}"
assert len(low) == 15, f"Expected 15 low_L1 tiles, got {len(low)}"

seed = 0

def split_site(ids: list[str], n_train: int, n_val: int, n_test: int) -> tuple[list[str], list[str], list[str]]:
  ids = list(ids)
  random.Random(seed).shuffle(ids)
  train = ids[:n_train]
  val = ids[n_train:n_train + n_val]
  test = ids[n_train + n_val:n_train + n_val + n_test]
  assert len(train) == n_train and len(val) == n_val and len(test) == n_test
  return train, val, test

# Strict held-out split (stratified by site)
# medium_L1: 10 train / 3 val / 2 test
# low_L1   : 10 train / 2 val / 3 test
med_train, med_val, med_test = split_site(medium, n_train=10, n_val=3, n_test=2)
low_train, low_val, low_test = split_site(low, n_train=10, n_val=2, n_test=3)

train_ids = med_train + low_train
val_ids = med_val + low_val
test_med = med_test
test_low = low_test
test_all = test_med + test_low

(meta / 'train_list_exp2_mediumL1_lowL1_fewshot.txt').write_text('\n'.join(train_ids) + '\n', encoding='utf-8')
(meta / 'val_list_exp2_mediumL1_lowL1_fewshot.txt').write_text('\n'.join(val_ids) + '\n', encoding='utf-8')
(meta / 'test_list_exp2_holdout_medium_L1.txt').write_text('\n'.join(test_med) + '\n', encoding='utf-8')
(meta / 'test_list_exp2_holdout_low_L1.txt').write_text('\n'.join(test_low) + '\n', encoding='utf-8')
(meta / 'test_list_exp2_holdout_mediumL1_lowL1.txt').write_text('\n'.join(test_all) + '\n', encoding='utf-8')

# Keep this empty on purpose for training-only cache export
(meta / 'test_list_exp2_mediumL1_lowL1.txt').write_text('', encoding='utf-8')

print('Wrote (strict held-out):')
print('- train_list_exp2_mediumL1_lowL1_fewshot.txt', len(train_ids))
print('- val_list_exp2_mediumL1_lowL1_fewshot.txt', len(val_ids))
print('- test_list_exp2_holdout_medium_L1.txt', len(test_med))
print('- test_list_exp2_holdout_low_L1.txt', len(test_low))
print('- test_list_exp2_holdout_mediumL1_lowL1.txt', len(test_all))
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
  --work-dir /workspace/work_dirs/exp2A_epoch3000_train
```

Optional: if you prefer to explicitly pin the PKLs and data-prefix paths via `--cfg-options` (as in your runs), use:

```bash
FORAINETV2_PROTOCOL=forest \
FORAINETV2_INSTANCE_DATA_DIR=forainetv2_instance_data_exp2A \
FORAINETV2_BIN_BASE_DIR=forainetv2_instance_data_exp2A \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,7 \
bash tools/dist_train.sh configs/exp2_finetune_l1_eval_gt.py 7 \
  --work-dir work_dirs/exp2A_epoch3000_train \
  --cfg-options \
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

Run `tools/zero_shot_infer_split.py` with the **best validation checkpoint** saved during Step 5 (`save_best='mIoU'`).
Keep `FORAINETV2_INSTANCE_DATA_DIR=forainetv2_instance_data_exp2A` so test exports land in the same cache.

Select the checkpoint that achieved the highest validation mIoU:

```bash
BEST_CKPT=$(ls -1t /workspace/work_dirs/exp2A_epoch3000_train/best_mIoU*.pth | head -n1)
echo "Using best checkpoint: ${BEST_CKPT}"
```

Tiny alternative (balanced checkpoint): pick and test `best_harmonic*.pth` directly:

```bash
BEST_HARMONIC_CKPT=/workspace/work_dirs/exp2A_epoch3000_train/best_harmonic_epoch_2000_fix.pth
echo "Using best harmonic checkpoint: ${BEST_HARMONIC_CKPT}"
BEST_CKPT=${BEST_HARMONIC_CKPT}
```

Tiny alternative (last epoch baseline): force `epoch_3000` for comparison:

```bash
BEST_CKPT=/workspace/work_dirs/exp2A_epoch3000_train/epoch_3000.pth
echo "Using last-epoch checkpoint: ${BEST_CKPT}"
```

Important: if you see spconv `size mismatch ... weight` errors when loading `epoch_3000.pth`, fix it once:

```bash
python /workspace/tools/fix_spconv_checkpoint.py \
  --in-path /workspace/work_dirs/exp2A_epoch3000_train/epoch_3000.pth \
  --out-path /workspace/work_dirs/exp2A_epoch3000_train/epoch_3000_fix.pth
```

Medium L1 held-out (test 3000ep):

```bash
FORAINETV2_PROTOCOL=forest \
FORAINETV2_INSTANCE_DATA_DIR=forainetv2_instance_data_exp2A \
FORAINETV2_BIN_BASE_DIR=forainetv2_instance_data_exp2A \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,7 \
python /workspace/tools/zero_shot_infer_split.py \
  --split medium \
  --scene-list /workspace/data/ForAINetV2/meta_data/test_list_exp2_holdout_medium_L1.txt \
  --ply-dir /workspace/data/ForAINetV2/test_data_medium_L1 \
  --extra-tag exp2A_epoch3000_medium_L1 \
  --config /workspace/configs/oneformer3d_qs_radius16_qp300_2many.py \
  --checkpoint ${BEST_CKPT} \
  --work-dir /workspace/work_dirs/exp2A_epoch3000_eval_medium_L1 \
  --dist --gpus 7 \
  --overwrite-pkl
```

Medium L1 held-out (test on best harmonic):

```bash
FORAINETV2_PROTOCOL=forest \
FORAINETV2_INSTANCE_DATA_DIR=forainetv2_instance_data_exp2A \
FORAINETV2_BIN_BASE_DIR=forainetv2_instance_data_exp2A \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,7 \
python /workspace/tools/zero_shot_infer_split.py \
  --split medium \
  --scene-list /workspace/data/ForAINetV2/meta_data/test_list_exp2_holdout_medium_L1.txt \
  --ply-dir /workspace/data/ForAINetV2/test_data_medium_L1 \
  --extra-tag exp2A_epoch2000_medium_L1 \
  --config /workspace/configs/oneformer3d_qs_radius16_qp300_2many.py \
  --checkpoint ${BEST_CKPT} \
  --work-dir /workspace/work_dirs/exp2A_epoch2000_eval_medium_L1 \
  --dist --gpus 7 \
  --overwrite-pkl
```

Low L1 held-out (test 3000ep):

```bash
FORAINETV2_PROTOCOL=forest \
FORAINETV2_INSTANCE_DATA_DIR=forainetv2_instance_data_exp2A \
FORAINETV2_BIN_BASE_DIR=forainetv2_instance_data_exp2A \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,7 \
python /workspace/tools/zero_shot_infer_split.py \
  --split low \
  --scene-list /workspace/data/ForAINetV2/meta_data/test_list_exp2_holdout_low_L1.txt \
  --ply-dir /workspace/data/ForAINetV2/test_data_low_L1 \
  --extra-tag exp2A_epoch3000_low_L1 \
  --config /workspace/configs/oneformer3d_qs_radius16_qp300_2many.py \
  --checkpoint ${BEST_CKPT} \
  --work-dir /workspace/work_dirs/exp2A_epoch3000_eval_low_L1 \
  --dist --gpus 7 \
  --overwrite-pkl
```

Low L1 held-out (test on best hamornic):

```bash
FORAINETV2_PROTOCOL=forest \
FORAINETV2_INSTANCE_DATA_DIR=forainetv2_instance_data_exp2A \
FORAINETV2_BIN_BASE_DIR=forainetv2_instance_data_exp2A \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,7 \
python /workspace/tools/zero_shot_infer_split.py \
  --split low \
  --scene-list /workspace/data/ForAINetV2/meta_data/test_list_exp2_holdout_low_L1.txt \
  --ply-dir /workspace/data/ForAINetV2/test_data_low_L1 \
  --extra-tag exp2A_epoch2000_low_L1 \
  --config /workspace/configs/oneformer3d_qs_radius16_qp300_2many.py \
  --checkpoint ${BEST_CKPT} \
  --work-dir /workspace/work_dirs/exp2A_epoch2000_eval_low_L1 \
  --dist --gpus 7 \
  --overwrite-pkl
```

Primary selection is `best_harmonic` for balanced semantic+instance quality; report `epoch_3000` as a baseline and `best_mIoU` as a semantic-focused reference.

### Step 6b — Validation metrics (from training log)

Validation runs every 100 epochs during training. To re-print all val lines:

```bash
grep "Epoch(val)" \
  work_dirs/exp2_variantA_joint/20260312_033733/20260312_033733.log
```

To find the best val epoch by mIoU:

```bash
grep "Epoch(val)" \
  work_dirs/exp2_variantA_joint/20260312_033733/20260312_033733.log \
  | python3 -c "
import sys, re
rows = []
for line in sys.stdin:
    ep = re.search(r'Epoch\(val\) \[(\d+)\]', line)
    miou = re.search(r'mIoU: ([0-9.]+)', line)
    f1 = re.search(r'F1: ([0-9.]+)', line)
    if ep and miou and f1:
        rows.append((int(ep.group(1)), float(miou.group(1)), float(f1.group(1)), line.strip()))
rows.sort(key=lambda x: x[1], reverse=True)
print('Top 5 by mIoU:')
for ep, miou, f1, _ in rows[:5]:
    print(f'  epoch={ep:4d}  mIoU={miou:.4f}  F1={f1:.4f}')
"
```

To print a full validation summary (best harmonic + epoch 3000 + top-5 mIoU), run:

```bash
python3 /workspace/tools/print_val_metrics_summary.py \
  --log-glob "/workspace/work_dirs/exp2A_epoch3000_train/*/*.log" \
  --epoch 3000 \
  --top-k 5
```

To print a compact **training** summary (epoch 3000 + final epoch + top-k lowest loss), run:

```bash
python3 /workspace/tools/print_train_metrics_summary.py \
  --log-glob "/workspace/work_dirs/exp2A_epoch3000_train/*/*.log" \
  --epoch 3000 \
  --metric loss \
  --top-k 5
```

To generate **plots** (train loss + val metrics) for Exp2A, run:

```bash
python3 /workspace/tools/plot_train_val_curves.py \
  --log-glob "/workspace/work_dirs/exp2A_epoch3000_train/*/*.log" \
  --out-dir "/workspace/work_dirs/exp2A_epoch3000_train/plots"
```

#### Validation results (run: 2026-03-12, best and final epochs)

**Best val epoch — epoch 1200** (highest mIoU across all 3000 epochs):

| Metric | Value |
|---|---|
| IoU ground | 71.1% |
| IoU wood | 10.3% |
| IoU leaf | 74.2% |
| mIoU | 51.9% |
| mIoU binary (non-tree / tree) | 71.1% / 93.0% = 82.0% |
| F1 / Precision / Recall | 45.0% / 58.7% / 36.5% |
| mMWCov | 41.9% |
| mMUCov | 31.1% |

**Final val epoch — epoch 3000** (checkpoint used for test evaluation):

| Metric | Value |
|---|---|
| IoU ground | 65.2% |
| IoU wood | 7.8% |
| IoU leaf | 60.5% |
| mIoU | 44.5% |
| mIoU binary (non-tree / tree) | 65.2% / 92.7% = 78.9% |
| F1 / Precision / Recall | 55.4% / 64.7% / 48.4% |
| mMWCov | 55.1% |
| mMUCov | 41.7% |

> **Note:** The best mIoU is at epoch 1200, but epoch 3000 has better F1 (55.4% vs 45.0%) and mMWCov (55.1% vs 41.9%).
> This suggests the model trades semantic accuracy for better instance grouping at later epochs.
> In this run, only `epoch_2998.pth`, `epoch_2999.pth`, `epoch_3000.pth` (plus `last_checkpoint`) were kept,
> so `epoch_1200.pth` is not available for re-test.
> If you want to compare checkpoints now, re-run Step 6 with `epoch_2998.pth` and `epoch_2999.pth` as additional baselines.

### Step 7 — Collect test metrics from eval run dirs

Run on the **host** (not inside the container) using `sudo` because the eval dirs are root-owned:

```bash
cd /home/nvidia/projects/fvc_structure/ForestFormer3D

sudo /home/nvidia/micromamba/bin/python tools/dvc_collect_metrics.py \
  --work-dir work_dirs/exp2A_epoch3000_eval_medium_L1 \
  --out work_dirs/exp2A_epoch3000_eval_medium_L1/metrics_test_mediumL1.json

sudo /home/nvidia/micromamba/bin/python tools/dvc_collect_metrics.py \
  --work-dir work_dirs/exp2A_epoch3000_eval_low_L1 \
  --out work_dirs/exp2A_epoch3000_eval_low_L1/metrics_test_lowL1.json
```

Alternative (inside the container, no `sudo` / micromamba path required):

```bash
cd /workspace

python3 tools/dvc_collect_metrics.py \
  --work-dir work_dirs/exp2A_epoch3000_eval_medium_L1 \
  --out work_dirs/exp2A_epoch3000_eval_medium_L1/metrics_test_mediumL1.json

python3 tools/dvc_collect_metrics.py \
  --work-dir work_dirs/exp2A_epoch3000_eval_low_L1 \
  --out work_dirs/exp2A_epoch3000_eval_low_L1/metrics_test_lowL1.json
```

To print Ground / Wood / Leaf IoU directly from the MMEngine log (no JSON needed):

```bash
grep -E "Table1Row|Stage1Row|Exp1Row|IoU_ground|IoU_wood|IoU_leaf" \
  work_dirs/exp2A_epoch3000_eval_medium_L1/20260316_041848/20260316_041848.log

grep -E "Table1Row|Stage1Row|Exp1Row|IoU_ground|IoU_wood|IoU_leaf" \
  work_dirs/exp2A_epoch3000_eval_low_L1/20260316_043048/20260316_043048.log
```

#### Val vs Test quick-check

Validation metrics (from training log):

```bash
grep "Epoch(val)" /home/nvidia/projects/fvc_structure/ForestFormer3D/work_dirs/exp2A_epoch3000_train/20260316_021318/20260316_021318.log
```

Test metrics (from held-out eval logs):

```bash
grep "Epoch(test)" /home/nvidia/projects/fvc_structure/ForestFormer3D/work_dirs/exp2A_epoch3000_eval_medium_L1/20260316_041848/20260316_041848.log
grep "Epoch(test)" /home/nvidia/projects/fvc_structure/ForestFormer3D/work_dirs/exp2A_epoch3000_eval_low_L1/20260316_043048/20260316_043048.log
```

To print a full **test-medium** summary (best harmonic + target run + top-k mIoU), run:

```bash
BEST_HARMONIC_CKPT=$(ls -1t /workspace/work_dirs/exp2A_epoch3000_train/best_harmonic*.pth | head -n1)
echo "Using best harmonic checkpoint: ${BEST_HARMONIC_CKPT}"

FORAINETV2_PROTOCOL=forest \
FORAINETV2_INSTANCE_DATA_DIR=forainetv2_instance_data_exp2A \
FORAINETV2_BIN_BASE_DIR=forainetv2_instance_data_exp2A \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,7 \
python /workspace/tools/zero_shot_infer_split.py \
  --split medium \
  --scene-list /workspace/data/ForAINetV2/meta_data/test_list_exp2_holdout_medium_L1.txt \
  --ply-dir /workspace/data/ForAINetV2/test_data_medium_L1 \
  --extra-tag exp2A_best_harmonic_medium_L1 \
  --config /workspace/configs/oneformer3d_qs_radius16_qp300_2many.py \
  --checkpoint ${BEST_HARMONIC_CKPT} \
  --work-dir /workspace/work_dirs/exp2A_best_harmonic_eval_medium_L1 \
  --dist --gpus 7 \
  --overwrite-pkl
```

Then summarize test-medium metrics:

```bash
python3 /workspace/tools/print_test_metrics_summary.py \
  --log-glob "/workspace/work_dirs/exp2A*_eval_medium_L1*/*/*.log" \
  --target-substring "best_harmonic" \
  --top-k 5

python3 /workspace/tools/print_test_metrics_summary.py \
  --log-glob "/workspace/work_dirs/exp2A*_eval_medium_L1*/*/*.log" \
  --target-substring "epoch_3000_fix.pth" \
  --top-k 5
```


Then summarize test-low metrics:

```bash
python3 /workspace/tools/print_test_metrics_summary.py \
  --log-glob "/workspace/work_dirs/exp2A*_eval_low_L1*/*/*.log" \
  --target-substring "best_harmonic" \
  --top-k 5

python3 /workspace/tools/print_test_metrics_summary.py \
  --log-glob "/workspace/work_dirs/exp2A*_eval_low_L1*/*/*.log" \
  --target-substring "epoch_3000_fix.pth" \
  --top-k 5
```

Test JSON summaries:

```bash
cat /home/nvidia/projects/fvc_structure/ForestFormer3D/work_dirs/exp2A_epoch3000_eval_medium_L1/metrics_test_mediumL1.json
cat /home/nvidia/projects/fvc_structure/ForestFormer3D/work_dirs/exp2A_epoch3000_eval_low_L1/metrics_test_lowL1.json
```

Qualitative viewer across experiments:

```bash
# If your current directory is /workspace/ForestFormer3D
python3 tools/qualitative_results_app.py \
  --work-dirs-root /media/laura/8402326D023263F8/calperumResearch/site1_1_DD0001/lidar/ForestFormer3D/data/ForAINetV2/work_dirs \
  --host 0.0.0.0 \
  --port 8051

# If your current directory is the parent workspace root
python3 ForestFormer3D/tools/qualitative_results_app.py \
  --work-dirs-root /media/laura/8402326D023263F8/calperumResearch/site1_1_DD0001/lidar/ForestFormer3D/data/ForAINetV2/work_dirs \
  --host 0.0.0.0 \
  --port 8051
```

This app scans all experiment folders under `work_dirs`, lists scenes with top-level `.ply` or `.las`
outputs, and renders side-by-side GT and Prediction views for both semantic and instance segmentation.

### Step 7 results (run: 2026-03-12)

**medium_L1** (`exp2A_epoch3000_eval_medium_L1`):

| Metric | Value |
|---|---|
| IoU ground | 56.3% |
| IoU wood | 4.1% |
| IoU leaf | 86.3% |
| mIoU | 48.9% |
| mIoU binary (non-tree / tree) | 56.3% / 96.8% = 76.5% |
| F1 / Precision / Recall | 39.1% / 58.9% / 29.2% |
| mMWCov | 30.0% |
| mMUCov | 26.2% |

**low_L1** (`exp2A_epoch3000_eval_low_L1`):

| Metric | Value |
|---|---|
| IoU ground | 74.3% |
| IoU wood | 26.7% |
| IoU leaf | 0.0% |
| mIoU | 50.5% |
| mIoU binary (non-tree / tree) | 74.3% / 86.5% = 80.4% |
| F1 / Precision / Recall | 0.0% / 0.0% / 0.0% |
| mMWCov | 0.0% |
| mMUCov | 0.0% |

> **Note on low_L1 instance metrics:** All instance metrics (F1, PQ, Cov) are 0 for low_L1.
> Semantic IoU is healthy (80.4% binary), meaning the backbone correctly classifies points,
> but the instance segmentation head predicted zero valid instances above the threshold.
> Likely cause: the 3 held-out low_L1 tiles contain no `leaf`-class points (`IoU_leaf=0`),
> which breaks the multi-class instance matcher. Consider re-running with the full low_L1
> pool (all 15 tiles) to get stable instance metrics.

### Checkpoint comparison on held-out test (updated)

#### medium_L1 (2 scenes)

| Checkpoint | Prec (%) | Rec (%) | F1 (%) | Cov (%) | Ground (%) | Wood (%) | Leaf (%) | mIoU (%) | Harmonic (%) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `best_harmonic_epoch_2000` | 50.7 | 31.0 | 38.5 | 30.1 | 52.8 | 5.7 | 88.2 | 48.9 | 43.1 |
| `best_mIoU_epoch_300` | 50.0 | 27.4 | 35.4 | 25.2 | 50.3 | 7.0 | 84.4 | 47.2 | 40.5 |
| `epoch_3000` | 58.9 | 29.2 | 39.1 | 26.2 | 56.3 | 4.1 | 86.3 | 48.9 | 43.4 |

#### low_L1 (3 scenes, `epoch_3000`)

| Prec (%) | Rec (%) | F1 (%) | Cov (%) | Ground (%) | Wood (%) | Leaf (%) | mIoU (%) | Harmonic (%) |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.0 | 0.0 | 0.0 | 0.0 | 74.3 | 26.7 | 0.0 | 50.5 | 0.0 |

> **Interpretation:** For balanced medium_L1 performance, `epoch_3000` and `best_harmonic_epoch_2000` are very close; `epoch_3000` is higher in F1 and harmonic, while `best_harmonic_epoch_2000` is higher in recall and coverage.
> Low_L1 instance metrics are not directly comparable because GT instances are absent (`Scenes with any GT instances: 0/3`).

## Variant B — Curriculum fine-tuning (medium-only → low-only)

This variant uses a two-stage curriculum with strict held-out testing:

- **Stage 1 (medium-only):** start from forest-pretrained checkpoint, train on medium-only train/val.
- **Stage 2 (low-only):** initialize from Stage-1 best checkpoint, then train on low-only train/val.

Key rules:

- Never test on tiles used for training.
- Keep strict held-out lists (`test_list_exp2_holdout_medium_L1.txt`, `test_list_exp2_holdout_low_L1.txt`).
- Prefer `best_harmonic` for model selection; also report `best_mIoU`.

### Variant B.1 — Build medium-only train/val lists (from existing exp2 split)

```bash
python - <<'PY'
from pathlib import Path

meta = Path('/workspace/data/ForAINetV2/meta_data')
train_all = (meta / 'train_list_exp2_mediumL1_lowL1_fewshot.txt').read_text(encoding='utf-8').splitlines()
val_all = (meta / 'val_list_exp2_mediumL1_lowL1_fewshot.txt').read_text(encoding='utf-8').splitlines()

train_med = [x for x in train_all if x.strip().startswith('znorm_3Dlabels_m_L1_')]
val_med = [x for x in val_all if x.strip().startswith('znorm_3Dlabels_m_L1_')]

(meta / 'train_list_exp2_mediumL1_only.txt').write_text('\n'.join(train_med) + '\n', encoding='utf-8')
(meta / 'val_list_exp2_mediumL1_only.txt').write_text('\n'.join(val_med) + '\n', encoding='utf-8')

print('Wrote medium-only lists:')
print('- train_list_exp2_mediumL1_only.txt', len(train_med))
print('- val_list_exp2_mediumL1_only.txt', len(val_med))
PY
```

### Variant B.2 — Create Stage-1 medium-only PKLs

```bash
FORAINETV2_PROTOCOL=forest \
FORAINETV2_INSTANCE_DATA_DIR=forainetv2_instance_data_exp2A \
FORAINETV2_BIN_BASE_DIR=forainetv2_instance_data_exp2A \
python /workspace/tools/create_forainetv2_train_val_from_lists.py \
  --root-path /workspace/data/ForAINetV2 \
  --out-dir /workspace/data/ForAINetV2 \
  --train-list /workspace/data/ForAINetV2/meta_data/train_list_exp2_mediumL1_only.txt \
  --val-list /workspace/data/ForAINetV2/meta_data/val_list_exp2_mediumL1_only.txt \
  --train-out forainetv2_oneformer3d_infos_train_exp2B_stage1_medium.pkl \
  --val-out forainetv2_oneformer3d_infos_val_exp2B_stage1_medium.pkl \
  --overwrite
```

### Variant B.3 — Stage-1 training (medium-only, from forest-pretrained)

```bash
FORAINETV2_PROTOCOL=forest \
FORAINETV2_INSTANCE_DATA_DIR=forainetv2_instance_data_exp2A \
FORAINETV2_BIN_BASE_DIR=forainetv2_instance_data_exp2A \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,7 \
bash /workspace/tools/dist_train.sh \
  /workspace/configs/exp2_finetune_l1_eval_gt.py \
  7 \
  --work-dir /workspace/work_dirs/exp2B_stage1_medium_train \
  --cfg-options \
    load_from=/workspace/work_dirs/clean_forestformer/epoch_3000_fix.pth \
    train_dataloader.dataset.ann_file=forainetv2_oneformer3d_infos_train_exp2B_stage1_medium.pkl \
    val_dataloader.dataset.ann_file=forainetv2_oneformer3d_infos_val_exp2B_stage1_medium.pkl \
    train_dataloader.dataset.data_prefix.pts=forainetv2_instance_data_exp2A/points \
    train_dataloader.dataset.data_prefix.pts_instance_mask=forainetv2_instance_data_exp2A/instance_mask \
    train_dataloader.dataset.data_prefix.pts_semantic_mask=forainetv2_instance_data_exp2A/semantic_mask \
    val_dataloader.dataset.data_prefix.pts=forainetv2_instance_data_exp2A/points \
    val_dataloader.dataset.data_prefix.pts_instance_mask=forainetv2_instance_data_exp2A/instance_mask \
    val_dataloader.dataset.data_prefix.pts_semantic_mask=forainetv2_instance_data_exp2A/semantic_mask
```

To print Stage-1 **training** metrics (epoch 3000 + final epoch + top-k lowest loss), run:

```bash
python3 /workspace/tools/print_train_metrics_summary.py \
  --log-glob "/workspace/work_dirs/exp2B_stage1_medium_train/*/*.log" \
  --epoch 3000 \
  --metric loss \
  --top-k 5
```

To print Stage-1 **validation** metrics (best harmonic + epoch 3000 + top-5 mIoU), run:

```bash
python3 /workspace/tools/print_val_metrics_summary.py \
  --log-glob "/workspace/work_dirs/exp2B_stage1_medium_train/*/*.log" \
  --epoch 3000 \
  --top-k 5
```

To generate Stage-1 **plots** (train loss + val metrics), run:

```bash
python3 /workspace/tools/plot_train_val_curves.py \
  --log-glob "/workspace/work_dirs/exp2B_stage1_medium_train/*/*.log" \
  --out-dir "/workspace/work_dirs/exp2B_stage1_medium_train/plots"
```

### Variant B.4 — Stage-1 evaluation (held-out medium)

```bash
python tools/fix_spconv_checkpoint.py \
  --in-path /workspace/work_dirs/exp2B_stage1_medium_train/epoch_3000.pth \
  --out-path /workspace/work_dirs/exp2B_stage1_medium_train/epoch_3000_fix.pth
```

```bash
STAGE1_BEST_CKPT=/workspace/work_dirs/exp2B_stage1_medium_train/epoch_3000_fix.pth

FORAINETV2_PROTOCOL=forest \
FORAINETV2_INSTANCE_DATA_DIR=forainetv2_instance_data_exp2A \
FORAINETV2_BIN_BASE_DIR=forainetv2_instance_data_exp2A \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,7 \
python /workspace/tools/zero_shot_infer_split.py \
  --split medium \
  --scene-list /workspace/data/ForAINetV2/meta_data/test_list_exp2_holdout_medium_L1.txt \
  --ply-dir /workspace/data/ForAINetV2/test_data_medium_L1 \
  --extra-tag exp2B_stage1_medium_only \
  --config /workspace/configs/oneformer3d_qs_radius16_qp300_2many.py \
  --checkpoint ${STAGE1_BEST_CKPT} \
  --work-dir /workspace/work_dirs/exp2B_stage1_eval_medium_L1 \
  --dist --gpus 7 \
  --overwrite-pkl
```

Then summarize Stage-1 **test-medium** metrics (target run + top-k mIoU), run:

```bash
python3 /workspace/tools/print_test_metrics_summary.py \
  --log-glob "/workspace/work_dirs/exp2B_stage1_eval_medium_L1/*/*.log" \
  --target-substring "epoch_3000_fix.pth" \
  --top-k 5
```

Stage-1 best-harmonic checkpoint (epoch 2400):

```bash
python tools/fix_spconv_checkpoint.py \
  --in-path /workspace/work_dirs/exp2B_stage1_medium_train/best_harmonic_epoch_2400.pth \
  --out-path /workspace/work_dirs/exp2B_stage1_medium_train/best_harmonic_epoch_2400_fix.pth
```

```bash
STAGE1_BEST_HARMONIC_CKPT=/workspace/work_dirs/exp2B_stage1_medium_train/best_harmonic_epoch_2400_fix.pth

FORAINETV2_PROTOCOL=forest \
FORAINETV2_INSTANCE_DATA_DIR=forainetv2_instance_data_exp2A \
FORAINETV2_BIN_BASE_DIR=forainetv2_instance_data_exp2A \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,7 \
python /workspace/tools/zero_shot_infer_split.py \
  --split medium \
  --scene-list /workspace/data/ForAINetV2/meta_data/test_list_exp2_holdout_medium_L1.txt \
  --ply-dir /workspace/data/ForAINetV2/test_data_medium_L1 \
  --extra-tag exp2B_stage1_best_harmonic_ep2400_medium_L1 \
  --config /workspace/configs/oneformer3d_qs_radius16_qp300_2many.py \
  --checkpoint ${STAGE1_BEST_HARMONIC_CKPT} \
  --work-dir /workspace/work_dirs/exp2B_stage1_best_harmonic_eval_medium_L1 \
  --dist --gpus 7 \
  --overwrite-pkl
```

Then summarize Stage-1 **test-medium** metrics for best-harmonic (epoch 2400), run:

```bash
python3 /workspace/tools/print_test_metrics_summary.py \
  --log-glob "/workspace/work_dirs/exp2B_stage1*_eval_medium_L1*/*/*.log" \
  --target-substring "best_harmonic_epoch_2400_fix.pth" \
  --top-k 5
```

### Variant B.5 — Build low-only lists and create Stage-2 low-only PKLs

```bash
python - <<'PY'
from pathlib import Path

meta = Path('/workspace/data/ForAINetV2/meta_data')
train_all = (meta / 'train_list_exp2_mediumL1_lowL1_fewshot.txt').read_text(encoding='utf-8').splitlines()
val_all = (meta / 'val_list_exp2_mediumL1_lowL1_fewshot.txt').read_text(encoding='utf-8').splitlines()

train_low = [x for x in train_all if x.strip().startswith('znorm_3Dlabels_l_L1_')]
val_low = [x for x in val_all if x.strip().startswith('znorm_3Dlabels_l_L1_')]

(meta / 'train_list_exp2B_lowL1_only.txt').write_text('\n'.join(train_low) + '\n', encoding='utf-8')
(meta / 'val_list_exp2B_lowL1_only.txt').write_text('\n'.join(val_low) + '\n', encoding='utf-8')

print('Wrote low-only lists:')
print('- train_list_exp2B_lowL1_only.txt', len(train_low))
print('- val_list_exp2B_lowL1_only.txt', len(val_low))
PY
```

```bash
FORAINETV2_PROTOCOL=forest \
FORAINETV2_INSTANCE_DATA_DIR=forainetv2_instance_data_exp2A \
FORAINETV2_BIN_BASE_DIR=forainetv2_instance_data_exp2A \
python /workspace/tools/create_forainetv2_train_val_from_lists.py \
  --root-path /workspace/data/ForAINetV2 \
  --out-dir /workspace/data/ForAINetV2 \
  --train-list /workspace/data/ForAINetV2/meta_data/train_list_exp2B_lowL1_only.txt \
  --val-list /workspace/data/ForAINetV2/meta_data/val_list_exp2B_lowL1_only.txt \
  --train-out forainetv2_oneformer3d_infos_train_exp2B_stage2_low.pkl \
  --val-out forainetv2_oneformer3d_infos_val_exp2B_stage2_low.pkl \
  --overwrite
```

### Variant B.6 — Stage-2 training (low-only, MUST initialize from B.3 output)

```bash
# Use the pretrained checkpoint produced by Variant B.3 (Stage-1 medium-only run),
# not the original forest checkpoint.
STAGE1_PRETRAIN_CKPT=/workspace/work_dirs/exp2B_stage1_medium_train/best_harmonic_epoch_2400_fix.pth
echo "Initializing Stage-2 from B.3 checkpoint: ${STAGE1_PRETRAIN_CKPT}"

FORAINETV2_PROTOCOL=forest \
FORAINETV2_INSTANCE_DATA_DIR=forainetv2_instance_data_exp2A \
FORAINETV2_BIN_BASE_DIR=forainetv2_instance_data_exp2A \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,7 \
bash /workspace/tools/dist_train.sh \
  /workspace/configs/exp2_finetune_l1_eval_gt.py \
  7 \
  --work-dir /workspace/work_dirs/exp2B_stage2_low_train_from_ep2400 \
  --cfg-options \
    load_from=${STAGE1_PRETRAIN_CKPT} \
    train_dataloader.dataset.ann_file=forainetv2_oneformer3d_infos_train_exp2B_stage2_low.pkl \
    val_dataloader.dataset.ann_file=forainetv2_oneformer3d_infos_val_exp2B_stage2_low.pkl \
    train_dataloader.dataset.data_prefix.pts=forainetv2_instance_data_exp2A/points \
    train_dataloader.dataset.data_prefix.pts_instance_mask=forainetv2_instance_data_exp2A/instance_mask \
    train_dataloader.dataset.data_prefix.pts_semantic_mask=forainetv2_instance_data_exp2A/semantic_mask \
    val_dataloader.dataset.data_prefix.pts=forainetv2_instance_data_exp2A/points \
    val_dataloader.dataset.data_prefix.pts_instance_mask=forainetv2_instance_data_exp2A/instance_mask \
    val_dataloader.dataset.data_prefix.pts_semantic_mask=forainetv2_instance_data_exp2A/semantic_mask
```

### Variant B.7 — Stage-2 evaluation after low-only fine-tuning (held-out medium + held-out low)

```bash
python tools/fix_spconv_checkpoint.py \
  --in-path /workspace/work_dirs/exp2B_stage2_low_train_from_ep2400/best_mIoU_epoch_2200.pth \
  --out-path /workspace/work_dirs/exp2B_stage2_low_train_from_ep2400/best_mIoU_epoch_2200_fix.pth

STAGE2_BEST_CKPT=/workspace/work_dirs/exp2B_stage2_low_train_from_ep2400/best_mIoU_epoch_2200_fix.pth
echo "Using Stage-2 checkpoint: ${STAGE2_BEST_CKPT}"

# held-out medium
FORAINETV2_PROTOCOL=forest \
FORAINETV2_INSTANCE_DATA_DIR=forainetv2_instance_data_exp2A \
FORAINETV2_BIN_BASE_DIR=forainetv2_instance_data_exp2A \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,7 \
python /workspace/tools/zero_shot_infer_split.py \
  --split medium \
  --scene-list /workspace/data/ForAINetV2/meta_data/test_list_exp2_holdout_medium_L1.txt \
  --ply-dir /workspace/data/ForAINetV2/test_data_medium_L1 \
  --extra-tag exp2B_stage2_best_mIoU_ep2200_eval_medium_L1 \
  --config /workspace/configs/oneformer3d_qs_radius16_qp300_2many.py \
  --checkpoint ${STAGE2_BEST_CKPT} \
  --work-dir /workspace/work_dirs/exp2B_stage2_best_mIoU_ep2200_eval_medium_L1 \
  --dist --gpus 7 \
  --overwrite-pkl

# held-out low
FORAINETV2_PROTOCOL=forest \
FORAINETV2_INSTANCE_DATA_DIR=forainetv2_instance_data_exp2A \
FORAINETV2_BIN_BASE_DIR=forainetv2_instance_data_exp2A \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,7 \
python /workspace/tools/zero_shot_infer_split.py \
  --split low \
  --scene-list /workspace/data/ForAINetV2/meta_data/test_list_exp2_holdout_low_L1.txt \
  --ply-dir /workspace/data/ForAINetV2/test_data_low_L1 \
  --extra-tag exp2B_stage2_best_mIoU_ep2200_eval_low_L1 \
  --config /workspace/configs/oneformer3d_qs_radius16_qp300_2many.py \
  --checkpoint ${STAGE2_BEST_CKPT} \
  --work-dir /workspace/work_dirs/exp2B_stage2_best_mIoU_ep2200_eval_low_L1 \
  --dist --gpus 7 \
  --overwrite-pkl
```

### Variant B.8 — Collect Variant B metrics

```bash
cd /workspace

python3 tools/dvc_collect_metrics.py \
  --work-dir work_dirs/exp2B_stage1_eval_medium_L1 \
  --out work_dirs/exp2B_stage1_eval_medium_L1/metrics_test_mediumL1.json

python3 tools/dvc_collect_metrics.py \
  --work-dir work_dirs/exp2B_stage2_best_mIoU_ep2200_eval_medium_L1 \
  --out work_dirs/exp2B_stage2_best_mIoU_ep2200_eval_medium_L1/metrics_test_mediumL1.json

python3 tools/dvc_collect_metrics.py \
  --work-dir work_dirs/exp2B_stage2_best_mIoU_ep2200_eval_low_L1 \
  --out work_dirs/exp2B_stage2_best_mIoU_ep2200_eval_low_L1/metrics_test_lowL1.json
```

# Experiment 3: Training from scratch (random initialization, same split as exp2A)

Yes — this experiment trains the model **without** ForAINet forest-pretrained weights.
It uses the **same train/val/held-out test split as Experiment 2A** to provide a fair baseline.

Training from scratch here means:
- No `clean_forestformer` checkpoint initialization.
- Model weights are randomly initialized.

### Step 3.1 — Train from scratch (same data split as exp2A)

Reuse Experiment 2A data preparation (Steps 1–4) as-is.

Important: `configs/exp2_finetune_l1_eval_gt.py` sets `load_from` to a forest checkpoint,
so we explicitly override it with `load_from=None`.
Also pin `data_prefix` to `forainetv2_instance_data_exp2A/*` to avoid fallback to
`data/ForAINetV2/points` (which can trigger missing `.bin` errors).

```bash
FORAINETV2_PROTOCOL=forest \
FORAINETV2_INSTANCE_DATA_DIR=forainetv2_instance_data_exp2A \
FORAINETV2_BIN_BASE_DIR=forainetv2_instance_data_exp2A \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,7 \
bash /workspace/tools/dist_train.sh \
  /workspace/configs/exp2_finetune_l1_eval_gt.py \
  7 \
  --work-dir /workspace/work_dirs/exp3A_scratch_train \
  --cfg-options \
    load_from=None \
    train_dataloader.dataset.data_prefix.pts=forainetv2_instance_data_exp2A/points \
    train_dataloader.dataset.data_prefix.pts_instance_mask=forainetv2_instance_data_exp2A/instance_mask \
    train_dataloader.dataset.data_prefix.pts_semantic_mask=forainetv2_instance_data_exp2A/semantic_mask \
    val_dataloader.dataset.data_prefix.pts=forainetv2_instance_data_exp2A/points \
    val_dataloader.dataset.data_prefix.pts_instance_mask=forainetv2_instance_data_exp2A/instance_mask \
    val_dataloader.dataset.data_prefix.pts_semantic_mask=forainetv2_instance_data_exp2A/semantic_mask
```

Equivalent relative-path form:

```bash
FORAINETV2_PROTOCOL=forest \
FORAINETV2_INSTANCE_DATA_DIR=forainetv2_instance_data_exp2A \
FORAINETV2_BIN_BASE_DIR=forainetv2_instance_data_exp2A \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,7 \
bash tools/dist_train.sh configs/exp2_finetune_l1_eval_gt.py 7 \
  --work-dir work_dirs/exp3A_scratch_train \
  --cfg-options \
    load_from=None \
    train_dataloader.dataset.data_prefix.pts=forainetv2_instance_data_exp2A/points \
    train_dataloader.dataset.data_prefix.pts_instance_mask=forainetv2_instance_data_exp2A/instance_mask \
    train_dataloader.dataset.data_prefix.pts_semantic_mask=forainetv2_instance_data_exp2A/semantic_mask \
    val_dataloader.dataset.data_prefix.pts=forainetv2_instance_data_exp2A/points \
    val_dataloader.dataset.data_prefix.pts_instance_mask=forainetv2_instance_data_exp2A/instance_mask \
    val_dataloader.dataset.data_prefix.pts_semantic_mask=forainetv2_instance_data_exp2A/semantic_mask
```

### Step 3.2 — Select checkpoint (same policy as exp2A)

```bash
BEST_CKPT=$(ls -1t /workspace/work_dirs/exp3A_scratch_train/best_harmonic*.pth | head -n1)
echo "Using scratch best-harmonic checkpoint: ${BEST_CKPT}"
```

Alternatives:

```bash
# Semantic-focused
BEST_CKPT=$(ls -1t /workspace/work_dirs/exp3A_scratch_train/best_mIoU*.pth | head -n1)

# Last-epoch baseline
BEST_CKPT=/workspace/work_dirs/exp3A_scratch_train/epoch_3000_fix.pth
```

### Step 3.2b — Metrics summaries (train + val)

Training metrics (loss etc.) for scratch run:

```bash
python3 /workspace/tools/print_train_metrics_summary.py \
  --log-glob "/workspace/work_dirs/exp3A_scratch_train/*/*.log" \
  --epoch 3000 \
  --metric loss \
  --top-k 5
```

Validation metrics (best harmonic + epoch 3000 + top-k mIoU) for scratch run:

```bash
python3 /workspace/tools/print_val_metrics_summary.py \
  --log-glob "/workspace/work_dirs/exp3A_scratch_train/*/*.log" \
  --epoch 3000 \
  --top-k 5
```

### Step 3.3 — Held-out test evaluation (same split files as exp2A)

Medium L1:

```bash
FORAINETV2_PROTOCOL=forest \
FORAINETV2_INSTANCE_DATA_DIR=forainetv2_instance_data_exp2A \
FORAINETV2_BIN_BASE_DIR=forainetv2_instance_data_exp2A \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,7 \
python /workspace/tools/zero_shot_infer_split.py \
  --split medium \
  --scene-list /workspace/data/ForAINetV2/meta_data/test_list_exp2_holdout_medium_L1.txt \
  --ply-dir /workspace/data/ForAINetV2/test_data_medium_L1 \
  --extra-tag exp3A_scratch_medium_L1 \
  --config /workspace/configs/oneformer3d_qs_radius16_qp300_2many.py \
  --checkpoint ${BEST_CKPT} \
  --work-dir /workspace/work_dirs/exp3A_scratch_eval_medium_L1 \
  --dist --gpus 7 \
  --overwrite-pkl

# Instance-eval note (UnifiedSegMetric): by default `binary_mode='collapsed'`,
# meaning wood+leaf are both treated as "tree" for instance metrics.
# If you want leaf-only ("strict") instance evaluation (e.g. rangeland where only
# mallee/leaf has GT instances), add:
#   --cfg-options test_evaluator.binary_mode=strict test_evaluator.binary_tree_eval_ids=[3]
```

#### Medium_L1 scratch checkpoint comparison (epoch_3000 vs best_harmonic)

Interpretation note:

- The held-out `medium_L1` scenes do contain GT instances. If scratch reports `F1=0` here, that does **not** mean the test split has no instances; it means the model produced no usable positive instance ids at evaluation time (for example, `Pred unique ids sample: [-1]` and `Scenes with any Pred instances: 0/2`).
- When using the base forest config (`oneformer3d_qs_radius16_qp300_2many.py`), the semantic names `ground/wood/leaf` should be read mainly as model output slots `0/1/2`. For Calperum-style rangeland data, those slots are only an approximate transfer target rather than a literal taxonomy match.
- In practice, the forest-pretrained checkpoint can still transfer useful instance structure to `medium_L1`, while random-init scratch may learn some vegetation semantics but fail to emit valid instance masks. So for the scratch-vs-pretrained comparison, treat `F1=0` primarily as an **instance-decoder failure**, not evidence that `medium_L1` lacks tree/mallee instances.

Run **two** evals (medium) so both checkpoints have their own logs:

```bash
# 1) best_harmonic (balanced)
BEST_HARMONIC_CKPT=$(ls -1t /workspace/work_dirs/exp3A_scratch_train/best_harmonic*.pth | head -n1)
echo "Using scratch best harmonic: ${BEST_HARMONIC_CKPT}"

FORAINETV2_PROTOCOL=forest \
FORAINETV2_INSTANCE_DATA_DIR=forainetv2_instance_data_exp2A \
FORAINETV2_BIN_BASE_DIR=forainetv2_instance_data_exp2A \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,7 \
python /workspace/tools/zero_shot_infer_split.py \
  --split medium \
  --scene-list /workspace/data/ForAINetV2/meta_data/test_list_exp2_holdout_medium_L1.txt \
  --ply-dir /workspace/data/ForAINetV2/test_data_medium_L1 \
  --extra-tag exp3A_scratch_best_harmonic_medium_L1 \
  --config /workspace/configs/oneformer3d_qs_radius16_qp300_2many.py \
  --checkpoint ${BEST_HARMONIC_CKPT} \
  --work-dir /workspace/work_dirs/exp3A_scratch_best_harmonic_eval_medium_L1 \
  --dist --gpus 7 \
  --overwrite-pkl

# 2) epoch_3000 baseline
EPOCH_3000_CKPT=/workspace/work_dirs/exp3A_scratch_train/epoch_3000_fix.pth
echo "Using scratch epoch 3000: ${EPOCH_3000_CKPT}"

FORAINETV2_PROTOCOL=forest \
FORAINETV2_INSTANCE_DATA_DIR=forainetv2_instance_data_exp2A \
FORAINETV2_BIN_BASE_DIR=forainetv2_instance_data_exp2A \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,7 \
python /workspace/tools/zero_shot_infer_split.py \
  --split medium \
  --scene-list /workspace/data/ForAINetV2/meta_data/test_list_exp2_holdout_medium_L1.txt \
  --ply-dir /workspace/data/ForAINetV2/test_data_medium_L1 \
  --extra-tag exp3A_scratch_epoch_3000_medium_L1 \
  --config /workspace/configs/oneformer3d_qs_radius16_qp300_2many.py \
  --checkpoint ${EPOCH_3000_CKPT} \
  --work-dir /workspace/work_dirs/exp3A_scratch_epoch_3000_eval_medium_L1 \
  --dist --gpus 7 \
  --overwrite-pkl
```

Then summarize scratch **test-medium** metrics for both checkpoints:

```bash
python3 /workspace/tools/print_test_metrics_summary.py \
  --log-glob "/workspace/work_dirs/exp3A_scratch*_eval_medium_L1*/*/*.log" \
  --target-substring "best_harmonic" \
  --top-k 5

python3 /workspace/tools/print_test_metrics_summary.py \
  --log-glob "/workspace/work_dirs/exp3A_scratch*_eval_medium_L1*/*/*.log" \
  --target-substring "epoch_3000_fix.pth" \
  --top-k 5
```

Low L1:

```bash
FORAINETV2_PROTOCOL=forest \
FORAINETV2_INSTANCE_DATA_DIR=forainetv2_instance_data_exp2A \
FORAINETV2_BIN_BASE_DIR=forainetv2_instance_data_exp2A \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,7 \
python /workspace/tools/zero_shot_infer_split.py \
  --split low \
  --scene-list /workspace/data/ForAINetV2/meta_data/test_list_exp2_holdout_low_L1.txt \
  --ply-dir /workspace/data/ForAINetV2/test_data_low_L1 \
  --extra-tag exp3A_scratch_low_L1 \
  --config /workspace/configs/oneformer3d_qs_radius16_qp300_2many.py \
  --checkpoint ${BEST_CKPT} \
  --work-dir /workspace/work_dirs/exp3A_scratch_eval_low_L1 \
  --dist --gpus 7 \
  --overwrite-pkl
```

Then summarize scratch **test-low** metrics for both checkpoints:

```bash
python3 /workspace/tools/print_test_metrics_summary.py \
  --log-glob "/workspace/work_dirs/exp3A_scratch*_eval_low_L1*/*/*.log" \
  --target-substring "best_harmonic" \
  --top-k 5

python3 /workspace/tools/print_test_metrics_summary.py \
  --log-glob "/workspace/work_dirs/exp3A_scratch*_eval_low_L1*/*/*.log" \
  --target-substring "epoch_3000_fix.pth" \
  --top-k 5
```

### Step 3.4 — Collect scratch test metrics

Inside container:

```bash
cd /workspace

python3 tools/dvc_collect_metrics.py \
  --work-dir work_dirs/exp3A_scratch_eval_medium_L1 \
  --out work_dirs/exp3A_scratch_eval_medium_L1/metrics_test_mediumL1.json

python3 tools/dvc_collect_metrics.py \
  --work-dir work_dirs/exp3A_scratch_eval_low_L1 \
  --out work_dirs/exp3A_scratch_eval_low_L1/metrics_test_lowL1.json
```

Quick val vs test check for scratch run:

```bash
grep "Epoch(val)" /workspace/work_dirs/exp3A_scratch_train/*/*.log
grep "Epoch(test)" /workspace/work_dirs/exp3A_scratch_eval_medium_L1/*/*.log
grep "Epoch(test)" /workspace/work_dirs/exp3A_scratch_eval_low_L1/*/*.log
```

## Experiment 3B: Training from scratch (random initialization, medium-only)

This scratch run trains **only on medium_L1 tiles** using the strict held-out split:
- Train: **10** medium_L1 tiles
- Val: **3** medium_L1 tiles
- Held-out Test: **2** medium_L1 tiles (`test_list_exp2_holdout_medium_L1.txt`)

### Step 3B.1 — (Re)create the medium-only train/val lists

If you already created these in Variant B.1, you can skip this.

```bash
python - <<'PY'
from pathlib import Path

meta = Path('/workspace/data/ForAINetV2/meta_data')
train_all = (meta / 'train_list_exp2_mediumL1_lowL1_fewshot.txt').read_text(encoding='utf-8').splitlines()
val_all = (meta / 'val_list_exp2_mediumL1_lowL1_fewshot.txt').read_text(encoding='utf-8').splitlines()

train_med = [x for x in train_all if x.strip().startswith('znorm_3Dlabels_m_L1_')]
val_med = [x for x in val_all if x.strip().startswith('znorm_3Dlabels_m_L1_')]

(meta / 'train_list_exp2_mediumL1_only.txt').write_text('\n'.join(train_med) + '\n', encoding='utf-8')
(meta / 'val_list_exp2_mediumL1_only.txt').write_text('\n'.join(val_med) + '\n', encoding='utf-8')

test_med = (meta / 'test_list_exp2_holdout_medium_L1.txt').read_text(encoding='utf-8').splitlines()
test_med = [x.strip() for x in test_med if x.strip()]

print('Medium-only split sizes:')
print('- train_list_exp2_mediumL1_only.txt', len(train_med))
print('- val_list_exp2_mediumL1_only.txt', len(val_med))
print('- test_list_exp2_holdout_medium_L1.txt', len(test_med))
PY
```

Expected: train=10, val=3, test=2.

### Step 3B.2 — Create scratch medium-only PKLs

```bash
FORAINETV2_PROTOCOL=forest \
FORAINETV2_INSTANCE_DATA_DIR=forainetv2_instance_data_exp2A \
FORAINETV2_BIN_BASE_DIR=forainetv2_instance_data_exp2A \
python /workspace/tools/create_forainetv2_train_val_from_lists.py \
  --root-path /workspace/data/ForAINetV2 \
  --out-dir /workspace/data/ForAINetV2 \
  --train-list /workspace/data/ForAINetV2/meta_data/train_list_exp2B_mediumL1_only.txt \
  --val-list /workspace/data/ForAINetV2/meta_data/val_list_exp2B_mediumL1_only.txt \
  --train-out forainetv2_oneformer3d_infos_train_exp3B_scratch_medium.pkl \
  --val-out forainetv2_oneformer3d_infos_val_exp3B_scratch_medium.pkl \
  --overwrite
```

### Step 3B.3 — Train from scratch (medium-only)

```bash
FORAINETV2_PROTOCOL=forest \
FORAINETV2_INSTANCE_DATA_DIR=forainetv2_instance_data_exp2A \
FORAINETV2_BIN_BASE_DIR=forainetv2_instance_data_exp2A \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,7 \
bash /workspace/tools/dist_train.sh \
  /workspace/configs/exp2_finetune_l1_eval_gt.py \
  7 \
  --work-dir /workspace/work_dirs/exp3B_scratch_medium_train \
  --cfg-options \
    load_from=None \
    train_dataloader.dataset.ann_file=forainetv2_oneformer3d_infos_train_exp3B_scratch_medium.pkl \
    val_dataloader.dataset.ann_file=forainetv2_oneformer3d_infos_val_exp3B_scratch_medium.pkl \
    train_dataloader.dataset.data_prefix.pts=forainetv2_instance_data_exp2A/points \
    train_dataloader.dataset.data_prefix.pts_instance_mask=forainetv2_instance_data_exp2A/instance_mask \
    train_dataloader.dataset.data_prefix.pts_semantic_mask=forainetv2_instance_data_exp2A/semantic_mask \
    val_dataloader.dataset.data_prefix.pts=forainetv2_instance_data_exp2A/points \
    val_dataloader.dataset.data_prefix.pts_instance_mask=forainetv2_instance_data_exp2A/instance_mask \
    val_dataloader.dataset.data_prefix.pts_semantic_mask=forainetv2_instance_data_exp2A/semantic_mask
```

### Step 3B.3b — (Optional) Fix the epoch-3000 checkpoint for spconv

If you see spconv "size mismatch" errors when loading `epoch_3000.pth`, fix it once:

```bash
python /workspace/tools/fix_spconv_checkpoint.py \
  --in-path /workspace/work_dirs/exp3B_scratch_medium_train/epoch_3000.pth \
  --out-path /workspace/work_dirs/exp3B_scratch_medium_train/epoch_3000_fix.pth

EPOCH_3000_CKPT=/workspace/work_dirs/exp3B_scratch_medium_train/epoch_3000_fix.pth
```

### Step 3B.4 — Plots + summaries (train + val)

```bash
python3 /workspace/tools/plot_train_val_curves.py \
  --log-glob "/workspace/work_dirs/exp3B_scratch_medium_train/*/*.log" \
  --out-dir "/workspace/work_dirs/exp3B_scratch_medium_train/plots" \
  --title "exp3B_scratch_medium_train"

python3 /workspace/tools/print_train_metrics_summary.py \
  --log-glob "/workspace/work_dirs/exp3B_scratch_medium_train/*/*.log" \
  --epoch 3000 \
  --metric loss \
  --top-k 5

python3 /workspace/tools/print_val_metrics_summary.py \
  --log-glob "/workspace/work_dirs/exp3B_scratch_medium_train/*/*.log" \
  --epoch 3000 \
  --top-k 5
```

### Step 3B.5 — Held-out test evaluation (medium-only test = 2 scenes)

```bash
BEST_CKPT=$(ls -1t /workspace/work_dirs/exp3B_scratch_medium_train/best_harmonic*.pth | head -n1)
echo "Using exp3B scratch best harmonic: ${BEST_CKPT}"

FORAINETV2_PROTOCOL=forest \
FORAINETV2_INSTANCE_DATA_DIR=forainetv2_instance_data_exp2A \
FORAINETV2_BIN_BASE_DIR=forainetv2_instance_data_exp2A \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,7 \
python /workspace/tools/zero_shot_infer_split.py \
  --split medium \
  --scene-list /workspace/data/ForAINetV2/meta_data/test_list_exp2_holdout_medium_L1.txt \
  --ply-dir /workspace/data/ForAINetV2/test_data_medium_L1 \
  --extra-tag exp3B_scratch_medium_only_medium_L1 \
  --config /workspace/configs/oneformer3d_qs_radius16_qp300_2many.py \
  --checkpoint ${BEST_CKPT} \
  --work-dir /workspace/work_dirs/exp3B_scratch_medium_eval_medium_L1 \
  --dist --gpus 7 \
  --overwrite-pkl

# Same instance-eval override applies here if needed:
#   --cfg-options test_evaluator.binary_mode=strict test_evaluator.binary_tree_eval_ids=[3]
```

Then summarize test-medium metrics:

```bash
python3 /workspace/tools/print_test_metrics_summary.py \
  --log-glob "/workspace/work_dirs/exp3B_scratch_medium_eval_medium_L1/*/*.log" \
  --target-substring "best_harmonic" \
  --top-k 5
```

## Experiment 3C: Training from scratch (random initialization, low-only)

This scratch run trains **only on low_L1 tiles** using the strict held-out split:
- Train: **10** low_L1 tiles
- Val: **2** low_L1 tiles
- Held-out Test: **3** low_L1 tiles (`test_list_exp2_holdout_low_L1.txt`)

### Step 3C.1 — (Re)create the low-only train/val lists

```bash
python - <<'PY'
from pathlib import Path

meta = Path('/workspace/data/ForAINetV2/meta_data')
train_all = (meta / 'train_list_exp2_mediumL1_lowL1_fewshot.txt').read_text(encoding='utf-8').splitlines()
val_all = (meta / 'val_list_exp2_mediumL1_lowL1_fewshot.txt').read_text(encoding='utf-8').splitlines()

# NOTE: low tiles use the prefix below (confirmed in the existing fewshot lists)
train_low = [x.strip() for x in train_all if x.strip().startswith('znorm_3Dlabels_l_L1_')]
val_low = [x.strip() for x in val_all if x.strip().startswith('znorm_3Dlabels_l_L1_')]

(meta / 'train_list_exp3C_lowL1_only.txt').write_text('\n'.join(train_low) + '\n', encoding='utf-8')
(meta / 'val_list_exp3C_lowL1_only.txt').write_text('\n'.join(val_low) + '\n', encoding='utf-8')

test_low = (meta / 'test_list_exp2_holdout_low_L1.txt').read_text(encoding='utf-8').splitlines()
test_low = [x.strip() for x in test_low if x.strip()]

print('Low-only split sizes:')
print('- train_list_exp3C_lowL1_only.txt', len(train_low))
print('- val_list_exp3C_lowL1_only.txt', len(val_low))
print('- test_list_exp2_holdout_low_L1.txt', len(test_low))
PY
```

Expected: train=10, val=2, test=3.

### Step 3C.2 — Create scratch low-only PKLs

```bash
FORAINETV2_PROTOCOL=forest \
FORAINETV2_INSTANCE_DATA_DIR=forainetv2_instance_data_exp2A \
FORAINETV2_BIN_BASE_DIR=forainetv2_instance_data_exp2A \
python /workspace/tools/create_forainetv2_train_val_from_lists.py \
  --root-path /workspace/data/ForAINetV2 \
  --out-dir /workspace/data/ForAINetV2 \
  --train-list /workspace/data/ForAINetV2/meta_data/train_list_exp3C_lowL1_only.txt \
  --val-list /workspace/data/ForAINetV2/meta_data/val_list_exp3C_lowL1_only.txt \
  --train-out forainetv2_oneformer3d_infos_train_exp3C_scratch_low.pkl \
  --val-out forainetv2_oneformer3d_infos_val_exp3C_scratch_low.pkl \
  --overwrite
```

### Step 3C.3 — Train from scratch (low-only)

```bash
FORAINETV2_PROTOCOL=forest \
FORAINETV2_INSTANCE_DATA_DIR=forainetv2_instance_data_exp2A \
FORAINETV2_BIN_BASE_DIR=forainetv2_instance_data_exp2A \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,7 \
bash /workspace/tools/dist_train.sh \
  /workspace/configs/exp2_finetune_l1_eval_gt.py \
  7 \
  --work-dir /workspace/work_dirs/exp3C_scratch_low_train \
  --cfg-options \
    load_from=None \
    train_dataloader.dataset.ann_file=forainetv2_oneformer3d_infos_train_exp3C_scratch_low.pkl \
    val_dataloader.dataset.ann_file=forainetv2_oneformer3d_infos_val_exp3C_scratch_low.pkl \
    train_dataloader.dataset.data_prefix.pts=forainetv2_instance_data_exp2A/points \
    train_dataloader.dataset.data_prefix.pts_instance_mask=forainetv2_instance_data_exp2A/instance_mask \
    train_dataloader.dataset.data_prefix.pts_semantic_mask=forainetv2_instance_data_exp2A/semantic_mask \
    val_dataloader.dataset.data_prefix.pts=forainetv2_instance_data_exp2A/points \
    val_dataloader.dataset.data_prefix.pts_instance_mask=forainetv2_instance_data_exp2A/instance_mask \
    val_dataloader.dataset.data_prefix.pts_semantic_mask=forainetv2_instance_data_exp2A/semantic_mask
```

### Step 3C.3b — (Optional) Fix the epoch-3000 checkpoint for spconv

If you see spconv "size mismatch" errors when loading `epoch_3000.pth`, fix it once:

```bash
python /workspace/tools/fix_spconv_checkpoint.py \
  --in-path /workspace/work_dirs/exp3C_scratch_low_train/epoch_3000.pth \
  --out-path /workspace/work_dirs/exp3C_scratch_low_train/epoch_3000_fix.pth

EPOCH_3000_CKPT=/workspace/work_dirs/exp3C_scratch_low_train/epoch_3000_fix.pth
```

### Step 3C.4 — Plots + summaries (train + val)

```bash
python3 /workspace/tools/plot_train_val_curves.py \
  --log-glob "/workspace/work_dirs/exp3C_scratch_low_train/*/*.log" \
  --out-dir "/workspace/work_dirs/exp3C_scratch_low_train/plots" \
  --title "exp3C_scratch_low_train"

python3 /workspace/tools/print_train_metrics_summary.py \
  --log-glob "/workspace/work_dirs/exp3C_scratch_low_train/*/*.log" \
  --epoch 3000 \
  --metric loss \
  --top-k 5

python3 /workspace/tools/print_val_metrics_summary.py \
  --log-glob "/workspace/work_dirs/exp3C_scratch_low_train/*/*.log" \
  --epoch 3000 \
  --top-k 5
```

### Step 3C.5 — Held-out test evaluation (low-only test = 3 scenes)

```bash
BEST_CKPT=$(ls -1t /workspace/work_dirs/exp3C_scratch_low_train/best_harmonic*.pth | head -n1)
echo "Using exp3C scratch best harmonic: ${BEST_CKPT}"

FORAINETV2_PROTOCOL=forest \
FORAINETV2_INSTANCE_DATA_DIR=forainetv2_instance_data_exp2A \
FORAINETV2_BIN_BASE_DIR=forainetv2_instance_data_exp2A \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,7 \
python /workspace/tools/zero_shot_infer_split.py \
  --split low \
  --scene-list /workspace/data/ForAINetV2/meta_data/test_list_exp2_holdout_low_L1.txt \
  --ply-dir /workspace/data/ForAINetV2/test_data_low_L1 \
  --extra-tag exp3C_scratch_low_only_low_L1 \
  --config /workspace/configs/oneformer3d_qs_radius16_qp300_2many.py \
  --checkpoint ${BEST_CKPT} \
  --work-dir /workspace/work_dirs/exp3C_scratch_low_eval_low_L1 \
  --dist --gpus 7 \
  --overwrite-pkl

# Same instance-eval override applies here if needed:
#   --cfg-options test_evaluator.binary_mode=strict test_evaluator.binary_tree_eval_ids=[3]
```

Then summarize test-low metrics:

```bash
python3 /workspace/tools/print_test_metrics_summary.py \
  --log-glob "/workspace/work_dirs/exp3C_scratch_low_eval_low_L1/*/*.log" \
  --target-substring "best_harmonic" \
  --top-k 5
```
