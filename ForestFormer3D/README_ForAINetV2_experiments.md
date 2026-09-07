# ForestFormer3D — ForAINetV2 experiments (baseline + Calperum)

Exact commands for the three experiments, run **inside the ForestFormer3D Docker container**
(repo mounted at `/workspace`).

| Experiment | What | Variants |
|---|---|---|
| [1](#experiment-1--zero-shot-transfer) | Zero-shot transfer of the forest-trained model to unseen forest and semi-arid rangeland test sets | baseline forest, Calperum low/medium × L1/L2 |
| [2](#experiment-2--few-shot-fine-tuning) | Few-shot fine-tuning on Calperum L1 | A: joint (medium+low), B: curriculum (medium → low) |
| [3](#experiment-3--training-from-scratch) | Training from scratch (random init), same splits as 2A | 3A joint, 3B medium-only, 3C low-only |

## How to read this guide

Every experiment is split into numbered steps that share the same shape:

| Field | Meaning |
|---|---|
| **Goal** | what the step produces |
| **Run** | the exact command(s) to execute |
| **Verify** | a cheap check to run before starting the next step |

Step ids (`E1.2`, `E2A.3`, `E3B.2`) are stable, so run notes and logs can reference them directly.
Sections titled **Recorded results** are outputs from past runs kept for comparison, not commands.

## Stage 0 — Prerequisites

### 0.1 Fixed paths and settings

| Item | Value |
|---|---|
| Working directory | `/workspace` (repo mounted inside the container) |
| Evaluation config | `/workspace/configs/oneformer3d_qs_radius16_qp300_2many.py` |
| Training config | `/workspace/configs/exp2_finetune_l1_eval_gt.py` |
| Forest-pretrained init | `/workspace/work_dirs/clean_forestformer/epoch_3000_fix.pth` |
| Cache — Experiment 1 | `forainetv2_instance_data_forest` (already in the repo) |
| Cache — Experiments 2 and 3 | `forainetv2_instance_data_exp2A` (built in [E2A.3](#e2a3--export-the-exp2a-instance-data-cache)) |
| GPUs | `CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,7` exposes 7 GPUs, so always pass `7` to `--gpus` / `dist_train.sh` |

Two conventions that silently break runs if ignored:

- A line-continuation backslash `\` must be the **last** character on the line (no trailing spaces).
- Checkpoint selection: prefer `best_harmonic` (balanced semantic + instance), and report
  `epoch_3000` as the last-epoch baseline and `best_mIoU` as a semantic-focused reference.

If loading a checkpoint raises spconv `size mismatch ... weight`, convert it once:

```bash
python /workspace/tools/fix_spconv_checkpoint.py \
  --in-path  <run-dir>/epoch_3000.pth \
  --out-path <run-dir>/epoch_3000_fix.pth
```

### 0.2 Shell helpers (run once per shell)

**Goal:** define the helpers every later step depends on. `use_cache` sets the shared environment,
and `PREFIX_OPTS` pins `data_prefix` to the selected cache so training never falls back to
`data/ForAINetV2/points` (which triggers missing `.bin` errors).

**Run:**

```bash
use_cache () {   # $1 = instance-data cache dir name
  export FORAINETV2_PROTOCOL=forest
  export FORAINETV2_INSTANCE_DATA_DIR=$1
  export FORAINETV2_BIN_BASE_DIR=$1
  export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,7
  export PREFIX_OPTS="\
train_dataloader.dataset.data_prefix.pts=$1/points \
train_dataloader.dataset.data_prefix.pts_instance_mask=$1/instance_mask \
train_dataloader.dataset.data_prefix.pts_semantic_mask=$1/semantic_mask \
val_dataloader.dataset.data_prefix.pts=$1/points \
val_dataloader.dataset.data_prefix.pts_instance_mask=$1/instance_mask \
val_dataloader.dataset.data_prefix.pts_semantic_mask=$1/semantic_mask"
}

ff_eval () {     # SPLIT SCENE_LIST PLY_DIR TAG WORK_DIR CKPT   (paths relative to data/work dirs)
  python /workspace/tools/zero_shot_infer_split.py \
    --split "$1" \
    --scene-list "/workspace/data/ForAINetV2/meta_data/$2" \
    --ply-dir "/workspace/data/ForAINetV2/$3" \
    --extra-tag "$4" \
    --config /workspace/configs/oneformer3d_qs_radius16_qp300_2many.py \
    --checkpoint "$6" \
    --work-dir "/workspace/work_dirs/$5" \
    --dist --gpus 7 \
    --overwrite-pkl
}

ff_train () {    # WORK_DIR [extra --cfg-options ...]
  local wd=$1; shift
  bash /workspace/tools/dist_train.sh \
    /workspace/configs/exp2_finetune_l1_eval_gt.py \
    7 \
    --work-dir "/workspace/work_dirs/$wd" \
    --cfg-options "$@" $PREFIX_OPTS
}

ff_pkls () {     # TRAIN_LIST VAL_LIST TRAIN_PKL VAL_PKL   (lists relative to meta_data/)
  python /workspace/tools/create_forainetv2_train_val_from_lists.py \
    --root-path /workspace/data/ForAINetV2 \
    --out-dir /workspace/data/ForAINetV2 \
    --train-list "/workspace/data/ForAINetV2/meta_data/$1" \
    --val-list "/workspace/data/ForAINetV2/meta_data/$2" \
    ${3:+--train-out "$3"} ${4:+--val-out "$4"} \
    --overwrite
}

ff_summarize () {  # WORK_DIR — train summary + val summary + curves
  local g="/workspace/work_dirs/$1/*/*.log"
  python3 /workspace/tools/print_train_metrics_summary.py --log-glob "$g" --epoch 3000 --metric loss --top-k 5
  python3 /workspace/tools/print_val_metrics_summary.py   --log-glob "$g" --epoch 3000 --top-k 5
  python3 /workspace/tools/plot_train_val_curves.py       --log-glob "$g" \
    --out-dir "/workspace/work_dirs/$1/plots" --title "$1"
}

ff_test_summary () {  # WORK_DIR_GLOB TARGET_SUBSTRING
  python3 /workspace/tools/print_test_metrics_summary.py \
    --log-glob "/workspace/work_dirs/$1/*/*.log" \
    --target-substring "$2" \
    --top-k 5
}
```

`ff_pkls` with only two arguments writes the default `*_train_L1.pkl` / `*_val_L1.pkl` that the
fine-tune config expects; pass the third and fourth to name them explicitly.

**Verify:**

```bash
type use_cache ff_eval ff_train ff_pkls ff_summarize ff_test_summary >/dev/null \
  && echo "helpers loaded"
```

### 0.3 Evaluator reference (`UnifiedSegMetric`)

The reusable evaluation entry point is `UnifiedSegMetric` in `oneformer3d/unified_metric.py`, wired
through the active configs as `type='UnifiedSegMetric'`. It reports semantic IoU/mIoU, binary
non-tree/tree IoU, instance precision/recall/F1, MUCov/MWCov, PQ/SQ/RQ, and the `harmonic` score
used for checkpoint selection.

For a new split or dataset, keep the active base config as the template and change only:

| Option | Purpose |
|---|---|
| `metric_meta['classes']` | semantic class names in model/output order |
| `binary_ground_eval_id` | eval id treated as non-tree/ground after label normalisation |
| `binary_mode='collapsed'` | all non-ground classes become tree for binary/instance evaluation |
| `binary_mode='strict'` + `binary_tree_eval_ids=[...]` | only the listed eval ids become tree |
| `report_instance_metrics=False` | semantic-only evaluation when instance masks are unavailable |

Do not use the old `*copy*.py` configs or copied modules; they were local backups and are not part
of the reusable experiment path.

---

# Experiment 1 — Zero-shot transfer

Forest-trained model, no fine-tuning, evaluated in-domain (forest) and cross-domain (Calperum
rangeland). All five runs use `--split forest` and the forest instance-data cache.

### E1.1 — Select the cache and checkpoint

**Run:**

```bash
use_cache forainetv2_instance_data_forest
CKPT=/workspace/work_dirs/clean_forestformer/epoch_3000_fix.pth
```

### E1.2 — Run the five evaluations

**Goal:** one work-dir, and one log, per split.

**Run:**

```bash
# 1.1 Baseline ForAINetV2 — forest (in-domain)
ff_eval forest test_list.txt           test_data           forainetv2_baseline_forest  zero_shot_baseline_forest  "$CKPT"
# 1.2 Calperum low_L1     (cross-domain)
ff_eval forest test_list_low_L1.txt    test_data_low_L1    forainetv2_low_L1_forest    zero_shot_low_L1_forest    "$CKPT"
# 1.3 Calperum low_L2     (cross-domain)
ff_eval forest test_list_low_L2.txt    test_data_low_L2    forainetv2_low_L2_forest    zero_shot_low_L2_forest    "$CKPT"
# 1.4 Calperum medium_L1  (cross-domain)
ff_eval forest test_list_medium_L1.txt test_data_medium_L1 forainetv2_medium_L1_forest zero_shot_medium_L1_forest "$CKPT"
# 1.5 Calperum medium_L2  (cross-domain)
ff_eval forest test_list_medium_L2.txt test_data_medium_L2 forainetv2_medium_L2_forest zero_shot_medium_L2_forest "$CKPT"
```

**Verify:** every split should print a metric row.

```bash
grep -l "Exp1Row" /workspace/work_dirs/zero_shot_*/*/*.log
```

---

# Experiment 2 — Few-shot fine-tuning

Adapts the forest-pretrained model to Calperum **L1 only** (`medium_L2` and `low_L2` are not used).

### Strict held-out split

Pool: **30 L1 tiles** = 15 `medium_L1` + 15 `low_L1`, stratified by site so both appear everywhere.

| Site | Train | Val | Held-out test |
|---|---:|---:|---:|
| `medium_L1` | 10 | 3 | 2 |
| `low_L1` | 10 | 2 | 3 |
| **Total** | **20** | **5** | **5** |

Helper scripts used here (in this repo):
`tools/make_ply_subset_dir.py` (symlinks the selected PLYs into one folder) and
`tools/create_forainetv2_train_val_from_lists.py` (PKLs from explicit scene-id lists).

```bash
use_cache forainetv2_instance_data_exp2A
```

## Variant A — Joint fine-tuning (medium_L1 + low_L1)

### E2A.1 — Generate the exp2 lists

Skip if the list files already exist. Writes `train_list_exp2_mediumL1_lowL1_fewshot.txt`,
`val_list_exp2_mediumL1_lowL1_fewshot.txt`, the three `test_list_exp2_holdout_*` files, and
`test_list_exp2_mediumL1_lowL1.txt` (**deliberately empty**, so E2A.3 exports train/val only).

```bash
python - <<'PY'
from pathlib import Path
import random

meta = Path('/workspace/data/ForAINetV2/meta_data')
meta.mkdir(parents=True, exist_ok=True)

def read_ids(p): return [ln.strip() for ln in p.read_text(encoding='utf-8').splitlines() if ln.strip()]

medium = read_ids(meta / 'test_list_medium_L1.txt')
low = read_ids(meta / 'test_list_low_L1.txt')
assert len(medium) == 15, f"Expected 15 medium_L1 tiles, got {len(medium)}"
assert len(low) == 15, f"Expected 15 low_L1 tiles, got {len(low)}"

seed = 0

def split_site(ids, n_train, n_val, n_test):
    ids = list(ids)
    random.Random(seed).shuffle(ids)
    train, val = ids[:n_train], ids[n_train:n_train + n_val]
    test = ids[n_train + n_val:n_train + n_val + n_test]
    assert len(train) == n_train and len(val) == n_val and len(test) == n_test
    return train, val, test

med_train, med_val, med_test = split_site(medium, 10, 3, 2)
low_train, low_val, low_test = split_site(low, 10, 2, 3)

out = {
    'train_list_exp2_mediumL1_lowL1_fewshot.txt': med_train + low_train,
    'val_list_exp2_mediumL1_lowL1_fewshot.txt': med_val + low_val,
    'test_list_exp2_holdout_medium_L1.txt': med_test,
    'test_list_exp2_holdout_low_L1.txt': low_test,
    'test_list_exp2_holdout_mediumL1_lowL1.txt': med_test + low_test,
}
for name, ids in out.items():
    (meta / name).write_text('\n'.join(ids) + '\n', encoding='utf-8')
    print(f'- {name}', len(ids))

# Empty on purpose: training-only cache export
(meta / 'test_list_exp2_mediumL1_lowL1.txt').write_text('', encoding='utf-8')
print('- test_list_exp2_mediumL1_lowL1.txt (empty)')
PY
```

**Verify:** 20 train ids and 5 val ids.

```bash
wc -l /workspace/data/ForAINetV2/meta_data/train_list_exp2_mediumL1_lowL1_fewshot.txt \
      /workspace/data/ForAINetV2/meta_data/val_list_exp2_mediumL1_lowL1_fewshot.txt
```

### E2A.2 — Stage the train+val PLYs into one folder

**Run:**

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

**Verify:** 25 staged scenes (20 train + 5 val).

```bash
ls /workspace/data/ForAINetV2/train_val_data_exp2A | wc -l
```

### E2A.3 — Export the exp2A instance-data cache

Writes the npys into `/workspace/data/ForAINetV2/forainetv2_instance_data_exp2A/`.

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

**Verify:** the cache directory is populated.

```bash
ls /workspace/data/ForAINetV2/forainetv2_instance_data_exp2A | head
```

### E2A.4 — Create the train/val PKLs

```bash
ff_pkls train_list_exp2_mediumL1_lowL1_fewshot.txt val_list_exp2_mediumL1_lowL1_fewshot.txt
```

> The repo has the L1-only lists (`*_mediumL1_lowL1_fewshot.txt`). The older `*_mediumL1L2lowL1_*`
> names do not exist here and would re-introduce `medium_L2`.

**Verify:**

```bash
ls -l /workspace/data/ForAINetV2/forainetv2_oneformer3d_infos_*_L1.pkl
```

### E2A.5 — Fine-tune

**Run:**

```bash
ff_train exp2A_epoch3000_train
```

**Verify:** checkpoints were written.

```bash
ls /workspace/work_dirs/exp2A_epoch3000_train/*.pth | head
```

### E2A.6 — Held-out evaluation

Pick a checkpoint, then run the two site evals. `save_best='mIoU'` runs during E2A.5, so
`best_mIoU*.pth` and `best_harmonic*.pth` both exist.

```bash
# Balanced (primary selection)
CKPT=/workspace/work_dirs/exp2A_epoch3000_train/best_harmonic_epoch_2000_fix.pth ; TAG=epoch2000
# Semantic-focused:  CKPT=$(ls -1t /workspace/work_dirs/exp2A_epoch3000_train/best_mIoU*.pth | head -n1)
# Last-epoch:        CKPT=/workspace/work_dirs/exp2A_epoch3000_train/epoch_3000.pth ; TAG=epoch3000

ff_eval medium test_list_exp2_holdout_medium_L1.txt test_data_medium_L1 \
  "exp2A_${TAG}_medium_L1" "exp2A_${TAG}_eval_medium_L1" "$CKPT"

ff_eval low    test_list_exp2_holdout_low_L1.txt    test_data_low_L1 \
  "exp2A_${TAG}_low_L1"    "exp2A_${TAG}_eval_low_L1"    "$CKPT"
```

### E2A.7 — Training and validation metrics

```bash
ff_summarize exp2A_epoch3000_train
```

Raw validation lines (validation runs every 100 epochs), and the top-5 val epochs by mIoU:

```bash
grep "Epoch(val)" work_dirs/exp2A_epoch3000_train/*/*.log

grep "Epoch(val)" work_dirs/exp2_variantA_joint/20260312_033733/20260312_033733.log \
  | python3 -c "
import sys, re
rows = []
for line in sys.stdin:
    ep = re.search(r'Epoch\(val\) \[(\d+)\]', line)
    miou = re.search(r'mIoU: ([0-9.]+)', line)
    f1 = re.search(r'F1: ([0-9.]+)', line)
    if ep and miou and f1:
        rows.append((int(ep.group(1)), float(miou.group(1)), float(f1.group(1))))
rows.sort(key=lambda x: x[1], reverse=True)
print('Top 5 by mIoU:')
for ep, miou, f1 in rows[:5]:
    print(f'  epoch={ep:4d}  mIoU={miou:.4f}  F1={f1:.4f}')
"
```

#### Recorded results — E2A.7 (run: 2026-03-12)

| Metric | Best val epoch (1200) | Final val epoch (3000) |
|---|---|---|
| IoU ground | 71.1% | 65.2% |
| IoU wood | 10.3% | 7.8% |
| IoU leaf | 74.2% | 60.5% |
| mIoU | 51.9% | 44.5% |
| mIoU binary (non-tree / tree) | 71.1% / 93.0% = 82.0% | 65.2% / 92.7% = 78.9% |
| F1 / Precision / Recall | 45.0% / 58.7% / 36.5% | 55.4% / 64.7% / 48.4% |
| mMWCov | 41.9% | 55.1% |
| mMUCov | 31.1% | 41.7% |

> **Note:** Best mIoU is at epoch 1200, but epoch 3000 has better F1 (55.4% vs 45.0%) and mMWCov
> (55.1% vs 41.9%) — the model trades semantic accuracy for better instance grouping at later
> epochs. Only `epoch_2998/2999/3000.pth` (plus `last_checkpoint`) were kept in this run, so
> `epoch_1200.pth` cannot be re-tested; use `epoch_2998`/`epoch_2999` if you want extra baselines.

### E2A.8 — Collect test metrics

Inside the container:

```bash
cd /workspace
python3 tools/dvc_collect_metrics.py \
  --work-dir work_dirs/exp2A_epoch3000_eval_medium_L1 \
  --out work_dirs/exp2A_epoch3000_eval_medium_L1/metrics_test_mediumL1.json

python3 tools/dvc_collect_metrics.py \
  --work-dir work_dirs/exp2A_epoch3000_eval_low_L1 \
  --out work_dirs/exp2A_epoch3000_eval_low_L1/metrics_test_lowL1.json
```

On the **host** the eval dirs are root-owned, so use `sudo` and the micromamba interpreter:

```bash
cd /home/nvidia/projects/fvc_structure/ForestFormer3D
sudo /home/nvidia/micromamba/bin/python tools/dvc_collect_metrics.py \
  --work-dir work_dirs/exp2A_epoch3000_eval_medium_L1 \
  --out work_dirs/exp2A_epoch3000_eval_medium_L1/metrics_test_mediumL1.json
```

Per-checkpoint summaries straight from the logs:

```bash
ff_test_summary "exp2A*_eval_medium_L1*" best_harmonic
ff_test_summary "exp2A*_eval_medium_L1*" epoch_3000_fix.pth
ff_test_summary "exp2A*_eval_low_L1*"    best_harmonic
ff_test_summary "exp2A*_eval_low_L1*"    epoch_3000_fix.pth
```

Read IoU / val / test lines directly from an MMEngine log:

```bash
grep -E "Table1Row|Stage1Row|Exp1Row|IoU_ground|IoU_wood|IoU_leaf" work_dirs/exp2A_epoch3000_eval_medium_L1/*/*.log
grep "Epoch(val)"  work_dirs/exp2A_epoch3000_train/*/*.log
grep "Epoch(test)" work_dirs/exp2A_epoch3000_eval_medium_L1/*/*.log
grep "Epoch(test)" work_dirs/exp2A_epoch3000_eval_low_L1/*/*.log
```

#### Recorded results — E2A.8 (run: 2026-03-12, `epoch_3000`)

| Metric | medium_L1 | low_L1 |
|---|---|---|
| IoU ground | 56.3% | 74.3% |
| IoU wood | 4.1% | 26.7% |
| IoU leaf | 86.3% | 0.0% |
| mIoU | 48.9% | 50.5% |
| mIoU binary (non-tree / tree) | 56.3% / 96.8% = 76.5% | 74.3% / 86.5% = 80.4% |
| F1 / Precision / Recall | 39.1% / 58.9% / 29.2% | 0.0% / 0.0% / 0.0% |
| mMWCov | 30.0% | 0.0% |
| mMUCov | 26.2% | 0.0% |

> **Note on low_L1 instance metrics:** all instance metrics (F1, PQ, Cov) are 0 while semantic IoU
> is healthy (80.4% binary) — the backbone classifies points correctly but the instance head
> predicted no valid instances above threshold. Likely cause: the 3 held-out low_L1 tiles contain
> no `leaf`-class points (`IoU_leaf=0`), which breaks the multi-class instance matcher. Re-run with
> the full 15-tile low_L1 pool for stable instance metrics.

#### Checkpoint comparison on held-out test

**medium_L1 (2 scenes)**

| Checkpoint | Prec (%) | Rec (%) | F1 (%) | Cov (%) | Ground (%) | Wood (%) | Leaf (%) | mIoU (%) | Harmonic (%) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `best_harmonic_epoch_2000` | 50.7 | 31.0 | 38.5 | 30.1 | 52.8 | 5.7 | 88.2 | 48.9 | 43.1 |
| `best_mIoU_epoch_300` | 50.0 | 27.4 | 35.4 | 25.2 | 50.3 | 7.0 | 84.4 | 47.2 | 40.5 |
| `epoch_3000` | 58.9 | 29.2 | 39.1 | 26.2 | 56.3 | 4.1 | 86.3 | 48.9 | 43.4 |

**low_L1 (3 scenes, `epoch_3000`)**

| Prec (%) | Rec (%) | F1 (%) | Cov (%) | Ground (%) | Wood (%) | Leaf (%) | mIoU (%) | Harmonic (%) |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.0 | 0.0 | 0.0 | 0.0 | 74.3 | 26.7 | 0.0 | 50.5 | 0.0 |

> **Interpretation:** on medium_L1, `epoch_3000` and `best_harmonic_epoch_2000` are very close —
> `epoch_3000` wins on F1 and harmonic, `best_harmonic_epoch_2000` on recall and coverage. low_L1
> instance metrics are not comparable because GT instances are absent (`Scenes with any GT
> instances: 0/3`).

### Qualitative viewer

Scans all experiment folders under `work_dirs`, lists scenes with top-level `.ply`/`.las` outputs,
and renders side-by-side GT vs prediction for semantic and instance segmentation.

```bash
python3 tools/qualitative_results_app.py \
  --work-dirs-root /media/laura/8402326D023263F8/calperumResearch/site1_1_DD0001/lidar/ForestFormer3D/data/ForAINetV2/work_dirs \
  --host 0.0.0.0 --port 8051
```

(From the parent workspace root, prefix the script path with `ForestFormer3D/`.)

## Variant B — Curriculum fine-tuning (medium-only → low-only)

Two stages with strict held-out testing:

- **Stage 1** — start from the forest-pretrained checkpoint, train on medium-only train/val.
- **Stage 2** — initialize from the **Stage-1** checkpoint (not the forest one), train on low-only.

Rules: never test on tiles used for training; keep the strict held-out lists
(`test_list_exp2_holdout_medium_L1.txt`, `test_list_exp2_holdout_low_L1.txt`); prefer
`best_harmonic` for selection and also report `best_mIoU`.

### E2B.1 — Site-only list generator

Splits the joint exp2 train/val lists by tile prefix. Used by Variant B and by Experiments 3B/3C —
only the prefix and the output names change.

```bash
make_site_lists () {   # PREFIX TRAIN_OUT VAL_OUT
  python - "$1" "$2" "$3" <<'PY'
import sys
from pathlib import Path

prefix, train_out, val_out = sys.argv[1], sys.argv[2], sys.argv[3]
meta = Path('/workspace/data/ForAINetV2/meta_data')
train_all = (meta / 'train_list_exp2_mediumL1_lowL1_fewshot.txt').read_text(encoding='utf-8').splitlines()
val_all = (meta / 'val_list_exp2_mediumL1_lowL1_fewshot.txt').read_text(encoding='utf-8').splitlines()

train = [x.strip() for x in train_all if x.strip().startswith(prefix)]
val = [x.strip() for x in val_all if x.strip().startswith(prefix)]

(meta / train_out).write_text('\n'.join(train) + '\n', encoding='utf-8')
(meta / val_out).write_text('\n'.join(val) + '\n', encoding='utf-8')
print(f'- {train_out}', len(train))
print(f'- {val_out}', len(val))
PY
}
```

| Use | Tile prefix | Train list | Val list | Expected |
|---|---|---|---|---|
| B Stage 1 (medium) | `znorm_3Dlabels_m_L1_` | `train_list_exp2_mediumL1_only.txt` | `val_list_exp2_mediumL1_only.txt` | 10 / 3 |
| B Stage 2 (low) | `znorm_3Dlabels_l_L1_` | `train_list_exp2B_lowL1_only.txt` | `val_list_exp2B_lowL1_only.txt` | 10 / 2 |
| Exp 3B (medium) | `znorm_3Dlabels_m_L1_` | `train_list_exp3B_mediumL1_only.txt` | `val_list_exp3B_mediumL1_only.txt` | 10 / 3 |
| Exp 3C (low) | `znorm_3Dlabels_l_L1_` | `train_list_exp3C_lowL1_only.txt` | `val_list_exp3C_lowL1_only.txt` | 10 / 2 |

### E2B.2 — Stage 1: medium-only, from forest-pretrained

```bash
make_site_lists znorm_3Dlabels_m_L1_ train_list_exp2_mediumL1_only.txt val_list_exp2_mediumL1_only.txt

ff_pkls train_list_exp2_mediumL1_only.txt val_list_exp2_mediumL1_only.txt \
  forainetv2_oneformer3d_infos_train_exp2B_stage1_medium.pkl \
  forainetv2_oneformer3d_infos_val_exp2B_stage1_medium.pkl

ff_train exp2B_stage1_medium_train \
  load_from=/workspace/work_dirs/clean_forestformer/epoch_3000_fix.pth \
  train_dataloader.dataset.ann_file=forainetv2_oneformer3d_infos_train_exp2B_stage1_medium.pkl \
  val_dataloader.dataset.ann_file=forainetv2_oneformer3d_infos_val_exp2B_stage1_medium.pkl

ff_summarize exp2B_stage1_medium_train
```

### E2B.3 — Stage-1 evaluation (held-out medium)

Both checkpoints, each into its own work-dir so the logs stay separate:

```bash
for CK in epoch_3000 best_harmonic_epoch_2400; do
  python /workspace/tools/fix_spconv_checkpoint.py \
    --in-path  /workspace/work_dirs/exp2B_stage1_medium_train/${CK}.pth \
    --out-path /workspace/work_dirs/exp2B_stage1_medium_train/${CK}_fix.pth
done

ff_eval medium test_list_exp2_holdout_medium_L1.txt test_data_medium_L1 \
  exp2B_stage1_medium_only exp2B_stage1_eval_medium_L1 \
  /workspace/work_dirs/exp2B_stage1_medium_train/epoch_3000_fix.pth

ff_eval medium test_list_exp2_holdout_medium_L1.txt test_data_medium_L1 \
  exp2B_stage1_best_harmonic_ep2400_medium_L1 exp2B_stage1_best_harmonic_eval_medium_L1 \
  /workspace/work_dirs/exp2B_stage1_medium_train/best_harmonic_epoch_2400_fix.pth

ff_test_summary "exp2B_stage1*_eval_medium_L1*" epoch_3000_fix.pth
ff_test_summary "exp2B_stage1*_eval_medium_L1*" best_harmonic_epoch_2400_fix.pth
```

### E2B.4 — Stage 2: low-only, initialised from Stage 1

```bash
make_site_lists znorm_3Dlabels_l_L1_ train_list_exp2B_lowL1_only.txt val_list_exp2B_lowL1_only.txt

ff_pkls train_list_exp2B_lowL1_only.txt val_list_exp2B_lowL1_only.txt \
  forainetv2_oneformer3d_infos_train_exp2B_stage2_low.pkl \
  forainetv2_oneformer3d_infos_val_exp2B_stage2_low.pkl

# MUST initialize from the E2B.2 output, not the original forest checkpoint
ff_train exp2B_stage2_low_train_from_ep2400 \
  load_from=/workspace/work_dirs/exp2B_stage1_medium_train/best_harmonic_epoch_2400_fix.pth \
  train_dataloader.dataset.ann_file=forainetv2_oneformer3d_infos_train_exp2B_stage2_low.pkl \
  val_dataloader.dataset.ann_file=forainetv2_oneformer3d_infos_val_exp2B_stage2_low.pkl
```

### E2B.5 — Stage-2 evaluation and metric collection

```bash
S2=/workspace/work_dirs/exp2B_stage2_low_train_from_ep2400
python /workspace/tools/fix_spconv_checkpoint.py \
  --in-path  $S2/best_mIoU_epoch_2200.pth \
  --out-path $S2/best_mIoU_epoch_2200_fix.pth
CKPT=$S2/best_mIoU_epoch_2200_fix.pth

ff_eval medium test_list_exp2_holdout_medium_L1.txt test_data_medium_L1 \
  exp2B_stage2_best_mIoU_ep2200_eval_medium_L1 exp2B_stage2_best_mIoU_ep2200_eval_medium_L1 "$CKPT"

ff_eval low    test_list_exp2_holdout_low_L1.txt    test_data_low_L1 \
  exp2B_stage2_best_mIoU_ep2200_eval_low_L1    exp2B_stage2_best_mIoU_ep2200_eval_low_L1    "$CKPT"

cd /workspace
for d in exp2B_stage1_eval_medium_L1 exp2B_stage2_best_mIoU_ep2200_eval_medium_L1 exp2B_stage2_best_mIoU_ep2200_eval_low_L1; do
  python3 tools/dvc_collect_metrics.py --work-dir work_dirs/$d --out work_dirs/$d/metrics_test.json
done
```

---

# Experiment 3 — Training from scratch

Same data and splits as Experiment 2A, but **no** forest-pretrained weights: `load_from=None`
overrides the `load_from` set by `configs/exp2_finetune_l1_eval_gt.py`, so weights are randomly
initialized. `PREFIX_OPTS` (from `use_cache`) pins `data_prefix` to the exp2A cache.

```bash
use_cache forainetv2_instance_data_exp2A
```

> **Instance-eval note (`UnifiedSegMetric`):** the default `binary_mode='collapsed'` treats wood +
> leaf as one "tree" class for instance metrics. For leaf-only ("strict") evaluation — e.g.
> rangeland where only mallee/leaf has GT instances — append to any eval command:
> `--cfg-options test_evaluator.binary_mode=strict test_evaluator.binary_tree_eval_ids=[3]`

## E3A — Scratch, joint split (medium + low)

### E3A.1 — Train from random init

Reuses E2A.1–E2A.4 as-is.

**Run:**

```bash
ff_train exp3A_scratch_train load_from=None
ff_summarize exp3A_scratch_train
```

### E3A.2 — Evaluate and collect metrics

**Goal:** evaluate both checkpoints on held-out medium (separate work-dirs → separate logs), and low.

**Run:**

```bash
BEST_HARMONIC_CKPT=$(ls -1t /workspace/work_dirs/exp3A_scratch_train/best_harmonic*.pth | head -n1)
EPOCH_3000_CKPT=/workspace/work_dirs/exp3A_scratch_train/epoch_3000_fix.pth

ff_eval medium test_list_exp2_holdout_medium_L1.txt test_data_medium_L1 \
  exp3A_scratch_best_harmonic_medium_L1 exp3A_scratch_best_harmonic_eval_medium_L1 "$BEST_HARMONIC_CKPT"

ff_eval medium test_list_exp2_holdout_medium_L1.txt test_data_medium_L1 \
  exp3A_scratch_epoch_3000_medium_L1 exp3A_scratch_epoch_3000_eval_medium_L1 "$EPOCH_3000_CKPT"

ff_eval low    test_list_exp2_holdout_low_L1.txt    test_data_low_L1 \
  exp3A_scratch_low_L1 exp3A_scratch_eval_low_L1 "$BEST_HARMONIC_CKPT"

ff_test_summary "exp3A_scratch*_eval_medium_L1*" best_harmonic
ff_test_summary "exp3A_scratch*_eval_medium_L1*" epoch_3000_fix.pth
ff_test_summary "exp3A_scratch*_eval_low_L1*"    best_harmonic
ff_test_summary "exp3A_scratch*_eval_low_L1*"    epoch_3000_fix.pth

cd /workspace
for d in exp3A_scratch_eval_medium_L1 exp3A_scratch_eval_low_L1; do
  python3 tools/dvc_collect_metrics.py --work-dir work_dirs/$d --out work_dirs/$d/metrics_test.json
done
```

> **Reading scratch `F1=0` on medium_L1:** the held-out medium_L1 scenes *do* contain GT instances,
> so `F1=0` means the model emitted no usable positive instance ids (e.g. `Pred unique ids sample:
> [-1]`, `Scenes with any Pred instances: 0/2`) — an **instance-decoder failure**, not a missing-GT
> artifact. Note also that with the base forest config the class names `ground/wood/leaf` are really
> output slots `0/1/2`; for Calperum rangeland they are an approximate transfer target, not a
> literal taxonomy. The forest-pretrained checkpoint can still transfer usable instance structure to
> medium_L1 while random init learns some semantics but fails to produce valid instance masks.

## E3B / E3C — Scratch, single-site

Identical recipe; set the variant block, then run the shared commands.

### E3B/E3C.1 — Select the variant block

```bash
# --- Experiment 3B: medium-only (train 10 / val 3 / test 2) ---
PREFIX=znorm_3Dlabels_m_L1_ ; SPLIT=medium
TRAIN_LIST=train_list_exp3B_mediumL1_only.txt ; VAL_LIST=val_list_exp3B_mediumL1_only.txt
TRAIN_PKL=forainetv2_oneformer3d_infos_train_exp3B_scratch_medium.pkl
VAL_PKL=forainetv2_oneformer3d_infos_val_exp3B_scratch_medium.pkl
TEST_LIST=test_list_exp2_holdout_medium_L1.txt ; PLY_DIR=test_data_medium_L1
WORK=exp3B_scratch_medium_train ; EVAL_WORK=exp3B_scratch_medium_eval_medium_L1
TAG=exp3B_scratch_medium_only_medium_L1

# --- Experiment 3C: low-only (train 10 / val 2 / test 3) ---
PREFIX=znorm_3Dlabels_l_L1_ ; SPLIT=low
TRAIN_LIST=train_list_exp3C_lowL1_only.txt ; VAL_LIST=val_list_exp3C_lowL1_only.txt
TRAIN_PKL=forainetv2_oneformer3d_infos_train_exp3C_scratch_low.pkl
VAL_PKL=forainetv2_oneformer3d_infos_val_exp3C_scratch_low.pkl
TEST_LIST=test_list_exp2_holdout_low_L1.txt ; PLY_DIR=test_data_low_L1
WORK=exp3C_scratch_low_train ; EVAL_WORK=exp3C_scratch_low_eval_low_L1
TAG=exp3C_scratch_low_only_low_L1
```

### E3B/E3C.2 — Shared commands

```bash
make_site_lists "$PREFIX" "$TRAIN_LIST" "$VAL_LIST"
ff_pkls "$TRAIN_LIST" "$VAL_LIST" "$TRAIN_PKL" "$VAL_PKL"

ff_train "$WORK" load_from=None \
  train_dataloader.dataset.ann_file="$TRAIN_PKL" \
  val_dataloader.dataset.ann_file="$VAL_PKL"

# Optional, if epoch_3000.pth raises spconv size-mismatch errors
python /workspace/tools/fix_spconv_checkpoint.py \
  --in-path  /workspace/work_dirs/$WORK/epoch_3000.pth \
  --out-path /workspace/work_dirs/$WORK/epoch_3000_fix.pth

ff_summarize "$WORK"

BEST_CKPT=$(ls -1t /workspace/work_dirs/$WORK/best_harmonic*.pth | head -n1)
ff_eval "$SPLIT" "$TEST_LIST" "$PLY_DIR" "$TAG" "$EVAL_WORK" "$BEST_CKPT"
ff_test_summary "$EVAL_WORK" best_harmonic
```
