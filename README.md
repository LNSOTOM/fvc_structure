# fvc_structure

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

## Reproduction (ForestFormer3D)

If you are running in the ForestFormer3D Docker container, follow the **Quickstart (ForestFormer3D Docker)** section above.

You can also reproduce ForestFormer3D runs with **plain commands** (no extra tooling), or optionally via **DVC stages** (see [dvc.yaml](dvc.yaml) + [params.yaml](params.yaml)).

### 1) Environment

Pick one:

- **Docker (recommended for GPU / ForestFormer3D):** follow [ForestFormer3D/readme.md](ForestFormer3D/readme.md).
- **Conda:**

```bash
conda env create -f environment.yml
conda activate fvc_structure
```

### 2) Data + checkpoint placement (ForestFormer3D)

ForestFormer3D expects the ForAINetV2 dataset + a checkpoint to exist locally. Follow the directory structure in [ForestFormer3D/readme.md](ForestFormer3D/readme.md) (Zenodo download + unzip), especially:

```text
ForestFormer3D/data/ForAINetV2/...
ForestFormer3D/work_dirs/clean_forestformer/epoch_3000_fix.pth
```

### 3) Run the DVC pipeline (ForestFormer3D)

#### Option A: Plain commands (no DVC)

Data preprocessing:

```bash
cd ForestFormer3D/data/ForAINetV2
python batch_load_ForAINetV2_data.py

cd ../../
PYTHONPATH=. python tools/create_data_forainetv2.py forainetv2
```

Quick smoke test (single GPU):

```bash
cd ForestFormer3D
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. \
	python tools/test.py \
	configs/oneformer3d_qs_radius16_qp300_2many.py \
	work_dirs/clean_forestformer/epoch_3000_fix.pth \
	--work-dir work_dirs/local_test/single
```

Training (long-running):

```bash
cd ForestFormer3D
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. \
	python tools/train.py \
	configs/oneformer3d_qs_radius16_qp300_2many.py \
	--work-dir work_dirs/local_train/my_run
```

#### Option B: DVC (optional convenience wrapper)

Install/verify:

```bash
dvc --version || pip install "dvc>=3,<4"
```

Run the same steps via named stages:

```bash
dvc repro prep_forainetv2_instances
dvc repro prep_forainetv2_infos
dvc repro test_ff3d_single

# Optional:
dvc repro test_ff3d_dist
dvc repro train_ff3d
```

### 4) Change configs/GPUs/work dirs

If you use the optional DVC route, edit [params.yaml](params.yaml) (recommended), or override at runtime:

```bash
dvc exp run \
	--set-param train.cuda_devices=0 \
	--set-param train.work_dir=work_dirs/experiments/my_run \
	train_ff3d
```