# Quick Setup (ForestFormer3D Docker)

This is a minimal, copy-paste workflow showing:
- how to enter the running container
- data preparation (ForAINetV2)
- training
- testing (your own checkpoint or the provided checkpoint)

For environment/Docker build details see [readme.md](readme.md); for the full experiment
runbook see [README_ForAINetV2_experiments.md](README_ForAINetV2_experiments.md).

## 0) Enter the running container (host)

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

## 1) Data preparation (inside container)

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

## 2) Train (inside container)

Multi-GPU training example (3 GPUs exposed):

```bash
root@<container-id>:/workspace# export PYTHONPATH=/workspace

root@<container-id>:/workspace# CUDA_VISIBLE_DEVICES=0,1,2 \
	bash tools/dist_train.sh \
	configs/oneformer3d_qs_radius16_qp300_2many.py \
	3 \
	--work-dir work_dirs/train_forestformer_3gpu
```

## 3) Test your own trained checkpoint (inside container)

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

### Output directory for predictions (no code edits)

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

## 4) Test the provided pre-trained checkpoint (inside container)

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

## 5) Useful utilities (metrics + plots)

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
