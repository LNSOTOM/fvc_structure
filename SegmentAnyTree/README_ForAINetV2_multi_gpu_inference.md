# ForAINetV2 multi-GPU inference (FOR-instanceV2)

This workspace runs SegmentAnyTree inference on the **ForAINetV2** test tiles (`.ply`) using **one Docker container per GPU** and then merges the results.

The entrypoint script is:
- `SegmentAnyTree/run_multi_gpu_inference.sh`

## Prerequisites

- Linux host with NVIDIA GPUs
- Docker + NVIDIA Container Toolkit working (so `docker run --gpus ...` works)
- Input tiles exist as `.ply` files in a directory
- Docker image available locally (default used by the script):
  - `segment-any-tree:depsfix`

If your local image name is different, set `IMAGE=...` in the command below.

## One command to run (recommended)

From the repo root (`/home/nvidia/projects/fvc_structure`):

```bash
OUTPUT_BASE=/data/calperumResearch/segmentanytree/output/ForAINetV2 \
INPUT_DIR=/home/nvidia/projects/fvc_structure/ForestFormer3D/data/ForAINetV2/test_data \
GPUS_OVERRIDE="1 2 3 4 5 6 7" \
MODE=parallel \
HEARTBEAT_SECS=60 \
INPUT_DISTRIBUTION=symlink \
CLEANUP_TEMP=1 \
bash SegmentAnyTree/run_multi_gpu_inference.sh
```

What it does:
- Splits `INPUT_DIR/*.ply` across the GPUs in `GPUS_OVERRIDE`
- Runs one container per GPU
- Writes per-GPU logs under the run folder
- Extracts each GPU’s `results.zip` and merges all `.laz/.las` into a single results directory

## Example: run a second experiment (medium_L1)

To run a new experiment on a different test set and keep outputs separate, set:
- `INPUT_DIR` to the new tiles directory
- `OUTPUT_BASE` to a new output root
- (recommended) `RUN_ID` to a descriptive name

Example (writes outputs under `/data/calperumResearch/segmentanytree/output/calperum/medium_L1/...`):

```bash
OUTPUT_BASE=/data/calperumResearch/segmentanytree/output/calperum/medium_L1 \
INPUT_DIR=/home/nvidia/projects/fvc_structure/ForestFormer3D/data/ForAINetV2/test_data_medium_L1 \
GPUS_OVERRIDE="1 2 3 4 5 6 7" \
MODE=parallel \
HEARTBEAT_SECS=60 \
INPUT_DISTRIBUTION=symlink \
CLEANUP_TEMP=1 \
RUN_ID=medium_L1_$(date +%Y%m%d_%H%M%S) \
bash SegmentAnyTree/run_multi_gpu_inference.sh
```

Same command (single line):

```bash
OUTPUT_BASE=/data/calperumResearch/segmentanytree/output/calperum/medium_L1 INPUT_DIR=/home/nvidia/projects/fvc_structure/ForestFormer3D/data/ForAINetV2/test_data_medium_L1 GPUS_OVERRIDE="1 2 3 4 5 6 7" MODE=parallel HEARTBEAT_SECS=60 INPUT_DISTRIBUTION=symlink CLEANUP_TEMP=1 RUN_ID=medium_L1_$(date +%Y%m%d_%H%M%S) bash SegmentAnyTree/run_multi_gpu_inference.sh
```

## Example: run a third experiment (low_L1)

Example (writes outputs under `/data/calperumResearch/segmentanytree/output/calperum/low_L1/...`):

```bash
OUTPUT_BASE=/data/calperumResearch/segmentanytree/output/calperum/low_L1 \
INPUT_DIR=/home/nvidia/projects/fvc_structure/ForestFormer3D/data/ForAINetV2/test_data_low_L1 \
GPUS_OVERRIDE="1 2 3 4 5 6 7" \
MODE=parallel \
HEARTBEAT_SECS=60 \
INPUT_DISTRIBUTION=symlink \
CLEANUP_TEMP=1 \
RUN_ID=low_L1_$(date +%Y%m%d_%H%M%S) \
bash SegmentAnyTree/run_multi_gpu_inference.sh
```

Same command (single line):

```bash
OUTPUT_BASE=/data/calperumResearch/segmentanytree/output/calperum/low_L1 INPUT_DIR=/home/nvidia/projects/fvc_structure/ForestFormer3D/data/ForAINetV2/test_data_low_L1 GPUS_OVERRIDE="1 2 3 4 5 6 7" MODE=parallel HEARTBEAT_SECS=60 INPUT_DISTRIBUTION=symlink CLEANUP_TEMP=1 RUN_ID=low_L1_$(date +%Y%m%d_%H%M%S) bash SegmentAnyTree/run_multi_gpu_inference.sh
```

## Where outputs go

The script chooses a run id like `YYYYMMDD_HHMMSS` (or you can supply `RUN_ID=...`).

- Per-run working directory:
  - `${OUTPUT_BASE}/runs/${RUN_ID}/`
  - Logs:
    - `${OUTPUT_BASE}/runs/${RUN_ID}/gpu<id>.log`
- Merged final results:
  - `${OUTPUT_BASE}/final_results/${RUN_ID}/results/*.laz`

## Post-run evaluation (offline metrics + plots)

After inference completes, you can compute instance-level metrics **directly from the merged LAZ outputs** (no log parsing required).

### 1) Export metrics to CSV

This reads `${OUTPUT_BASE}/final_results/${RUN_ID}/results/*.laz` and writes two CSVs:
- `outputs/<name>_per_file.csv`
- `outputs/<name>_per_region.csv`

Example (matches the run id used above):

```bash
python SegmentAnyTree/metrics/instance_metrics_from_laz.py \
  --input /data/calperumResearch/segmentanytree/output/ForAINetV2/final_results/20260310_052457/results \
  --iou 0.5 \
  --matching per_pred \
  --out-csv outputs/metrics_20260310_iou0p5.csv
```

Optional: include height error metrics (adds extra columns to the per-file CSV), including:
- `height_rmse` (RMSE of $(H_{gt} - H_{pred})$ over matched instances)

Example (Medium_L1 run):

```bash
python SegmentAnyTree/metrics/instance_metrics_from_laz.py \
  --input /data/calperumResearch/segmentanytree/output/calperum/medium_L1/final_results/medium_L1_20260311_012821/results \
  --iou 0.5 \
  --matching per_pred \
  --height-metrics \
  --out-csv outputs/metrics_medium_L1_20260311_012821_iou0p5.csv
```

Outputs:
- `outputs/metrics_20260310_iou0p5_per_file.csv`
- `outputs/metrics_20260310_iou0p5_per_region.csv`

### 2) Plot the CSV metrics

This generates paper-ready PNGs:
- region bar charts: MicroF1 / MacroF1 / MacroCov / MacroWCov
- per-file scatter: F1 vs Coverage (colored by region)

```bash
python plot_instance_metrics_from_csv.py \
  --per-file-csv outputs/metrics_20260310_iou0p5_per_file.csv \
  --per-region-csv outputs/metrics_20260310_iou0p5_per_region.csv \
  --out-dir outputs/plots_iou0p5 \
  --region-sort-by macro_f1
```

Outputs:
- `outputs/plots_iou0p5/region_metrics.png`
- `outputs/plots_iou0p5/file_scatter_f1_cov.png`

## Monitoring progress

In another terminal:

```bash
tail -f /data/calperumResearch/segmentanytree/output/ForAINetV2/runs/<RUN_ID>/gpu*.log
```

The script also prints a **heartbeat** every `HEARTBEAT_SECS` with:
- `docker stats` for the containers
- whether each GPU has produced a `results.zip` yet

## Key environment variables

These are the most useful knobs when re-running:

- `INPUT_DIR` (default: `ForestFormer3D/data/ForAINetV2/test_data`)
  - Directory containing `*.ply` tiles.
- `OUTPUT_BASE` (default: `/data/calperumResearch/segmentanytree/output/ForAINetV2`)
  - Root output directory on the host.
- `GPUS_OVERRIDE` (default: `1 2 3 4 5 6 7`)
  - Which GPU indices to use.
- `MODE` (`parallel` | `sequential`, default: `parallel`)
  - `parallel`: all GPUs run concurrently.
  - `sequential`: one GPU container at a time (useful for debugging).
- `HEARTBEAT_SECS` (default: `60`, set `0` to disable)
- `INPUT_DISTRIBUTION` (`copy` | `symlink`, default: `copy`)
  - `copy`: copies tiles into per-GPU input folders (can use reflinks if supported).
  - `symlink`: creates symlinks and bind-mounts `INPUT_DIR` read-only into the container (saves disk).
- `CLEANUP_TEMP` (`0` | `1`, default: `0`)
  - If `1`, deletes large intermediate folders after merging.
- `RUN_ID` (default: timestamp)
  - Use this to make runs reproducible / avoid overwriting.
- `IMAGE` (default: `segment-any-tree:depsfix`)
  - Docker image name.

Less common (performance / stability):
- `OMP_NUM_THREADS` (default: `2`)
- `MKL_NUM_THREADS` (default: `1`)
- `OPENBLAS_NUM_THREADS` (default: `1`)
- `NUMEXPR_NUM_THREADS` (default: `1`)

## Troubleshooting

- **No tiles found**: ensure `INPUT_DIR` contains `*.ply`.
- **Permission errors under OUTPUT_BASE**:
  - The script runs containers as your user via `DOCKER_USER=$(id -u):$(id -g)`.
  - If you still get permission issues, try: `DOCKER_USER=0:0 ... bash SegmentAnyTree/run_multi_gpu_inference.sh`.
- **Docker image not found**:
  - Set `IMAGE=<your_image_tag>`.
- **Want to re-run without deleting previous output**:
  - Provide a new `RUN_ID=...`.
