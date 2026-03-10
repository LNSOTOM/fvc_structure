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

## Where outputs go

The script chooses a run id like `YYYYMMDD_HHMMSS` (or you can supply `RUN_ID=...`).

- Per-run working directory:
  - `${OUTPUT_BASE}/runs/${RUN_ID}/`
  - Logs:
    - `${OUTPUT_BASE}/runs/${RUN_ID}/gpu<id>.log`
- Merged final results:
  - `${OUTPUT_BASE}/final_results/${RUN_ID}/results/*.laz`

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
