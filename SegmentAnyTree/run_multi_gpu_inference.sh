#!/bin/bash
# Run SegmentAnyTree inference across multiple GPUs in parallel.
# Each GPU processes a subset of tiles in its own Docker container.
#
# Usage: bash run_multi_gpu_inference.sh
#
# The container entrypoint (run_oracle_pipeline.sh) expects:
#   bucket_in_folder  -> input PLY files
#   bucket_out_folder -> receives results.zip at the end
#
# We give each GPU its own in/out folders, then merge.

set -euo pipefail

shopt -s nullglob

INPUT_DIR=${INPUT_DIR:-"/home/nvidia/projects/fvc_structure/ForestFormer3D/data/ForAINetV2/test_data"}
OUTPUT_BASE=${OUTPUT_BASE:-"/data/calperumResearch/segmentanytree/output/ForAINetV2"}
IMAGE=${IMAGE:-"segment-any-tree:depsfix"}
EVAL_YAML=${EVAL_YAML:-"/home/nvidia/projects/fvc_structure/SegmentAnyTree/conf/eval.yaml"}
EVAL_PY=${EVAL_PY:-"/home/nvidia/projects/fvc_structure/SegmentAnyTree/eval.py"}

# Input distribution method:
# - copy (default): copy tiles into per-GPU input folders (uses reflink when supported)
# - symlink: create symlinks pointing to /input_all/<file> inside the container and mount INPUT_DIR -> /input_all (read-only)
INPUT_DISTRIBUTION=${INPUT_DISTRIBUTION:-copy}

# If set to 1, remove large intermediate folders under RUN_ROOT after results are merged.
# This can save a lot of disk (e.g., docker_in_folder/tmp_{in,out}_folder).
CLEANUP_TEMP=${CLEANUP_TEMP:-0}

# Run containers as this user to avoid root-owned outputs on bind mounts.
# You can override: DOCKER_USER=0:0 bash run_multi_gpu_inference.sh
DOCKER_USER=${DOCKER_USER:-"$(id -u):$(id -g)"}

# Execution mode: parallel (default) or sequential (one GPU container at a time)
MODE=${MODE:-parallel}

# Heartbeat interval in seconds. Set to 0 to disable.
HEARTBEAT_SECS=${HEARTBEAT_SECS:-60}

# Thread caps inside the container (helps avoid OpenMP oversubscription / stalls)
OMP_NUM_THREADS=${OMP_NUM_THREADS:-2}
MKL_NUM_THREADS=${MKL_NUM_THREADS:-1}
OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-1}
NUMEXPR_NUM_THREADS=${NUMEXPR_NUM_THREADS:-1}

# Unique run id for container names
RUN_ID=${RUN_ID:-$(date +%Y%m%d_%H%M%S)}

# Run-scoped root directory (keeps bind-mount source paths stable even if OUTPUT_BASE is cleaned)
RUN_ROOT="$OUTPUT_BASE/runs/$RUN_ID"

# GPUs to use
# Override with: GPUS_OVERRIDE="0 1 2" bash run_multi_gpu_inference.sh
if [ -n "${GPUS_OVERRIDE:-}" ]; then
    read -r -a GPUS <<<"$GPUS_OVERRIDE"
else
    GPUS=(1 2 3 4 5 6 7)
fi

# Collect all PLY files
FILES=($(ls "$INPUT_DIR"/*.ply 2>/dev/null | sort))
NUM_FILES=${#FILES[@]}
NUM_GPUS=${#GPUS[@]}

if [ "$NUM_FILES" -eq 0 ]; then
    echo "No .ply files found in $INPUT_DIR"
    exit 1
fi

echo "Distributing $NUM_FILES tiles across $NUM_GPUS GPUs..."

echo "INPUT_DIR:  $INPUT_DIR"
echo "OUTPUT_BASE: $OUTPUT_BASE"
echo "RUN_ROOT:   $RUN_ROOT"
echo "DOCKER_USER: $DOCKER_USER"

rm -rf "$RUN_ROOT" || {
    echo "ERROR: failed to remove existing RUN_ROOT (permission issue)."
    echo "Fix with: sudo rm -rf '$RUN_ROOT'  OR set a new RUN_ID."
    exit 1
}
mkdir -p "$RUN_ROOT"

# Create per-GPU input/output directories and distribute files
for i in "${!GPUS[@]}"; do
    gpu_id=${GPUS[$i]}
    gpu_input="$RUN_ROOT/gpu${gpu_id}_input"
    gpu_output="$RUN_ROOT/gpu${gpu_id}_output"
    rm -rf "$gpu_input" "$gpu_output"
    mkdir -p "$gpu_input" "$gpu_output"
done

# Round-robin distribute files across GPUs
for i in "${!FILES[@]}"; do
    gpu_idx=$((i % NUM_GPUS))
    gpu_id=${GPUS[$gpu_idx]}
    gpu_input="$RUN_ROOT/gpu${gpu_id}_input"
    bn=$(basename "${FILES[$i]}")
    if [ "$INPUT_DISTRIBUTION" = "symlink" ]; then
        ln -sf "/input_all/$bn" "$gpu_input/$bn"
    else
        # Try cheap CoW copies when supported by the filesystem, otherwise fall back to normal copy.
        cp --reflink=auto "${FILES[$i]}" "$gpu_input/"
    fi
done

# Print distribution
for i in "${!GPUS[@]}"; do
    gpu_id=${GPUS[$i]}
    gpu_input="$RUN_ROOT/gpu${gpu_id}_input"
    count=$(ls "$gpu_input"/*.ply 2>/dev/null | wc -l)
    echo "  GPU $gpu_id: $count tiles"
done

docker_run_one() {
    local gpu_id gpu_input gpu_output gpu_tmp gpu_datascience gpu_ckpt_dataroot cname
    gpu_id="$1"
    gpu_input="$RUN_ROOT/gpu${gpu_id}_input"
    gpu_output="$RUN_ROOT/gpu${gpu_id}_output"
    gpu_tmp="$RUN_ROOT/gpu${gpu_id}_tmp"
    gpu_datascience="$RUN_ROOT/gpu${gpu_id}_datascience"
    # The checkpoint config expects to write processed cache under this path.
    # We bind-mount it to a writable location on /data.
    gpu_ckpt_dataroot="$RUN_ROOT/gpu${gpu_id}_ckpt_dataroot"
    cname="sat_${RUN_ID}_gpu${gpu_id}"

    mkdir -p "$gpu_tmp" "$gpu_datascience" "$gpu_ckpt_dataroot/treeinsfused/processed_0.2_test"

    local input_all_mount=()
    if [ "$INPUT_DISTRIBUTION" = "symlink" ]; then
        input_all_mount=(--mount "type=bind,source=$INPUT_DIR,target=/input_all,readonly")
    fi

    # Clean up any old container name (in case of reruns)
    docker rm -f "$cname" >/dev/null 2>&1 || true

    docker run --rm --name "$cname" --gpus "device=$gpu_id" \
        --user "$DOCKER_USER" \
        --ipc=host \
        --shm-size 8g \
        -e DEBUG_MODE=true \
        -e OMP_NUM_THREADS="$OMP_NUM_THREADS" \
        -e MKL_NUM_THREADS="$MKL_NUM_THREADS" \
        -e OPENBLAS_NUM_THREADS="$OPENBLAS_NUM_THREADS" \
        -e NUMEXPR_NUM_THREADS="$NUMEXPR_NUM_THREADS" \
        -e MEANSHIFT_USE_MP=0 \
        -e MEANSHIFT_NPROCS=1 \
        -e NUMBA_DISABLE_CACHE=1 \
        -e NUMBA_CACHE_DIR=/tmp/numba_cache \
        -e PYTHONUNBUFFERED=1 \
        -e TMPDIR=/tmp \
        --mount type=bind,source="$gpu_tmp",target=/tmp \
        --mount type=bind,source="$gpu_datascience",target=/home/datascience \
        "${input_all_mount[@]}" \
        --mount type=bind,source="$gpu_input",target=/home/nibio/mutable-outside-world/bucket_in_folder \
        --mount type=bind,source="$gpu_output",target=/home/nibio/mutable-outside-world/bucket_out_folder \
        --mount type=bind,source="$gpu_ckpt_dataroot",target=/home/nibio/mutable-outside-world/processed_data_ready_for_training_sparse_1000_500_100_10 \
        --mount type=bind,source="$EVAL_YAML",target=/home/nibio/mutable-outside-world/conf/eval.yaml \
        --mount type=bind,source="$EVAL_PY",target=/home/nibio/mutable-outside-world/eval.py \
        --mount type=bind,source="/home/nvidia/projects/fvc_structure/SegmentAnyTree/run_oracle_pipeline.sh",target=/home/nibio/mutable-outside-world/run_oracle_pipeline.sh \
        --mount type=bind,source="/home/nvidia/projects/fvc_structure/SegmentAnyTree/nibio_inference/modify_eval.py",target=/home/nibio/mutable-outside-world/nibio_inference/modify_eval.py \
        --mount type=bind,source="/home/nvidia/projects/fvc_structure/SegmentAnyTree/nibio_inference/merge_pt_ss_is.py",target=/home/nibio/mutable-outside-world/nibio_inference/merge_pt_ss_is.py \
        --mount type=bind,source="/home/nvidia/projects/fvc_structure/SegmentAnyTree/nibio_inference/merge_pt_ss_is_knn.py",target=/home/nibio/mutable-outside-world/nibio_inference/merge_pt_ss_is_knn.py \
        --mount type=bind,source="/home/nvidia/projects/fvc_structure/SegmentAnyTree/torch_points3d/datasets/panoptic/treeins.py",target=/home/nibio/mutable-outside-world/torch_points3d/datasets/panoptic/treeins.py \
        --mount type=bind,source="/home/nvidia/projects/fvc_structure/SegmentAnyTree/torch_points3d/utils/meanshift_cluster.py",target=/home/nibio/mutable-outside-world/torch_points3d/utils/meanshift_cluster.py \
        --mount type=bind,source="/home/nvidia/projects/fvc_structure/SegmentAnyTree/torch_points3d/metrics/confusion_matrix.py",target=/home/nibio/mutable-outside-world/torch_points3d/metrics/confusion_matrix.py \
        --mount type=bind,source="/home/nvidia/projects/fvc_structure/SegmentAnyTree/torch_points3d/metrics/panoptic_tracker_pointgroup_treeins.py",target=/home/nibio/mutable-outside-world/torch_points3d/metrics/panoptic_tracker_pointgroup_treeins.py \
        "$IMAGE" > "$RUN_ROOT/gpu${gpu_id}.log" 2>&1
}

heartbeat_one() {
    local gpu_id cname gpu_output zip_path zip_size epoch zip_age
    gpu_id="$1"
    cname="sat_${RUN_ID}_gpu${gpu_id}"
    gpu_output="$RUN_ROOT/gpu${gpu_id}_output"
    zip_path="$gpu_output/results.zip"
    epoch=$(date +%s)

    echo ""
    echo "==== HEARTBEAT $(date) (RUN_ID=$RUN_ID, GPU=$gpu_id, MODE=$MODE) ===="
    docker stats --no-stream --format "{{.Name}}\t{{.CPUPerc}}\t{{.MemUsage}}" 2>/dev/null | grep -E "^(${cname})\b" || true

    # Prefer the zip in the mounted output folder, but some runs may only
    # have it under the /home/datascience bind mount.
    if [ -f "$zip_path" ]; then
        zip_size=$(du -h "$zip_path" 2>/dev/null | awk '{print $1}')
        zip_age=$((epoch - $(stat -c %Y "$zip_path" 2>/dev/null || echo "$epoch")))
        echo "GPU $gpu_id: results.zip present in gpu_output (${zip_size}), age=${zip_age}s"
        return 0
    fi

    alt_zip="$RUN_ROOT/gpu${gpu_id}_datascience/results.zip"
    if [ -f "$alt_zip" ]; then
        zip_size=$(du -h "$alt_zip" 2>/dev/null | awk '{print $1}')
        zip_age=$((epoch - $(stat -c %Y "$alt_zip" 2>/dev/null || echo "$epoch")))
        echo "GPU $gpu_id: results.zip present in gpu_datascience (${zip_size}), age=${zip_age}s"
        return 0
    fi

    echo "GPU $gpu_id: results.zip not yet"
}

heartbeat_all() {
    local now epoch regex
    now=$(date)
    epoch=$(date +%s)

    echo ""
    echo "==== HEARTBEAT $now (RUN_ID=$RUN_ID, MODE=$MODE) ===="

    regex=$(IFS='|'; echo "${CONTAINER_NAMES[*]}")
    docker stats --no-stream --format "{{.Name}}\t{{.CPUPerc}}\t{{.MemUsage}}" 2>/dev/null | grep -E "^(${regex})\b" || true

    for i in "${!GPUS[@]}"; do
        local gpu_id gpu_output zip_path zip_size zip_age alt_zip
        gpu_id=${GPUS[$i]}
        gpu_output="$RUN_ROOT/gpu${gpu_id}_output"
        zip_path="$gpu_output/results.zip"
        if [ -f "$zip_path" ]; then
            zip_size=$(du -h "$zip_path" 2>/dev/null | awk '{print $1}')
            zip_age=$((epoch - $(stat -c %Y "$zip_path" 2>/dev/null || echo "$epoch")))
            echo "GPU $gpu_id: results.zip present in gpu_output (${zip_size}), age=${zip_age}s"
        else
            alt_zip="$RUN_ROOT/gpu${gpu_id}_datascience/results.zip"
            if [ -f "$alt_zip" ]; then
                zip_size=$(du -h "$alt_zip" 2>/dev/null | awk '{print $1}')
                zip_age=$((epoch - $(stat -c %Y "$alt_zip" 2>/dev/null || echo "$epoch")))
                echo "GPU $gpu_id: results.zip present in gpu_datascience (${zip_size}), age=${zip_age}s"
            else
                echo "GPU $gpu_id: results.zip not yet"
            fi
        fi
    done
}

PIDS=()
CONTAINER_NAMES=()

if [ "$MODE" = "parallel" ]; then
    # Launch one container per GPU in parallel
    for i in "${!GPUS[@]}"; do
        gpu_id=${GPUS[$i]}
        cname="sat_${RUN_ID}_gpu${gpu_id}"

        echo "Starting container on GPU $gpu_id..."
        docker_run_one "$gpu_id" &
        pid=$!
        PIDS+=($pid)
        CONTAINER_NAMES+=("$cname")
        echo "  PID: $pid, log: $RUN_ROOT/gpu${gpu_id}.log"
    done

    echo ""
    echo "All $NUM_GPUS containers launched. Waiting for completion..."
    echo "Monitor with: tail -f $RUN_ROOT/gpu*.log"
else
    echo ""
    echo "MODE=sequential: running one GPU container at a time"
    echo "Monitor with: tail -f $RUN_ROOT/gpu*.log"
fi

# Wait for all containers
FAILED=0

if [ "$MODE" = "parallel" ]; then
    if [ "${HEARTBEAT_SECS}" -gt 0 ]; then
        heartbeat_all
    fi

    while true; do
        any_alive=0
        for pid in "${PIDS[@]}"; do
            if kill -0 "$pid" >/dev/null 2>&1; then
                any_alive=1
                break
            fi
        done
        if [ "$any_alive" -eq 0 ]; then
            break
        fi
        if [ "${HEARTBEAT_SECS}" -gt 0 ]; then
            sleep "${HEARTBEAT_SECS}"
            heartbeat_all
        else
            sleep 5
        fi
    done

    for i in "${!PIDS[@]}"; do
        pid=${PIDS[$i]}
        gpu_id=${GPUS[$i]}
        if ! wait "$pid"; then
            echo "GPU $gpu_id container (PID $pid) had errors. Check $RUN_ROOT/gpu${gpu_id}.log"
            FAILED=$((FAILED + 1))
        else
            echo "GPU $gpu_id container finished successfully."
        fi
    done
else
    for i in "${!GPUS[@]}"; do
        gpu_id=${GPUS[$i]}
        echo ""
        echo "Starting container on GPU $gpu_id (sequential)..."
        docker_run_one "$gpu_id" &
        pid=$!
        echo "  PID: $pid, log: $RUN_ROOT/gpu${gpu_id}.log"

        while kill -0 "$pid" >/dev/null 2>&1; do
            if [ "${HEARTBEAT_SECS}" -gt 0 ]; then
                heartbeat_one "$gpu_id"
                sleep "${HEARTBEAT_SECS}"
            else
                sleep 5
            fi
        done

        if ! wait "$pid"; then
            echo "GPU $gpu_id container (PID $pid) had errors. Check $RUN_ROOT/gpu${gpu_id}.log"
            FAILED=$((FAILED + 1))
        else
            echo "GPU $gpu_id container finished successfully."
        fi
    done
fi

# Merge results: container writes results.zip to bucket_out_folder
MERGED="$OUTPUT_BASE/final_results/$RUN_ID"
MERGED_RESULTS="$MERGED/results"
mkdir -p "$MERGED_RESULTS"

ensure_results_zip() {
    # Ensure "$RUN_ROOT/gpu${gpu_id}_output/results.zip" exists.
    # Some runs might only have the zip under the gpu*_datascience mount, or
    # only have final_results files but no zip.
    local gpu_id gpu_output gpu_datascience out_zip alt_zip alt_zip2 final_dir2 final_dir3
    gpu_id="$1"
    gpu_output="$RUN_ROOT/gpu${gpu_id}_output"
    gpu_datascience="$RUN_ROOT/gpu${gpu_id}_datascience"
    out_zip="$gpu_output/results.zip"
    alt_zip="$gpu_datascience/results.zip"
    alt_zip2="$gpu_datascience/results/results.zip"

    if [ -f "$out_zip" ]; then
        return 0
    fi
    if [ -f "$alt_zip" ]; then
        cp -f "$alt_zip" "$out_zip"
        return 0
    fi
    if [ -f "$alt_zip2" ]; then
        cp -f "$alt_zip2" "$out_zip"
        return 0
    fi

    # Fallback: build results.zip from final_results laz/las if present.
    final_dir2="$gpu_datascience/docker_out_folder/final_results"
    final_dir3="$gpu_datascience/tmp_out_folder/final_results"

    if [ -d "$final_dir2" ] || [ -d "$final_dir3" ]; then
        python3 - "$gpu_output" "$final_dir2" "$final_dir3" <<'PY'
import os
import sys
import zipfile

gpu_output = sys.argv[1]
final_dirs = sys.argv[2:]
out_zip = os.path.join(gpu_output, "results.zip")

files = []
for d in final_dirs:
    if not d or not os.path.isdir(d):
        continue
    for name in os.listdir(d):
        if name.lower().endswith((".laz", ".las")):
            p = os.path.join(d, name)
            if os.path.isfile(p):
                files.append(p)

files = sorted(set(files))
if not files:
    sys.exit(1)

os.makedirs(gpu_output, exist_ok=True)
with zipfile.ZipFile(out_zip, "w", compression=zipfile.ZIP_DEFLATED) as zf:
    for p in files:
        zf.write(p, arcname=os.path.basename(p))
print(out_zip)
PY
        >/dev/null 2>&1 || true
    fi
}

for i in "${!GPUS[@]}"; do
    gpu_id=${GPUS[$i]}
    gpu_output="$RUN_ROOT/gpu${gpu_id}_output"

    ensure_results_zip "$gpu_id" || true

    # The container outputs a results.zip
    if [ -f "$gpu_output/results.zip" ]; then
        echo "Extracting GPU $gpu_id results..."
        unzip -o "$gpu_output/results.zip" -d "$MERGED/" 2>/dev/null || true
    else
        echo "WARNING: No results.zip from GPU $gpu_id"
    fi
done

# Flatten extracted paths (zips often contain absolute-like paths such as home/datascience/results/...)
find "$MERGED" -type f \( -iname '*.laz' -o -iname '*.las' \) -print0 2>/dev/null \
  | while IFS= read -r -d '' f; do
        bn=$(basename "$f")
        if [ ! -f "$MERGED_RESULTS/$bn" ]; then
            mv "$f" "$MERGED_RESULTS/$bn" 2>/dev/null || true
        fi
    done

echo ""
if [ "$FAILED" -gt 0 ]; then
    echo "$FAILED container(s) had errors. Check logs in $RUN_ROOT/gpu*.log"
fi
echo "Merged results in: $MERGED"
echo "Total result files: $(find "$MERGED_RESULTS" -maxdepth 1 -type f \( -iname '*.laz' -o -iname '*.las' \) | wc -l)"

if [ "$CLEANUP_TEMP" = "1" ]; then
    echo ""
    echo "CLEANUP_TEMP=1: removing large intermediate folders under $RUN_ROOT ..."
    for i in "${!GPUS[@]}"; do
        gpu_id=${GPUS[$i]}
        rm -rf \
            "$RUN_ROOT/gpu${gpu_id}_tmp" \
            "$RUN_ROOT/gpu${gpu_id}_datascience/docker_in_folder" \
            "$RUN_ROOT/gpu${gpu_id}_datascience/tmp_in_folder" \
            "$RUN_ROOT/gpu${gpu_id}_datascience/tmp_out_folder" \
            "$RUN_ROOT/gpu${gpu_id}_datascience/results" \
            "$RUN_ROOT/gpu${gpu_id}_datascience/results.zip" \
            2>/dev/null || true
    done
fi
