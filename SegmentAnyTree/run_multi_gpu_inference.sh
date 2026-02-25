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

set -e

INPUT_DIR="/data/calperumResearch/segmentanytree/input/medium"
OUTPUT_BASE="/data/calperumResearch/segmentanytree/output/medium"
IMAGE="segment-any-tree:depsfix"
EVAL_YAML="/home/nvidia/projects/fvc_structure/SegmentAnyTree/conf/eval.yaml"
EVAL_PY="/home/nvidia/projects/fvc_structure/SegmentAnyTree/eval.py"

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

# GPUs to use
GPUS=(0 1 2 3 4 5 7)

# Collect all PLY files
FILES=($(ls "$INPUT_DIR"/*.ply 2>/dev/null | sort))
NUM_FILES=${#FILES[@]}
NUM_GPUS=${#GPUS[@]}

if [ "$NUM_FILES" -eq 0 ]; then
    echo "No .ply files found in $INPUT_DIR"
    exit 1
fi

echo "Distributing $NUM_FILES tiles across $NUM_GPUS GPUs..."

# Create per-GPU input/output directories and distribute files
for i in "${!GPUS[@]}"; do
    gpu_id=${GPUS[$i]}
    gpu_input="$OUTPUT_BASE/gpu${gpu_id}_input"
    gpu_output="$OUTPUT_BASE/gpu${gpu_id}_output"
    rm -rf "$gpu_input" "$gpu_output"
    mkdir -p "$gpu_input" "$gpu_output"
done

# Round-robin distribute files across GPUs (copy, not symlink)
for i in "${!FILES[@]}"; do
    gpu_idx=$((i % NUM_GPUS))
    gpu_id=${GPUS[$gpu_idx]}
    gpu_input="$OUTPUT_BASE/gpu${gpu_id}_input"
    cp "${FILES[$i]}" "$gpu_input/"
done

# Print distribution
for i in "${!GPUS[@]}"; do
    gpu_id=${GPUS[$i]}
    gpu_input="$OUTPUT_BASE/gpu${gpu_id}_input"
    count=$(ls "$gpu_input"/*.ply 2>/dev/null | wc -l)
    echo "  GPU $gpu_id: $count tiles"
done

docker_run_one() {
    local gpu_id gpu_input gpu_output cname
    gpu_id="$1"
    gpu_input="$OUTPUT_BASE/gpu${gpu_id}_input"
    gpu_output="$OUTPUT_BASE/gpu${gpu_id}_output"
    cname="sat_${RUN_ID}_gpu${gpu_id}"

    # Clean up any old container name (in case of reruns)
    docker rm -f "$cname" >/dev/null 2>&1 || true

    docker run --rm --name "$cname" --gpus "device=$gpu_id" \
        --ipc=host \
        --shm-size 8g \
        -e OMP_NUM_THREADS="$OMP_NUM_THREADS" \
        -e MKL_NUM_THREADS="$MKL_NUM_THREADS" \
        -e OPENBLAS_NUM_THREADS="$OPENBLAS_NUM_THREADS" \
        -e NUMEXPR_NUM_THREADS="$NUMEXPR_NUM_THREADS" \
        -e PYTHONUNBUFFERED=1 \
        --mount type=bind,source="$gpu_input",target=/home/nibio/mutable-outside-world/bucket_in_folder \
        --mount type=bind,source="$gpu_output",target=/home/nibio/mutable-outside-world/bucket_out_folder \
        --mount type=bind,source="$EVAL_YAML",target=/home/nibio/mutable-outside-world/conf/eval.yaml \
        --mount type=bind,source="$EVAL_PY",target=/home/nibio/mutable-outside-world/eval.py \
        --mount type=bind,source="/home/nvidia/projects/fvc_structure/SegmentAnyTree/torch_points3d/metrics/confusion_matrix.py",target=/home/nibio/mutable-outside-world/torch_points3d/metrics/confusion_matrix.py \
        --mount type=bind,source="/home/nvidia/projects/fvc_structure/SegmentAnyTree/torch_points3d/metrics/panoptic_tracker_pointgroup_treeins.py",target=/home/nibio/mutable-outside-world/torch_points3d/metrics/panoptic_tracker_pointgroup_treeins.py \
        "$IMAGE" > "$OUTPUT_BASE/gpu${gpu_id}.log" 2>&1
}

heartbeat_one() {
    local gpu_id cname gpu_output zip_path zip_size epoch zip_age
    gpu_id="$1"
    cname="sat_${RUN_ID}_gpu${gpu_id}"
    gpu_output="$OUTPUT_BASE/gpu${gpu_id}_output"
    zip_path="$gpu_output/results.zip"
    epoch=$(date +%s)

    echo ""
    echo "==== HEARTBEAT $(date) (RUN_ID=$RUN_ID, GPU=$gpu_id, MODE=$MODE) ===="
    docker stats --no-stream --format "{{.Name}}\t{{.CPUPerc}}\t{{.MemUsage}}" 2>/dev/null | grep -E "^(${cname})\b" || true
    if [ -f "$zip_path" ]; then
        zip_size=$(du -h "$zip_path" 2>/dev/null | awk '{print $1}')
        zip_age=$((epoch - $(stat -c %Y "$zip_path" 2>/dev/null || echo "$epoch")))
        echo "GPU $gpu_id: results.zip present (${zip_size}), age=${zip_age}s"
    else
        echo "GPU $gpu_id: results.zip not yet"
    fi
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
        local gpu_id gpu_output zip_path zip_size zip_age
        gpu_id=${GPUS[$i]}
        gpu_output="$OUTPUT_BASE/gpu${gpu_id}_output"
        zip_path="$gpu_output/results.zip"
        if [ -f "$zip_path" ]; then
            zip_size=$(du -h "$zip_path" 2>/dev/null | awk '{print $1}')
            zip_age=$((epoch - $(stat -c %Y "$zip_path" 2>/dev/null || echo "$epoch")))
            echo "GPU $gpu_id: results.zip present (${zip_size}), age=${zip_age}s"
        else
            echo "GPU $gpu_id: results.zip not yet"
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
        echo "  PID: $pid, log: $OUTPUT_BASE/gpu${gpu_id}.log"
    done

    echo ""
    echo "All $NUM_GPUS containers launched. Waiting for completion..."
    echo "Monitor with: tail -f $OUTPUT_BASE/gpu*.log"
else
    echo ""
    echo "MODE=sequential: running one GPU container at a time"
    echo "Monitor with: tail -f $OUTPUT_BASE/gpu*.log"
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
            echo "GPU $gpu_id container (PID $pid) had errors. Check $OUTPUT_BASE/gpu${gpu_id}.log"
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
        echo "  PID: $pid, log: $OUTPUT_BASE/gpu${gpu_id}.log"

        while kill -0 "$pid" >/dev/null 2>&1; do
            if [ "${HEARTBEAT_SECS}" -gt 0 ]; then
                heartbeat_one "$gpu_id"
                sleep "${HEARTBEAT_SECS}"
            else
                sleep 5
            fi
        done

        if ! wait "$pid"; then
            echo "GPU $gpu_id container (PID $pid) had errors. Check $OUTPUT_BASE/gpu${gpu_id}.log"
            FAILED=$((FAILED + 1))
        else
            echo "GPU $gpu_id container finished successfully."
        fi
    done
fi

# Merge results: container writes results.zip to bucket_out_folder
MERGED="$OUTPUT_BASE/final_results"
mkdir -p "$MERGED"

for i in "${!GPUS[@]}"; do
    gpu_id=${GPUS[$i]}
    gpu_output="$OUTPUT_BASE/gpu${gpu_id}_output"

    # The container outputs a results.zip
    if [ -f "$gpu_output/results.zip" ]; then
        echo "Extracting GPU $gpu_id results..."
        unzip -o "$gpu_output/results.zip" -d "$MERGED/" 2>/dev/null || true
    else
        echo "WARNING: No results.zip from GPU $gpu_id"
    fi
done

echo ""
if [ "$FAILED" -gt 0 ]; then
    echo "$FAILED container(s) had errors. Check logs in $OUTPUT_BASE/gpu*.log"
fi
echo "Merged results in: $MERGED"
echo "Total result files: $(find "$MERGED" -maxdepth 1 -type f | wc -l)"
