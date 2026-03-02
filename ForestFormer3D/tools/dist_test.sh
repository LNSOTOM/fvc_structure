#!/usr/bin/env bash

set -euo pipefail

if [[ $# -lt 3 ]]; then
    echo "Usage: $0 <CONFIG> <CHECKPOINT> <GPUS> [-- other args passed to test.py]" >&2
    exit 2
fi

CONFIG=$1
CHECKPOINT=$2
GPUS=$3
NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
PORT=${PORT:-29600}
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="$SCRIPT_DIR/..":${PYTHONPATH:-}

if command -v torchrun >/dev/null 2>&1; then
    torchrun \
        --nnodes="$NNODES" \
        --node_rank="$NODE_RANK" \
        --master_addr="$MASTER_ADDR" \
        --nproc_per_node="$GPUS" \
        --master_port="$PORT" \
        "$SCRIPT_DIR/test.py" \
        "$CONFIG" \
        "$CHECKPOINT" \
        --launcher pytorch "${@:4}"
else
    python -m torch.distributed.launch \
        --nnodes="$NNODES" \
        --node_rank="$NODE_RANK" \
        --master_addr="$MASTER_ADDR" \
        --nproc_per_node="$GPUS" \
        --master_port="$PORT" \
        "$SCRIPT_DIR/test.py" \
        "$CONFIG" \
        "$CHECKPOINT" \
        --launcher pytorch "${@:4}"
fi
