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

is_port_free() {
    local port="$1"
    python - "$port" <<'PY'
import socket
import sys

port = int(sys.argv[1])

def can_bind(family, host):
    s = socket.socket(family, socket.SOCK_STREAM)
    try:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        if family == socket.AF_INET6:
            s.bind((host, port, 0, 0))
        else:
            s.bind((host, port))
        return True
    except OSError:
        return False
    finally:
        s.close()

ok_v4 = can_bind(socket.AF_INET, '0.0.0.0')
ok_v6 = True
try:
    ok_v6 = can_bind(socket.AF_INET6, '::')
except OSError:
    ok_v6 = False

sys.exit(0 if (ok_v4 and ok_v6) else 1)
PY
}

find_free_port() {
    local start_port="$1"
    python - "$start_port" <<'PY'
import socket
import sys

start_port = int(sys.argv[1])

def can_bind(port: int, family, host: str) -> bool:
    s = socket.socket(family, socket.SOCK_STREAM)
    try:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        if family == socket.AF_INET6:
            s.bind((host, port, 0, 0))
        else:
            s.bind((host, port))
        return True
    except OSError:
        return False
    finally:
        s.close()

for port in range(start_port, start_port + 200):
    ok_v4 = can_bind(port, socket.AF_INET, '0.0.0.0')
    ok_v6 = True
    try:
        ok_v6 = can_bind(port, socket.AF_INET6, '::')
    except OSError:
        ok_v6 = False
    if ok_v4 and ok_v6:
        print(port)
        raise SystemExit(0)

raise SystemExit(f"No free port found in range {start_port}..{start_port + 199}")
PY
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="$SCRIPT_DIR/..":${PYTHONPATH:-}

if ! is_port_free "$PORT"; then
    if [[ "$NNODES" != "1" ]]; then
        echo "Error: PORT=$PORT is already in use. For multi-node runs you must pick a free port and export the same PORT on every node." >&2
        echo "Example: PORT=29601 NNODES=$NNODES NODE_RANK=$NODE_RANK MASTER_ADDR=$MASTER_ADDR $0 $CONFIG $CHECKPOINT $GPUS" >&2
        exit 1
    fi

    NEW_PORT="$(find_free_port "$PORT")"
    echo "Info: PORT=$PORT is in use; switching to PORT=$NEW_PORT" >&2
    PORT="$NEW_PORT"
fi

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
