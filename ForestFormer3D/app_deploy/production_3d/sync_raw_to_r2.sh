#!/usr/bin/env bash
set -euo pipefail

raw_root=${1:?Usage: sync_raw_to_r2.sh <raw-root> [bucket] [prefix]}
bucket=${2:-terrascientia}
prefix=${3:-fvc_structure-viewer/lidar/raw}

: "${R2_ACCOUNT_ID:?Set R2_ACCOUNT_ID in the environment}"
: "${AWS_ACCESS_KEY_ID:?Set AWS_ACCESS_KEY_ID in the environment}"
: "${AWS_SECRET_ACCESS_KEY:?Set AWS_SECRET_ACCESS_KEY in the environment}"

endpoint_url="https://${R2_ACCOUNT_ID}.r2.cloudflarestorage.com"

aws s3 sync "${raw_root}" "s3://${bucket}/${prefix}/" \
  --endpoint-url "${endpoint_url}" \
  --exclude "*" \
  --include "*.ply" \
  --include "*.las" \
  --include "*.laz"