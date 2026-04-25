#!/usr/bin/env bash
set -euo pipefail

bundle_root=${1:?Usage: sync_copc_to_r2.sh <bundle-root> [bucket] [prefix]}
bucket=${2:-terrascientia}
prefix=${3:-fvc_structure-viewer/lidar}

access_key_id=${AWS_ACCESS_KEY_ID:-${R2_ACCESS_KEY_ID:-}}
secret_access_key=${AWS_SECRET_ACCESS_KEY:-${R2_SECRET_ACCESS_KEY:-}}

: "${R2_ACCOUNT_ID:?Set R2_ACCOUNT_ID in the environment}"
: "${access_key_id:?Set AWS_ACCESS_KEY_ID or R2_ACCESS_KEY_ID in the environment}"
: "${secret_access_key:?Set AWS_SECRET_ACCESS_KEY or R2_SECRET_ACCESS_KEY in the environment}"

endpoint_url="https://${R2_ACCOUNT_ID}.r2.cloudflarestorage.com"

AWS_ACCESS_KEY_ID="${access_key_id}" AWS_SECRET_ACCESS_KEY="${secret_access_key}" \
  aws s3 sync "${bundle_root}/copc" "s3://${bucket}/${prefix}/copc/" --endpoint-url "${endpoint_url}"
AWS_ACCESS_KEY_ID="${access_key_id}" AWS_SECRET_ACCESS_KEY="${secret_access_key}" \
  aws s3 sync "${bundle_root}/stac" "s3://${bucket}/${prefix}/stac/" --endpoint-url "${endpoint_url}"
