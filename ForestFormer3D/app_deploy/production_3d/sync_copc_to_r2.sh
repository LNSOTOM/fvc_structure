#!/usr/bin/env bash
set -euo pipefail

bundle_root=${1:?Usage: sync_copc_to_r2.sh <bundle-root> [bucket] [prefix]}
bucket=${2:-terrascientia}
prefix=${3:-fvc_structure-viewer}

: "${R2_ACCOUNT_ID:?Set R2_ACCOUNT_ID in the environment}"
: "${AWS_ACCESS_KEY_ID:?Set AWS_ACCESS_KEY_ID in the environment}"
: "${AWS_SECRET_ACCESS_KEY:?Set AWS_SECRET_ACCESS_KEY in the environment}"

endpoint_url="https://${R2_ACCOUNT_ID}.r2.cloudflarestorage.com"

aws s3 sync "${bundle_root}/copc" "s3://${bucket}/${prefix}/copc/" --endpoint-url "${endpoint_url}"
aws s3 sync "${bundle_root}/stac" "s3://${bucket}/${prefix}/stac/" --endpoint-url "${endpoint_url}"