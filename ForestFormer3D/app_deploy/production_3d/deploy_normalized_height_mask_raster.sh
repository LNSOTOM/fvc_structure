#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd "${script_dir}/../../.." && pwd)

bundle_root=${1:-${script_dir}/generated/normalized_height_mask_raster}
bucket=${2:-terrascientia}
prefix=${3:-fvc_structure-viewer/lidar/copc}
python_cmd=${PYTHON:-python}
level_namespace=${LEVEL_NAMESPACE:-lidar_l1}
run_mask=${RUN_MASK:-1}
run_raster=${RUN_RASTER:-1}
run_znorm=${RUN_ZNORM:-1}

medium_mask_input=${MEDIUM_MASK_INPUT:-/media/laura/8402326D023263F8/calperumResearch/site1_1_DD0001/lidar/clipped/3d_structure_mask_malleid}
low_mask_input=${LOW_MASK_INPUT:-/media/laura/8402326D023263F8/calperumResearch/site2_1_DD0011/lidar/clipped/3d_structure_mask_malleid}
medium_raster_input=${MEDIUM_RASTER_INPUT:-/media/laura/8402326D023263F8/calperumResearch/site1_1_DD0001/lidar/clipped/raster/coreg/icp}
low_raster_input=${LOW_RASTER_INPUT:-/media/laura/8402326D023263F8/calperumResearch/site2_1_DD0011/lidar/clipped/raster/coreg/icp}
medium_znorm_input=${MEDIUM_ZNORM_INPUT:-/media/laura/8402326D023263F8/calperumResearch/site1_1_DD0001/lidar/clipped/raster/z_norm/csf}
low_znorm_input=${LOW_ZNORM_INPUT:-/media/laura/8402326D023263F8/calperumResearch/site2_1_DD0011/lidar/clipped/raster/z_norm/csf}

access_key_id=${AWS_ACCESS_KEY_ID:-${R2_ACCESS_KEY_ID:-}}
secret_access_key=${AWS_SECRET_ACCESS_KEY:-${R2_SECRET_ACCESS_KEY:-}}

: "${R2_ACCOUNT_ID:?Set R2_ACCOUNT_ID in the environment}"
: "${access_key_id:?Set AWS_ACCESS_KEY_ID or R2_ACCESS_KEY_ID in the environment}"
: "${secret_access_key:?Set AWS_SECRET_ACCESS_KEY or R2_SECRET_ACCESS_KEY in the environment}"

if ! command -v rclone >/dev/null 2>&1; then
  printf 'rclone is required but was not found in PATH.\n' >&2
  exit 1
fi

stage_top_level_mask_inputs() {
  local source_root=$1
  local staged_root=$2

  mkdir -p "${staged_root}"
  find "${source_root}" -maxdepth 1 -type f \( -name '*.las' -o -name '*.laz' -o -name '*.ply' \) -exec cp '{}' "${staged_root}/" ';'
}

cd "${repo_root}"

rm -rf "${bundle_root}"
mkdir -p \
  "${bundle_root}/copc/${level_namespace}/medium/mask" \
  "${bundle_root}/copc/${level_namespace}/low/mask" \
  "${bundle_root}/copc/${level_namespace}/medium/raster" \
  "${bundle_root}/copc/${level_namespace}/low/raster" \
  "${bundle_root}/copc/${level_namespace}/medium/z_norm" \
  "${bundle_root}/copc/${level_namespace}/low/z_norm"

if [[ "${run_mask}" == "1" ]]; then
  stage_top_level_mask_inputs "${medium_mask_input}" "${bundle_root}/mask_inputs/medium"
  stage_top_level_mask_inputs "${low_mask_input}" "${bundle_root}/mask_inputs/low"

  ${python_cmd} ForestFormer3D/app_deploy/production_3d/build_copc_from_ply.py \
    --input-root "${bundle_root}/mask_inputs/medium" \
    --output-root "${bundle_root}/copc/${level_namespace}/medium/mask" \
    --overwrite

  ${python_cmd} ForestFormer3D/app_deploy/production_3d/build_copc_from_ply.py \
    --input-root "${bundle_root}/mask_inputs/low" \
    --output-root "${bundle_root}/copc/${level_namespace}/low/mask" \
    --overwrite
fi

if [[ "${run_raster}" == "1" ]]; then
  ${python_cmd} ForestFormer3D/app_deploy/production_3d/build_copc_from_ply.py \
    --input-root "${medium_raster_input}" \
    --output-root "${bundle_root}/copc/${level_namespace}/medium/raster" \
    --overwrite

  ${python_cmd} ForestFormer3D/app_deploy/production_3d/build_copc_from_ply.py \
    --input-root "${low_raster_input}" \
    --output-root "${bundle_root}/copc/${level_namespace}/low/raster" \
    --overwrite
fi

if [[ "${run_znorm}" == "1" ]]; then
  ${python_cmd} ForestFormer3D/app_deploy/production_3d/build_copc_from_ply.py \
    --input-root "${medium_znorm_input}" \
    --output-root "${bundle_root}/copc/${level_namespace}/medium/z_norm" \
    --use-z-dimension Z_norm \
    --overwrite

  ${python_cmd} ForestFormer3D/app_deploy/production_3d/build_copc_from_ply.py \
    --input-root "${low_znorm_input}" \
    --output-root "${bundle_root}/copc/${level_namespace}/low/z_norm" \
    --use-z-dimension Z_norm \
    --overwrite
fi

if [[ "${run_mask}" == "1" ]]; then
  RCLONE_CONFIG_R2REMOTE_TYPE=s3 \
  RCLONE_CONFIG_R2REMOTE_PROVIDER=Cloudflare \
  RCLONE_CONFIG_R2REMOTE_ACCESS_KEY_ID="${access_key_id}" \
  RCLONE_CONFIG_R2REMOTE_SECRET_ACCESS_KEY="${secret_access_key}" \
  RCLONE_CONFIG_R2REMOTE_ENDPOINT="https://${R2_ACCOUNT_ID}.r2.cloudflarestorage.com" \
  RCLONE_CONFIG_R2REMOTE_REGION=auto \
    rclone sync "${bundle_root}/copc/${level_namespace}/medium/mask" "r2remote:${bucket}/${prefix}/${level_namespace}/medium/mask" --progress

  RCLONE_CONFIG_R2REMOTE_TYPE=s3 \
  RCLONE_CONFIG_R2REMOTE_PROVIDER=Cloudflare \
  RCLONE_CONFIG_R2REMOTE_ACCESS_KEY_ID="${access_key_id}" \
  RCLONE_CONFIG_R2REMOTE_SECRET_ACCESS_KEY="${secret_access_key}" \
  RCLONE_CONFIG_R2REMOTE_ENDPOINT="https://${R2_ACCOUNT_ID}.r2.cloudflarestorage.com" \
  RCLONE_CONFIG_R2REMOTE_REGION=auto \
    rclone sync "${bundle_root}/copc/${level_namespace}/low/mask" "r2remote:${bucket}/${prefix}/${level_namespace}/low/mask" --progress
fi

if [[ "${run_raster}" == "1" ]]; then
  RCLONE_CONFIG_R2REMOTE_TYPE=s3 \
  RCLONE_CONFIG_R2REMOTE_PROVIDER=Cloudflare \
  RCLONE_CONFIG_R2REMOTE_ACCESS_KEY_ID="${access_key_id}" \
  RCLONE_CONFIG_R2REMOTE_SECRET_ACCESS_KEY="${secret_access_key}" \
  RCLONE_CONFIG_R2REMOTE_ENDPOINT="https://${R2_ACCOUNT_ID}.r2.cloudflarestorage.com" \
  RCLONE_CONFIG_R2REMOTE_REGION=auto \
    rclone sync "${bundle_root}/copc/${level_namespace}/medium/raster" "r2remote:${bucket}/${prefix}/${level_namespace}/medium/raster" --progress

  RCLONE_CONFIG_R2REMOTE_TYPE=s3 \
  RCLONE_CONFIG_R2REMOTE_PROVIDER=Cloudflare \
  RCLONE_CONFIG_R2REMOTE_ACCESS_KEY_ID="${access_key_id}" \
  RCLONE_CONFIG_R2REMOTE_SECRET_ACCESS_KEY="${secret_access_key}" \
  RCLONE_CONFIG_R2REMOTE_ENDPOINT="https://${R2_ACCOUNT_ID}.r2.cloudflarestorage.com" \
  RCLONE_CONFIG_R2REMOTE_REGION=auto \
    rclone sync "${bundle_root}/copc/${level_namespace}/low/raster" "r2remote:${bucket}/${prefix}/${level_namespace}/low/raster" --progress
fi

if [[ "${run_znorm}" == "1" ]]; then
  RCLONE_CONFIG_R2REMOTE_TYPE=s3 \
  RCLONE_CONFIG_R2REMOTE_PROVIDER=Cloudflare \
  RCLONE_CONFIG_R2REMOTE_ACCESS_KEY_ID="${access_key_id}" \
  RCLONE_CONFIG_R2REMOTE_SECRET_ACCESS_KEY="${secret_access_key}" \
  RCLONE_CONFIG_R2REMOTE_ENDPOINT="https://${R2_ACCOUNT_ID}.r2.cloudflarestorage.com" \
  RCLONE_CONFIG_R2REMOTE_REGION=auto \
    rclone sync "${bundle_root}/copc/${level_namespace}/medium/z_norm" "r2remote:${bucket}/${prefix}/${level_namespace}/medium/z_norm" --progress

  RCLONE_CONFIG_R2REMOTE_TYPE=s3 \
  RCLONE_CONFIG_R2REMOTE_PROVIDER=Cloudflare \
  RCLONE_CONFIG_R2REMOTE_ACCESS_KEY_ID="${access_key_id}" \
  RCLONE_CONFIG_R2REMOTE_SECRET_ACCESS_KEY="${secret_access_key}" \
  RCLONE_CONFIG_R2REMOTE_ENDPOINT="https://${R2_ACCOUNT_ID}.r2.cloudflarestorage.com" \
  RCLONE_CONFIG_R2REMOTE_REGION=auto \
    rclone sync "${bundle_root}/copc/${level_namespace}/low/z_norm" "r2remote:${bucket}/${prefix}/${level_namespace}/low/z_norm" --progress
fi

printf 'Published normalized-height mask/raster COPCs under %s from:\n' "${level_namespace}"
printf '  medium mask  -> %s\n' "${medium_mask_input}"
printf '  low mask     -> %s\n' "${low_mask_input}"
printf '  medium raster-> %s\n' "${medium_raster_input}"
printf '  low raster   -> %s\n' "${low_raster_input}"
printf '  medium z_norm-> %s\n' "${medium_znorm_input}"
printf '  low z_norm   -> %s\n' "${low_znorm_input}"
printf 'Bundle root: %s\n' "${bundle_root}"