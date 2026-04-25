#!/usr/bin/env bash
set -euo pipefail

public_base_url=${1:?Usage: deploy_site1_znorm_csf.sh <public-base-url> [heroku-app] [bundle-root]}
heroku_app=${2:-lauransotomayor}
bundle_root=${3:-/home/laura/Documents/code/fvc_structure/ForestFormer3D/app_deploy/production_3d/generated/site1_znorm_csf}
low_bundle_root=${4:-/media/laura/8402326D023263F8/calperumResearch/site2_1_DD0011/lidar/clipped/normalized_height_copc}

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd "${script_dir}/../../.." && pwd)

lidar_base_url="${public_base_url%/}/fvc_structure-viewer/lidar"
catalog_url="${lidar_base_url}/stac/catalog.json"

python "${script_dir}/generate_stac_for_copc.py" \
  --copc-root "${bundle_root}/copc" \
  --stac-root "${bundle_root}/stac" \
  --asset-href-mode absolute \
  --public-base-url "${lidar_base_url}" \
  --default-asset-group medium

access_key_id=${AWS_ACCESS_KEY_ID:-${R2_ACCESS_KEY_ID:-}}
secret_access_key=${AWS_SECRET_ACCESS_KEY:-${R2_SECRET_ACCESS_KEY:-}}

: "${R2_ACCOUNT_ID:?Set R2_ACCOUNT_ID in the environment}"
: "${access_key_id:?Set AWS_ACCESS_KEY_ID or R2_ACCESS_KEY_ID in the environment}"
: "${secret_access_key:?Set AWS_SECRET_ACCESS_KEY or R2_SECRET_ACCESS_KEY in the environment}"

RCLONE_CONFIG_R2REMOTE_TYPE=s3 \
RCLONE_CONFIG_R2REMOTE_PROVIDER=Cloudflare \
RCLONE_CONFIG_R2REMOTE_ACCESS_KEY_ID="${access_key_id}" \
RCLONE_CONFIG_R2REMOTE_SECRET_ACCESS_KEY="${secret_access_key}" \
RCLONE_CONFIG_R2REMOTE_ENDPOINT="https://${R2_ACCOUNT_ID}.r2.cloudflarestorage.com" \
RCLONE_CONFIG_R2REMOTE_REGION=auto \
  rclone sync "${bundle_root}/copc/z_norm_csf" "r2remote:terrascientia/fvc_structure-viewer/lidar/copc/medium" --progress

if [ -d "${low_bundle_root}" ]; then
  RCLONE_CONFIG_R2REMOTE_TYPE=s3 \
  RCLONE_CONFIG_R2REMOTE_PROVIDER=Cloudflare \
  RCLONE_CONFIG_R2REMOTE_ACCESS_KEY_ID="${access_key_id}" \
  RCLONE_CONFIG_R2REMOTE_SECRET_ACCESS_KEY="${secret_access_key}" \
  RCLONE_CONFIG_R2REMOTE_ENDPOINT="https://${R2_ACCOUNT_ID}.r2.cloudflarestorage.com" \
  RCLONE_CONFIG_R2REMOTE_REGION=auto \
    rclone sync "${low_bundle_root}" "r2remote:terrascientia/fvc_structure-viewer/lidar/copc/low" --progress
else
  printf 'Warning: low bundle root does not exist, skipping low upload: %s\n' "${low_bundle_root}"
fi

RCLONE_CONFIG_R2REMOTE_TYPE=s3 \
RCLONE_CONFIG_R2REMOTE_PROVIDER=Cloudflare \
RCLONE_CONFIG_R2REMOTE_ACCESS_KEY_ID="${access_key_id}" \
RCLONE_CONFIG_R2REMOTE_SECRET_ACCESS_KEY="${secret_access_key}" \
RCLONE_CONFIG_R2REMOTE_ENDPOINT="https://${R2_ACCOUNT_ID}.r2.cloudflarestorage.com" \
RCLONE_CONFIG_R2REMOTE_REGION=auto \
  rclone sync "${bundle_root}/stac" "r2remote:terrascientia/fvc_structure-viewer/lidar/stac" --progress

if command -v heroku >/dev/null 2>&1; then
  heroku config:set FVC_STRUCTURE_STAC_CATALOG_URL="${catalog_url}" --app "${heroku_app}"
else
  printf 'Set the Heroku config var manually:\n'
  printf '  heroku config:set FVC_STRUCTURE_STAC_CATALOG_URL=%q --app %q\n' "${catalog_url}" "${heroku_app}"
fi

printf '\nDeployment prepared.\n'
printf 'Catalog URL: %s\n' "${catalog_url}"
printf 'Example COPC URL: %s\n' "${lidar_base_url}/copc/medium/z_norm_csf_tiles_multispectral.106.copc.laz"
printf 'Example low COPC URL: %s\n' "${lidar_base_url}/copc/low/tiles_multispectral.106.copc.laz"
printf 'Repo root: %s\n' "${repo_root}"
