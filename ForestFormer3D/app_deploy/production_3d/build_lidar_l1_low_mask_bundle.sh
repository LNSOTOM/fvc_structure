#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd "${script_dir}/../../.." && pwd)

bundle_root=${1:-${script_dir}/generated/lidar_l1_low_mask}
public_base_url=${2:-https://pub-020b0b37031f434591da1fc0459021f8.r2.dev/fvc_structure-viewer/lidar}
copc_public_prefix=${3:-copc/lidar_l1/low/mask}
python_cmd=${PYTHON:-python}

mask_input=/media/laura/8402326D023263F8/calperumResearch/site2_1_DD0011/lidar/clipped/mask/coreg/icp

cd "${repo_root}"

rm -rf "${bundle_root}"
mkdir -p "${bundle_root}/copc/lidar_l1/low/mask"

tmp_input_root=$(mktemp -d)
trap 'rm -rf "${tmp_input_root}"' EXIT
${python_cmd} ForestFormer3D/app_deploy/production_3d/remap_mask_classification.py \
  --input-root "${mask_input}" \
  --output-root "${tmp_input_root}" \
  --remap "64:2,65:1,66:2,67:4,68:5"

${python_cmd} ForestFormer3D/app_deploy/production_3d/build_copc_from_ply.py \
  --input-root "${tmp_input_root}" \
  --output-root "${bundle_root}/copc/lidar_l1/low/mask" \
  --overwrite

${python_cmd} ForestFormer3D/app_deploy/production_3d/generate_stac_for_copc.py \
  --copc-root "${bundle_root}/copc" \
  --stac-root "${bundle_root}/stac" \
  --asset-href-mode absolute \
  --public-base-url "${public_base_url}" \
  --copc-public-prefix "${copc_public_prefix}" \
  --asset-group-map '{"lidar_l1/low/mask":"lidar_l1_low_mask"}' \
  --metadata-root-map "{\"lidar_l1/low/mask\":\"${mask_input}\"}"

printf 'Built lidar_l1 low mask bundle at %s\n' "${bundle_root}"
