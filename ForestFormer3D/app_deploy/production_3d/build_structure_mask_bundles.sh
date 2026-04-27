#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd "${script_dir}/../../.." && pwd)

bundle_root=${1:-${script_dir}/generated/structure_mask_medium_low}
public_base_url=${2:-https://pub-020b0b37031f434591da1fc0459021f8.r2.dev/fvc_structure-viewer/lidar}
copc_public_prefix=${3:-copc}
python_cmd=${PYTHON:-python}

medium_input=/media/laura/8402326D023263F8/calperumResearch/site1_1_DD0001/lidar/clipped/3d_structure_mask_malleid/new
low_input=/media/laura/8402326D023263F8/calperumResearch/site2_1_DD0011/lidar/clipped/3d_structure_mask_malleid
medium_meta=${medium_input}
low_meta=${low_input}

cd "${repo_root}"

rm -rf "${bundle_root}"
mkdir -p "${bundle_root}/copc"

${python_cmd} ForestFormer3D/app_deploy/production_3d/build_copc_from_ply.py \
  --input-root "${medium_input}" \
  --output-root "${bundle_root}/copc/medium" \
  --overwrite

tmp_low_root=$(mktemp -d)
trap 'rm -rf "${tmp_low_root}"' EXIT
find "${low_input}" -maxdepth 1 -type f -name 'znorm_3Dlabels_l_*.las' -exec ln -sf {} "${tmp_low_root}"/ \;

${python_cmd} ForestFormer3D/app_deploy/production_3d/build_copc_from_ply.py \
  --input-root "${tmp_low_root}" \
  --output-root "${bundle_root}/copc/low" \
  --overwrite

${python_cmd} ForestFormer3D/app_deploy/production_3d/generate_stac_for_copc.py \
  --copc-root "${bundle_root}/copc" \
  --stac-root "${bundle_root}/stac" \
  --asset-href-mode absolute \
  --public-base-url "${public_base_url}" \
  --copc-public-prefix "${copc_public_prefix}" \
  --asset-group-map '{"medium":"medium","low":"low"}' \
  --metadata-root-map "{\"medium\":\"${medium_meta}\",\"low\":\"${low_meta}\"}"

printf 'Built structure-mask bundle at %s\n' "${bundle_root}"