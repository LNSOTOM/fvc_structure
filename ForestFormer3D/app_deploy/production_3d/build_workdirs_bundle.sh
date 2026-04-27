#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd "${script_dir}/../../.." && pwd)

input_root=${1:-/media/laura/8402326D023263F8/calperumResearch/site1_1_DD0001/lidar/ForestFormer3D/data/ForAINetV2/work_dirs}
bundle_root=${2:-${script_dir}/generated/work_dirs_bundle}
public_base_url=${3:-https://pub-020b0b37031f434591da1fc0459021f8.r2.dev/fvc_structure-viewer/lidar}
python_cmd=${PYTHON:-}
structure_root=${STRUCTURE_ROOT:-/media/laura/8402326D023263F8/calperumResearch/site1_1_DD0001/lidar/clipped/3d_structure_mask_malleid/new}
forestformer_npy_root=${FORESTFORMER_NPY_ROOT:-/media/laura/8402326D023263F8/calperumResearch/site1_1_DD0001/lidar/clipped/preprocessed/forestformer3d_npy}

if [[ -z "${python_cmd}" ]]; then
  for candidate in \
    /home/laura/.local/share/mamba/envs/fvc_structure/bin/python \
    "${repo_root}/.venv/bin/python" \
    "$(command -v python 2>/dev/null || true)" \
    "$(command -v python3 2>/dev/null || true)"; do
    if [[ -n "${candidate}" && -x "${candidate}" ]]; then
      if "${candidate}" - <<'PY' >/dev/null 2>&1
import importlib.util
required = ('laspy', 'plyfile', 'numpy')
raise SystemExit(0 if all(importlib.util.find_spec(name) for name in required) else 1)
PY
      then
        python_cmd="${candidate}"
        break
      fi
    fi
  done
fi

: "${python_cmd:?Set PYTHON to an interpreter with laspy, plyfile, and numpy installed.}"

cd "${repo_root}"

rm -rf "${bundle_root}"
mkdir -p "${bundle_root}"

${python_cmd} ForestFormer3D/app_deploy/production_3d/build_copc_from_ply.py \
  --input-root "${input_root}" \
  --output-root "${bundle_root}/copc" \
  --structure-root "${structure_root}" \
  --forestformer-npy-root "${forestformer_npy_root}" \
  --overwrite

${python_cmd} ForestFormer3D/app_deploy/production_3d/generate_stac_for_copc.py \
  --copc-root "${bundle_root}/copc" \
  --stac-root "${bundle_root}/stac" \
  --asset-href-mode absolute \
  --public-base-url "${public_base_url}"

printf 'Built work_dirs bundle at %s\n' "${bundle_root}"