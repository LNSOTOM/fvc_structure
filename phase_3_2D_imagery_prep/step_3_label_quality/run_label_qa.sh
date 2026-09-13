#!/usr/bin/env bash
# Run label_qa.py for one site.
#
#   ./run_label_qa.sh site1_1_DD0001 Medium annotations_shp [extra args...]
#
# Paths follow the layout on D:. When D: is not mounted in WSL the script falls
# back to the /tmp/wsl_drive_d mirror, matching wsl_path() in label_qa.py.

set -euo pipefail

SITE="${1:?usage: run_label_qa.sh SITE CONDITION SHP_SUBDIR [extra args]}"
CONDITION="${2:?}"
SHP_SUBDIR="${3:?}"
shift 3

if [ -d /mnt/d/paper2 ]; then
    ROOT=/mnt/d/paper2/data/calperumResearch
else
    ROOT=/tmp/wsl_drive_d/paper2/data/calperumResearch
fi

BASE="$ROOT/$SITE"
MASK="$BASE/imagery/inputs/masks/tiles_3072/raw/structure_class"
PYTHON="${PYTHON:-$HOME/micromamba/envs/fvc_structure/bin/python}"

# The predictor stack the models read. DD0001 keeps it under multispec/res_05,
# DD0011 directly under tiles_3072, so take whichever layout is on disk.
PRED="$BASE/imagery/inputs/predictors/tiles_3072"
IMAGE_ARGS=()
for CANDIDATE in "$PRED/multispec/res_05/stacked" "$PRED/stacked"; do
    if [ -d "$CANDIDATE" ]; then
        IMAGE_ARGS=(--image-dir "$CANDIDATE")
        break
    fi
done

exec "$PYTHON" "$(dirname "$0")/label_qa.py" \
    --site "$SITE" \
    --site-condition "$CONDITION" \
    --shp-dir "$MASK/$SHP_SUBDIR" \
    --raster-dir "$MASK/annotation_raster" \
    --downsampled-dir "$MASK/annotation_raster/downsample_5cm_nearest" \
    --output-dir "$BASE/lidar/clipped/validation/label_quality" \
    ${IMAGE_ARGS[@]+"${IMAGE_ARGS[@]}"} \
    "$@"
