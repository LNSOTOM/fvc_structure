#!/bin/bash
# Syncs fixed host files to the container and regenerates pkl files.
# Run this after any container restart or when pkl files need to be regenerated.
# Usage: bash sync_to_container.sh [container_name]

CONTAINER=${1:-forestformer3d-container}
HOST_BASE=/home/nvidia/projects/fvc_structure/ForestFormer3D

echo "=== Syncing fixed files to container: $CONTAINER ==="

# 1. Tools
docker cp $HOST_BASE/tools/update_infos_to_v2.py       $CONTAINER:/workspace/tools/update_infos_to_v2.py
docker cp $HOST_BASE/tools/forainetv2_data_utils.py    $CONTAINER:/workspace/tools/forainetv2_data_utils.py

# 2. Data prep scripts
docker cp $HOST_BASE/data/ForAINetV2/load_forainetv2_data.py      $CONTAINER:/workspace/data/ForAINetV2/load_forainetv2_data.py
docker cp $HOST_BASE/data/ForAINetV2/batch_load_ForAINetV2_data.py $CONTAINER:/workspace/data/ForAINetV2/batch_load_ForAINetV2_data.py

# 3. Config
docker cp $HOST_BASE/configs/oneformer3d_qs_radius16_qp300_2many.py \
          $CONTAINER:/workspace/configs/oneformer3d_qs_radius16_qp300_2many.py

# 4. Metric (size-mismatch fix)
docker cp $HOST_BASE/oneformer3d/unified_metric.py \
          $CONTAINER:/workspace/oneformer3d/unified_metric.py

# 5. Model (full-scene pred_pts_seg fix)
docker cp $HOST_BASE/oneformer3d/oneformer3d.py \
          $CONTAINER:/workspace/oneformer3d/oneformer3d.py

echo "=== Verifying container has updated predict() ==="
docker exec $CONTAINER bash -c "grep -n 'merged full-scene arrays\|empty_inst_masks' /workspace/oneformer3d/oneformer3d.py | head -5 || true"

echo "=== Files synced. Regenerating pkl files ==="

# 5. Regenerate pkl files
docker exec $CONTAINER bash -c "cd /workspace && python tools/create_data_forainetv2.py forainetv2 2>&1 | tail -15"

# 6. Copy pkl files back to host
echo "=== Copying pkl files back to host ==="
docker cp $CONTAINER:/workspace/data/ForAINetV2/forainetv2_oneformer3d_infos_train.pkl \
          $HOST_BASE/data/ForAINetV2/forainetv2_oneformer3d_infos_train.pkl
docker cp $CONTAINER:/workspace/data/ForAINetV2/forainetv2_oneformer3d_infos_val.pkl \
          $HOST_BASE/data/ForAINetV2/forainetv2_oneformer3d_infos_val.pkl
docker cp $CONTAINER:/workspace/data/ForAINetV2/forainetv2_oneformer3d_infos_test.pkl \
          $HOST_BASE/data/ForAINetV2/forainetv2_oneformer3d_infos_test.pkl

echo "=== Verifying pkl labels ==="
# 7. Verify
docker exec $CONTAINER python3 -c "
import pickle
from collections import Counter
for split in ['train', 'val', 'test']:
    with open(f'/workspace/data/ForAINetV2/forainetv2_oneformer3d_infos_{split}.pkl', 'rb') as f:
        data = pickle.load(f)
    c = Counter(inst.get('bbox_label_3d') for s in data['data_list'] for inst in s.get('instances', []))
    print(f'{split}: ground={c.get(0,0)}, wood={c.get(1,0)}, leaf={c.get(2,0)}')
"

echo "=== Done ==="
