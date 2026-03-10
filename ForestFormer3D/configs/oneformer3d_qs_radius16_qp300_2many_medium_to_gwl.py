_base_ = ['./oneformer3d_qs_radius16_qp300_2many.py']

# Remap medium labels (hummock/shrub/mallee) into the ground/wood/leaf space
# to make results more comparable with forest runs:
#   hummock -> ground
#   shrub   -> ground
#   mallee  -> leaf
# (wood will typically be absent in medium.)
sem_remap_hsm_to_gwl = {0: 0, 1: 0, 2: 2, -1: -1}

# Override test_pipeline to apply the semantic remap before packing inputs.
# NOTE: This remaps GT masks only (predictions are unchanged).
test_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='DEPTH',
        shift_height=False,
        use_color=False,
        load_dim=3,
        use_dim=[0, 1, 2]),
    dict(
        type='LoadAnnotations3D',
        with_bbox_3d=False,
        with_label_3d=False,
        with_mask_3d=True,
        with_seg_3d=True),
    dict(type='RemapSemanticMask', remap=sem_remap_hsm_to_gwl),
    dict(type='Pack3DDetInputs_', keys=['points', 'gt_labels_3d', 'pts_semantic_mask', 'pts_instance_mask', 'instance_mask'])
]
