_base_ = ['./oneformer3d_qs_radius16_qp300_2many.py']

# -------------------------
# MEDIUM split = H/S/M semantics after loader normalization
# raw semantic_seg {3,4,5} -> normalized -> label_ids {0,1,2}
# meaning: 0=hummock, 1=shrub, 2=mallee
# -------------------------

# 1) Point test loader to the medium PKL
test_dataloader = dict(
    dataset=dict(
        ann_file='forainetv2_medium_oneformer3d_infos_test.pkl'
    )
)

# (Optional, but recommended) also point val to medium if you ever run val
val_dataloader = dict(
    dataset=dict(
        ann_file='forainetv2_medium_oneformer3d_infos_test.pkl'
    )
)

# 2) Rename classes for correct reporting
class_names = ['hummock', 'shrub', 'mallee']
label2cat = {i: name for i, name in enumerate(class_names)}

metric_meta = dict(
    label2cat=label2cat,
    ignore_index=[],
    classes=class_names,
    dataset_name='ForAINetV2_medium',
)

# 3) Reuse your existing evaluator, just swap meta
val_evaluator = dict(metric_meta=metric_meta)
test_evaluator = dict(metric_meta=metric_meta)

# 4) Separate work dir so outputs don't collide with forest
work_dir = 'work_dirs/zero_shot_medium_hsm'