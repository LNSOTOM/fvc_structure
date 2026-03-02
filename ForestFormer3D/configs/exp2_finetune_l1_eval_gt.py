_base_ = [
    './oneformer3d_qs_radius16_qp300_2many.py',
]

# Experiment 2: Fine-tune on noisy (2D→3D transferred) L1-only supervision,
# then evaluate on held-out Medium+Low tiles with human-verified GT masks.

# Forest-trained checkpoint (out-of-domain clean supervision) used as initialization.
load_from = 'work_dirs/clean_forestformer/epoch_3000_fix.pth'

data_root_noisy = 'data/ForAINetV2/'
data_root_gt = 'data/ForAINetV2_GT/'

train_dataloader = dict(
    dataset=dict(
        data_root=data_root_noisy,
        ann_file='forainetv2_oneformer3d_infos_train_L1.pkl',
    )
)

val_dataloader = dict(
    dataset=dict(
        data_root=data_root_noisy,
        ann_file='forainetv2_oneformer3d_infos_val_L1.pkl',
    )
)

test_dataloader = dict(
    dataset=dict(
        data_root=data_root_gt,
        ann_file='forainetv2_oneformer3d_infos_test_ML.pkl',
        # Expected folder layout under data_root_gt:
        #   points/ (can be a symlink to ../ForAINetV2/points)
        #   semantic_mask/ (clean GT)
        #   instance_mask/ (clean GT)
        data_prefix=dict(
            pts='points',
            pts_instance_mask='instance_mask',
            pts_semantic_mask='semantic_mask',
        ),
    )
)
