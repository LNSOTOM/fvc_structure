_base_ = [
    './oneformer3d_qs_radius16_qp300_2many.py',
]

# Experiment 2: Fine-tune on noisy (2D→3D transferred) L1-only supervision,
# then evaluate on held-out Medium+Low tiles with human-verified GT masks.

# Forest-trained checkpoint (out-of-domain clean supervision) used as initialization.
load_from = 'work_dirs/clean_forestformer/epoch_3000_fix.pth'

# Validation cadence is inherited from the base config train_cfg:
#   val_interval=100  (i.e., validate every 100 epochs, not every epoch).

default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook',
        interval=1,
        max_keep_ckpts=3,
        # Save both semantic-best and instance-oriented-best checkpoints.
        # This keeps two selection options for final test-time evaluation:
        #   - best_mIoU*.pth  (semantic quality)
        #   - best_F1*.pth    (instance quality / balance)
        #   - best_harmonic*.pth (balanced score = harmonic mean of mIoU and F1)
        save_best=['mIoU', 'F1', 'harmonic'],
        rule=['greater', 'greater', 'greater'],
        save_optimizer=True,
    ))

data_root_noisy = 'data/ForAINetV2/'
data_root_gt = 'data/ForAINetV2/'

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
        # Default fallback for direct `tools/test.py` runs with this config.
        # For Experiment 2 Step 6, `tools/zero_shot_infer_split.py` overrides
        # `test_dataloader.dataset.ann_file` at runtime with a generated split-specific pkl.
        ann_file='forainetv2_oneformer3d_infos_test.pkl',
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
