default_scope = 'mmdet3d'
default_hooks = dict(
    timer=dict(type='IterTimerHook', _scope_='mmdet3d'),
    logger=dict(type='LoggerHook', interval=20, _scope_='mmdet3d'),
    param_scheduler=dict(type='ParamSchedulerHook', _scope_='mmdet3d'),
    checkpoint=dict(
        type='CheckpointHook',
        interval=1,
        _scope_='mmdet3d',
        max_keep_ckpts=3,
        save_optimizer=True),
    sampler_seed=dict(type='DistSamplerSeedHook', _scope_='mmdet3d'),
    visualization=dict(
        type='Det3DVisualizationHook', _scope_='mmdet3d', draw=False))
env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'))
log_processor = dict(
    type='LogProcessor', window_size=50, by_epoch=True, _scope_='mmdet3d')
log_level = 'INFO'
load_from = None
resume = False
custom_imports = dict(imports=['oneformer3d'])
num_channels = 32
num_instance_classes = 1
num_semantic_classes = 3
radius = 16
voxel_size = 0.2
score_th = 0.4
chunk = 20000
model = dict(
    type='ForAINetV2OneFormer3D_XAwarequery',
    data_preprocessor=dict(type='Det3DDataPreprocessor'),
    in_channels=3,
    num_channels=32,
    voxel_size=0.2,
    num_classes=1,
    min_spatial_shape=128,
    stuff_classes=[0],
    thing_cls=[1, 2],
    prepare_epoch=1000,
    query_point_num=300,
    radius=16,
    score_th=0.4,
    backbone=dict(
        type='SpConvUNet',
        num_planes=[32, 64, 96, 128, 160],
        return_blocks=True),
    decoder=dict(
        type='ForAINetv2QueryDecoder_XAwarequery',
        num_layers=6,
        num_classes=1,
        num_instance_queries=0,
        num_semantic_queries=3,
        num_instance_classes=1,
        in_channels=32,
        d_model=256,
        num_heads=8,
        hidden_dim=1024,
        dropout=0.0,
        activation_fn='gelu',
        iter_pred=True,
        attn_mask=True,
        fix_attention=True,
        objectness_flag=True),
    criterion=dict(
        type='ForAINetv2UnifiedCriterion_XAwarequery',
        num_semantic_classes=3,
        sem_criterion=dict(type='S3DISSemanticCriterion', loss_weight=0.2),
        inst_criterion=dict(
            type='InstanceCriterionForAI_OneToManyMatch',
            matcher=dict(type='One2ManyMatcher'),
            loss_weight=[1.0, 1.0, 0.5],
            fix_dice_loss_weight=True,
            iter_matcher=True,
            fix_mean_loss=True)),
    train_cfg=dict(),
    test_cfg=dict(
        topk_insts=300,
        inst_score_thr=0.0,
        pan_score_thr=0.0,
        npoint_thr=10,
        obj_normalization=True,
        obj_normalization_thr=0.01,
        sp_score_thr=0.15,
        nms=True,
        matrix_nms_kernel='linear',
        num_sem_cls=3,
        stuff_cls=[0],
        thing_cls=[1, 2],
        output_dir='work_dirs/test_results'))
dataset_type = 'ForAINetV2SegDataset_'
data_root_forainetv2 = 'data/ForAINetV2/'
data_prefix = dict(
    pts='points',
    pts_instance_mask='instance_mask',
    pts_semantic_mask='semantic_mask')
train_pipeline = [
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
    dict(type='CylinderCrop', radius=16),
    dict(type='GridSample', grid_size=0.2),
    dict(type='PointSample_', num_points=640000),
    dict(type='SkipEmptyScene_'),
    dict(type='PointInstClassMapping_', num_classes=1),
    dict(
        type='RandomFlip3D',
        sync_2d=False,
        flip_ratio_bev_horizontal=0.5,
        flip_ratio_bev_vertical=0.0),
    dict(
        type='GlobalRotScaleTrans',
        rot_range=[-3.14, 3.14],
        scale_ratio_range=[0.8, 1.2],
        translation_std=[0.1, 0.1, 0.1],
        shift_height=False),
    dict(
        type='Pack3DDetInputs_',
        keys=[
            'points', 'gt_labels_3d', 'pts_semantic_mask', 'pts_instance_mask',
            'ratio_inspoint', 'vote_label', 'instance_mask'
        ])
]
val_pipeline = [
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
    dict(type='CylinderCrop', radius=16),
    dict(type='GridSample', grid_size=0.2),
    dict(type='PointSample_', num_points=640000),
    dict(type='PointInstClassMapping_', num_classes=1),
    dict(
        type='Pack3DDetInputs_',
        keys=[
            'points', 'gt_labels_3d', 'pts_semantic_mask', 'pts_instance_mask',
            'instance_mask'
        ])
]
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
    dict(
        type='Pack3DDetInputs_',
        keys=[
            'points', 'gt_labels_3d', 'pts_semantic_mask', 'pts_instance_mask',
            'instance_mask'
        ])
]
train_dataloader = dict(
    batch_size=2,
    num_workers=12,
    prefetch_factor=10,
    pin_memory=True,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type='ForAINetV2SegDataset_',
        data_root='data/ForAINetV2/',
        ann_file='forainetv2_oneformer3d_infos_train.pkl',
        data_prefix=dict(
            pts='points',
            pts_instance_mask='instance_mask',
            pts_semantic_mask='semantic_mask'),
        pipeline=[
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
            dict(type='CylinderCrop', radius=16),
            dict(type='GridSample', grid_size=0.2),
            dict(type='PointSample_', num_points=640000),
            dict(type='SkipEmptyScene_'),
            dict(type='PointInstClassMapping_', num_classes=1),
            dict(
                type='RandomFlip3D',
                sync_2d=False,
                flip_ratio_bev_horizontal=0.5,
                flip_ratio_bev_vertical=0.0),
            dict(
                type='GlobalRotScaleTrans',
                rot_range=[-3.14, 3.14],
                scale_ratio_range=[0.8, 1.2],
                translation_std=[0.1, 0.1, 0.1],
                shift_height=False),
            dict(
                type='Pack3DDetInputs_',
                keys=[
                    'points', 'gt_labels_3d', 'pts_semantic_mask',
                    'pts_instance_mask', 'ratio_inspoint', 'vote_label',
                    'instance_mask'
                ])
        ],
        filter_empty_gt=True,
        box_type_3d='Depth',
        backend_args=None))
val_dataloader = dict(
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='ForAINetV2SegDataset_',
        data_root='data/ForAINetV2/',
        ann_file='forainetv2_oneformer3d_infos_val.pkl',
        data_prefix=dict(
            pts='points',
            pts_instance_mask='instance_mask',
            pts_semantic_mask='semantic_mask'),
        pipeline=[
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
            dict(type='CylinderCrop', radius=16),
            dict(type='GridSample', grid_size=0.2),
            dict(type='PointSample_', num_points=640000),
            dict(type='PointInstClassMapping_', num_classes=1),
            dict(
                type='Pack3DDetInputs_',
                keys=[
                    'points', 'gt_labels_3d', 'pts_semantic_mask',
                    'pts_instance_mask', 'instance_mask'
                ])
        ],
        box_type_3d='Depth',
        test_mode=True,
        backend_args=None))
test_dataloader = dict(
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='ForAINetV2SegDataset_',
        data_root='data/ForAINetV2/',
        ann_file='forainetv2_oneformer3d_infos_test.pkl',
        data_prefix=dict(
            pts='points',
            pts_instance_mask='instance_mask',
            pts_semantic_mask='semantic_mask'),
        pipeline=[
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
            dict(
                type='Pack3DDetInputs_',
                keys=[
                    'points', 'gt_labels_3d', 'pts_semantic_mask',
                    'pts_instance_mask', 'instance_mask'
                ])
        ],
        box_type_3d='Depth',
        test_mode=True,
        backend_args=None))
instance_class_names = ['tree']
class_names = ['tree']
label2cat = dict({0: 'tree'})
metric_meta = dict(
    label2cat=dict({0: 'tree'}),
    ignore_index=[],
    classes=['tree'],
    dataset_name='ForAINetV2')
semantic_class_names = ['ground', 'wood', 'leaf']
semantic_label2cat = dict({0: 'ground', 1: 'wood', 2: 'leaf'})
semantic_metric_meta = dict(
    label2cat=dict({
        0: 'ground',
        1: 'wood',
        2: 'leaf'
    }),
    ignore_index=[],
    classes=['ground', 'wood', 'leaf'],
    dataset_name='ForAINetV2')
sem_mapping = [0, 1, 2]
inst_mapping = [1, 2]
val_evaluator = dict(
    type='UnifiedSegMetric',
    stuff_class_inds=[0],
    thing_class_inds=[1, 2],
    min_num_points=1,
    id_offset=65536,
    sem_mapping=[0, 1, 2],
    inst_mapping=[1, 2],
    metric_meta=dict(
        label2cat=dict({
            0: 'ground',
            1: 'wood',
            2: 'leaf'
        }),
        ignore_index=[],
        classes=['ground', 'wood', 'leaf'],
        dataset_name='ForAINetV2'))
test_evaluator = dict(
    type='UnifiedSegMetric',
    stuff_class_inds=[0],
    thing_class_inds=[1, 2],
    min_num_points=1,
    id_offset=65536,
    sem_mapping=[0, 1, 2],
    inst_mapping=[1, 2],
    metric_meta=dict(
        label2cat=dict({
            0: 'ground',
            1: 'wood',
            2: 'leaf'
        }),
        ignore_index=[],
        classes=['ground', 'wood', 'leaf'],
        dataset_name='ForAINetV2'))
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=0.0001, weight_decay=0.05),
    clip_grad=dict(max_norm=10, norm_type=2))
param_scheduler = dict(
    type='PolyLR', begin=0, end=450000, power=0.9, by_epoch=False)
custom_hooks = [dict(type='EmptyCacheHook', after_iter=True)]
vis_backends = [
    dict(type='LocalVisBackend'),
    dict(type='TensorboardVisBackend')
]
visualizer = dict(
    type='Det3DLocalVisualizer',
    vis_backends=[
        dict(type='LocalVisBackend'),
        dict(type='TensorboardVisBackend')
    ],
    name='visualizer')
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=10, val_interval=100)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')
find_unused_parameters = True
launcher = 'pytorch'
work_dir = 'work_dirs/train_forestformer_3gpu'
