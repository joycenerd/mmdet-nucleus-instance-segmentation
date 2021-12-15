# The new config inherits a base config to highlight the necessary modification
_base_ = '../cascade_rcnn/cascade_mask_rcnn_r50_caffe_fpn_mstrain_3x_coco.py'

# We also need to change the num_classes in head to match the dataset's annotation
model = dict(
    roi_head=dict(
        bbox_head=[
            dict(type='Shared2FCBBoxHead', num_classes=1),
            dict(type='Shared2FCBBoxHead', num_classes=1),
            dict(type='Shared2FCBBoxHead', num_classes=1)
        ],
        mask_head=dict(num_classes=1)
    ),
    backbone=dict(
        with_cp=True
    ),
    # train_cfg=dict(
    #     rpn=dict(
    #         assigner=dict(
    #             type='MaxIoUAssigner',
    #             gpu_assign_thr=1000)),
    #     rcnn=dict(
    #         assigner=dict(
    #             type='MaxIoUAssigner',
    #             gpu_assign_thr=1000))
    # )
)

# Modify dataset related settings
dataset_type = 'CocoDataset'
classes = ('nucleus',)
runner = dict(type='EpochBasedRunner', max_epochs=200)

# use caffe img_norm
img_norm_cfg = dict(
    mean=[103.530, 116.280, 123.675], std=[1.0, 1.0, 1.0], to_rgb=False)

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1333, 800),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='DefaultFormatBundle'),
            dict(type='Collect', keys=['img']),
        ])
]

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='LoadAnnotations',
        with_bbox=True,
        with_mask=True,
        poly2mask=False),
    dict(
        type='Resize',
        img_scale=[(1333, 640), (1333, 800)],
        multiscale_mode='range',
        keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels', 'gt_masks']),
]

data = dict(
    samples_per_gpu=1,
    workers_per_gpu=1,
    train=dict(
        type='RepeatDataset',
        times=3,
        dataset=dict(
            type='CocoDataset',
            ann_file='/work/zchin31415/nucleus_data/annotations/instance_all_train.json',
            img_prefix='/work/zchin31415/nucleus_data/all_train',
            # classes=('tennis', )
            pipeline=train_pipeline
        ),
        classes=classes,
        ann_file='/work/zchin31415/nucleus_data/annotations/instance_all_train.json',
        img_prefix='/work/zchin31415/nucleus_data/all_train'),
    val=dict(
        type=dataset_type,
        ann_file='/work/zchin31415/nucleus_data/annotations/instance_all_train.json',
        img_prefix='/work/zchin31415/nucleus_data/all_train',
        classes=classes),
    test=dict(
        type=dataset_type,
        ann_file='/work/zchin31415/nucleus_data/annotations/instance_test.json',
        img_prefix='/work/zchin31415/nucleus_data/test',
        pipeline=test_pipeline,
        classes=classes)
)

load_from = '/home/zchin31415/mmdet-nucleus-instance-segmentation/mmdetection/checkpoints/cascade_mask_rcnn_r50_caffe_fpn_mstrain_3x_coco_20210707_002651-6e29b3a6.pth'
