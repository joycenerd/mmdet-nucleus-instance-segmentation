# The new config inherits a base config to highlight the necessary modification
_base_ = '../mask_rcnn/mask_rcnn_x101_32x8d_fpn_mstrain-poly_3x_coco.py'

# We also need to change the num_classes in head to match the dataset's annotation
model = dict(
    roi_head=dict(
        bbox_head=dict(num_classes=1),
        mask_head=dict(num_classes=1)),
    # backbone=dict(
    #     with_cp=True
    # ),
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
            dict(type='DefaultFormatBundle', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
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

load_from = '/home/zchin31415/mmdet-nucleus-instance-segmentation/mmdetection/checkpoints/mask_rcnn_x101_32x8d_fpn_mstrain-poly_3x_coco_20210607_161042-8bd2c639.pth'
