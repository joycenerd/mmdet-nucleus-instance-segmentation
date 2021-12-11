_base_='../ms_rcnn/ms_rcnn_x101_32x4d_fpn_1x_coco.py'

# We also need to change the num_classes in head to match the dataset's annotation
model = dict(
    type='MaskScoringRCNN',
    roi_head=dict(
        type='MaskScoringRoIHead',
        mask_iou_head=dict(
            type='MaskIoUHead',
            num_classes=1),
        bbox_head=dict(
            type='Shared2FCBBoxHead',
            num_classes=1),
        mask_head=dict(
            type='FCNMaskHead',
            num_classes=1)
    ),    
)

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True,poly2mask=False),
    dict(type='Resize', img_scale=(1333, 800), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels', 'gt_masks']),
]
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

# Modify dataset related settings
dataset_type = 'CocoDataset'
classes = ('nucleus',)
runner = dict(type='EpochBasedRunner', max_epochs=200)

data = dict(
    samples_per_gpu=1,
    workers_per_gpu=1,
    train=dict(
        img_prefix='/work/zchin31415/nucleus_data/all_train',
        classes=classes,
        ann_file='/work/zchin31415/nucleus_data/annotations/nuclei.json',
        pipeline=train_pipeline),
    val=dict(
        img_prefix='/work/zchin31415/nucleus_data/all_train',
        classes=classes,
        ann_file='/work/zchin31415/nucleus_data/annotations/nuclei.json',
        pipeline=test_pipeline),
    test=dict(
        img_prefix='/work/zchin31415/nucleus_data/test',
        classes=classes,
        ann_file='/work/zchin31415/nucleus_data/annotations/instance_test.json',
        pipeline=test_pipeline)
    )

# We can use the pre-trained Mask RCNN model to obtain higher performance
load_from = '/home/zchin31415/mmdet-nucleus-instance-segmentation/mmdetection/checkpoints/ms_rcnn_x101_32x4d_fpn_1x_coco_20200206-81fd1740.pth'