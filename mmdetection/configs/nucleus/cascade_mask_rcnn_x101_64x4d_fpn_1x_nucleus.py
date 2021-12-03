# The new config inherits a base config to highlight the necessary modification
_base_ = '../cascade_rcnn/cascade_mask_rcnn_x101_64x4d_fpn_1x_coco.py'

# We also need to change the num_classes in head to match the dataset's annotation
model = dict(
    roi_head=dict(
        bbox_head=[
            dict(type='Shared2FCBBoxHead',num_classes=1),
            dict(type='Shared2FCBBoxHead',num_classes=1),
            dict(type='Shared2FCBBoxHead',num_classes=1)
        ],
        mask_head=dict(num_classes=1)
    )
)

# Modify dataset related settings
# dataset_type = 'COCODataset'
classes = ('nucleus',)
runner = dict(type='EpochBasedRunner', max_epochs=200)

test_pipeline = [
   dict(type='LoadImageFromFile'),
   dict(
       type='MultiScaleFlipAug',
       img_scale=(1333, 800),
       flip=False,
       transforms=[
           dict(type='Resize', keep_ratio=True),
           dict(type='RandomFlip'),
           dict(type='Normalize', mean=[0, 0, 0], std=[1, 1, 1]),
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
    dict(type='Resize', img_scale=(1333, 800), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(
        type='Normalize',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        to_rgb=True),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels', 'gt_masks'])
]

data = dict(
    samples_per_gpu=1,
    workers_per_gpu=1,
    train=dict(
        img_prefix='/work/zchin31415/nucleus_data/train',
        classes=classes,
        ann_file='/work/zchin31415/nucleus_data/annotations/instance_train_copy.json',
        pipeline=train_pipeline),   
    val=dict(
        img_prefix='/work/zchin31415/nucleus_data/val',
        classes=classes,
        ann_file='/work/zchin31415/nucleus_data/annotations/instance_val.json',
        # pipeline = test_pipeline
        ),
    test=dict(
        img_prefix='/work/zchin31415/nucleus_data/test',
        classes=classes,
        ann_file='/work/zchin31415/nucleus_data/annotations/instance_test.json'),
        # pipeline = test_pipeline
        )
    
load_from ='/home/zchin31415/mmdet-nucleus-instance-segmentation/mmdetection/checkpoints/cascade_mask_rcnn_x101_64x4d_fpn_1x_coco_20200203-9a2db89d.pth'