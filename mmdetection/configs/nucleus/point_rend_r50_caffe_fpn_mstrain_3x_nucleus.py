_base_ = '../point_rend/point_rend_r50_caffe_fpn_mstrain_3x_coco.py'
# learning policy
runner = dict(type='EpochBasedRunner', max_epochs=200)

model = dict(
    type='PointRend',
    roi_head=dict(
        type='PointRendRoIHead',
        mask_head=dict(
            _delete_=True,
            type='CoarseMaskHead',
            num_classes=1,
        ),
        point_head=dict(
            type='MaskPointHead',
            num_classes=1,
            coarse_pred_each_layer=True
        ),
        bbox_head=dict(
            type='Shared2FCBBoxHead',
            num_classes=1,
            reg_class_agnostic=False
        )))

# use caffe img_norm
img_norm_cfg = dict(
    mean=[103.530, 116.280, 123.675], std=[1.0, 1.0, 1.0], to_rgb=False)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True, poly2mask=False),
    dict(
        type='Resize',
        img_scale=[(1333, 640), (1333, 672), (1333, 704), (1333, 736),
                   (1333, 768), (1333, 800)],
        multiscale_mode='value',
        keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(
        type='Normalize',
        mean=[103.53, 116.28, 123.675],
        std=[1.0, 1.0, 1.0],
        to_rgb=False),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels', 'gt_masks'])
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

classes = ('nucleus',)

data = dict(
    samples_per_gpu=1,
    workers_per_gpu=1,
    train=dict(
        img_prefix='/work/zchin31415/nucleus_data/all_train',
        classes=classes,
        ann_file='/work/zchin31415/nucleus_data/annotations/instance_all_train.json',
        pipeline=train_pipeline),
    val=dict(
        img_prefix='/work/zchin31415/nucleus_data/all_train',
        classes=classes,
        ann_file='/work/zchin31415/nucleus_data/annotations/instance_all_train.json',
        pipeline=test_pipeline),
    test=dict(
        img_prefix='/work/zchin31415/nucleus_data/test',
        classes=classes,
        ann_file='/work/zchin31415/nucleus_data/annotations/instance_test.json'),
    pipeline=test_pipeline)

load_from = '/home/zchin31415/mmdet-nucleus-instance-segmentation/mmdetection/checkpoints/point_rend_r50_caffe_fpn_mstrain_3x_coco-e0ebb6b7.pth'
