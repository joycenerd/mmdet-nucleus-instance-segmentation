# The new config inherits a base config to highlight the necessary modification
_base_ = '../mask_rcnn/mask_rcnn_r50_caffe_fpn_mstrain-poly_1x_coco.py'

# We also need to change the num_classes in head to match the dataset's annotation
model = dict(
    roi_head=dict(
        bbox_head=dict(num_classes=1),
        mask_head=dict(num_classes=1)),
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

test_pipelines = [
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

data = dict(
    samples_per_gpu=1,
    workers_per_gpu=1,
    train=dict(
        img_prefix='/work/zchin31415/nucleus_data/all_train/',
        classes=classes,
        ann_file='/work/zchin31415/nucleus_data/annotations/instance_all_train.json'),
    val=dict(
        img_prefix='/work/zchin31415/nucleus_data/all_train',
        classes=classes,
        ann_file='/work/zchin31415/nucleus_data/annotations/instance_all_train.json'),
    test=dict(
        img_prefix='/work/zchin31415/nucleus_data/test',
        classes=classes,
        ann_file='/work/zchin31415/nucleus_data/annotations/instance_test.json'),
    pipelines=[
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
    ])

load_from = '/home/zchin31415/mmdet-nucleus-instance-segmentation/mmdetection/checkpoints/mask_rcnn_r50_caffe_fpn_mstrain-poly_3x_coco_bbox_mAP-0.408__segm_mAP-0.37_20200504_163245-42aa3d00.pth'
