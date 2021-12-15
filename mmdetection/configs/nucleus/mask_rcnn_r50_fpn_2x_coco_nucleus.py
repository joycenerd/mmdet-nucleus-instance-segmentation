# The new config inherits a base config to highlight the necessary modification
_base_ = '../mask_rcnn/mask_rcnn_r50_fpn_mstrain-poly_3x_coco.py'

# We also need to change the num_classes in head to match the dataset's annotation
model = dict(
    roi_head=dict(
        bbox_head=dict(num_classes=1),
        mask_head=dict(num_classes=1)))

# Modify dataset related settings
dataset_type = 'Cocoataset'
classes = ('nucleus',)
data = dict(
    train=dict(
        img_prefix='/eva_data/zchin/nucleus_data/all_train',
        classes=classes,
        ann_file='/eva_data/zchin/nucleus_data/annotations/instance_all_train.json'),
    val=dict(
        img_prefix='/eva_data/zchin/nucleus_data/all_train',
        classes=classes,
        ann_file='/eva_data/zchin/nucleus_data/annotations/instance_all_train.json'),
    test=dict(
        img_prefix='/eva_data/zchin/nucleus_data/test',
        classes=classes,
        ann_file='/eva_data/zchin/nucleus_data/annotations/instance_test.json')
)

# We can use the pre-trained Mask RCNN model to obtain higher performance
load_from = '/home/zchin/mmdet-nucleus-instance-segmentation/mmdetection/checkpoints/mask_rcnn_r50_caffe_fpn_1x_coco_bbox_mAP-0.38__segm_mAP-0.344_20200504_231812-0ebd1859.pth'
