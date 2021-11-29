# The new config inherits a base config to highlight the necessary modification
_base_ = 'mask_rcnn/mask_rcnn_r50_fpn_2x_coco.py'

# We also need to change the num_classes in head to match the dataset's annotation
model = dict(
    roi_head=dict(
        bbox_head=dict(num_classes=1),
        mask_head=dict(num_classes=1)))

# Modify dataset related settings
dataset_type = 'COCODataset'
classes = ('nucleus',)
data = dict(
    train=dict(
        img_prefix='/eva_data/zchin/nucleus_data/train',
        classes=classes,
        ann_file='/eva_data/zchin/nucleus_data/annotations/instance_train.json'),
    val=dict(
        img_prefix='/eva_data/zchin/nucleus_data/val',
        classes=classes,
        ann_file='/eva_data/zchin/nucleus_data/annotations/instance_val.json'),
    test=dict(
        img_prefix='/eva_data/zchin/nucleus_data/val',
        classes=classes,
        ann_file='/eva_data/zchin/nucleus_data/annotations/instance_val.json'))

# We can use the pre-trained Mask RCNN model to obtain higher performance
load_from = 'checkpoints/mask_rcnn_r50_fpn_2x_coco_bbox_mAP-0.392__segm_mAP-0.354_20200505_003907-3e542a40.pth'