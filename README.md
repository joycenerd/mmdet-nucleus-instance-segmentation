# mmdet-nucleus-instance-segmentation
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)

### [Report](./docs/REPORT.pdf)

by [Zhi-Yi Chin](https://joycenerd.github.io/)

This repository is implementation of homework3 for IOC5008 Selected Topics in Visual Recognition using Deep Learning course in 2021 fall semester at National Yang Ming Chiao Tung University.

In this homework, we participate in nuclei segmentation challenge on [CodaLab](https://codalab.lisn.upsaclay.fr/competitions/333?secret_key=3b31d945-289d-4da6-939d-39435b506ee5). In this challenge, we perform instance segmentation on TCGA nuclei dataset from the 2018 Kaggle Data Science Bowl. This dataset contains 24 training images with 14,598 nuclei and 6 test images with 2,360 nuclei. For training, pre-trained models are allowed, but no external data should be used. We apply four existing methods to solve this challenge.



## Getting the code

You can download a copy of all the files in this repository by cloning this repository:

```
git clone https://github.com/joycenerd/mmdet-nucleus-instance-segmentation.git
```

## Requirements

You need to have [Anaconda](https://www.anaconda.com/) or Miniconda already installed in your environment. To install requirements:

1. Create a conda environment
```
conda create -n openmmlab python=3.7 -y
conda activate openmmlab
```

2. Install mmdetection
```
conda install pytorch==1.7.0 torchvision==0.8.0 torchaudio==0.7.0 cudatoolkit=11.0 -c pytorch
pip install openmim
mim install mmdet
```

3. Update mmdetection to custom need
```
cd mmdetection
python setup.py install
``` 

For more information please visit: [get_started.md](https://github.com/open-mmlab/mmdetection/blob/master/docs/get_started.md)

3. Install imantics (for converting COCO segmentation annotation)
```
pip install imantics
```

## Dataset

You can choose to download the data that we have pre-processed already or you can download the raw data.

### Option#1: Download the data that have been pre-processed
1. Download the data from the Google drive link: [nucleus_data.zip](https://drive.google.com/file/d/1Twy3XZgUhEVMklp8lrx5z5u9CRG-a7Ro/view?usp=sharing)  
2. After decompress the zip file, the data folder structure should look like this:
```
nucleus_data
├── all_train
│   ├── TCGA-18-5592-01Z-00-DX1.png
│   ├── TCGA-21-5784-01Z-00-DX1.png
│   ├── TCGA-21-5786-01Z-00-DX1.png
│   ├── ......
├── annotations
│   ├── instance_all_train.json
│   ├── instance_test.json
│   ├── instance_train.json
│   ├── instance_val.json
│   └── test_img_ids.json
├── classes.txt
├── test
│   ├── TCGA-50-5931-01Z-00-DX1.png
│   ├── TCGA-A7-A13E-01Z-00-DX1.png
│   ├── ......
├── train
│   ├── TCGA-18-5592-01Z-00-DX1.png
│   ├── TCGA-21-5786-01Z-00-DX1.png
│   ├── ......
└── val
    ├── TCGA-21-5784-01Z-00-DX1.png
    ├── TCGA-B0-5711-01Z-00-DX1.png
    ├── ......
```

### Option#2: Download the raw data
1. Download the data from the Google drive link: [dataset.zip](https://drive.google.com/file/d/1nEJ7NTtHcCHNQqUXaoPk55VH3Uwh4QGG/view?usp=sharing)
2. After decompress the zip file, the data folder structure should look like this: 
```
dataset
├── test
│   ├── .ipynb_checkpoints
│   │   ├── TCGA-50-5931-01Z-00-DX1-checkpoint.png
│   │   ├── TCGA-AY-A8YK-01A-01-TS1-checkpoint.png
│   │   ├── TCGA-G9-6336-01Z-00-DX1-checkpoint.png
│   │   └── TCGA-G9-6348-01Z-00-DX1-checkpoint.png
│   ├── TCGA-50-5931-01Z-00-DX1.png
│   ├── TCGA-A7-A13E-01Z-00-DX1.png
│   ├── TCGA-AY-A8YK-01A-01-TS1.png
│   ├── TCGA-G2-A2EK-01A-02-TSB.png
│   ├── TCGA-G9-6336-01Z-00-DX1.png
│   └── TCGA-G9-6348-01Z-00-DX1.png
├── test_img_ids.json
└── train
    ├── TCGA-18-5592-01Z-00-DX1
    │   ├── images
    │   │   └── TCGA-18-5592-01Z-00-DX1.png
    │   └── masks
    │       ├── .ipynb_checkpoints
    │       │   └── mask_0002-checkpoint.png
    │       ├── mask_0001.png
    │       ├── mask_0002.png
    │       ├── ......
    ├── TCGA-RD-A8N9-01A-01-TS1
    │   ├── images
    │   │   └── TCGA-RD-A8N9-01A-01-TS1.png
    │   └── masks
    │       ├── mask_0001.png
    │       ├── mask_0002.png
    │       ├── ......
    └── ......
```

### Data pre-processing

**Note: If you download the data by following option#1 you can split this step.**

If your raw data folder structure is different, you will need to modify [train_val_split.py](./train_val_split.py) and [mask2coco.py](./mask2coco.py) before executing the code.

1. train valid split
In default we split the whole training set to 80% for training and 20% for validation.
```
python train_valid_split.py --data-root <save_dir>/dataset/train --ratio 0.2 --out-dir <save_dir>/nucleus_data
```
  * input: original whole training image directory
  * output: new data dir name `nucleus_data`, inside this directory there will be to folders `train/` and `val/` with images inside 

2. convert binary mask images into COCO segmentation annotation.
```
python mask2coco.py --mode <train_or_val> --data_root <save_dir>/nucleus_data/<train_or_val> --mask_root <save_dir>/dataset/train --out_dir <save_dir>/nucleus_data/annotations
```
  * input: 
    1. train or val folder path from the last step
    2. binary mask saving root directory
  * output: `instance_train.json` or `instance_val.json` in `nucleus_data/annotations/`

## Training

You should have Graphics card to train the model. For your reference, we trained on a single NVIDIA Tesla V100.

1. Download the pre-trained weights (pre-trained on COCO)
| Model             | Backbone | Lr_schd | Download                                                                                                                                                                                                                |
|:-----------------:|:--------:|:-------:|:-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|
| Mask RCNN         | R50      | 3x      | [model](https://download.openmmlab.com/mmdetection/v2.0/mask_rcnn/mask_rcnn_r50_caffe_fpn_mstrain-poly_3x_coco/mask_rcnn_r50_caffe_fpn_mstrain-poly_3x_coco_bbox_mAP-0.408__segm_mAP-0.37_20200504_163245-42aa3d00.pth) |
| Mask RCNN         | X101     | 3x      | [model](https://download.openmmlab.com/mmdetection/v2.0/mask_rcnn/mask_rcnn_x101_32x8d_fpn_mstrain-poly_3x_coco/mask_rcnn_x101_32x8d_fpn_mstrain-poly_3x_coco_20210607_161042-8bd2c639.pth)                             |
| Cascade Mask RCNN | R50      | 3x      | [model](https://download.openmmlab.com/mmdetection/v2.0/cascade_rcnn/cascade_mask_rcnn_r50_caffe_fpn_mstrain_3x_coco/cascade_mask_rcnn_r50_caffe_fpn_mstrain_3x_coco_20210707_002651-6e29b3a6.pth)                      |
| Cascade Mask RCNN | X101     | 3x      | [model](https://download.openmmlab.com/mmdetection/v2.0/cascade_rcnn/cascade_mask_rcnn_x101_32x4d_fpn_mstrain_3x_coco/cascade_mask_rcnn_x101_32x4d_fpn_mstrain_3x_coco_20210706_225234-40773067.pth)                    |
| PointRend         | R50      | 3x      | [model](https://download.openmmlab.com/mmdetection/v2.0/point_rend/point_rend_r50_caffe_fpn_mstrain_3x_coco/point_rend_r50_caffe_fpn_mstrain_3x_coco-e0ebb6b7.pth)                                                      |
| Mask Scoring RCNN | X101     | 1x      | [model](https://download.openmmlab.com/mmdetection/v2.0/ms_rcnn/ms_rcnn_x101_32x4d_fpn_1x_coco/ms_rcnn_x101_32x4d_fpn_1x_coco_20200206-81fd1740.pth)                                                                    |




## Validation
You can validate your training results by the following recommendation command:
```
cd yolov5
python val.py --data data/custom-data.yaml --weights <ckpt_path> --device <gpu_ids> --project <val_log_dir>
```

* input: your model checkpoint path

## Testing

You can do detection on the testing set by the following recommendation commend:
```
cd yolov5
python detect.py --weights <ckpt_path> --source <test_data_dir_path> --save-txt --device <gpu_id> --save-conf --nosave
```

* input: 
  * trained model checkpoint
  * testing images 
* output: `yolov5/runs/detect/exp<X>/labels/`will be generated, inside this folder will have text files with the same name as the testing images, and inside each text file is the detection results of the correspoding testing image in YOLO format.

There is another way that you don't need to do post-processing afterward:
```
cd yolov5
python val.py --data data/custom-data.yaml --weights <ckpt_path> --device <gpu_id> --project <test_log_dir> --task test --save-txt --save-conf --save-json
```

* input: training model checkpoint
* output: `test_log_dir/exp<X>/<ckpt_name>.json` -> this is the COCO format detection result of the test set.

## Post-processing

Turn YOLO format detection results into COCO format.
```
python yolo2coco.py --yolo-path <detect_label_dir>
```
* input: detection results in the testing step.
* output: `answer.json`

## Submit the results
Run this command to compress your submission file:
```
zip answer.zip answer.json
```
You can upload `answer.zip` to the challenge. Then you can get your testing score.

## Pre-trained models

Go to [Releases](https://github.com/yolov5-svhn-detection/releases). Under **My YOLOv5s model** download `yolov5_best.pt`. This pre-trained model get score 0.4217 on the SVHN testing set.

## Inference
To reproduce our results, run this command:
```
cd yolov5
python val.py --data data/custom-data.yaml --weights <yolov5_best.pt_path> --device <gpu_id> --project <test_log_dir> --task test --save-txt --save-conf --save-json
```

## Benchmark the speed
Open `inference.ipynb` using Google Colab and follow the instruction in it.

## Reproducing Submission

To reproduce our submission without retraining, do the following steps

1. [Getting the code](#getting-the-code)
2. [Install the dependencies](#requirements)
3. [Download the data and data pre-processing](#dataset)
4. [Download pre-trained models](#pre-trained-models)
5. [Inference](#inference)
6. [Submit the results](#submit-the-results)

## Results

* Testing score:

| conf_thres | 0.25   | 0.01   | 0.001  |
|------------|--------|--------|--------|
| score      | 0.4067 | 0.4172 | 0.4217 |
* Detection speed: 22.7ms per image

## GitHub Acknowledgement
We thank the authors of these repositories:
* [ultralytics/yolov5](https://github.com/ultralytics/yolov5)

## Citation
If you find our work useful in your project, please cite:

```bibtex
@misc{
    title = {yolov5-schn-detection},
    author = {Zhi-Yi Chin},
    url = {https://github.com/joycenerd/yolov5-schn-detection},
    year = {2021}
}
```

## Contributing

If you'd like to contribute, or have any suggestions, you can contact us at [joycenerd.cs09@nycu.edu.tw](mailto:joycenerd.cs09@nycu.edu.tw) or open an issue on this GitHub repository.

All contributions welcome! All content in this repository is licensed under the MIT license.
