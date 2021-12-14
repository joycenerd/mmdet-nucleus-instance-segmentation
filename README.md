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

**Note: If you download the data by following option#1 you can skip this step.**

If your raw data folder structure is different, you will need to modify [train_val_split.py](./train_val_split.py) and [mask2coco.py](./mask2coco.py) before executing the code.

#### 1. train valid split
In default we split the whole training set to 80% for training and 20% for validation.
```
python train_valid_split.py --data-root <save_dir>/dataset/train --ratio 0.2 --out-dir <save_dir>/nucleus_data
```
  * input: original whole training image directory
  * output: new data dir name `nucleus_data`, inside this directory there will be to folders `train/` and `val/` with images inside 

#### 2. convert binary mask images into COCO segmentation annotation.
```
python mask2coco.py --mode <train_or_val> --data_root <save_dir>/nucleus_data/<train_or_val> --mask_root <save_dir>/dataset/train --out_dir <save_dir>/nucleus_data/annotations
```
  * input: 
    1. train or val folder path from the last step
    2. binary mask saving root directory
  * output: `instance_train.json` or `instance_val.json` in `nucleus_data/annotations/`

## Training

You should have Graphics card to train the model. For your reference, we trained on a single NVIDIA Tesla V100.

### 1. Download the pre-trained weights (pre-trained on COCO)
<div id="pre-trained"></div>

| **Model** | **Backbone** | **Lr_schd** | **Download** |
|:---:|:---:|:---:|:---:|
| Mask RCNN | R50 | 3x | [model](https://download.openmmlab.com/mmdetection/v2.0/mask_rcnn/mask_rcnn_r50_caffe_fpn_mstrain-poly_3x_coco/mask_rcnn_r50_caffe_fpn_mstrain-poly_3x_coco_bbox_mAP-0.408__segm_mAP-0.37_20200504_163245-42aa3d00.pth) |
| Mask RCNN | X101 | 3x | [model](https://download.openmmlab.com/mmdetection/v2.0/mask_rcnn/mask_rcnn_x101_32x8d_fpn_mstrain-poly_3x_coco/mask_rcnn_x101_32x8d_fpn_mstrain-poly_3x_coco_20210607_161042-8bd2c639.pth) |
| Cascade Mask RCNN | R50 | 3x | [model](https://download.openmmlab.com/mmdetection/v2.0/cascade_rcnn/cascade_mask_rcnn_r50_caffe_fpn_mstrain_3x_coco/cascade_mask_rcnn_r50_caffe_fpn_mstrain_3x_coco_20210707_002651-6e29b3a6.pth) |
| Cascade Mask RCNN | X101 | 3x | [model](https://download.openmmlab.com/mmdetection/v2.0/cascade_rcnn/cascade_mask_rcnn_x101_32x4d_fpn_mstrain_3x_coco/cascade_mask_rcnn_x101_32x4d_fpn_mstrain_3x_coco_20210706_225234-40773067.pth) |
| PointRend | R50 | 3x | [model](https://download.openmmlab.com/mmdetection/v2.0/point_rend/point_rend_r50_caffe_fpn_mstrain_3x_coco/point_rend_r50_caffe_fpn_mstrain_3x_coco-e0ebb6b7.pth) |
| Mask Scoring RCNN | X101 | 1x | [model](https://download.openmmlab.com/mmdetection/v2.0/ms_rcnn/ms_rcnn_x101_32x4d_fpn_1x_coco/ms_rcnn_x101_32x4d_fpn_1x_coco_20200206-81fd1740.pth) |

### 2. Modify config file
<div id='config'></div>

Go to [Results and Models](#results-and-models) and find model configuration you want to train. You will need to modify the configuration file in order to train the model. Things you need to modify are:
* `ann_file` and `img_prefix` in the data section
* Put the downloaded pre-trained weights path in `load_from`

### 3. Train the model
```
python tools/train.py <config_file_path> --work-dir <save_dir>/train
```
* input: model configuration file
* output: checkpoints every epoch and training logs will be saved in `<save_dir>/train`

## Validation
In the configuration file, the testing `ann_file` and `img_prefix` should put the validation data path, not the testing data path because test data doesn't has ground truth.
```
python tools/test.py <config_file_path> <save_dir>/train/epoch<X>.pth --eval bbox segm --work-dir <save_dir>/val
```
* input: 
  * model configuration file
  *  checkpoint you save at the last step
* output: validation logs

## Testing

### 1. Convert image to coco format
**Note: If you download the data by following option#1 in [Dataset](#dataset) section you can skip this step.**
```
python tools/dataset_converters/images2coco.py <data_dir>/nucleus_data/test <data_dir>/nucleus_data/classes.txt instance_test.json --imgid_json <data_dir>/nucleus_data/annotations/test_img_ids.json
```
* input: 
  * test image directory
  * `classes.txt`: class names
  * `test_img_ids.json`: test image id
* output: `instance_test.json`

### 2. Generate testing results
```
python tools/test.py <config_file_path> <save_dir>/train/epoch_<X>.json --format-only --options "jsonfile_prefix=test" --show
```
* input: 
  * model configuration file
  * trained model checkpoint
* output:
  * `test.segm.json`: instance segmentation results
  * `test.bbox.json`: detection results

## Submit the results
1. rename the result file: `mv test.segm.json answer.json`
2. compress the file: `zip answer.zip answer.json`
3. upload the result to CodaLab to get the testing score

## Results and Models
<div id='results'></div>
| **Model** | **Backbone** | **Lr_schd** | **Mask AP** | **Config** | **Download** |
|:---:|:---:|:---:|:---:|:---:|:---:|
| Mask RCNN | R50 | 3x | 0.2323 | [config](mmdetection/configs/nucleus/mask_rcnn_r50_caffe_fpn_mstrain-poly_3x_nucleus.py) | [model](https://drive.google.com/file/d/1SbvaWoegYhEm9nUAdJVGnZEIy_inyAeW/view?usp=sharing) |
| Mask RCNN | X101 | 3x | 0.2316 | [config](mmdetection/configs/nucleus/mask_rcnn_x101_32x8d_fpn_mstrain-poly_3x_nucleus.py) | - |
| Cascade Mask RCNN | R50 | 3x | 0.2428 | [config](mmdetection/configs/nucleus/cascade_mask_rcnn_r50_caffe_fpn_mstrain_3x_nucleus.py) | [model](https://drive.google.com/file/d/1-Qjsfxg5_gG9PpwyOXBa4_ogygJyxxN7/view?usp=sharing) |
| Cascade Mask RCNN | X101 | 3x | 0.2444 | [config](mmdetection/configs/nucleus/cascade_mask_rcnn_x101_32x4d_fpn_mstrain_3x_nucleus.py) | [model](https://drive.google.com/file/d/1wSgynRlrb9Y8yK1r_xmRQloYFlN7KPRN/view?usp=sharing) |
| PointRend | R50 | 3x | 0.2439 | [config](mmdetection/configs/nucleus/point_rend_r50_caffe_fpn_mstrain_3x_nucleus.py) | [model](https://drive.google.com/file/d/1XrkKaJdMoOXZodlzAj4lMx8gccBQu7Rs/view?usp=sharing) |
| Mask Scoring RCNN | X101 | 1x | 0.2420 | [config](mmdetection/configs/nucleus/ms_rcnn_x101_32x4d_fpn_1x_nucleus.py) | [model](https://drive.google.com/file/d/1-lftdJXJRVpDhzhIMfXrfDNaoqUoRvFP/view?usp=sharing) |


## Inference
**Note we use Cascade Mask RCNN as our model with X101 as our backbone**
To reproduce our best results, do the following steps:
1. [Getting the code](#getting-the-code)
2. [Install the dependencies](#requirements)
3. [Download the data](#dataset): please download the data by following **option#1**
4. <a href="#pre-trained">Download pre-trained weights</a>
5. <a href="#config">Modify config file</a>: 
6. <a href='#results'>Download checkpoints</a>
7. [Testing](#testing)
8. [Submit the results](#submit-the-results)

## GitHub Acknowledgement
We thank the authors of these repositories:
* [open-mmlab/mmdetection](https://github.com/open-mmlab/mmdetection)
* [jsbroks/imantics](https://github.com/jsbroks/imantics)

## Citation
If you find our work useful in your project, please cite:

```bibtex
@misc{
    title = {mmdet-nucleus-instance-segmentation},
    author = {Zhi-Yi Chin},
    url = {https://github.com/joycenerd/mmdet-nucleus-instance-segmentation},
    year = {2021}
}
```

## Contributing

If you'd like to contribute, or have any suggestions, you can contact us at [joycenerd.cs09@nycu.edu.tw](mailto:joycenerd.cs09@nycu.edu.tw) or open an issue on this GitHub repository.

All contributions welcome! All content in this repository is licensed under the MIT license.
