import glob
import argparse
import math
import random
import os
import shutil

parser = argparse.ArgumentParser()
parser.add_argument('--data-root', type=str, default='/eva_data/zchin/dataset/train',
                    help='trainig image saving directory')
parser.add_argument('--ratio', type=float, default=0.2, help='validation data ratio')
parser.add_argument('--out-dir', type=str, default='/eva_data/zchin/nucleus_data')
args = parser.parse_args()

if __name__ == '__main__':
    train_dir = os.path.join(args.out_dir, 'train')
    val_dir = os.path.join(args.out_dir, 'val')
    if not os.path.isdir(args.out_dir):
        os.makedirs(args.out_dir)
        os.mkdir(train_dir)
        os.mkdir(val_dir)

    data_size = len(os.listdir(args.data_root))
    print(f'data size: {data_size}')
    valid_size = math.floor(data_size * args.ratio)

    img_list = []
    for img_dir_path in glob.glob(f'{args.data_root}/*'):
        img_name = img_dir_path.split('/')[-1]
        img_path = os.path.join(img_dir_path, 'images', img_name + '.png')
        if os.path.isfile(img_path):
            img_list.append(img_path)
    print(f'img_list size: {len(img_list)}')

    idx = random.sample(range(data_size), valid_size)

    for i, src_img_path in enumerate(img_list):
        img_name = src_img_path.split('/')[-1]
        if i in idx:
            dest_img_path = os.path.join(args.out_dir, 'val', img_name)
        else:
            dest_img_path = os.path.join(args.out_dir, 'train', img_name)
        shutil.copyfile(src_img_path, dest_img_path)

    train_size = len(glob.glob1(train_dir, "*.png"))
    valid_size = len(glob.glob1(val_dir, "*.png"))
    print(f'train size: {train_size}\tvalid size: {valid_size}')
