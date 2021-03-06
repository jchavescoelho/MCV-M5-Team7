
# import some common libraries
import os
import sys
import json
import glob
import random

from hurry.filesize import size as fsize
import cv2
import numpy as np
import pandas as pd
import pickle as pkl
import PIL.Image as Image

# import some common detectron2 utilities
import detectron2
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.structures import BoxMode
from detectron2.engine import DefaultPredictor
from detectron2.utils.logger import setup_logger
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
setup_logger()

MOTS_PATH = '/home/mcv/datasets/MOTSChallenge/train/images/'
KITTI_MOTS_PATH = '/home/mcv/datasets/KITTI-MOTS/training/image_02/'
KITTI_MOTS_PATH = '/home/mcv/datasets/KITTI-MOTS/training/image_02/'
PKLS_PATH = './pkls/'

MOTS_CLASSES = {
    '0': 'background',
    '1': 'car',
    '2': 'pedestrian',
    '10': 'ignore'
}

def get_mots_dicts(ds_path, ds_name):

    ds_train_dicts = []
    ds_val_dicts = []

    val_sets = [2, 6, 7, 8, 10, 13, 14, 16, 18]
    if ds_name == 'mots':
        val_sets = [2]

    if not os.path.exists(f'{PKLS_PATH}{ds_name}_train_dict.pkl') or not os.path.exists(f'{PKLS_PATH}{ds_name}_val_dict.pkl'):
        print("Pkl file does not exist. Generating dics from dataset:")

        for idx, img_path in enumerate(glob.glob(f'{ds_path}*/*.*g')):
            mask_path = img_path.replace('training', 'instances').replace('images', 'instances').replace('jpg', 'png').replace('image_02/', '')

            seq = int(mask_path.split('/')[-2])

            print(f'Images: {img_path}')
            print(f'Masks: {mask_path}')
            print('seq:', seq)

            im = np.array(Image.open(mask_path))
            if im is None:
                continue
        
            height, width = im.shape[:2]
            
            record = {}
            record["file_name"] = img_path
            record["image_id"] = idx
            record["height"] = height
            record["width"] = width

            class_im, ids_im = im // 1000, im % 1000

            # Memory stuff
            del im
            class_im = class_im.astype(np.uint8)
            ids_im = ids_im.astype(np.uint8)

            f = 1
            new_size = (int(width*f), int(height*f))
            class_im = cv2.resize(class_im, new_size, interpolation=cv2.INTER_NEAREST)
            ids_im = cv2.resize(ids_im, new_size, interpolation=cv2.INTER_NEAREST)

            class_list = np.unique(class_im)
            id_list = np.unique(ids_im)

            objs = []

            if 1 not in class_list and 2 not in class_list:
                continue

            print('Objs in image:', id_list)
            print('Classes in image:', class_list)

            for id in id_list:
                w = np.where(ids_im == id)

                # Segmentation
                pts = [(int(x), int(y)) for x,y in zip(w[1], w[0])]
                poly = [i for p in pts for i in p]
                del pts
                # print('Computed poly')

                bbox = [np.min(w[1]), np.min(w[0]), np.max(w[1]), np.max(w[0])]
                category = class_im[w[0][0], w[1][0]]

                if category == 1 or category == 2:

                    print(f'id {id} is a {MOTS_CLASSES[str(category)]}')
                    obj = {
                        "bbox": bbox,
                        "bbox_mode": BoxMode.XYXY_ABS,
                        "segmentation": [poly],
                        "category_id": category,
                    }

                    objs.append(obj)
                    print('#')

            record["annotations"] = objs

            if seq in val_sets:
                ds_val_dicts.append(record)
                print(fsize(sys.getsizeof(ds_val_dicts)))
                print()
            else:
                ds_train_dicts.append(record)
                print(fsize(sys.getsizeof(ds_train_dicts)))
                print()

        #Saving dicts
        if not os.path.exists(PKLS_PATH):
            os.makedirs(PKLS_PATH)

        print(f'Saving ds in {PKLS_PATH}{ds_name}_train_dict.pkl...')
        with open(f'{PKLS_PATH}{ds_name}_train_dict.pkl', 'wb') as fp:
            pkl.dump(ds_train_dicts, fp)
        print('Done!')

        print(f'Saving ds in {PKLS_PATH}{ds_name}_val_dict.pkl...')
        with open(f'{PKLS_PATH}{ds_name}_val_dict.pkl', 'wb') as fp:
            pkl.dump(ds_val_dicts, fp)
        print('Done!')

    else:
        print(f'Loading ds from {PKLS_PATH}{ds_name}_train_dict.pkl...')
        with open(f'{PKLS_PATH}{ds_name}_train_dict.pkl', 'rb') as f:
            ds_train_dicts = pkl.load(f)
            print(f'Loading ds from {PKLS_PATH}{ds_name}_val_dict.pkl...')
        with open(f'{PKLS_PATH}{ds_name}_val_dict.pkl', 'rb') as f:
            ds_val_dicts = pkl.load(f)
        print('Done!')

    return ds_train_dicts, ds_val_dicts


# d = get_mots_dicts(MOTS_PATH, ds_name, train)

# ds_name = 'mots_test'
# DatasetCatalog.register(ds_name, lambda d=d: get_mots_dicts(MOTS_PATH))
# MetadataCatalog.get(ds_name).set(thing_classes=['car', 'pedestrian'])
# ds_metadata = MetadataCatalog.get(ds_name)

# show single pair
# for im_path in glob.glob(MOTS_PATH + 'train/images/*/*.jpg'):
#     mask_path = im_path.replace('images', 'instances').replace('jpg', 'png')
#     # print(im_path)
#     # print(mask_path)
#     # print()

#     img = np.array(Image.open(mask_path))
#     pix = np.unique(img)

#     print(im_path.split('/')[-1], pix)


# print(glob.glob('/home/mcv/datasets/MOTSChallenge/train/images/*/*.jpg'))
