
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

MOTS_PATH = '/home/mcv/datasets/MOTSChallenge/'
KITTI_MOTS_PATH = '/home/mcv/datasets/MOTSChallenge/KITTI-MOTS/'

MOTS_CLASSES = {
    '0': 'background',
    '1': 'car',
    '2': 'pedestrian',
    '10': 'ignore'
}

def get_mots_dicts(path):
    dataset_dicts = []

    for idx, img_path in enumerate(glob.glob(MOTS_PATH + 'train/images/*/*.jpg')):
        mask_path = img_path.replace('images', 'instances').replace('jpg', 'png')
        print(img_path)
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
        # class_im = cv2.resize(class_im, (width//2, height//2), interpolation=cv2.INTER_NEAREST)
        # ids_im = cv2.resize(ids_im, (width//2, height//2), interpolation=cv2.INTER_NEAREST)

        class_list = np.unique(class_im)
        id_list = np.unique(ids_im)

        objs = []

        print('Objs in image:', id_list)
        print('Classes in image:', class_list)

        if 1 not in class_list and 2 not in class_list:
            continue

        for id in id_list:
            w = np.where(ids_im == id)
            # print(f'{len(w[0])} matching points')
            px = w[1]
            py = w[0]

            # Segmentation
            # pts = [(int(x), int(y)) for x,y in zip(px, py)]
            # poly = [i for p in pts for i in p]
            # print('Computed poly')

            bbox = [np.min(px), np.min(py), np.max(px), np.max(py)]
            category = class_im[bbox[1], bbox[0]]
            print(f'id {id} is a {MOTS_CLASSES[str(category)]}')

            if category == 1 or category == 2:
                obj = {
                    "bbox": bbox,
                    "bbox_mode": BoxMode.XYXY_ABS,
                    # "segmentation": [poly],
                    "category_id": category,
                }

                objs.append(obj)
                print('#')

        record["annotations"] = objs
        dataset_dicts.append(record)
        print(fsize(sys.getsizeof(dataset_dicts)))
        print()
    return dataset_dicts

d = get_mots_dicts(MOTS_PATH)
print('Saving tp pkl...')
with open('mots_dict.pkl', 'wb') as fp:
    pkl.dump(d, fp)
print('Done')

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
