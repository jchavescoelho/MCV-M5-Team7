
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

import parse_ds as ds



MOTS_PATH = '/home/mcv/datasets/MOTSChallenge/train/images/'
KITTI_MOTS_PATH = '/home/mcv/datasets/MOTSChallenge/KITTI-MOTS/training/image_02/'
PKLS_PATH = './pkls/'

MOTS_CLASSES = {
    '0': 'background',
    '1': 'car',
    '2': 'pedestrian',
    '10': 'ignore'
}
# Load/Register datasets

# mots
# train
ds_name = 'mots'
mots_train_dicts, mots_val_dicts = ds.get_mots_dicts(MOTS_PATH, ds_name)

print('Registering...')
DatasetCatalog.register(ds_name+'_train', lambda : mots_train_dicts)
MetadataCatalog.get(ds_name+'_train').set(thing_classes=['ignore', 'car', 'pedestrian'])
ds_metadata = MetadataCatalog.get(ds_name+'_train')

DatasetCatalog.register(ds_name+'_val', lambda : mots_val_dicts)
MetadataCatalog.get(ds_name+'_val').set(thing_classes=['ignore', 'car', 'pedestrian'])


# visualize
print('Save some visualizations...')
os.makedirs('./samplegt/', exist_ok=True)
for d in random.sample(mots_train_dicts, 5):
    print('\n', d["file_name"])
    img = cv2.imread(d["file_name"])
    visualizer = Visualizer(img[:, :, ::-1], metadata=ds_metadata, scale=0.5)
    out = visualizer.draw_dataset_dict(d)
    name = os.path.split(d['file_name'])[-1]
    saveto = '/home/group07/code/MCV-M5-Team7/week3/samplegt/gt_' + name
    print(out.get_image()[:, :, ::-1])
    cv2.imwrite(saveto, out.get_image()[:, :, ::-1])

print('What now')