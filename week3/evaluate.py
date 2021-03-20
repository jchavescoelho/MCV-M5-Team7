
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


MOTS_PATH = '/home/mcv/datasets/MOTSChallenge/'
MOTS_ALL_DICT_PATH = '/home/group07/code/save/mots_dict.pkl'
MOTS_TRAIN_DICT_PATH = '/home/group07/code/save/mots_dict_train.pkl'
MOTS_VAL_DICT_PATH = '/home/group07/code/save/mots_dict_val.pkl'
KITTI_MOTS_PATH = '/home/mcv/datasets/MOTSChallenge/KITTI-MOTS/'
KITTI_MOTS_DICT = '/home/group07/code/save/kitti_mots_dict.pkl'

MOTS_CLASSES = {
    '0': 'background',
    '1': 'car',
    '2': 'pedestrian',
    '10': 'ignore'
}
# Load/Register datasets

# mots
# train
dataset_dicts = ds.get_mots_dicts(MOTS_PATH, 'train', MOTS_ALL_DICT_PATH)

ds_name = 'mots_all'
DatasetCatalog.register(ds_name, lambda : dataset_dicts)
MetadataCatalog.get(ds_name).set(thing_classes=['ignore', 'car', 'pedestrian'])
ds_metadata = MetadataCatalog.get(ds_name)

# visualize
os.makedirs('./samplegt/', exist_ok=True)
for d in random.sample(dataset_dicts, 3):
    img = cv2.imread(d["file_name"])
    visualizer = Visualizer(img[:, :, ::-1], metadata=ds_metadata, scale=0.5)
    out = visualizer.draw_dataset_dict(d)
    cv2.imwrite('./samplegt/gt_'+d["file_name"], out.get_image()[:, :, ::-1])