
# Some basic setup:
# Setup detectron2 logger
import detectron2
from detectron2.utils.logger import setup_logger

setup_logger()

# import some common libraries
import numpy as np
import os, json, cv2, random

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog

from detectron2.engine import DefaultTrainer
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader

import parse_ds as ds

OUTPUT_DIR = './experiments'

MOTS_PATH = '/home/mcv/datasets/MOTSChallenge/train/images/'
KITTI_MOTS_PATH = '/home/mcv/datasets/KITTI-MOTS/training/image_02/'
PKLS_PATH = './pkls/'

#config
models = ["COCO-Detection/retinanet_R_50_FPN_3x.yaml", "COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"]
datasets = []

# Load/Register datasets
ds_name = 'kitti-mots'
kittimots_train_dicts, kittimots_val_dicts = ds.get_mots_dicts(KITTI_MOTS_PATH, ds_name)

labels = set()
# remap dataset class labels
for dataset in [kittimots_train_dicts, kittimots_val_dicts]:
    for image in dataset:
        for obj in image['annotations']:
            if obj['category_id'] == 1:
                obj['category_id'] = 2
            elif obj['category_id'] == 2:
                obj['category_id'] = 0
            labels.add(obj['category_id'])
print('LABELS', labels)

print('Registering...')
DatasetCatalog.register(ds_name+'_val', lambda : kittimots_val_dicts)
MetadataCatalog.get(ds_name+'_val').set(thing_classes=['pedestrian', 'bike', 'car'])

datasets.append(ds_name+'_val')

ds_name = 'mots'
mots_train_dicts, mots_val_dicts = ds.get_mots_dicts(MOTS_PATH, ds_name)

labels = set()
# remap dataset class labels
for dataset in [mots_train_dicts, mots_val_dicts]:
    for image in dataset:
        for obj in image['annotations']:
            if obj['category_id'] == 1:
                obj['category_id'] = 2
            elif obj['category_id'] == 2:
                obj['category_id'] = 0
            labels.add(obj['category_id'])
print('LABELS', labels)

allmots_train_dicts = kittimots_train_dicts + mots_train_dicts
allmots_val_dicts = kittimots_val_dicts + mots_val_dicts

print('Registering...')
DatasetCatalog.register(ds_name+'_val', lambda : allmots_val_dicts)
MetadataCatalog.get(ds_name+'_val').set(thing_classes=['pedestrian', 'bike', 'car'])

datasets.append(ds_name+'_val')

# Inference should use the config with parameters that are used in training
# cfg now already contains everything we've set previously. We changed it a little bit for inference:
print('Evaluating...')

for model in models:
    for dts in datasets:

        cfg = get_cfg()
        cfg.merge_from_file(model_zoo.get_config_file(model))
        cfg.DATASETS.TEST = (dts,)
        cfg.DATALOADER.NUM_WORKERS = 2
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(model)  # Let training initialize from model zoo
        cfg.SOLVER.IMS_PER_BATCH = 2
        cfg.SOLVER.BASE_LR = 0.00025  # pick a good LR
        cfg.SOLVER.MAX_ITER = 300    # 300 iterations seems good enough for this toy dataset; you will need to train longer for a practical dataset
        cfg.SOLVER.STEPS = []        # do not decay learning rate
        cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128   # faster, and good enough for this toy dataset (default: 512)
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2  # only has one class (ballon). (see https://detectron2.readthedocs.io/tutorials/datasets.html#update-the-config-for-new-datasets)
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5   # set a custom testing threshold
        

        predictor = DefaultPredictor(cfg)

        eval_name = f'{dts}_{model[15:-5]}_pretrained'

        os.makedirs(f'./output_eval_pretrained/{eval_name}', exist_ok=True)
        evaluator = COCOEvaluator(dts, ("bbox", ), False, output_dir=f'./output_eval_pretrained/{eval_name}')
        val_loader = build_detection_test_loader(cfg, dts)

        print(inference_on_dataset(predictor.model, val_loader, evaluator))
        # another equivalent way to evaluate the model is to use `trainer.test`