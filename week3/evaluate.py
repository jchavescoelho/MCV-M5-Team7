
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

<<<<<<< HEAD


MOTS_PATH = '/home/mcv/datasets/MOTSChallenge/train/images/'
KITTI_MOTS_PATH = '/home/mcv/datasets/KITTI-MOTS/training/image_02/'
PKLS_PATH = './pkls/'
=======
MOTS_PATH = '/home/mcv/datasets/MOTSChallenge/train/images/'
KITTI_MOTS_PATH = '/home/mcv/datasets/KITTI-MOTS/training/image_02/'
PKLS_PATH = './pkls/'
TRAIN = True
>>>>>>> ea5e7eb909b43829db845fa2d016fa31f1ac8d4e

MOTS_CLASSES = {
    '0': 'background',
    '1': 'car',
    '2': 'pedestrian',
    '10': 'ignore'
}
# Load/Register datasets

# mots
# train
ds_name = 'kitti-mots'
mots_train_dicts, mots_val_dicts = ds.get_mots_dicts(KITTI_MOTS_PATH, ds_name)

<<<<<<< HEAD
print('Registering...')
DatasetCatalog.register(ds_name+'_train', lambda : mots_train_dicts)
MetadataCatalog.get(ds_name+'_train').set(thing_classes=['ignore', 'car', 'pedestrian'])
ds_metadata = MetadataCatalog.get(ds_name+'_train')

DatasetCatalog.register(ds_name+'_val', lambda : mots_val_dicts)
MetadataCatalog.get(ds_name+'_val').set(thing_classes=['ignore', 'car', 'pedestrian'])

=======
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

print('Registering...')
DatasetCatalog.register(ds_name+'_train', lambda : mots_train_dicts)
MetadataCatalog.get(ds_name+'_train').set(thing_classes=['pedestrian', 'bike', 'car'])
ds_metadata = MetadataCatalog.get(ds_name+'_train')

DatasetCatalog.register(ds_name+'_val', lambda : mots_val_dicts)
MetadataCatalog.get(ds_name+'_val').set(thing_classes=['pedestrian', 'bike', 'car'])
>>>>>>> ea5e7eb909b43829db845fa2d016fa31f1ac8d4e

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
    cv2.imwrite(saveto, out.get_image()[:, :, ::-1])

# Pre-trained
print('Loading pre-trained models...')

#Select model
classes = ('pedestrian', 'car')
model_zoo_yml = "COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"
model_zoo_yml = "COCO-Detection/retinanet_R_50_FPN_3x.yaml"
folder_name = model_zoo_yml.split('/')[-1].split('.')[0]

cfg = get_cfg()
cfg.OUTPUT_DIR = os.path.join('models', folder_name)

# cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/retinanet_R_50_FPN_3x.yaml"))
cfg.merge_from_file(model_zoo.get_config_file(model_zoo_yml))

# cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/retinanet_R_50_FPN_3x.yaml")
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(model_zoo_yml)


# Training goes here
if TRAIN:
    from detectron2.engine import DefaultTrainer
    print('Configuring training...')
    cfg.DATASETS.TRAIN = (ds_name + "_train",)
    cfg.DATASETS.TEST = ()
    cfg.DATALOADER.NUM_WORKERS = 4
    cfg.SOLVER.IMS_PER_BATCH = 8
    cfg.SOLVER.BASE_LR = 0.00025  # pick a good LR
    cfg.SOLVER.MAX_ITER = 300    # 300 iterations seems good enough for this toy dataset; you will need to train longer for a practical dataset
    cfg.SOLVER.STEPS = []        # do not decay learning rate
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512   # faster, and good enough for this toy dataset (default: 512)
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(classes)  

    print('Preparing training...')
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    trainer = DefaultTrainer(cfg) 
    trainer.resume_or_load(resume=False)

    print('Launching training...')
    trainer.train()
    
    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")  # path to the model we just trained

# Random inferences
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model

predictor = DefaultPredictor(cfg)
os.makedirs(f'./sampleinfer/{folder_name}', exist_ok=True)

print('Running some random inferences...')
for d in random.sample(mots_train_dicts, 5):
    file_name = d['file_name']
    im = cv2.imread(file_name)
    outputs = predictor(im)
<<<<<<< HEAD
    # instances = outputs["instances"][outputs["instances"].scores > 0.5]
    v = Visualizer(im[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1)
    out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    name = os.path.split(d['file_name'])[-1]
    saveto = f'/home/group07/code/MCV-M5-Team7/week3/sampleinfer/{folder_name}/gt_' + name
    cv2.imwrite(saveto, out.get_image()[:, :, ::-1])

print('What now?')
=======
    inst = outputs["instances"].to('cpu')
    inst = inst[[True if c == 0 or c == 2 else False for c in inst.pred_classes]]
    # instances = outputs["instances"][outputs["instances"].scores > 0.5]
    v = Visualizer(im[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1)
    out = v.draw_instance_predictions(inst.to("cpu"))
    name = os.path.split(d['file_name'])[-1]
    saveto = f'/home/group07/code/MCV-M5-Team7/week3/sampleinfer/{folder_name}/infer_' + name
    cv2.imwrite(saveto, out.get_image()[:, :, ::-1])


# Evaluate
print('Evaluating...')
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader

evaluator = COCOEvaluator(ds_name + "_val", ("bbox",), False, output_dir="./output/")
val_loader = build_detection_test_loader(cfg, ds_name + "_val")
print(inference_on_dataset(predictor.model, val_loader, evaluator))
# another equivalent way to evaluate the model is to use `trainer.test`
>>>>>>> ea5e7eb909b43829db845fa2d016fa31f1ac8d4e
