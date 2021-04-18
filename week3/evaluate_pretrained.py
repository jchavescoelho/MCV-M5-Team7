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
KITTI_MOTS_PATH = '/home/mcv/datasets/KITTI-MOTS/training/image_02/'
PKLS_PATH = './pkls_big/'

# Load conf

# Pre-trained
print('Loading pre-trained models...')

#Select model
# classes = ('pedestrian', 'car')
model_zoo_yml = "COCO-InstanceSegmentation/mask_rcnn_R_101_C4_3x.yaml"

cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file(model_zoo_yml))
model_training_metadata = MetadataCatalog.get(cfg.DATASETS.TRAIN[0]) # Store current model training metadata
cfg.DATASETS.TRAIN = ('KITTIMOTS_train', )
cfg.DATASETS.TEST = ('MOTS_train', )
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
cfg.OUTPUT_DIR = SAVE_PATH
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(model_zoo_yml)

model = build_model(cfg)
DetectionCheckpointer(model).load(cfg.MODEL.WEIGHTS)

print('Evaluating')
evaluator = COCOEvaluator('MOTS_train', cfg, False, output_dir=SAVE_PATH)
trainer = DefaultTrainer(cfg)
trainer.test(cfg, model, evaluators=[evaluator])

predictor = DefaultPredictor(cfg)

import pandas as pd
modelclasses = MetadataCatalog.get(cfg.DATASETS.TRAIN[0]).thing_classes
df = pd.DataFrame(modelclasses,columns=['Model classes'])
print(df)



# # visualize
# print('Save some visualizations...')
# os.makedirs('./samplepretrained/', exist_ok=True)

# for d in random.sample(mots_val_dicts, 5):
#     print('\n', d["file_name"])
#     img = cv2.imread(d["file_name"])
#     v = Visualizer(im[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
#     out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
#     saveto = '/home/group07/code/MCV-M5-Team7/week4/samplegt/gt_' + name
#     cv2.imwrite(saveto, out.get_image()[:, :, ::-1])
    
# Evaluate
print('Evaluating...')

from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader

evaluator = COCOEvaluator(ds_name + "_val", ("bbox", "segm"), False, output_dir="./output/")
val_loader = build_detection_test_loader(cfg, ds_name + "_val")
print(inference_on_dataset(predictor.model, val_loader, evaluator))




