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
SAVE_PATH_RAW = '/home/group07/code/MCV-M5-Team7/week4/evaluation_pret_good/'

os.makedirs(SAVE_PATH_RAW, exist_ok=True)

# Load/Register datasets

ds_name = 'kitti-mots'
train_dicts, val_dicts = ds.get_mots_dicts(KITTI_MOTS_PATH, ds_name)

DatasetCatalog.register(ds_name+'_train', lambda : train_dicts)
MetadataCatalog.get(ds_name+'_train').set(thing_classes=['pedestrian', 'ignore', 'car'])

DatasetCatalog.register(ds_name+'_val', lambda : val_dicts)
MetadataCatalog.get(ds_name+'_val').set(thing_classes=['pedestrian', 'ignore', 'car'])

ds_metadata = MetadataCatalog.get(ds_name+'_train')


# Pre-trained
print('Loading pre-trained models...')

#Select model
classes = ('pedestrian', 'car')

models = [
    "COCO-InstanceSegmentation/mask_rcnn_R_50_C4_3x.yaml",
    "COCO-InstanceSegmentation/mask_rcnn_R_101_C4_3x.yaml",
    "COCO-InstanceSegmentation/mask_rcnn_R_50_DC5_3x.yaml",
    "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml",
    "Cityscapes/mask_rcnn_R_50_FPN.yaml"
    ]

for model_zoo_yml in models:

    model_name = model_zoo_yml.split('/')[-1][:-5]
    print(model_name)
    SAVE_PATH = os.path.join(SAVE_PATH_RAW, model_name)
    os.makedirs(SAVE_PATH, exist_ok=True)

    print('Running for model', model_zoo_yml)
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(model_zoo_yml))
    cfg.DATASETS.TRAIN = (ds_name + '_train', )
    cfg.DATASETS.TEST = (ds_name + '_val', )
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
    cfg.OUTPUT_DIR = SAVE_PATH
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(model_zoo_yml)

    predictor = DefaultPredictor(cfg)

    import pandas as pd
    modelclasses = MetadataCatalog.get(cfg.DATASETS.TRAIN[0]).thing_classes
    df = pd.DataFrame(modelclasses,columns=['Model classes'])
    print(df)

    # visualize
    # SAVE_PATH = 
    os.makedirs(os.path.join(SAVE_PATH, 'images'), exist_ok=True)

    print('Save some visualizations...')
    for d in random.sample(val_dicts, 5):
        print('\n', d["file_name"])
        img = cv2.imread(d["file_name"])
        visualizer = Visualizer(img[:, :, ::-1], metadata=ds_metadata, scale=0.5)
        out = visualizer.draw_dataset_dict(d)
        name = os.path.split(d['file_name'])[-1]
        saveto = os.path.join(os.path.join(SAVE_PATH, 'images'), 'gt_' + name)
        print(saveto)
        cv2.imwrite(saveto, out.get_image()[:, :, ::-1])

    print('Running some random inferences...')
    for d in random.sample(val_dicts, 5):
        file_name = d['file_name']
        im = cv2.imread(file_name)
        outputs = predictor(im)

        inst = outputs["instances"].to('cpu')
        inst = inst[[True if c == 0 or c == 2 else False for c in inst.pred_classes]]
        # instances = outputs["instances"][outputs["instances"].scores > 0.5]
        v = Visualizer(im[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1)
        out = v.draw_instance_predictions(inst.to("cpu"))
        name = os.path.split(d['file_name'])[-1]
        saveto = os.path.join(SAVE_PATH, 'images', 'infer_' + name)
        print(saveto)
        cv2.imwrite(saveto, out.get_image()[:, :, ::-1])

    # Evaluate
    print('Evaluating...')

    from detectron2.evaluation import COCOEvaluator, inference_on_dataset
    from detectron2.data import build_detection_test_loader

    evaluator = COCOEvaluator(ds_name + "_val", ("bbox", "segm",), False, output_dir=SAVE_PATH)
    val_loader = build_detection_test_loader(cfg, ds_name + "_val")
    results = inference_on_dataset(predictor.model, val_loader, evaluator)

    print(os.path.join(SAVE_PATH, 'results.json'))
    with open(os.path.join(SAVE_PATH, 'results.json'), 'w') as fp:
        json.dump(results, fp, indent=4)


