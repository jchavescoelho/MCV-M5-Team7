
# Some basic setup:
# Setup detectron2 logger
import detectron2
from detectron2.utils.logger import setup_logger

setup_logger()

# import some common libraries
import numpy as np
import os, json, cv2, random
#from google.colab.patches import cv2_imshow

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader

from detectron2.engine import DefaultTrainer

import parse_ds as ds

OUTPUT_DIR = './experiments'
MOTS_PATH = '/home/mcv/datasets/MOTSChallenge/train/images/'
KITTI_MOTS_PATH = '/home/mcv/datasets/KITTI-MOTS/training/image_02/'

#Fine tuning config
# learn_rates = [0.00025, 0.0005, 0.001, 0.01, 0.1, 1]
learn_rates = [0.001]
# models = ["COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml", "COCO-Detection/retinanet_R_50_FPN_3x.yaml"]
models = [
    "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml",
    "Cityscapes/mask_rcnn_R_50_FPN.yaml"]

# datasets = ['mots_train', 'kitti-mots_train']
datasets = ['kitti-mots', 'mots']
# batchs = [32, 64, 128, 254, 512]
batchs = [512]

# Load/Register datasets
ds_name = 'kitti-mots'
kittimots_train_dicts, kittimots_val_dicts = ds.get_mots_dicts(KITTI_MOTS_PATH, ds_name)

print('Registering...')
DatasetCatalog.register(ds_name+'_train', lambda : kittimots_train_dicts)
MetadataCatalog.get(ds_name+'_train').set(thing_classes=['pedestrian', 'ignore', 'car'])

DatasetCatalog.register(ds_name+'_val', lambda : kittimots_val_dicts)
MetadataCatalog.get(ds_name+'_val').set(thing_classes=['pedestrian', 'ignore', 'car'])

ds_name = 'mots'
mots_train_dicts, mots_val_dicts = ds.get_mots_dicts(MOTS_PATH, ds_name)

allmots_train_dicts = kittimots_train_dicts + mots_train_dicts
allmots_val_dicts = kittimots_val_dicts + mots_val_dicts

print('Registering...')
DatasetCatalog.register(ds_name+'_train', lambda : allmots_train_dicts)
MetadataCatalog.get(ds_name+'_train').set(thing_classes=['pedestrian', 'ignore', 'car'])

DatasetCatalog.register(ds_name+'_val', lambda : allmots_val_dicts)
MetadataCatalog.get(ds_name+'_val').set(thing_classes=['pedestrian', 'ignore', 'car'])


for lr in learn_rates:
    for model in models:
        for dts in datasets:
            for batch in batchs:
                model_name = model.split('/')[-1][:-5]
                experiment_name = f'{dts}_{model_name}_lr{lr}_batch{batch}'
                print('Launching', experiment_name)
                cfg = get_cfg()
                cfg.merge_from_file(model_zoo.get_config_file(model))
                cfg.DATASETS.TRAIN = (dts +'_train',)
                cfg.DATASETS.TEST = ('kitti-mots_val',)
                cfg.DATALOADER.NUM_WORKERS = 0
                cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
                cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(model)  # Let training initialize from model zoo
                cfg.SOLVER.IMS_PER_BATCH = 3
                cfg.SOLVER.BASE_LR = lr      # pick a good LR
                cfg.SOLVER.MAX_ITER = 100    # 300 iterations seems good enough for this toy dataset; you will need to train longer for a practical dataset
                cfg.SOLVER.STEPS = []        # do not decay learning rate
                cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = batch   # faster, and good enough for this toy dataset (default: 512)
                cfg.MODEL.ROI_HEADS.NUM_CLASSES = 3  # only has one class (ballon). (see https://detectron2.readthedocs.io/tutorials/datasets.html#update-the-config-for-new-datasets)
                # NOTE: this config means the number of classes, but a few popular unofficial tutorials incorrect uses num_classes+1 here.

#                cfg.INPUT.MAX_SIZE_TRAIN: 2048
#                cfg.INPUT.MIN_SIZE_TEST: 1024
#                cfg.INPUT.MAX_SIZE_TEST: 2048

                cfg.OUTPUT_DIR = os.path.join(OUTPUT_DIR, experiment_name)
                os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

                trainer = DefaultTrainer(cfg) 
                trainer.resume_or_load(resume=False)
                trainer.train()

                ds_val =  'kitti-mots_val'
                os.makedirs(f'./outputs_train/{model_name}', exist_ok=True)
                # evaluator = COCOEvaluator(ds_val, ("segm",), False, output_dir=f'./outputs_train/{model_name}')
                evaluator = COCOEvaluator(ds_val, cfg, False, output_dir=f'./outputs_train/{model_name}')

                results = trainer.test(cfg, trainer.model, evaluators=[evaluator])
                print(results)
#                predictor = DefaultPredictor(cfg) 
                # results = inference_on_dataset(trainer.model, val_loader, evaluator)
                # another equivalent way to evaluate the model is to use `trainer.test`   
                print(os.path.join(f'./outputs_train/{model_name}', 'results.json'))
                with open(os.path.join(f'./outputs_train/{model_name}', 'results.json'), 'w') as fp:
                    json.dump(results, fp, indent=4)

