import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import numpy as np
import cv2
import random

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog

import json
from detectron2.structures import BoxMode

from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg
import os

from detectron2.utils.visualizer import ColorMode
from detectron2.data import MetadataCatalog, DatasetCatalog

from detectron2.engine import DefaultTrainer

#import the COCO Evaluator to use the COCO Metrics
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader
import parse_ds as ds

OUTPUT_DIR = './faster_test'

MOTS_PATH = '/home/mcv/datasets/MOTSChallenge/train/images/'
KITTI_MOTS_PATH = '/home/mcv/datasets/KITTI-MOTS/training/image_02/'
PKLS_PATH = './pkls/'

#Register dataset

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
DatasetCatalog.register(ds_name+'_train', lambda : kittimots_train_dicts)
MetadataCatalog.get(ds_name+'_train').set(thing_classes=['pedestrian', 'bike', 'car'])

DatasetCatalog.register(ds_name+'_val', lambda : kittimots_val_dicts)
MetadataCatalog.get(ds_name+'_val').set(thing_classes=['pedestrian', 'bike', 'car'])

#Visualize
# dataset_dicts = get_board_dicts("Text_Detection_Dataset_COCO_Format/train")
# #Randomly choosing 3 images from the Set
# for d in random.sample(dataset_dicts, 3):
#     img = cv2.imread(d["file_name"])
#     visualizer = Visualizer(img[:, :, ::-1], metadata=board_metadata)
#     vis = visualizer.draw_dataset_dict(d)
#     cv2_imshow(vis.get_image()[:, :, ::-1])


cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")) #Get the basic model configuration from the model zoo 
#Passing the Train and Validation sets
cfg.DATASETS.TRAIN = (ds_name+'_train',)
cfg.DATASETS.TEST = (ds_name+'_val',)
# Number of data loading threads
cfg.DATALOADER.NUM_WORKERS = 4
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")  # Let training initialize from model zoo
# Number of images per batch across all machines.
cfg.SOLVER.IMS_PER_BATCH = 4
cfg.SOLVER.BASE_LR = 0.0125  # pick a good LearningRate
cfg.SOLVER.MAX_ITER = 500  #No. of iterations   
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 256  
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 3 # No. of classes = [HINDI, ENGLISH, OTHER]
cfg.TEST.EVAL_PERIOD = 300 # No. of iterations after which the Validation Set is evaluated. 


os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
trainer = DefaultTrainer(cfg) 
trainer.resume_or_load(resume=False)
trainer.train()

#Use the final weights generated after successful training for inference  
cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")

cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.8  # set the testing threshold for this model
#Pass the validation dataset
cfg.DATASETS.TEST = (ds_name+'_val', )

predictor = DefaultPredictor(cfg)

# dataset_dicts = get_board_dicts("Text_Detection_Dataset_COCO_Format/val")
# for d in random.sample(dataset_dicts, 3):    
#     im = cv2.imread(d["file_name"])
#     outputs = predictor(im)
#     v = Visualizer(im[:, :, ::-1],
#                    metadata=board_metadata, 
#                    scale=0.8,
#                    instance_mode=ColorMode.IMAGE   
#     )
#     v = v.draw_instance_predictions(outputs["instances"].to("cpu")) #Passing the predictions to CPU from the GPU
#     cv2_imshow(v.get_image()[:, :, ::-1])

# evaluat = DefaultTrainer.test(evaluators=)
#Call the COCO Evaluator function and pass the Validation Dataset
evaluator = COCOEvaluator(ds_name+'_val', cfg, False, output_dir="/output_faster/")
val_loader = build_detection_test_loader(cfg, ds_name+'_val')

#Use the created predicted model in the previous step
inference_on_dataset(predictor.model, val_loader, evaluator)
