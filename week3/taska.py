import os
import numpy as np
import cv2
import torch
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor, DefaultTrainer
from detectron2.data import DatasetCatalog
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.data import MetadataCatalog, build_detection_test_loader
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.modeling import build_model
from detectron2.checkpoint import DetectionCheckpointer

from dataset import KITTIMOTS_Dataloader
from dataset import KITTI_CATEGORIES
from dataset import MOTS_Dataloader
from dataset import MOTS_KITTI_Dataloader
from dataset import MOTS_CATEGORIES
from loss import ValidationLoss, draw_loss

__name__ == '__main__':

    # Loading data KITTI-MOTS
    print('Loading data')
    dataloader_kitti = KITTIMOTS_Dataloader()
    def kitti_train(): return dataloader_kitti.get_dicts(train_flag=True)
    def kitti_val(): return dataloader_kitti.get_dicts(train_flag=False)
    DatasetCatalog.register('KITTIMOTS_train', kitti_train)
    MetadataCatalog.get('KITTIMOTS_train').set(thing_classes=list(KITTI_CATEGORIES.keys()))
    DatasetCatalog.register('KITTIMOTS_val', kitti_val)
    MetadataCatalog.get('KITTIMOTS_val').set(thing_classes=list(KITTI_CATEGORIES.keys()))

    # Loading data MOTSChallenge
    print('Loading MOTSChallenge data')
    dataloader_mots = MOTS_Dataloader()
    def mots_train(): return dataloader_mots.get_dicts(train_flag=True)
    DatasetCatalog.register('MOTS_train', mots_train)
    MetadataCatalog.get('MOTS_train').set(thing_classes=list(MOTS_CATEGORIES.keys()))

    # Loading data MOTSChallenge and KITTI-MOTS
    print('Loading MOTSChallenge and KITTI data')
    dataloader_mots_kitti = MOTS_KITTI_Dataloader()
    def mots_kitti_train(): return dataloader_mots_kitti.get_dicts(train_flag=True)
    DatasetCatalog.register('MOTS_KITTI_train', mots_kitti_train)
    MetadataCatalog.get('MOTS_KITTI_train').set(thing_classes=list(KITTI_CATEGORIES.keys()))