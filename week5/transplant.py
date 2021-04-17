import os
import time
import random
import argparse
import pickle as pkl
import colorsys

import cv2
import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt

from torchvision.models.detection import maskrcnn_resnet50_fpn, fasterrcnn_resnet50_fpn
from cocods import paint_detections

DATA_DIR = '/home/capiguri/code/datasets/COCO/test2017/all'
OUTPUT_DIR = './results'
model = maskrcnn_resnet50_fpn(True)
model.eval()

im1 = '000000580537.jpg'
im2 = '000000581918.jpg'

def extract_obj(im_path, idx):
    original = cv2.imread(im_path)
    im = original.copy()

    if im is None:
        print(f'Error reading {im_path}')
        quit()

    cv2.imshow('Input', im)

    det = model(torchvision.transforms.ToTensor()(im).unsqueeze(0))[0]
    out = paint_detections(im, det)

    cv2.imshow('Output', out)

    crop = det['masks'][idx]
    crop = (crop > 0.5).float()
    crop = crop.cpu().numpy()
    crop = np.transpose(crop, (1, 2, 0))
    crop *= 255
    crop = cv2.cvtColor(crop, cv2.COLOR_GRAY2BGR)

    cv2.imshow('Crop', crop)
    cv2.waitKey(0)

    return original, crop


with torch.no_grad():
    # Task b
    im, mask = extract_obj(os.path.join(DATA_DIR, im1), 0)

    masked = im * (mask//255).astype(np.uint8)

    cv2.imshow('Masked', masked)
    cv2.waitKey(0)

    dst = cv2.imread(os.path.join(DATA_DIR, im2))


