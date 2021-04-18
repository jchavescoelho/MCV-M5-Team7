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

def overlay_rnd(img_ol, mask_ol, guide, alpha=0.6):
    pts = np.where((guide != [0, 0, 0]).all(axis=2))
    img_ol[pts] = (img_ol[pts]*(1 - alpha) + mask_ol[pts]*alpha).astype(np.uint8)
    return img_ol

DATA_DIR = '/home/capiguri/code/datasets/COCO/test2017/all'
OUTPUT_DIR = './taskbin'
model = maskrcnn_resnet50_fpn(True)
model.eval()

src_name = '000000052113.jpg'
dst_name = '000000052113.jpg'


def extract_obj(im_path, idx=None):
    name = os.path.split(im_path)[-1]
    print('Source image:', name)

    original = cv2.imread(im_path)
    im = original.copy()

    if im is None:
        print(f'Error reading {im_path}')
        quit()

    # cv2.imshow(f'Image {name}', im)
    # cv2.waitKey(0)

    det = model(torchvision.transforms.ToTensor()(im).unsqueeze(0))[0]
    out = paint_detections(im, det)

    cv2.imshow(f'Source {name}', out)
    cv2.waitKey(100)
    
    if not idx:
        idx = int(input('Select idx: '))

    crop = det['masks'][idx]
    crop = (crop > 0.5).float()
    crop = crop.cpu().numpy()
    crop = np.transpose(crop, (1, 2, 0))
    crop *= 255
    crop = cv2.cvtColor(crop, cv2.COLOR_GRAY2BGR)
    
    masked = original * (crop//255).astype(np.uint8)

    x1, y1, x2, y2 = np.int0(det['boxes'][idx])
    masked = masked[y1:y2, x1:x2]
    crop = crop[y1:y2, x1:x2]

    return masked, crop, idx


os.makedirs(OUTPUT_DIR, exist_ok=True)

with torch.no_grad():
    # Task b
    src_ok = False
    crop_ok = False
    dst_ok = False

    all_imgs = os.listdir(DATA_DIR)

    if not src_name:
        while not src_ok:
            cv2.destroyAllWindows()
            src_name  = random.sample(all_imgs, 1)[0]
            src = cv2.imread(os.path.join(DATA_DIR, src_name))

            cv2.imshow(f'Source {src_name}', src)
            print('Press s if you want use this as source image...')
            k = cv2.waitKey(0)

            if k == ord('s'):
                src_ok = True
    else:
        src = cv2.imread(os.path.join(DATA_DIR, src_name))

    while not crop_ok:
        masked, mask, idx = extract_obj(os.path.join(DATA_DIR, src_name))

        cv2.imshow(f'Masked {src_name}. Idx {idx}', masked)
        print('Press s if you want use this crop as source object...')
        k = cv2.waitKey(0)

        if k == ord('s'):
            crop_ok = True

    if not dst_name:
        lst_dst = 'efimero'
        cv2.namedWindow(lst_dst)
        while not dst_ok:
            cv2.destroyWindow(lst_dst)
            dst_name  = random.sample(all_imgs, 1)[0]
            dst = cv2.imread(os.path.join(DATA_DIR, dst_name))

            lst_dst = f'Destination {dst_name}'
            cv2.imshow(lst_dst, dst)
            print('Press s if you want to use this image as destination...')
            k = cv2.waitKey(0)

            if k == ord('s'):
                dst_ok = True
    else:
        dst = cv2.imread(os.path.join(DATA_DIR, dst_name))

    print('Target image:', dst_name)
    # Match sizes
    adapted = np.zeros_like(dst)
    adapted_guide = np.zeros_like(dst)
    if masked.shape[0] > dst.shape[0]:
        masked = cv2.resize(masked, (dst.shape[0], dst.shape[0]*masked.shape[1]//masked.shape[0]))
        mask = cv2.resize(mask, (dst.shape[0], dst.shape[0]*masked.shape[1]//masked.shape[0]))

    if masked.shape[1] > dst.shape[1]:
        masked = cv2.resize(masked, (dst.shape[1]*masked.shape[0]//masked.shape[1], dst.shape[1]))
        mask = cv2.resize(mask, (dst.shape[1]*masked.shape[0]//masked.shape[1], dst.shape[1]))

    y, x = random.randint(0, dst.shape[0] - masked.shape[0]), random.randint(0, dst.shape[1] - masked.shape[1])
    adapted[y:y+masked.shape[0], x:x+masked.shape[1]] = masked
    adapted_guide[y:y+masked.shape[0], x:x+masked.shape[1]] = mask

    cv2.destroyAllWindows()
    cv2.imshow('Final match - Dst', dst)
    cv2.imshow('Final match - Src', adapted)
    print('Source object and destination iamge. Press s if you want to resize the object...')
    k = cv2.waitKey(0)

    if k == ord('s'):
        sf = float(input('Desired scale factor on obj (0 - 1, pls): '))
        adapted = cv2.resize(adapted, tuple(np.int0(sf*np.array(adapted.shape[:2][::-1]))))
        adapted_guide = cv2.resize(adapted_guide, tuple(np.int0(sf*np.array(adapted_guide.shape[:2][::-1]))))

    print(dst.shape, adapted.shape, adapted_guide.shape)
    dst = overlay_rnd(dst, adapted, adapted_guide, 1)

    cv2.destroyAllWindows()
    cv2.imshow('Destiny', dst)
    print('Press s to save...')
    k = cv2.waitKey(0)

    if k == ord('s'):
        name = f'{src_name.split(".")[0]}_{idx}_to_{dst_name.split(".")[0]}.png'
        cv2.imwrite(os.path.join(OUTPUT_DIR, name), dst)



