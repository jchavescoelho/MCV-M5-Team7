import os
import time
import glob
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

# Globals
with open('coco_classes.pkl', 'rb') as f:
    CLASSES = pkl.load(f)

# print(CLASSES)
# quit()

# DATA_DIR is a folder with a subfolder inside for each class. 
# Even if you have no class labels or whatever just create a subfolder with all the images. The structure would look like:

# <DATA_DIR> -|
# ------------|- all
#-------------------| <img0>.jpg
#-------------------| <img1>.jpg
#-------------------| <img2>.jpg

DATA_NAME = 'taskD2' # images will be saved as <DATA_NAME>_<num>.png
MAX_NUM_IMG = 100
DATA_DIR = '/home/capiguri/code/datasets/COCO/test2017/'
DATA_DIR = './taskD2/'
# DATA_DIR = './img2infer/'
OUTPUT_DIR = './results'

MODELS = {
    'faster': fasterrcnn_resnet50_fpn(True),
    'mask': maskrcnn_resnet50_fpn(True)
}

def get_optimal_font_scale(text, width):
    for scale in reversed(range(0, 60, 1)):
        textSize = cv2.getTextSize(text, fontFace=cv2.FONT_HERSHEY_COMPLEX_SMALL, fontScale=scale/10, thickness=1)
        new_width = textSize[0][0]
        if (new_width <= width):
            return scale/10
    return 1

def get_random_col():
    h,s,l = random.random(), 0.5 + random.random()/2.0, 0.4 + random.random()/5.0
    r,g,b = [int(256*i) for i in colorsys.hls_to_rgb(h,l,s)]
    return (b, g, r)

def overlay(img_ol, mask_ol, alpha=0.6):
    pts = np.where((mask_ol != [0]))
    img_ol[pts] = (img_ol[pts]*(1 - alpha) + mask_ol[pts]*alpha).astype(np.uint8)
    return img_ol

def paint_detections(im, det, score_thresh=0.8, mask_thresh=0.5):
    colors = [get_random_col() for _ in range(len(det['scores']))]

    if 'masks' in det:
        print('Drawing masks...')

        out = np.zeros_like(im)
        m = 0

        for inst, score, color in zip(det['masks'], det['scores'], colors):
            if score < score_thresh:
                continue

            inst = (inst > 0.25).float()

            inst = inst.cpu().numpy()
            inst = np.transpose(inst, (1, 2, 0))
            inst *= 255
            inst = cv2.cvtColor(inst, cv2.COLOR_GRAY2BGR)

            inst[np.where((inst==[255,255,255]).all(axis=2))] = color
            im = overlay(im, inst, 0.5)
            m += 1

    if 'boxes' in det:
        print('Drawing boxes...')

        cont = 0
        for bbox, lab, score, color in zip(det['boxes'], det['labels'], det['scores'], colors):

            x1, y1, x2, y2 = np.int0(bbox.cpu().numpy())
            w = x2 - x1
            h = y2 - y1
            category = CLASSES[int(lab.cpu().numpy())]
            confidence = score.cpu().numpy()

            if confidence > score_thresh:
                print(f'id: {cont} - class: {category} - conf: {confidence}' )
                cv2.rectangle(im, (x1, y1), (x2, y2), color)

                text = f'{cont} - {category} - {int(100*confidence)} %'
                # "Background" for label
                im = cv2.rectangle(im, 
                    (x1, y1),
                    (int(x1+0.7*w), int(y1+0.2*h)),
                    color, -1)
                # Print name, id and conf
                cv2.putText(im, text, (x1, int(y1+0.15*h)), cv2.FONT_HERSHEY_COMPLEX_SMALL, get_optimal_font_scale(text, 0.7*w), (0,0,0))
            cont += 1

    return im

def inout_grid(im, out, savepath):
    f, axs = plt.subplots(1, 2)
    
    for img, ax in zip([(im, 'Input'), (out, 'Output')], axs):
        ax.axes.xaxis.set_visible(False)
        ax.axes.yaxis.set_visible(False)
        ax.set_title(img[1], fontsize=5)
        # img = cv2.cvtColor(i[0], cv2.COLOR_BGR2RGB)
        ax.imshow(img[0])
    
    plt.savefig(savepath, bbox_inches='tight',dpi=300)
    plt.show()


def main(args):
    os.makedirs(os.path.join(OUTPUT_DIR, DATA_NAME), exist_ok=True)

    # ds = torchvision.datasets.ImageFolder(DATA_DIR,
    #     transform=torchvision.transforms.Compose([torchvision.transforms.ToTensor()]))

    # generator = torch.utils.data.DataLoader(ds, 1, shuffle=args.shuffle)

    # model = fasterrcnn_resnet50_fpn(pretrained=True)
    model = MODELS[args.model]

    # GPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    c = 0
    with torch.no_grad():# GPU
        # for im, lab in generator:
        files = glob.glob(DATA_DIR + '/*/*.*')
        if args.shuffle:
            random.shuffle(files)

        for impath in files:
            print('impath', impath)
            im = cv2.imread(impath)
            im = torchvision.transforms.ToTensor()(im)
            im = im.unsqueeze(0)
            # Predict
            im = im.to(device)
            print('Inferring...')
            t0 = time.time()
            det = model(im)
            print('Inference time: ', time.time() - t0)
            
            # print('keys', det[0].keys())
            # labs = [CLASSES[int(lab.cpu().numpy())] for lab in det[0]['labels']]
            # print(labs, 'person' in labs)
            
            # if 'potted plant' not in labs or 'cat' not in labs:
            #     continue

            # Display
            disp = np.array(torchvision.transforms.ToPILImage()(im.squeeze(0)), dtype=np.uint8)
            out = paint_detections(disp.copy(), det[0], 0.8)
            inout_grid(disp, out, os.path.join(OUTPUT_DIR, DATA_NAME, f'{args.model}_{c}.png'))

            c += 1
            if c == MAX_NUM_IMG:
                quit()

            print('='*16)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('-m', '--model', type=str, required=False,
        default='faster', choices=['faster', 'mask'], help='Which model to run: faster or mask rcnn')
    parser.add_argument('-s', '--shuffle', action='store_true', help='Whether to shuffle the dataset or not')
    args = parser.parse_args()

    main(args)