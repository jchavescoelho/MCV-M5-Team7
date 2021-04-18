
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
from detectron2.evaluation import COCOEvaluator, inference_on_dataset, SemSegEvaluator
from detectron2.data import build_detection_test_loader

import parse_ds as ds

OUTPUT_DIR = '/home/group07/code/MCV-M5-Team7/week4/experiments'

MOTS_PATH = '/home/mcv/datasets/MOTSChallenge/train/images/'
KITTI_MOTS_PATH = '/home/mcv/datasets/KITTI-MOTS/training/image_02/'
PKLS_PATH = './pkls_big/'

#Fine tuning config
learn_rates = [0.001]
models = ["COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"]#, "Cityscapes/mask_rcnn_R_50_FPN.yaml"]
datasets = ['kitti-mots']#, 'mots']
batchs = [512]

# Load/Register datasets
ds_name = 'kitti-mots'
kittimots_train_dicts, kittimots_val_dicts = ds.get_mots_dicts(KITTI_MOTS_PATH, ds_name)

print('Registering...')
DatasetCatalog.register(ds_name+'_train', lambda : kittimots_val_dicts)
MetadataCatalog.get(ds_name+'_train').set(thing_classes=['pedestrian', 'ignore', 'car'])


DatasetCatalog.register(ds_name+'_val', lambda : kittimots_val_dicts)
MetadataCatalog.get(ds_name+'_val').set(thing_classes=['pedestrian', 'ignore', 'car'])


ds_name = 'mots'
mots_train_dicts, mots_val_dicts = ds.get_mots_dicts(MOTS_PATH, ds_name)


allmots_train_dicts = kittimots_train_dicts + mots_train_dicts
allmots_val_dicts = kittimots_val_dicts + mots_val_dicts

print('Registering...')
DatasetCatalog.register(ds_name+'_train', lambda : allmots_val_dicts)
MetadataCatalog.get(ds_name+'_train').set(thing_classes=['pedestrian', 'ignore', 'car'])

DatasetCatalog.register(ds_name+'_val', lambda : allmots_val_dicts)
MetadataCatalog.get(ds_name+'_val').set(thing_classes=['pedestrian', 'ignore', 'car'])


for lr in learn_rates:
    for model in models:
        for dts in datasets:
            for batch in batchs:

                model_name = model.split('/')[-1][:-5]
                lr_str = str(lr)#.replace('.', '-')
                experiment_name = f'{dts}_{model_name}_lr{lr_str}_batch{batch}'

                print(f'Now testing: {experiment_name}')

                cfg = get_cfg()
                cfg.merge_from_file(model_zoo.get_config_file(model))
                cfg.DATASETS.TRAIN = ()
                cfg.DATASETS.TEST = ('kitti-mots_val',)
                cfg.DATALOADER.NUM_WORKERS = 4
                cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(model)  # Let training initialize from model zoo
                cfg.SOLVER.IMS_PER_BATCH = 2
                cfg.SOLVER.BASE_LR = lr      # pick a good LR
                cfg.SOLVER.MAX_ITER = 300    # 300 iterations seems good enough for this toy dataset; you will need to train longer for a practical dataset
                cfg.SOLVER.STEPS = []        # do not decay learning rate
                cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = batch   # faster, and good enough for this toy dataset (default: 512)
                cfg.MODEL.ROI_HEADS.NUM_CLASSES = 3  # only has one class (ballon). (see https://detectron2.readthedocs.io/tutorials/datasets.html#update-the-config-for-new-datasets)
                # NOTE: this config means the number of classes, but a few popular unofficial tutorials incorrect uses num_classes+1 here.

                cfg.OUTPUT_DIR = os.path.join(OUTPUT_DIR, experiment_name)

                # Inference should use the config with parameters that are used in training
                # cfg now already contains everything we've set previously. We changed it a little bit for inference:
                print('Evaluating...')
                print(cfg.OUTPUT_DIR)
                cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")  # path to the model we just trained
                cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5   # set a custom testing threshold
                predictor = DefaultPredictor(cfg)

                ds_val = 'kitti-mots_val'
                eval_name = f'{ds_val}_{model_name}_{dts}_lr{lr}_batch{batch}'

                os.makedirs(f'./outputs_eval/{eval_name}', exist_ok=True)
                evaluator = COCOEvaluator(ds_val, ("segm",), False, output_dir=f'./outputs_eval/{eval_name}')
#                evaluator = COCOEvaluator(ds_val, cfg, False, output_dir=f'./outputs_eval/{eval_name}')

                ds_val = 'kitti-mots_val'
                eval_name = f'{ds_val}_{model_name}_{dts}_lr{lr}_batch{batch}'
                os.makedirs( os.path.join(f'./outputs_eval/{eval_name}', 'images'), exist_ok=True)
                print('Running some random inferences...')

                imgsout = [v for i, v in enumerate(val_dicts) if i in [20, 123, 169, 270, 237, 305]]


                for d in imgsout: # random.sample(kittimots_val_dicts, 5):
                    file_name = d['file_name']
                    im = cv2.imread(file_name)
                    outputs = predictor(im)

                    inst = outputs["instances"].to('cpu')
                    inst = inst[[True if c == 0 or c == 2 else False for c in inst.pred_classes]]
                    # instances = outputs["instances"][outputs["instances"].scores > 0.5]
                    v = Visualizer(im[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TEST[0]), scale=1)
                    out = v.draw_instance_predictions(inst.to("cpu"))
                    name = os.path.split(d['file_name'])[-1]
                    saveto = os.path.join(f'./outputs_eval/{eval_name}', 'images', 'infer_' + name)
                    print(saveto)
                    cv2.imwrite(saveto, out.get_image()[:, :, ::-1])


                print('ds VAL', ds_val)
                val_loader = build_detection_test_loader(cfg, ds_val)
             
                predictor = DefaultPredictor(cfg) 
                results = inference_on_dataset(predictor.model, val_loader, evaluator)
                # another equivalent way to evaluate the model is to use `trainer.test`   
                print(os.path.join(f'./outputs_eval/{eval_name}', 'results.json'))
                with open(os.path.join(f'./outputs_eval/{eval_name}', 'results.json'), 'w') as fp:
                    json.dump(results, fp, indent=4)

