#Data augmentation

import PIL
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms



def dataset_aug(image_path,img_width,img_height,hue,saturation,rotate_angle):
    transforms = torchvision.transforms.Compose([
    torchvision.transforms.Resize((img_width,img_height)),
    torchvision.transforms.ColorJitter(hue=hue, saturation=saturation), 
    torchvision.transforms.RandomHorizontalFlip(),
    torchvision.transforms.RandomVerticalFlip(),
    torchvision.transforms.RandomRotation(rotate_angle, resample=PIL.Image.BILINEAR),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),])
    #hue=.05, saturation=.05

    dataset = torchvision.datasets.ImageFolder(image_path,transform=transforms)
    return dataset



    

            

            

    

