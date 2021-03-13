import os
import time
import torch
import torchvision as tv
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
from matplotlib.pyplot import imshow
from torch.utils.tensorboard import SummaryWriter

from graphviz import Source
from torchviz import make_dot

class M3Net(nn.Module):
    def __init__(self):
        super(M3Net, self).__init__()
        #m3-layers
        self.conv0 = nn.Conv2d(3, 64, 3, padding=1) #dim in, dim out, size ker
        self.bn0 = nn.BatchNorm2d(64, eps=1e-03, momentum=0.99)
        self.pool0 = nn.MaxPool2d(2, 2)
        self.drop0 = nn.Dropout2d(0.2)

        #block1
        self.conv1_1 = nn.Conv2d(64, 64, 3, padding=1) #dim in, dim out, size ker
        self.bn1_1 = nn.BatchNorm2d(64, eps=1e-03, momentum=0.99)
        self.drop1_1 = nn.Dropout2d(0.2)

        self.conv1_2 = nn.Conv2d(64, 64, 3, padding=1)
        self.bn1_2 = nn.BatchNorm2d(64, eps=1e-03, momentum=0.99)
        self.drop1_2 = nn.Dropout2d(0.2)

        self.pool1 = nn.MaxPool2d(2, 2)
        self.drop1_p = nn.Dropout2d(0.2)

        #block2
        self.conv2_1 = nn.Conv2d(64, 64, 3, padding=1) #dim in, dim out, size ker
        self.bn2_1 = nn.BatchNorm2d(64, eps=1e-03, momentum=0.99)
        self.drop2_1 = nn.Dropout2d(0.2)

        self.conv2_2 = nn.Conv2d(64, 64, 3, padding=1)
        self.bn2_2 = nn.BatchNorm2d(64, eps=1e-03, momentum=0.99)
        self.drop2_2 = nn.Dropout2d(0.2)

        self.pool2 = nn.MaxPool2d(2, 2)
        self.drop2_p = nn.Dropout2d(0.2)

        self.fc = nn.Linear(64, 8)

        pass

    def forward(self, x):
        x = self.drop0(self.pool0(self.bn0(F.relu(self.conv0(x))))) #layer 0
        res = x

        #block1
        x = self.drop1_1(self.bn1_1(F.relu(self.conv1_1(x))))
        x = self.drop1_2(self.bn1_2(F.relu(self.conv1_2(x))))
        x += res
        x = self.drop1_p(self.pool1(x))

        res = x

        #block2
        x = self.drop2_1(self.bn2_1(F.relu(self.conv2_1(x))))
        x = self.drop2_2(self.bn2_2(F.relu(self.conv2_2(x))))
        x += res
        x = self.drop2_p(self.pool2(x))

        x = F.avg_pool2d(x, x.shape[-2:])
        x = x.view(-1, 64)
        x = F.softmax(self.fc(x))

        return x

def confusion_matrix(model, loader, classes, device='cpu'):
    # model.to(device)

    with torch.no_grad():
        all_preds = torch.tensor([]).to(device)
        all_labs = torch.tensor([]).to(device)

        for data in loader:
            inputs, labels = data[0].to(device), data[1].to(device)

            preds = model(inputs)
            all_preds = torch.cat(
                (all_preds, preds)
                ,dim=0
            )
            all_labs = torch.cat((all_labs, labels), dim=0)

    print('LEN', all_preds.shape, all_labs.shape)

    stacked = torch.stack(
    (
        all_labs
        ,all_preds.argmax(dim=1)
    )
    ,dim=1
    ).to('cpu')

    cmt = torch.zeros(len(classes),len(classes), dtype=torch.int64)

    for p in stacked:
        tl, pl = p.tolist()
        cmt[int(tl), int(pl)] += 1
    

    cmt = cmt.numpy()
    print(cmt)

    for i in range(cmt.shape[0]):
        print(np.sum(cmt[i,:]))
        cmt[i, :] = 100*cmt[i, :] / np.sum(cmt[i,:])
    
    print(cmt)