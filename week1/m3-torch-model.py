import torch
import torchvision
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        #m3-layers
        self.conv0 = nn.Conv2D(3, 64, 3) #dim in, dim out, size ker
        self.bn0 = nn.BatchNorm2d(64, eps=1e-03, momentum=0.99)
        self.pool0 = nn.MaxPool2d(2, 2)
        self.drop0 = nn.Dropout2d(0.2)

        #block1
        self.conv1_1 = nn.Conv2D(3, 64, 3) #dim in, dim out, size ker
        self.bn1_1 = nn.BatchNorm2d(64, eps=1e-03, momentum=0.99)
        self.drop1_1 = nn.Dropout2d(0.2)

        self.conv1_2 = nn.Conv2D(64, 64, 3)
        self.bn1_2 = nn.BatchNorm2d(64, eps=1e-03, momentum=0.99)
        self.drop1_2 = nn.Dropout2d(0.2)

        self.pool1 = nn.MaxPool2d(2, 2)
        self.drop1_p = nn.Dropout2d(0.2)

        #block2
        self.conv2_1 = nn.Conv2D(3, 64, 3) #dim in, dim out, size ker
        self.bn2_1 = nn.BatchNorm2d(64, eps=1e-03, momentum=0.99)
        self.drop2_1 = nn.Dropout2d(0.2)

        self.conv2_2 = nn.Conv2D(64, 64, 3)
        self.bn2_2 = nn.BatchNorm2d(64, eps=1e-03, momentum=0.99)
        self.drop2_2 = nn.Dropout2d(0.2)

        self.pool2 = nn.MaxPool2d(2, 2)
        self.drop2_p = nn.Dropout2d(0.2)

        self.avgPool = nn.AvgPool2d()
        self.fc = nn.Linear(64, 8)

        pass

    def forward(self, x):
        x = self.drop0(self.pool0(self.bn0(F.relu(self.conv0(x))))) #layer 0

        #block1
        x = self.drop1_1(self.bn1_1(F.relu(self.conv1_1(x))))
        x = self.drop1_2(self.bn1_2(F.relu(self.conv1_2(x))))
        x = self.drop1_p(self.pool1(x))

        #block2
        x = self.drop2_1(self.bn2_1(F.relu(self.conv2_1(x))))
        x = self.drop2_2(self.bn2_2(F.relu(self.conv2_2(x))))
        x = self.drop2_p(self.pool2(x))

        x = F.softmax(self.fc(self.avgPool(x)))

        return x

#TODO:RESIDUAL