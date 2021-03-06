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

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        #m3-layers
        self.conv0 = nn.Conv2d(3, 64, 3) #dim in, dim out, size ker
        self.bn0 = nn.BatchNorm2d(64, eps=1e-03, momentum=0.99)
        self.pool0 = nn.MaxPool2d(2, 2)
        self.drop0 = nn.Dropout2d(0.2)

        #block1
        self.conv1_1 = nn.Conv2d(64, 64, 3) #dim in, dim out, size ker
        self.bn1_1 = nn.BatchNorm2d(64, eps=1e-03, momentum=0.99)
        self.drop1_1 = nn.Dropout2d(0.2)

        self.conv1_2 = nn.Conv2d(64, 64, 3)
        self.bn1_2 = nn.BatchNorm2d(64, eps=1e-03, momentum=0.99)
        self.drop1_2 = nn.Dropout2d(0.2)

        self.pool1 = nn.MaxPool2d(2, 2)
        self.drop1_p = nn.Dropout2d(0.2)

        #block2
        self.conv2_1 = nn.Conv2d(64, 64, 3) #dim in, dim out, size ker
        self.bn2_1 = nn.BatchNorm2d(64, eps=1e-03, momentum=0.99)
        self.drop2_1 = nn.Dropout2d(0.2)

        self.conv2_2 = nn.Conv2d(64, 64, 3)
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


BATCH_SIZE = 32
LEARNING_RATE = 1e-5
EPOCHS = 10
MOMENTUM = 0.99

CLASSES = ('coast', 'forest', 'highway', 'inside_city', 'mountain', 'opencountry', 'street', 'tallbuilding')


class custom_preprocess(object):
    def __init__(self):
        pass

    def __call__(self, im):
        return im


def main():

    # Load datasets
    tv.transforms.Normalize
    DATASET_DIR = '~/code/uab_cv_master/m3/Databases/MIT_split'

    transfom_list = [tv.transforms.ToTensor(), tv.transforms.Resize((64, 64))]
    train_val_dataset = tv.datasets.ImageFolder(os.path.join(DATASET_DIR, 'train'), transform=tv.transforms.Compose(transfom_list))
    train_dataset, val_dataset = torch.utils.data.random_split(train_val_dataset, [4*len(train_val_dataset)//5, len(train_val_dataset) - 4*len(train_val_dataset)//5])
    test_dataset = tv.datasets.ImageFolder(os.path.join(DATASET_DIR, 'test'), transform=tv.transforms.Compose(transfom_list))

    train_generator = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True, num_workers=4)
    val_generator = torch.utils.data.DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True, num_workers=4)
    test_generator = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=True, num_workers=4)

    # # Visualize dataset
    # train_it = iter(train_generator)
    # ims, labels = train_it.next()

    # print(len(ims), len(labels))
    # imshow(np.moveaxis(ims[0].numpy(), 0, -1)) # imshow(tv.transforms.ToPILImage()(ims[0]))
    # print(CLASSES[labels[0].item()])
    # plt.show()

    # Instanciate model
    model = Net()

    # GPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Loss
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM)

    t0 = time.time()
    # Train
    for epoch in range(EPOCHS):  # loop over the dataset multiple times

        train_correct = 0
        val_correct = 0
        for i, data in enumerate(train_generator, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data[0].to(device), data[1].to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs) # forward pass
            loss = criterion(outputs, labels) # calculate loss
            loss.backward() # backpropagate loss
            optimizer.step() # update weights/parameters

            # Compute training accuracy
            train_correct += (torch.max(outputs, dim=1).indices == labels).float().sum()

        # Compute val accuracy
        with torch.no_grad():
            for i, data in enumerate(val_generator, 0):
                inputs, labels = data[0].to(device), data[1].to(device)
                outputs = model(inputs) # forward pass

                val_correct += (torch.max(outputs, dim=1).indices == labels).float().sum()


        accuracy = 100 * train_correct / len(train_dataset)
        val_accuracy = 100 * val_correct / len(val_dataset)
        
        # print statistics
        print(f'Epoch {epoch}. Training accuracy = {accuracy}. Validation accuracy = {val_accuracy}')

    print(f'Finished Training. Device: {device}. Elapsed time {round(time.time() - t0, 2)} s')



if __name__ == '__main__':
    main()