"""Class for implementing a reusable model trainer with cross validation"""
import os

import torch
import torchvision as tv
import numpy as np

from utils import M3Net
class CrosValTrainer():
    def __init__(self, dataset, model, k=5):
        self.dataset = dataset
        self.model = model
        self.k = k

        # GPU
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def train(self, epochs=50, batch_size=32, optimizer_fun=torch.optim.Adam, criterion=torch.nn.CrossEntropyLoss()):
        test_accuracy_list = []

        # Split dataset
        l = len(self.dataset)
        split_lens =  [round(l/self.k) for _ in range(self.k - bool(l%self.k))]

        if l%self.k:
            split_lens += [l%self.k]

        data_splits = torch.utils.data.random_split(self.dataset, split_lens)

        for k_i in range(self.k):

            test_data = data_splits[k_i]
            train_data = torch.utils.data.ConcatDataset([d for i, d in enumerate(data_splits) if  i != k_i])

            train_generator = torch.utils.data.DataLoader(train_data, batch_size=batch_size, 
                shuffle=True, drop_last=True)
            test_generator = torch.utils.data.DataLoader(test_data, batch_size=batch_size, 
                shuffle=False, drop_last=False)

            # Init optimizer 
            optimizer = optimizer_fun(self.model.parameters())

            for epoch in range(epochs):  # loop over the dataset multiple times

                train_correct = 0
                train_loss = 0

                for i, data in enumerate(train_generator, 0):
                    # get the inputs; data is a list of [inputs, labels]
                    inputs, labels = data[0].to(self.device), data[1].to(self.device)

                    # zero the parameter gradients
                    optimizer.zero_grad()

                    # forward + backward + optimize
                    outputs = self.model(inputs) # forward pass
                    loss = criterion(outputs, labels) # calculate loss
                    loss.backward() # backpropagate loss
                    optimizer.step() # update weights/parameters

                    # Compute training accuracy
                    train_correct += (torch.max(outputs, dim=1).indices == labels).float().sum()
                    train_loss += loss.item()

                train_accuracy = 100 * train_correct / len(train_data)
                print(f'CrossValSplit {k_i+1}/{self.k} - Epoch {epoch+1}/{epochs}. Training accuracy = {train_accuracy}. Training loss = {train_loss/len(train_data)}')
            print()
            # Test
            self.model.eval() # disabled layers such as dropout or batchnorm (not used during inference)
            test_correct = 0

            with torch.no_grad():
                for data in test_generator:
                    inputs, labels = data[0].to(self.device), data[1].to(self.device)
                    outputs = self.model(inputs)
                    test_correct += (torch.max(outputs, dim=1).indices == labels).float().sum()

            test_accuracy = 100 * test_correct / len(test_data)
            print(f'CrossValSplit {k_i} - Accuracy on test: {test_accuracy}\n')
            test_accuracy_list.append(test_accuracy.item())

            self.model.train()

        print('Final cross validation accuracy:', np.mean(test_accuracy_list))



MODEL_PATH = './models/test-crosval.pth'
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
EPOCHS = 3
MOMENTUM = 0.99

CLASSES = ('coast', 'forest', 'highway', 'inside_city', 'mountain', 'opencountry', 'street', 'tallbuilding')

DATASET_DIR = '/home/capiguri/code/uab_cv_master/m3/Databases/MIT_split'

transfom_list = [tv.transforms.ToTensor(), tv.transforms.Resize((64, 64))]

train_val_dataset = tv.datasets.ImageFolder(os.path.join(DATASET_DIR, 'train'), transform=tv.transforms.Compose(transfom_list))

model = M3Net()

validator = CrosValTrainer(train_val_dataset, model, 3)

validator.train(10)
