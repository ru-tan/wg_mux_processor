# This code worked with the following environment:
# OS: Windows 11, 64 bit
# Python version: 3.9.13 (anaconda)
# PyTorch version: 1.12.1
# Numpy version: 1.23.1

# fashion-mnist_test.csv can be downloaded from: https://www.kaggle.com/datasets/zalando-research/fashionmnist?select=fashion-mnist_test.csv
# fashion-mnist_train.csv can be downloaded from: https://www.kaggle.com/datasets/zalando-research/fashionmnist?select=fashion-mnist_train.csv

import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import confusion_matrix

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# print(device)

train_csv = pd.read_csv("fashion-mnist_train.csv")
test_csv = pd.read_csv("fashion-mnist_test.csv")


class FashionDataset(Dataset):
    """User defined class to build a datset using Pytorch class Dataset."""

    def __init__(self, data, transform=None):
        """Method to initilaize variables."""
        self.fashion_MNIST = list(data.values)
        self.transform = transform

        label = []
        image = []

        for i in self.fashion_MNIST:
            # first column is of labels.
            label.append(i[0])
            image.append(i[1:])
        self.labels = np.asarray(label)
        # Dimension of Images = 28 * 28 * 1. where height = width = 28 and color_channels = 1.
        self.images = np.asarray(image).reshape(-1, 28, 28, 1).astype('float32')

    def __getitem__(self, index):
        label = self.labels[index]
        image = self.images[index]

        if self.transform is not None:
            image = self.transform(image)

        return image, label

    def __len__(self):
        return len(self.images)


def output_label(label):
    output_mapping = {
                 0: "T-shirt/Top",
                 1: "Trouser",
                 2: "Pullover",
                 3: "Dress",
                 4: "Coat",
                 5: "Sandal",
                 6: "Shirt",
                 7: "Sneaker",
                 8: "Bag",
                 9: "Ankle Boot"
                 }
    input = (label.item() if type(label) == torch.Tensor else label)
    return output_mapping[input]


class FashionCNN(nn.Module):

    def __init__(self):
        super(FashionCNN, self).__init__()

        l1_out_ch = 30
        l1_kernel_size = 3
        l1_padding = 1
        l1_maxpool_size = 2
        l1_stride = 1

        l2_out_ch = 60
        l2_kernel_size = 3
        l2_padding = 0
        l2_maxpool_size = 2
        l2_stride = 1

        # after max-pool layer
        l1_out_size = ((28-l1_kernel_size+2*l1_padding)/l1_stride+1)/l1_maxpool_size
        if int(l1_out_size)-l1_out_size != 0:
            print('layer 1 output size not integer')
            os.system('pause')
        else:
            l1_out_size = int(l1_out_size)
        # after max-pool layer
        l2_out_size = ((l1_out_size-l2_kernel_size+2*l2_padding)/l2_stride+1)/l2_maxpool_size
        if int(l2_out_size)-l2_out_size != 0:
            print('layer 2 output size not integer')
            os.system('pause')
        else:
            l2_out_size = int(l2_out_size)

        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=l1_out_ch, kernel_size=l1_kernel_size, stride=l1_stride, padding=l1_padding),
            nn.BatchNorm2d(l1_out_ch),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=l1_maxpool_size, stride=l1_maxpool_size)
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=l1_out_ch, out_channels=l2_out_ch, kernel_size=l2_kernel_size, stride=l2_stride, padding=l2_padding),
            nn.BatchNorm2d(l2_out_ch),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=l2_maxpool_size, stride=l2_maxpool_size)
        )
        # print(l1_out_ch * l1_out_size * l1_out_size)

        self.fc1 = nn.Linear(in_features=l2_out_ch * l2_out_size * l2_out_size, out_features=500)
        self.drop = nn.Dropout2d(0.25)
        self.fc2 = nn.Linear(in_features=500, out_features=128)
        self.fc3 = nn.Linear(in_features=128, out_features=64)
        self.fc4 = nn.Linear(in_features=64, out_features=10)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        out = self.drop(out)
        out = F.relu(out)
        out = self.fc2(out)
        out = F.relu(out)
        out = self.fc3(out)
        out = F.relu(out)
        out = self.fc4(out)

        return out


# Transform data into Tensor that has a range from 0 to 1
train_set = FashionDataset(train_csv, transform=transforms.Compose([transforms.ToTensor()]))
test_set = FashionDataset(test_csv, transform=transforms.Compose([transforms.ToTensor()]))

train_loader = DataLoader(train_set, batch_size=100)
test_loader = DataLoader(test_set, batch_size=100)
# print(len(test_loader))
# os.system('pause')

model = FashionCNN()
model.to(device)

error = nn.CrossEntropyLoss()

learning_rate = 0.001
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
# print(model)

num_epochs = 10
count = 0
# Lists for visualization of loss and accuracy
loss_list = []
iteration_list = []
accuracy_list = []

# Lists for knowing classwise accuracy
predictions_list = []
labels_list = []

# load model
model.load_state_dict(torch.load('20240725_saved_model_best_1'))
# set to evaluation mode (very important)
model.eval()

# for name, param in model.named_parameters():
#     print(name, param)

total = 0
correct = 0

# test
confusion_mat = np.zeros((10, 10))
for images, labels in test_loader:
    images, labels = images.to(device), labels.to(device)
    labels_list.append(labels)

    test = Variable(images.view(100, 1, 28, 28))

    outputs = model(test)

    predictions = torch.max(outputs, 1)[1].to(device)
    # print(labels_list)
    # print(predictions)
    # os.system('pause')
    predictions_list.append(predictions)
    correct += (predictions == labels).sum()
    total += len(labels)

    # update confusion matrix
    for j in range(len(labels)):
        confusion_mat[labels[j].data, predictions[j].data] += 1

accuracy = correct * 100 / total
print(accuracy.data)
print(confusion_mat)
# print(total)

# save the confusion matrix into file
# np.save('confusion_matrix_using_computer.npy', confusion_mat)





