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

import math
from typing import Any

from torch import Tensor
from torch.nn.parameter import Parameter, UninitializedParameter
from torch.nn import init, Module
# from torch.nn import Module
from torch.nn.modules.lazy import LazyModuleMixin


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device_gpu = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
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


# custom Linear module
class myLinear(Module):
    r"""Applies a linear transformation to the incoming data: :math:`y = xA^T + b`

    This module supports :ref:`TensorFloat32<tf32_on_ampere>`.

    On certain ROCm devices, when using float16 inputs this module will use :ref:`different precision<fp16_on_mi200>` for backward.

    Args:
        in_features: size of each input sample
        out_features: size of each output sample
        bias: If set to ``False``, the layer will not learn an additive bias.
            Default: ``True``

    Shape:
        - Input: :math:`(*, H_{in})` where :math:`*` means any number of
          dimensions including none and :math:`H_{in} = \text{in\_features}`.
        - Output: :math:`(*, H_{out})` where all but the last dimension
          are the same shape as the input and :math:`H_{out} = \text{out\_features}`.

    Attributes:
        weight: the learnable weights of the module of shape
            :math:`(\text{out\_features}, \text{in\_features})`. The values are
            initialized from :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})`, where
            :math:`k = \frac{1}{\text{in\_features}}`
        bias:   the learnable bias of the module of shape :math:`(\text{out\_features})`.
                If :attr:`bias` is ``True``, the values are initialized from
                :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})` where
                :math:`k = \frac{1}{\text{in\_features}}`

    Examples::

        # >>> m = nn.Linear(20, 30)
        # >>> input = torch.randn(128, 20)
        # >>> output = m(input)
        # >>> print(output.size())
        torch.Size([128, 30])
    """
    __constants__ = ['in_features', 'out_features']
    in_features: int
    out_features: int
    weight: Tensor

    def __init__(self, in_features: int, out_features: int, bias: bool = True,
                 device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(myLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.empty((out_features, in_features), **factory_kwargs))
        if bias:
            self.bias = Parameter(torch.empty(out_features, **factory_kwargs))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

        # added parameters (Rui Tang)
        self.is_weight_converted = False
        self.t_min_mzi = 1.0e-3
        self.bit = 8
        self.t_mzi = torch.linspace(self.t_min_mzi, 1.0, 2**self.bit, device=device_gpu)
        self.m_one = torch.ones(self.weight.shape, device=device_gpu)
        self.m_one = self.m_one.T
        self.out_matrix_element_max = in_features/1.0
        self.adc_bit = 8
        self.out_lut = torch.linspace(0, self.out_matrix_element_max, 2**self.adc_bit, device=device_gpu)

        # for normalize weight matrix (Rui Tang): current weight matrix include negative value
        self.max_temp = 1
        self.wei_new = torch.ones(self.weight.shape)
        self.max_temp2 = 1

        # self.out_matrix_element_max = 0
        # self.out_matrix_element_min = 100000

        # print(self.wei_new)

        # if any value in the converted matrix is less than the minimum transmittance of MZI,
        # change the value (turns out to be not necessary, because the search_multi_mzis function will automatically perform this)
        # self.wei_new[self.wei_new<self.t_min_mzi] = self.t_min_mzi
        # print(self.wei_new)

        # replace each value with XX-bit approximation
        # print('wei_new')

        # self.wei_new = self.search_multi_mzis(self.wei_new)

        # print(self.wei_new)
        #
        # os.system('pause')

        # print(wei_new.min())

        ##### continue here########
        # print(self.weight)
        # print(self.wei_new)
        # os.system('pause')

    def reset_parameters(self) -> None:
        # Setting a=sqrt(5) in kaiming_uniform is the same as initializing with
        # uniform(-1/sqrt(in_features), 1/sqrt(in_features)). For details, see
        # https://github.com/pytorch/pytorch/issues/57109
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            init.uniform_(self.bias, -bound, bound)


    def search_multi_mzis(self, values):
        # Find nearest neighbors in a torch array(self.t_mzi) for multiple values, self.t_mzi is a 1D array that stores the
        # values of MZI transmittances, values is a 2D array that stores the batched input vectors, one row is one input vector
        nearest_indices = torch.abs(self.t_mzi[:, None, None] - values).argmin(axis=0)
        nearest_values = self.t_mzi[nearest_indices]
        return nearest_values


    def search_out_values(self, values):
        nearest_indices = torch.abs(self.out_lut[:, None, None] - values).argmin(axis=0)
        nearest_values = self.out_lut[nearest_indices]
        return nearest_values


    # def forward(self, input: Tensor) -> Tensor:
    #     # print(input.shape)
    #     # print(self.weight.shape)
    #     # print(self.bias.shape)
    #
    #     # original
    #     # y = F.linear(input, self.weight, self.bias)
    #
    #     # modified (returns the same result as the original version)
    #     y = torch.matmul(input, self.weight.T) + self.bias
    #     return y

    def forward(self, input: Tensor) -> Tensor:

        # # check if the weight matrix has been converted, this cannot be done in the initialization
        if not self.is_weight_converted:
            self.is_weight_converted = True
            # convert the weight matrix
            self.max_temp = torch.abs(self.weight).max()
            self.wei_new = (self.weight / self.max_temp + torch.ones(self.weight.shape, device=device_gpu)) * 0.5
            self.max_temp2 = self.wei_new.max()
            if self.max_temp2 < 1:
                self.wei_new = self.wei_new / self.max_temp2
            # replace each element with XX-bit approximation
            self.wei_new = self.search_multi_mzis(self.wei_new)
            # check if in GPU
            if not self.wei_new.is_cuda:
                self.wei_new = self.wei_new.to(device_gpu)

        # normalize input
        input_max_temp, max_indices = input.max(dim=1)
        input_max_temp[input_max_temp==0] = 1
        input_new = input/input_max_temp[:, None]
        # replace each element with XX-bit approximation
        input_new = self.search_multi_mzis(input_new)

        mat_temp = torch.matmul(input_new, self.wei_new.T)
        # add Gaussian noise, std is the standard deviation
        std = 0.015
        noise = torch.normal(1, std, size=mat_temp.shape, device=device_gpu)
        mat_temp = mat_temp * noise
        # search closest values (quantization)
        mat_temp = self.search_out_values(mat_temp)

        # print(mat_temp)
        # os.system('pause')
        mat_temp = mat_temp * self.max_temp2


        noise2 = torch.normal(1, std, size=mat_temp.shape, device=device_gpu)
        mat_temp2 = torch.matmul(input_new, self.m_one)
        # add Gaussian noise
        mat_temp2 = mat_temp2 * noise2
        # search closest values (quantization)
        mat_temp2 = self.search_out_values(mat_temp2)

        out_temp = 2*mat_temp-mat_temp2
        out = out_temp*input_max_temp[:, None]*self.max_temp
        out = out + self.bias
        return out

    def extra_repr(self) -> str:
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )


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
        # self.fc3 = nn.Linear(in_features=128, out_features=64)
        # self.fc4 = nn.Linear(in_features=64, out_features=10)

        # custom Linear layer
        self.fc3 = myLinear(in_features=128, out_features=64)
        self.fc4 = myLinear(in_features=64, out_features=10)

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

# import saved model parameters
state_dict = torch.load('20240725_saved_model_best_1')
# shape = state_dict['fc3.weight'].shape
# state_dict['fc3.weight'] = torch.zeros(shape)

# print(state_dict)
# os.system('pause')

# load modified model
model.load_state_dict(state_dict)

# set to evaluation mode (very important)
model.eval()

# modified_param_names = ['fc3.weight', 'fc4.weight']
# shape = model.state_dict()['fc3.weight'].shape
# model.state_dict()['fc3.weight'] = torch.zeros(shape)
# print(model.state_dict()['fc3.weight'])

# print(model.state_dict()['fc3.weight'])
# print(model.named_parameters())
# for name, param in model.named_parameters():
#     print(name)

# os.system('pause')

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

# save the confusion matrix into file
# np.save('confusion_matrix_using_PIC.npy', confusion_mat)

# print(total)





