# This code worked with the following environment:
# OS: Windows 11, 64 bit
# Python version: 3.9.13 (anaconda)
# PyTorch version: 1.12.1
# Numpy version: 1.23.1
# scikit-learn: 1.0.2
# pandas: 1.5.0
# matplotlib: 3.6.1

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class Model(nn.Module):
    def __init__(self, input_features=4, hidden_layer1=4, output_features=3):
        super().__init__()
        self.fc1 = nn.Linear(input_features, hidden_layer1)
        self.out = nn.Linear(hidden_layer1, output_features)

    def forward(self, x):
        x = F.sigmoid(self.fc1(x))
        x = self.out(x)
        return x


dataset_train = pd.read_csv('Iris_normalized_train_202409.csv')
dataset_test = pd.read_csv('Iris_normalized_test_202409.csv')
dataset_train.columns = ["sepal length", "sepal width", "petal length", "petal width", "species"]
dataset_test.columns = ["sepal length", "sepal width", "petal length", "petal width", "species"]
# print(dataset.head())

mappings = {"Iris-setosa": 0, "Iris-versicolor": 1, "Iris-virginica": 2}
dataset_train["species"] = dataset_train["species"].apply(lambda x: mappings[x])
dataset_test["species"] = dataset_test["species"].apply(lambda x: mappings[x])
# print(dataset.head())

X_train = dataset_train.drop("species", axis=1).values
X_test = dataset_test.drop("species", axis=1).values
y_train = dataset_train["species"].values
y_test = dataset_test["species"].values

# print(X_train[0])
# print(y_test)
# os.system('pause')

X_train = torch.FloatTensor(X_train)
X_test = torch.FloatTensor(X_test)
y_train = torch.LongTensor(y_train)
y_test = torch.LongTensor(y_test)

model = Model()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.03)

epochs = 200
losses = []

for i in range(epochs):
    y_pred = model.forward(X_train)
    loss = criterion(y_pred, y_train)
    losses.append(loss.item())
    print(f'epoch: {i:2}  loss: {loss.item():10.8f}')
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    for name, p in model.named_parameters():
        if 'weight' in name:
            p.data.clamp_(0.6, 4)
            # weights will be normalized in the experiments

# plt.plot(range(epochs), losses)
# plt.ylabel('Loss')
# plt.xlabel('epoch')
# plt.show()

preds = []
with torch.no_grad():
    for val in X_test:
        y_hat = model.forward(val)
        preds.append(y_hat.argmax().item())

df = pd.DataFrame({'Y': y_test, 'YHat': preds})
df['Correct'] = [1 if corr == pred else 0 for corr, pred in zip(df['Y'], df['YHat'])]
print(df)

# confusion matrix


print(df['Correct'].sum() / len(df))
print('Parameters:')
for name, param in model.named_parameters():
    if param.requires_grad:
        print([name, param.data])