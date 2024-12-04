import torch
from torchvision import datasets, transforms
from torch import nn, optim
from torchvision import models

def classification_layer(input_size, hidden_units):
    classifier = nn.Sequential(nn.Linear(input_size, hidden_units),
                                      nn.ReLU(),
                                      nn.Dropout(0.2),
                                      nn.Linear(hidden_units, 102),
                                      nn.LogSoftmax(dim=1))
    return classifier
