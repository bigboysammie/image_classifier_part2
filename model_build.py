import torch
from torchvision import datasets, transforms
from torch import nn, optim
from torchvision import models
from classification_layer import classification_layer

def model_build( arch, hidden_units):

#Load correct model based on user input 
    if arch == "vgg":
        model = models.vgg16(pretrained=True)

    elif arch == "resnet":
        model = models.resnet50(pretrained=True)

    elif arch == "densenet":
        model = models.densenet121(pretrained=True)

    else:
        print("Architecture not supported")

# Freeze parameter in pre-trained model
    for parameter in model.parameters():
        parameter.requires_grad = False

#Check for model type and modify classification layer according to build new model
    if hasattr(model, 'fc'):
    # For models like ResNet
        input_size = model.fc.in_features
        model.fc = classification_layer(input_size, hidden_units)

    elif hasattr(model, 'classifier'):
        # For models like VGG and DenseNet
        input_size = model.classier[0].in_features
        model.classifier = classification_layer(input_size, hidden_units)
    
    return model, input_size
