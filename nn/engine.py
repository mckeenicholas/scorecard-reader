import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import resnet18, ResNet18_Weights
from torch import Tensor

def train(model, data_loader, optimizer):
    