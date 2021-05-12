import time
import copy
import sys
import random
from collections import OrderedDict
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import numpy as np
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

# simple ConvNet used in the DLG paper, adapted for CIFAR10 instead of CIFAR100
class DLGNet(nn.Module):
    def __init__(self):
        super(DLGNet, self).__init__()
        act = nn.Sigmoid
        self.body = nn.Sequential(
            nn.Conv2d(3, 12, kernel_size=5, padding=5//2, stride=2),
            act(),
            nn.Conv2d(12, 12, kernel_size=5, padding=5//2, stride=2),
            act(),
            nn.Conv2d(12, 12, kernel_size=5, padding=5//2, stride=1),
            act(),
        )
        self.fc = nn.Sequential(
            nn.Linear(768, 10)
        )
        
    def forward(self, x):
        out = self.body(x)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out

# initialize device with larger weights than pytorch default
def weights_init(model, minval, maxval):
    if hasattr(model, "weight"):
        model.weight.data.uniform_(minval, maxval)
    if hasattr(model, "bias") and model.bias is not None:
        model.bias.data.uniform_(minval, maxval)

# given a class number c, output a onehot label with length num_classes
def label_to_onehot(c, num_classes=10):
    target = torch.unsqueeze(c, 1)
    onehot_target = torch.zeros(target.size(0), num_classes, device=target.device)
    onehot_target.scatter_(1, target, 1)
    return onehot_target

# loss for onehot vector
def cross_entropy_for_onehot(pred, target):
    return torch.mean(torch.sum(- target * F.log_softmax(pred, dim=-1), 1))

# get a random image, label pair with the dimensions of CIFAR data
def get_random_pair():
  random_image = torch.randn((3, 32, 32), requires_grad=True).requires_grad_(True)
  random_label = torch.randn(10, requires_grad=True).requires_grad_(True)
  return random_image, random_label

# plot a CIFAR tensor image
def imshow(img):
  npimg = img.numpy()   # convert from tensor
  plt.imshow(np.transpose(npimg, (1, 2, 0))) 
  plt.show()

# plot a series of images
def plot_history(history):
  plt.figure(figsize=(12, 8))
  for i in range(len(history)):
      plt.subplot(3, 10, i + 1)
      plt.imshow(np.transpose(history[i], (1, 2, 0)).numpy())
      plt.title("iter=%d" % (i * 10))
      plt.axis('off')
  plt.show()
