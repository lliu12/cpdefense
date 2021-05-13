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

# loss for onehot vector
def cross_entropy_for_onehot(pred, target):
    return torch.mean(torch.sum(- target * F.log_softmax(pred, dim=-1), 1))

leakage_criterion = cross_entropy_for_onehot

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
def get_onehot_label(c, num_classes=10):
    target = torch.unsqueeze(c, 1)
    onehot_target = torch.zeros(target.size(0), num_classes, device=target.device)
    onehot_target.scatter_(1, target, 1)
    return onehot_target

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

# get a sample data/onehot label pair from the device's dataloader
def get_device_data_sample(device):
  for inputs, targets in device['dataloader']:
      sample_image = inputs[0]
      sample_label = targets[0]
      onehot_sample_label = get_onehot_label(torch.unsqueeze(sample_label, 0), num_classes = 10)
      break
  return sample_image, onehot_sample_label

# given a device and data sample, return device model's gradient on this data
def get_true_device_gradient(device, sample_image, onehot_sample_label):
  pred = device['net'](torch.unsqueeze(sample_image, dim = 0).cuda())
  y = leakage_criterion(pred, onehot_sample_label.cuda())
  dy_dx = torch.autograd.grad(y, device['net'].parameters())
  actual_grad = list((_.detach().clone() for _ in dy_dx))
  return actual_grad

# adds a pruned_gradient key to the dictionaries in device_gradients
# representing the pruned gradient for that device using the clustered pruning method
def add_pruned_gradients(device_gradients, clusters, overlap_factor):
  num_devices = len(device_gradients)
  # iterate through clusters
  for c in clusters.keys():
    cluster_size = len(clusters[c])
    if overlap_factor > cluster_size:
      for i in range(len(clusters[c])):
        d = clusters[c][i]
        # make a copy of the true gradient
        pruned_grad = copy.deepcopy(device_gradients[d]['orig_grad'])
        # prune this copy and store pruned version in the dict
        for k, layer in enumerate(pruned_grad):
          start = math.ceil((i)*(float(layer.shape[0])/float(cluster_size)) - 0.0001)
          end = math.floor((i+1)*(float(layer.shape[0])/float(cluster_size)) + 0.0001)
          layer[0:start] = 0
          layer[end+1:] = 0
        device_gradients[d]['gradient_multiplier'] = cluster_size/num_devices
        device_gradients[d]['pruned_grad'] = pruned_grad
        device_gradients[d]['group_size'] = 1
    else:
      num_groups = math.ceil(cluster_size/overlap_factor) 
      for g in range(num_groups):
        group = clusters[c][g*overlap_factor:(g+1)*overlap_factor]
        group_size = len(group)
        for d in group:
          pruned_grad = copy.deepcopy(device_gradients[d]['orig_grad'])
          for k, layer in enumerate(pruned_grad):
            start = math.ceil((g)*(float(layer.shape[0])/float(num_groups)) - 0.0001)
            end = math.floor((g+1)*(float(layer.shape[0])/float(num_groups)) + 0.0001)
            layer[0:start] = 0
            layer[end+1:] = 0
          device_gradients[d]['gradient_multiplier'] = (cluster_size/num_devices)/group_size
          device_gradients[d]['pruned_grad'] = pruned_grad
          device_gradients[d]['group_size'] = group_size

# initialize list of 0-tensors in shape of gradient
def init_empty_gradient(device_gradients):
  if len(device_gradients) == 0:
    print("No device gradients to match shape to...")
    return []
  empty_grad = []
  for dg in device_gradients[0]['orig_grad']:
    print(dg)
    empty_grad.append(torch.zeros(dg.shape).cuda())
  return empty_grad


# creates a list of cluster gradients
# for each cluster, want to: iterate through the pruned gradients of devices in it
# sum up pruned_grad divided by group_size for each device
def get_cluster_gradients(device_gradients, clusters):
  cluster_grads = [init_empty_gradient(device_gradients) for _ in clusters]
  for c in clusters:
    for d in clusters[c]:
      for layer_num, d_layer in enumerate(device_gradients[d]['pruned_grad']):
        scaled_grad = d_layer / device_gradients[d]['group_size']
        cluster_grads[c][layer_num] = torch.add(cluster_grads[c][layer_num], scaled_grad)
  return cluster_grads

