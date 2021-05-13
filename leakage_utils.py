import time
import copy
import sys
import random
import math
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
def weights_init(model, minval=-.5, maxval=.5):
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
      plt.title("iter=%d" % (i * 10 + 9))
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

mse_loss = torch.nn.MSELoss()
# calculate peak signal to noise ratio
def psnr(im1, im2, max_value = 1):
  mse = mse_loss(im1, im2).detach().numpy()
  return 10 * np.log10(max_value * max_value / mse)

# # adds a pruned_gradient key to the dictionaries in device_gradients
# # representing the pruned gradient for that device using the clustered pruning method
# # optional amplify factor will just multiply the gradients to scale them up
def add_pruned_gradients(device_gradients, clusters, overlap_factor, amplify_factor = 1, for_attack = False):
  gradient_type = 'pruned_grad_for_attack' if for_attack else 'pruned_grad'
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
          layer[start:end] = layer[start:end] * amplify_factor
          layer[end+1:] = 0
        device_gradients[d]['gradient_multiplier'] = cluster_size/num_devices
        device_gradients[d][gradient_type] = pruned_grad
        device_gradients[d]['group_size'] = 1
        device_gradients[d]['start_end_numerator'] = i
        device_gradients[d]['start_end_denominator'] = float(cluster_size)
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
            layer[start:end] = layer[start:end] * amplify_factor
            layer[end+1:] = 0
          device_gradients[d]['gradient_multiplier'] = (cluster_size/num_devices)/group_size
          device_gradients[d][gradient_type] = pruned_grad
          device_gradients[d]['group_size'] = group_size
          device_gradients[d]['start_end_numerator'] = g
          device_gradients[d]['start_end_denominator'] = float(num_groups)

# initialize list of 0-tensors in shape of gradient
def init_empty_gradient(device_gradients):
  if len(device_gradients) == 0:
    print("No device gradients to match shape to...")
    return []
  empty_grad = []
  for dg in device_gradients[0]['orig_grad']:
    empty_grad.append(torch.zeros(dg.shape).cuda())
  return empty_grad

# get gradient for a single cluster given by cluster_num
def get_cluster_gradient(cluster_num, device_gradients, for_attack = False):
  gradient_type = 'pruned_grad_for_attack' if for_attack else 'pruned_grad'
  cluster_grad = init_empty_gradient(device_gradients)
  for d in clusters[cluster_num]:
    for layer_num, d_layer in enumerate(device_gradients[d][gradient_type]):
        scaled_grad = d_layer / device_gradients[d]['group_size']
        cluster_grad[layer_num] = torch.add(cluster_grad[layer_num], scaled_grad)
  return cluster_grad


# # creates a list of cluster gradients
# # for each cluster, want to: iterate through the pruned gradients of devices in it
# sum up pruned_grad divided by group_size for each device
# for_attack: whether this clustering is for gradients from the random image used in the attack
def get_cluster_gradients(device_gradients, clusters, for_attack = False):
  gradient_type = 'pruned_grad_for_attack' if for_attack else 'pruned_grad'
  cluster_grads = []
  # cluster_grads = [init_empty_gradient(device_gradients) for _ in clusters]
  for c in clusters.keys():
    # for d in clusters[c]:
    cluster_grads.append(get_cluster_gradient(c, device_gradients, for_attack = for_attack))
      # for layer_num, d_layer in enumerate(device_gradients[d][gradient_type]):
      #   scaled_grad = d_layer / device_gradients[d]['group_size']
      #   cluster_grads[c][layer_num] = torch.add(cluster_grads[c][layer_num], scaled_grad)
  return cluster_grads


def prune_attack_gradients(device_gradients, cluster_items, overlap_factor, amplify_factor = 1):
  gradient_type = 'pruned_grad_for_attack'
  cluster_size = len(cluster_items)
  num_devices = len(device_gradients)
  # iterate through clusters
  if overlap_factor > cluster_size:
    for i in range(len(cluster_items)):
      d = cluster_items[i]
      # gradient
      pruned_grad = device_gradients[d]['grad_for_attack'] # copy.deepcopy(device_gradients[d]['grad_for_attack'])
      # prune it
      for k, layer in enumerate(pruned_grad):
        start = math.ceil((i)*(float(layer.shape[0])/float(cluster_size)) - 0.0001)
        end = math.floor((i+1)*(float(layer.shape[0])/float(cluster_size)) + 0.0001)
        layer[0:start] = layer[0:start] * 0
        layer[start:end] = layer[start:end] * amplify_factor
        layer[end+1:] = layer[end+1:] * 0
      device_gradients[d]['gradient_multiplier'] = cluster_size/num_devices
      device_gradients[d][gradient_type] = pruned_grad
      device_gradients[d]['group_size'] = 1
  else:
    num_groups = math.ceil(cluster_size/overlap_factor) 
    for g in range(num_groups):
      group = cluster_items[g*overlap_factor:(g+1)*overlap_factor]
      group_size = len(group)
      for d in group:
        pruned_grad = device_gradients[d]['grad_for_attack'] # copy.deepcopy(device_gradients[d]['grad_for_attack'])
        for k, layer in enumerate(pruned_grad):
          start = math.ceil((g)*(float(layer.shape[0])/float(num_groups)) - 0.0001)
          end = math.floor((g+1)*(float(layer.shape[0])/float(num_groups)) + 0.0001)
          layer[0:start] = layer[0:start] * 0
          layer[start:end] = layer[start:end] * amplify_factor
          layer[end+1:] = layer[end+1:] * 0
        device_gradients[d]['gradient_multiplier'] = (cluster_size/num_devices)/group_size
        device_gradients[d][gradient_type] = pruned_grad
        device_gradients[d]['group_size'] = group_size

# launch privacy attack on an individual device
def try_recovery_individual(device, actual_grad, num_iters = 100, save_every = 10):
  random_image, random_label = get_random_pair()
  leakage_optimizer = torch.optim.LBFGS([random_image, random_label], lr = 1)
  history = []
  for iters in range(num_iters):

    def closure():
      leakage_optimizer.zero_grad()
      dummy_pred = device['net'](torch.unsqueeze(random_image, dim = 0).cuda())
      dummy_onehot_label = F.softmax(random_label, dim=-1).cuda()
      dummy_loss = leakage_criterion(dummy_pred, dummy_onehot_label) 
      dummy_grad = torch.autograd.grad(dummy_loss, device['net'].parameters(), create_graph = True)
      grad_diff = 0
      for gx, gy in zip(dummy_grad, actual_grad): 
        grad_diff += ((gx - gy) ** 2).sum()
      grad_diff.backward()
      return grad_diff

    leakage_optimizer.step(closure)
    if (iters+1) % save_every == 0:
      diff = closure()
      # print(diff)
      history.append(copy.deepcopy(random_image.cpu().detach()))
  return random_image, random_label, history
          


