import time
import copy
import sys
import random
from collections import OrderedDict


import random
import math

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import numpy as np
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import cluster_utils as utils
import argparse



# Using same ConvNet as in Assignment 1
def conv_block(in_channels, out_channels, kernel_size=3, stride=1,
               padding=1):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding,
                  bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
        )

class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.model = nn.Sequential(
            conv_block(3, 32),
            conv_block(32, 32),
            conv_block(32, 64, stride=2),
            conv_block(64, 64),
            conv_block(64, 64),
            conv_block(64, 128, stride=2),
            nn.AdaptiveAvgPool2d(1)
            )

        self.classifier = nn.Linear(128, 10)

    def forward(self, x):
        h = self.model(x)
        B, C, _, _ = h.shape
        h = h.view(B, C)
        return self.classifier(h)


import copy


criterion = nn.CrossEntropyLoss()

class DatasetSplit(torch.utils.data.Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = [int(i) for i in idxs]

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return image, torch.tensor(label)
  
def create_device(net, device_id, trainset, data_idxs, lr=0.1,
                  milestones=None, batch_size=128):
    if milestones == None:
        milestones = [25, 50, 75]

    device_net = copy.deepcopy(net)
    optimizer = torch.optim.SGD(device_net.parameters(), lr=lr, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                     milestones=milestones,
                                                     gamma=0.1)
    device_trainset = DatasetSplit(trainset, data_idxs)
    device_trainloader = torch.utils.data.DataLoader(device_trainset,
                                                     batch_size=batch_size,
                                                     shuffle=True)
    return {
        'net': device_net,
        'id': device_id,
        'dataloader': device_trainloader, 
        'optimizer': optimizer,
        'scheduler': scheduler,
        'train_loss_tracker': [],
        'train_acc_tracker': [],
        'test_loss_tracker': [],
        'test_acc_tracker': [],
        'weights': []
        }
        
# Load testing data
transform_test = transforms.Compose([                                           
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])
testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                      download=True,
                                      transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False)

def cifar_noniid_group_test(dataset):

  dict_group_classes = {}
  dict_group_classes[0] = [0,1,2,3]
  dict_group_classes[1] = [4,5,6]
  dict_group_classes[2] = [7,8,9]

  
  group_indices = {}
  group_indices[0] = []
  group_indices[1] = []
  group_indices[2] = []

  # iterate through data, adding it to the right group list
  for j,num in enumerate(dataset.targets):
    if num in dict_group_classes[0]:
      group_indices[0].append(j)
    elif num in dict_group_classes[1]:
      group_indices[1].append(j)
    else:
      group_indices[2].append(j)

  for k in group_indices:
    group_indices[k] = set(group_indices[k])

  return group_indices


test_idxs = cifar_noniid_group_test(testset)


def train(epoch, device, net):
    net.train()
    device['net'].train()
    train_loss, correct, total = 0, 0, 0
    for batch_idx, (inputs, targets) in enumerate(device['dataloader']):
        inputs, targets = inputs.cuda(), targets.cuda()
        device['optimizer'].zero_grad()
        outputs = device['net'](inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        device['optimizer'].step()
        train_loss += loss.item()
        device['train_loss_tracker'].append(loss.item())
        loss = train_loss / (batch_idx + 1)
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        acc = 100. * correct / total
        dev_id = device['id']
        sys.stdout.write(f'\r(Device {dev_id}/Epoch {epoch}) ' + 
                         f'Train Loss: {loss:.3f} | Train Acc: {acc:.3f}')
        sys.stdout.flush()
    new_dic = {}
    for key, value in device['net'].named_parameters():
      if 'weight' in key:
        new_dic[key] =  value

    device['weights'] = new_dic
    device['train_acc_tracker'].append(acc)
    sys.stdout.flush()

def test(epoch, device, net):
    net.eval()
    test_loss, correct, total = 0, 0, 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.cuda(), targets.cuda()
            outputs = device['net'](inputs)
            loss = criterion(outputs, targets)
            test_loss += loss.item()
            device['test_loss_tracker'].append(loss.item())
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            loss = test_loss / (batch_idx + 1)
            acc = 100.* correct / total
    sys.stdout.write(f' | Test Loss: {loss:.3f} | Test Acc: {acc:.3f}\n')
    sys.stdout.flush()  
    acc = 100.*correct/total
    device['test_acc_tracker'].append(acc)
    return acc

def moving_average(a, n=100):
    '''Helper function used for visualization'''
    ret = torch.cumsum(torch.Tensor(a), 0)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n


def get_devices_for_round(devices, device_pct):
    '''
    '''
    # Randomly sample indices corresponding to devices based on total number of devices and device_pct
    indices = random.sample(range(len(devices)), int(len(devices)*device_pct))

    # Create list based on indices sampled above
    new_devices = [devices[i] for i in indices]
    return new_devices
    
def get_devices_for_round_cluster(devices, device_pct, preds):
    '''
    '''
    # Randomly sample indices corresponding to devices based on total number of devices and device_pct
    indices = random.sample(range(len(devices)), int(len(devices)*device_pct))

    # Create list based on indices sampled above
    new_devices = [devices[i] for i in indices]
    new_preds = [preds[i] for i in indices]
    return new_devices, new_preds


def average_weights_cluster_overlap(devices,preds,overlap_factor):
  num_devices = len(devices)
  clusters = {}
  for i in range(len(preds)):
    if preds[i] in clusters.keys():
      clusters[preds[i]].append(i)
    else:
      clusters[preds[i]] = [i]
  for k in clusters.keys():
    random.shuffle(clusters[k])
  # initialize master dict to 0
  new_dict = devices[0]['net'].state_dict()
  for k in new_dict.keys():
      new_dict[k] = torch.zeros(new_dict[k].shape).cuda()
  # iterate through clusters
  for c in clusters.keys():
    cluster_size = len(clusters[c])
    if overlap_factor > cluster_size:
      scale = cluster_size/num_devices
      for i in range(len(clusters[c])):
        device_dict = devices[clusters[c][i]]['net'].state_dict()
        for k in new_dict.keys():
          if 'weight' in k:
            if len(new_dict[k].shape) == 0: # if scalar
              new_dict[k] = torch.add(((1/num_devices)*device_dict[k]), new_dict[k])
            else:
              start = math.ceil((i)*(float(device_dict[k].shape[0])/float(cluster_size)) - 0.0001)
              end = math.floor((i+1)*(float(device_dict[k].shape[0])/float(cluster_size)) + 0.0001)
              device_dict[k][0:start] = 0
              device_dict[k][end+1:] = 0
              new_dict[k] = torch.add(scale*device_dict[k],new_dict[k])
          else:
            new_dict[k] = torch.add((1/num_devices)*device_dict[k],new_dict[k])
      devices[clusters[c][i]]['net'].load_state_dict(device_dict)
    else: # if the cluster is large enough that we can do overlap
      num_groups = math.ceil(cluster_size/overlap_factor) 
      # divide the cluster into further groups so that each partition has 
      # overlap_factor number of devices
      for g in range(num_groups):
        group = clusters[c][g*overlap_factor:(g+1)*overlap_factor]
        group_size = len(group)
        scale = (cluster_size/num_devices)/group_size
        for d in group:
          device_dict = devices[d]['net'].state_dict()
          for k in new_dict.keys():
            if 'weight' in k:
              if len(new_dict[k].shape) == 0: # if scalar
                new_dict[k] = torch.add(((1/num_devices)*device_dict[k]), new_dict[k])
              else:
                start = math.ceil((g)*(float(device_dict[k].shape[0])/float(num_groups)) - 0.0001)
                end = math.floor((g+1)*(float(device_dict[k].shape[0])/float(num_groups)) + 0.0001)
                device_dict[k][0:start] = 0
                device_dict[k][end+1:] = 0
                new_dict[k] = torch.add(scale*device_dict[k],new_dict[k])
            else:
              new_dict[k] = torch.add((1/num_devices)*device_dict[k],new_dict[k])
        devices[d]['net'].load_state_dict(device_dict)
  return new_dict
  

def average_weights(devices):
    '''
    devices: a list of devices generated by create_devices
    Returns an the average of the weights.
    '''
    # Create a new dict that corresponds to the weights of the first device (we assume > 0 devices, which is fair)
    new_dict = devices[0]['net'].state_dict()
    num_devices = len(devices)

    # Change all the weights to be 1/(num devices) * weight
    for k in new_dict:
      new_dict[k] = (1/num_devices)*new_dict[k]
    
    # For each other device add the tensor of 1/(num devices) * weight
    for d in devices[1:]:
      device_dict = d['net'].state_dict()
      for k in device_dict:
        new_dict[k] = torch.add(((1/num_devices)*device_dict[k]), new_dict[k])
    return new_dict

def iid_sampler(dataset, num_devices, data_pct):
    '''
    dataset: PyTorch Dataset (e.g., CIFAR-10 training set)
    num_devices: integer number of devices to create subsets for
    data_pct: percentage of training samples to give each device
              e.g., 0.1 represents 10%

    return: a dictionary of the following format:
      {
        0: [3, 65, 2233, ..., 22] // device 0 sample indexes
        1: [0, 2, 4, ..., 583] // device 1 sample indexes
        ...
      }

    iid (independent and identically distributed) means that the indexes
    should be drawn independently in a uniformly random fashion.
    '''

    # total number of samples in the dataset
    total_samples = len(dataset)

    # Initialize dictionary to store indices
    sampled = {}
    for i in range(num_devices):
      # Randomly sample indices uniformly for each device
      sampled[i] = random.sample(range(total_samples), int(data_pct*total_samples))

    return sampled


def main():
    parser = argparse.ArgumentParser(
        description="IID Clustered Pruning Federated Learning"
    )
    # model related parameters
    parser.add_argument("--initial-rounds", type=int, default=1)
    parser.add_argument("--initial-epochs", type=int, default=3)
    parser.add_argument("--num-devices", type=int, default=60)
    parser.add_argument("--fl-rounds", type=int, default=50)
    parser.add_argument("--fl-epochs", type=int, default=1)
    parser.add_argument("--device-percent", type=float, default=1.0)
    parser.add_argument("--data-percent", type=float, default=0.1)
    parser.add_argument("--num-clusters", type=int, default=9)
    parser.add_argument("--pca-dim", type=int, default=10)

    args = parser.parse_args()


    

    # Using CIFAR-10
    # Load training data
    transform_train = transforms.Compose([                                   
        transforms.RandomCrop(32, padding=4),                                       
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, 
                                            download=True,
                                            transform=transform_train)

    init_rounds = args.initial_rounds
    local_epochs = args.initial_epochs
    num_devices = args.num_devices
    num_clusters = args.num_clusters
    pca_dim = args.pca_dim
    data_pct = args.data_percent
    net = ConvNet().cuda()
    criterion = nn.CrossEntropyLoss()
    data_idxs = iid_sampler(trainset, num_devices, data_pct)
    device_pct = args.device_percent
    preds_arr = []

    # Device Creation
    devices = []
    for i in data_idxs:
        new_d = create_device(net, i, trainset, data_idxs[i],
                            milestones=[250, 500, 750])
        devices.append(new_d)
    # 
    ## Non-IID Federated Learning
    start_time = time.time()
    for round_num in range(init_rounds):
        round_devices = devices

        print('Round: ', round_num)
        for device in round_devices:
            for local_epoch in range(local_epochs):
                train(local_epoch, device, net)

        w_avg = average_weights(round_devices)

        for device in devices:
            device['net'].load_state_dict(w_avg)
            device['optimizer'].zero_grad()
            device['optimizer'].step()
            device['scheduler'].step()

        test(round_num, devices[0], net)

        pca_data = utils.pca_decomp(round_devices, pca_dim)

    preds = utils.kmeans_cluster(pca_data, num_clusters)
    print(preds)
    preds_arr.append(preds)

    fl_rounds = args.fl_rounds
    local_epochs = args.fl_epochs
    for round_num in range(fl_rounds):
        round_devices, round_preds = get_devices_for_round_cluster(devices, device_pct, preds)


        print('Round: ', round_num)
        for device in round_devices:
            for local_epoch in range(local_epochs):
                train(local_epoch, device, net)

        w_avg = average_weights_cluster_overlap(round_devices, round_preds, 2)
        

        for device in devices:
            device['net'].load_state_dict(w_avg)
            device['optimizer'].zero_grad()
            device['optimizer'].step()
            device['scheduler'].step()

        # test accuracy after aggregation
        test(round_num, devices[0], net)
        if round_num < 15:
            pca_data = utils.pca_decomp(round_devices, pca_dim)
            preds_new = utils.kmeans_cluster(pca_data, num_clusters)
            print(preds_new)
            preds_arr.append(preds_new)



    total_time = time.time() - start_time
    print('Total training time: {} seconds'.format(total_time))
    with open('test_acc_iid.npy', 'wb') as f:
        np.save(f, devices[0]['test_acc_tracker'])
    with open('train_loss_iid.npy', 'wb') as f:
        np.save(f, devices[0]['train_loss_tracker'])
    with open('cluster_labs_iid.npy', 'wb') as f:
        np.save(f, np.array(preds_arr))

if __name__ == "__main__":
    main()

