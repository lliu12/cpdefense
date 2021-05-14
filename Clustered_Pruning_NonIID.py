

## Code Cell 1.1

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

## Code Cell 1.2
import copy

# net  = ConvNet().cuda()
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


# creates noniid TRAINING datasets for each group
def noniid_group_sampler(dataset, num_items_per_device):
  '''
    dataset: PyTorch Dataset (e.g., CIFAR-10 training set)
    num_devices: integer number of devices to create subsets for
    num_items_per_device: how many samples to assign to each device

    return: a dictionary of the following format:
      {
        0: [3, 65, 2233, ..., 22] // device 0 sample indexes
        1: [0, 2, 4, ..., 583] // device 1 sample indexes
        ...
      }

    '''
  # how many devices per non-iid group
  devices_per_group = [20, 20, 20]

  # label assignment per group
  dict_group_classes = {}
  dict_group_classes[0] = [0,1,2,3]
  dict_group_classes[1] = [4,5,6]
  dict_group_classes[2] = [7,8,9]  

  devices_per_group = [20, 20, 20]

  dict_group_classes = {}
  dict_group_classes[0] = [0,1,2,3]
  dict_group_classes[1] = [4,5,6]
  dict_group_classes[2] = [7,8,9] 

  group_indices = {}
  group_indices[0] = []
  group_indices[1] = []
  group_indices[2] = []

  for j,num in enumerate(dataset.targets):
    if num in dict_group_classes[0]:
      group_indices[0].append(j)
    elif num in dict_group_classes[1]:
      group_indices[1].append(j)
    else:
      group_indices[2].append(j)

  sampled = {}
  offset = 0 
  for j,d in enumerate(devices_per_group):
    for i in range(d):
      sampled[offset + i] = random.sample(group_indices[j], num_items_per_device)
    offset = offset + d

  return sampled

# get which devices in each group should participate in a current round
# by explicitly saying number of each devices desired for each group 
def get_devices_for_round_GROUP(devices, device_nums, user_group_idxs = None):
  # PART 2.2: Implement! 
  # Assume first 20 are group 0, second 20 are group 1, third 20 are group 2

  # devices: list of devices
  # device_nums: list providing sample number desired per group
  # what is user_group_idxs??? function currently does not use it

  devices_per_group = [20,20,20]
  num_groups = len(devices_per_group)
  included_indices = []
  offset = 0
  for g in range(num_groups):
    # sample desired number of indices from group g
    included_indices += random.sample(range(offset, offset + devices_per_group[g]), device_nums[g]) 
    offset += devices_per_group[g]
  # get devices for selected indices
  included_devices = [devices[i] for i in included_indices]
  return included_devices


def average_weights(devices):
    '''
    devices: a list of devices generated by create_devices
    Returns an the average of the weights.
    '''
    # Part 1.2: Implement!
    # Hint: device['net'].state_dict() will return an OrderedDict of all
    #       tensors in the model. Return the average of each tensor using
    #       and OrderedDict so that you can update the global model using
    #       device['net'].load_state_dict(w_avg), where w_avg is the 
    #       averaged OrderedDict over all devices
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

# gets per-group accuracy of global model
def test_group(epoch, device, group_idxs_dict, net):

    # Part 2.3: Implement!
    # Hint: refer to test function in PART 1
    # Hint: check https://pytorch.org/docs/stable/data.html?highlight=subset#torch.utils.data.Subset
    
    num_groups = len(group_idxs_dict)

    # same testing/evaluation code from earlier, but adapted to support three groups
    net.eval()

    # keep lists instead of single numbers for metrics
    test_loss, correct, total = [0 for _ in range(num_groups)], [0 for _ in range(num_groups)], [0 for _ in range(num_groups)]
    with torch.no_grad():
      sys.stdout.write('\n')
      for g in range(num_groups):
        group_testset = torch.utils.data.Subset(testset, list(test_idxs[g]))
        group_dataloader = torch.utils.data.DataLoader(group_testset, batch_size=128, shuffle=False)
        for batch_idx, (inputs, targets) in enumerate(group_dataloader):
            inputs, targets = inputs.cuda(), targets.cuda()
            outputs = device['net'](inputs)
            loss = criterion(outputs, targets)
            test_loss[g] += loss.item()
            device['test_loss_tracker'].append(loss.item())
            _, predicted = outputs.max(1)
            total[g] += targets.size(0)
            correct[g] += predicted.eq(targets).sum().item()
        loss = test_loss[g] / (batch_idx + 1)
        acc = 100.* correct[g] / total[g]
        sys.stdout.write(f'| Group: {g} | Test Loss: {loss:.3f} | Test Acc: {acc:.3f}\n')

    sys.stdout.flush()  
    acc = [100.*correct[i] / total[i] for i in range(num_groups)]
    device['test_acc_tracker'].append(acc)

import random
import math

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




def main():
    parser = argparse.ArgumentParser(
        description="IID Clustered Pruning Federated Learning"
    )
    # model related parameters
    parser.add_argument("--initial-rounds", type=int, default=1)
    parser.add_argument("--initial-epochs", type=int, default=3)
    parser.add_argument("--num-items-per-device", type=int, default=5000)
    parser.add_argument("--fl-rounds", type=int, default=50)
    parser.add_argument("--fl-epochs", type=int, default=1)
    parser.add_argument("--device-percent", type=float, default=1.0)
    parser.add_argument("--data-percent", type=float, default=0.1)
    parser.add_argument("--num-clusters", type=int, default=9)
    parser.add_argument("--pca-dim", type=int, default=10)

    args = parser.parse_args()


    # Using CIFAR-10 again as in Assignment 1
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
    num_items_per_device = args.num_items_per_device
    num_clusters = args.num_clusters
    pca_dim = args.pca_dim
    data_pct = args.data_percent
    net = ConvNet().cuda()
    criterion = nn.CrossEntropyLoss()
    cluster_arr = []
    preds_arr = []
    device_nums = [1, 1, 1]
    net = ConvNet().cuda()
    criterion = nn.CrossEntropyLoss()
    device_pct = args.device_percent

    # Part 2.4: Implement non-iid sampling
    data_idxs = noniid_group_sampler(trainset, num_items_per_device)

    # Part 2.4: Implement device creation here
    devices = [create_device(net, i, trainset, data_idxs[i],
                        milestones=[250, 500, 750]) for i in range(len(data_idxs))]
    # 
    ## Non-IID Federated Learning
    start_time = time.time()
    for round_num in range(init_rounds):

        # Part 2.4: Implement getting devices for each round here
        round_devices = devices

        print('Round: ', round_num)
        for device in round_devices:
            for local_epoch in range(local_epochs):
                train(local_epoch, device, net)

        # Part 2.4: Implement weight averaging here 
        w_avg = average_weights(round_devices)


        for device in devices:
            device['net'].load_state_dict(w_avg)
            device['optimizer'].zero_grad()
            device['optimizer'].step()
            device['scheduler'].step()

        # Part 2.4: Implement test accuracy here
        test_group(round_num, devices[0], test_idxs, net)

        arr = []
        for i in range(len(devices)):
            new_torch = torch.empty([1]).cuda()
            for k in devices[i]['weights']:
                if 'weight'  in k:
                    new_torch = torch.cat((new_torch, devices[i]['weights'][k].flatten()), 0)
            arr.append(new_torch.detach().cpu().numpy())

        pca_data = utils.pca_decomp(round_devices, pca_dim)
    preds = utils.kmeans_cluster(pca_data, num_clusters)
    print(preds)
    preds_arr.append(preds)

    fl_rounds = args.fl_rounds
    local_epochs = args.fl_epochs
    for round_num in range(fl_rounds):
        # Part 1.3: Implement getting devices for each round here
        round_devices, round_preds = get_devices_for_round_cluster(devices, device_pct, preds)


        print('Round: ', round_num)
        for device in round_devices:
            for local_epoch in range(local_epochs):
                train(local_epoch, device, net)

        # Part 1.3: Implement weight averaging here
        w_avg = average_weights_cluster_overlap(round_devices, round_preds, 2)
        

        for device in devices:
            device['net'].load_state_dict(w_avg)
            device['optimizer'].zero_grad()
            device['optimizer'].step()
            device['scheduler'].step()

        # test accuracy after aggregation
        test_group(round_num, devices[0], test_idxs, net)
        if round_num < 15:
            pca_data = utils.pca_decomp(round_devices, pca_dim)
            preds_new = utils.kmeans_cluster(pca_data, num_clusters)
            print(preds_new)
            preds_arr.append(preds_new)



    total_time = time.time() - start_time
    print('Total training time: {} seconds'.format(total_time))
    with open('test_acc_noniid.npy', 'wb') as f:
        np.save(f, devices[0]['test_acc_tracker'])
    with open('train_loss_noniid.npy', 'wb') as f:
        np.save(f, devices[0]['train_loss_tracker'])
    with open('cluster_labs_noniid.npy', 'wb') as f:
        np.save(f, np.array(preds_arr))

if __name__ == "__main__":
    main()

