import numpy as np
import os
import sys
import random
import torch
import torchvision
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from PIL import Image
import torch.utils.data as data
from os import path 
# from six.moves import urllib
# from utils.dataset_utils import check, separate_data, split_data, save_file

random.seed(1)
np.random.seed(1)
num_clients = 20
num_classes = 10
dir_path = "mnist/"
batch_size = 64
train_size = 0.75
least_samples = batch_size / (1-train_size)

def split_data(X, y):
    # Split dataset
    train_data, test_data = [], []
    num_samples = {'train':[], 'test':[]}

    X_train, X_test, y_train, y_test = train_test_split(
            X, y, train_size=0.75, shuffle=True)

    train_data.append({'x': X_train, 'y': y_train})
    num_samples['train'].append(len(y_train))
    test_data.append({'x': X_test, 'y': y_test})
    num_samples['test'].append(len(y_test))

    print("Total number of samples:", sum(num_samples['train'] + num_samples['test']))
    print("The number of train samples:", num_samples['train'])
    print("The number of test samples:", num_samples['test'])
    print()
    del X, y
    # gc.collect()

    return train_data, test_data

# Allocate data to users
def generate_mnist(dir_path, num_classes, num_clients, niid , balance, partition, alpha):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
        
    # Setup directory for train/test data
    # config_path = dir_path + "config.json"
    # train_path = dir_path + "train/"
    # test_path = dir_path + "test/"

    # if check(config_path, train_path, test_path, num_clients, num_classes, niid, balance, partition):
    #     return

    # FIX HTTP Error 403: Forbidden
    
    # opener = urllib.request.build_opener()
    # opener.addheaders = [('User-agent', 'Mozilla/5.0')]
    # urllib.request.install_opener(opener)
    
    X, y = [], []
    global_loaders_train = {}
    party2loaders_train = {}
    party2loaders_test = {}
    
    # Get MNIST data
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])

    trainset = torchvision.datasets.MNIST(
        root=dir_path+"rawdata", train=True, download=True, transform=transform)
    testset = torchvision.datasets.MNIST(
        root=dir_path+"rawdata", train=False, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=len(trainset.data), shuffle=False)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=len(testset.data), shuffle=False)

    for _, train_data in enumerate(trainloader, 0):
        trainset.data, trainset.targets = train_data
    for _, test_data in enumerate(testloader, 0):
        testset.data, testset.targets = test_data

    dataset_image = []
    dataset_label = []
    # dataset_train_image = []
    # dataset_train_label = []
    # dataset_test_image = []
    # dataset_test_label = []

    # dataset_train_image = np.array(trainset.data.cpu().detach().numpy())
    # dataset_train_label = np.array(trainset.targets.cpu().detach().numpy())
    # dataset_test_image = np.array(testset.data.cpu().detach().numpy())
    # dataset_test_label = np.array(testset.targets.cpu().detach().numpy())
    dataset_image.extend(trainset.data.cpu().detach().numpy())
    dataset_image.extend(testset.data.cpu().detach().numpy())
    dataset_label.extend(trainset.targets.cpu().detach().numpy())
    dataset_label.extend(testset.targets.cpu().detach().numpy())

    # train_data, test_data = [], []
    # num_samples = {'train':[], 'test':[]}

    # train_data.append({'x': dataset_train_image, 'y': dataset_train_label})
    # num_samples['train'].append(len(dataset_train_label))
    # test_data.append({'x': dataset_test_image, 'y': dataset_test_label})
    # num_samples['test'].append(len(dataset_test_label))

    # print("Total number of samples:", sum(num_samples['train'] + num_samples['test']))
    # print("The number of train samples:", num_samples['train'])
    # print("The number of test samples:", num_samples['test'])
    # print()

    train_data, test_data = split_data(np.array(dataset_image), np.array(dataset_label))
    X, y, statistic = separate_data((train_data[0]['x'], train_data[0]['y']), num_clients, num_classes, alpha, 
                                    niid, balance, partition)
    # train_data = []
    for i in range(len(y)):
        train_data_now_x = X[i]
        train_data_now_y = y[i]
        X_train = torch.Tensor(train_data_now_x).type(torch.float32)
        y_train = torch.Tensor(train_data_now_y).type(torch.int64)
        train_data_use = [(x, y) for x, y in zip(X_train, y_train)]
        party2loaders_train[i] = (DataLoader(train_data_use, batch_size = 64, drop_last=True, shuffle=True))
        
    X_global_train = torch.Tensor(train_data[0]['x']).type(torch.float32)
    y_global_train = torch.Tensor(train_data[0]['y']).type(torch.int64)
    global_train_data_use = [(x, y) for x, y in zip(X_global_train, y_global_train)]
        
    global_loaders_train = (DataLoader(global_train_data_use, batch_size = 64, drop_last=True, shuffle=True))
    
    X_test = torch.Tensor(test_data[0]['x']).type(torch.float32)
    y_test = torch.Tensor(test_data[0]['y']).type(torch.int64)
    test_data_use = [(x, y) for x, y in zip(X_test, y_test)]
    party2loaders_test = (DataLoader(test_data_use, batch_size = 64, drop_last=True, shuffle=True))
    
    return party2loaders_train , party2loaders_test, global_loaders_train
        

def separate_data(data, num_clients, num_classes, alpha, niid=False, balance=False, partition=None, class_per_client=2):
    X = [[] for _ in range(num_clients)]
    y = [[] for _ in range(num_clients)]
    statistic = [[] for _ in range(num_clients)]

    dataset_content, dataset_label = data

    dataidx_map = {}
    
    n_train = dataset_label.shape[0]
    
    if not niid:
        partition = 'pat'
        class_per_client = num_classes
        
    if partition == "homo" or partition == "iid":
        idxs = np.random.permutation(n_train)
        batch_idxs = np.array_split(idxs, num_clients)
        dataidx_map = {i: batch_idxs[i] for i in range(num_clients)} #make dic

    elif partition == 'pat':
        idxs = np.array(range(len(dataset_label)))
        idx_for_each_class = []
        for i in range(num_classes):
            idx_for_each_class.append(idxs[dataset_label == i])

        class_num_per_client = [class_per_client for _ in range(num_clients)]
        for i in range(num_classes):
            selected_clients = []
            for client in range(num_clients):
                if class_num_per_client[client] > 0:
                    selected_clients.append(client)
                selected_clients = selected_clients[:int(num_clients/num_classes*class_per_client)]

            num_all_samples = len(idx_for_each_class[i])
            num_selected_clients = len(selected_clients)
            num_per = num_all_samples / num_selected_clients
            if balance:
                num_samples = [int(num_per) for _ in range(num_selected_clients-1)]
            else:
                num_samples = np.random.randint(max(num_per/10, least_samples/num_classes), num_per, num_selected_clients-1).tolist()
            num_samples.append(num_all_samples-sum(num_samples))

            idx = 0
            for client, num_sample in zip(selected_clients, num_samples):
                if client not in dataidx_map.keys():
                    dataidx_map[client] = idx_for_each_class[i][idx:idx+num_sample]
                else:
                    dataidx_map[client] = np.append(dataidx_map[client], idx_for_each_class[i][idx:idx+num_sample], axis=0)
                idx += num_sample
                class_num_per_client[client] -= 1

    elif partition == "noniid":
        # https://github.com/IBM/probabilistic-federated-neural-matching/blob/master/experiment.py
        min_size = 0
        K = num_classes
        N = len(dataset_label)

        while min_size < least_samples:
            idx_batch = [[] for _ in range(num_clients)]
            for k in range(K):
                idx_k = np.where(dataset_label == k)[0]
                np.random.shuffle(idx_k)
                proportions = np.random.dirichlet(np.repeat(alpha, num_clients))
                proportions = np.array([p*(len(idx_j)<N/num_clients) for p,idx_j in zip(proportions,idx_batch)])
                proportions = proportions/proportions.sum()
                proportions = (np.cumsum(proportions)*len(idx_k)).astype(int)[:-1]
                idx_batch = [idx_j + idx.tolist() for idx_j,idx in zip(idx_batch,np.split(idx_k,proportions))]
                min_size = min([len(idx_j) for idx_j in idx_batch])

        for j in range(num_clients):
            dataidx_map[j] = idx_batch[j]
    else:
        raise NotImplementedError

    # assign data
    for client in range(num_clients):
        idxs = dataidx_map[client]
        X[client] = dataset_content[idxs]
        y[client] = dataset_label[idxs]

        for i in np.unique(y[client]):
            statistic[client].append((int(i), int(sum(y[client]==i))))
            
    del data
    # gc.collect()

    for client in range(num_clients):
        print(f"Client {client}\t Size of data: {len(X[client])}\t Labels: ", np.unique(y[client]))
        print(f"\t\t Samples of labels: ", [i for i in statistic[client]])
        print("-" * 50)

    return X, y, statistic