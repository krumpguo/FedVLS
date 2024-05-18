import torch
import torch.nn as nn
import numpy as np
import time
import os
import copy
import torch.nn.functional as F
import random

def off_diagonal(x):
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

class MR_loss(nn.Module):

    def __init__(self,):
        super(MR_loss, self).__init__()
        self.eps = 1e-8
        self.reject_threshold = 1

    def forward(self, x, y):
        _, C = x.shape
        loss = 0.0
        uniq_l, uniq_c = y.unique(return_counts=True)
        n_count = 0
        for i, label in enumerate(uniq_l):
            if uniq_c[i] <= self.reject_threshold:
                continue
            x_label = x[y==label, :]
            x_label = x_label - x_label.mean(dim=0, keepdim=True)
            x_label = x_label / torch.sqrt(self.eps + x_label.var(dim=0, keepdim=True))

            N = x_label.shape[0]
            corr_mat = torch.matmul(x_label.t(), x_label)
            loss += (off_diagonal(corr_mat).pow(2)).mean()
            n_count += N

        if n_count == 0:
            return 0
        else:
            loss = loss / n_count
            return loss

class clientMR(object):
    def __init__(self, args, id, train_samples, **kwargs):
        
        self.model = copy.deepcopy(args.model)
        self.algorithm = args.algorithm
        self.dataset = args.dataset
        self.device = args.device 
        self.id = id  # integer
        self.num_classes = args.num_classes
        self.train_samples = train_samples
        self.batch_size = args.batch_size
        self.learning_rate = args.local_learning_rate
        self.local_epochs = args.local_epochs
        self.weight_decay = args.weight_decay
        self.mu = args.mu
        self.train_time_cost = {'num_rounds': 0, 'total_cost': 0.0}
        self.send_time_cost = {'num_rounds': 0, 'total_cost': 0.0}
        
        self.loss = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, self.model.parameters()), lr=self.learning_rate, momentum=0.9, weight_decay=self.weight_decay)
        
        self.criterion_deco = MR_loss().cuda()
    
    def train(self, data_this_client):
        start_time = time.time()
        trainloader = data_this_client
  
        self.model.train()
        print(f"\n-------------clinet: {self.id}-------------")        

        max_local_epochs = self.local_epochs

        for step in range(max_local_epochs):
            epoch_loss_collector = []
            for i, (x, y) in enumerate(trainloader):
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                y = y.long()

                self.optimizer.zero_grad()
                rep = self.model.base(x)
                   
                loss_deco = self.criterion_deco(rep, y) * self.mu
                
                output = self.model(x)
                loss = self.loss(output, y) + loss_deco
                
                loss.backward()
                self.optimizer.step()
                
                epoch_loss_collector.append(loss.item())
                
            epoch_loss = sum(epoch_loss_collector) / len(epoch_loss_collector)
            print('Epoch: %d Loss: %f' % (step, epoch_loss))
        
        self.train_time_cost['num_rounds'] += 1
        self.train_time_cost['total_cost'] += time.time() - start_time

    def set_parameters(self, model):
        global_w = model.state_dict()
        self.model.load_state_dict(global_w)
    




