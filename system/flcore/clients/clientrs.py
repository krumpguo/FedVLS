import torch
import torch.nn as nn
import numpy as np
import time
import os
import copy
import torch.nn.functional as F

class clientRS(object):
    def __init__(self, args, id, train_samples,  **kwargs):
        
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
        
        self.train_time_cost = {'num_rounds': 0, 'total_cost': 0.0}
        self.send_time_cost = {'num_rounds': 0, 'total_cost': 0.0}
        
        self.rs = args.restricted_strength
        
        self.loss = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, self.model.parameters()), lr=self.learning_rate, momentum=0.9, weight_decay=self.weight_decay)       
        
    def train(self, data_this_client):
        start_time = time.time()
        trainloader = data_this_client
  
        self.model.train()
        print(f"\n-------------clinet: {self.id}-------------")     
           
        # compute restrict strength for each class
        n_class = self.model.fc.out_features
        all_labels = np.empty((0,), dtype=np.int64)
        for data, targets in trainloader:
            labels = targets.numpy()  
            all_labels = np.concatenate((all_labels, labels), axis=0)
        uniq_val = np.unique(all_labels)
        class2data = self.rs * torch.ones(n_class)
        for c in uniq_val:
            class2data[c] = 1.0
        class2data = class2data.unsqueeze(dim=0).cuda()   
                       
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
                output = self.model(x)
                # apply restricted softmax
                output *= class2data
                loss = self.loss(output, y)
                
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