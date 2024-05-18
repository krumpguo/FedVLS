import copy
import torch
import torch.nn as nn
import numpy as np
import time
import torch.nn.functional as F


class ContrastiveModelWrapper(nn.Module):
    def __init__(self, base_model, use_proj_head, proj_dim):
        super(ContrastiveModelWrapper, self).__init__()
        self.features = base_model
        self.repres_dim = base_model.head.in_features
        self.n_class = base_model.head.out_features
        self.use_proj_head = use_proj_head
        self.proj_dim = proj_dim

        if use_proj_head:
            self.l1 = nn.Linear(self.repres_dim, self.repres_dim // 2)
            self.l2 = nn.Linear(self.repres_dim // 2, self.proj_dim)
            self.head = nn.Linear(self.proj_dim, self.n_class)
        else:
            self.head = nn.Linear(self.repres_dim, self.n_class)

        # remove the classifier of the original model
        self.features.head = nn.Sequential()

    def forward(self, x, return_features=False):
        h = self.features(x)
        if self.use_proj_head:
            h = self.l1(h)
            h = F.relu(h)
            h = self.l2(h)
        out = self.head(h)

        if return_features:
            return out, h
        else:
            return out

class clientMOON(object):
    def __init__(self, args, id, train_samples,  **kwargs):

        self.model = copy.deepcopy(args.model)
        self.algorithm = args.algorithm
        self.dataset = args.dataset
        self.device = args.device 
        self.id = id  
        self.num_classes = args.num_classes
        self.train_samples = train_samples
        self.batch_size = args.batch_size
        self.learning_rate = args.local_learning_rate
        self.local_epochs = args.local_epochs
        self.weight_decay = args.weight_decay
        
        self.train_time_cost = {'num_rounds': 0, 'total_cost': 0.0}
        self.send_time_cost = {'num_rounds': 0, 'total_cost': 0.0}
        #MOON  
        self.tem = args.temperature
        self.mu = args.mu
        
        self.global_model = copy.deepcopy(args.model)
        self.model = ContrastiveModelWrapper(self.model, args.use_proj_head, args.proj_dim)
        self.global_model = ContrastiveModelWrapper(self.global_model, args.use_proj_head, args.proj_dim)
        self.model.cuda()
        self.global_model.cuda()
        self.old_model = copy.deepcopy(self.model)
        for param in self.old_model.parameters():
            param.requires_grad = False

        self.loss = nn.CrossEntropyLoss()
        self.cos_sim = torch.nn.CosineSimilarity(dim=-1)
        self.optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, self.model.parameters()), lr=self.learning_rate, momentum=0.9, weight_decay=self.weight_decay)
        
    def train(self, data_this_client):
        start_time = time.time()
        trainloader = data_this_client
        self.model.train()
        
        print(f"\n-------------clinet: {self.id}-------------")
        
        max_local_epochs = self.local_epochs
        
        for step in range(max_local_epochs):
            epoch_loss_collector = []
            epoch_loss1_collector = []
            epoch_loss2_collector = []
            for i, (x, y) in enumerate(trainloader):
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                y = y.long()
                
                self.optimizer.zero_grad()
                
                out, features = self.model(x, return_features=True)
                _, features_global = self.global_model(x, return_features=True)
                _, features_prev_local = self.old_model(x, return_features=True)
                features_global = features_global.detach()
                features_prev_local = features_prev_local.detach()
                loss_cls = self.loss(out, y)
                # fedmoon loss 
                pos_similarity = self.cos_sim(features, features_global).view(-1, 1)
                neg_similarity = self.cos_sim(features, features_prev_local).view(-1, 1)
                repres_sim = torch.cat([pos_similarity, neg_similarity], dim=-1)
                repres_sim /= self.tem
                contrast_label = torch.zeros(repres_sim.size(0)).long().cuda()
                loss_con = self.loss(repres_sim, contrast_label)
                
                loss = loss_cls + self.mu*loss_con
                
                loss.backward()
                self.optimizer.step()
                
                epoch_loss_collector.append(loss.item())
                epoch_loss1_collector.append(loss_cls.item())
                epoch_loss2_collector.append(loss_con.item())
        
            epoch_loss = sum(epoch_loss_collector) / len(epoch_loss_collector)
            epoch_loss1 = sum(epoch_loss1_collector) / len(epoch_loss1_collector)
            epoch_loss2 = sum(epoch_loss2_collector) / len(epoch_loss2_collector)
            print('Epoch: %d Loss: %f Loss1: %f Loss2: %f' % 
                             (step, epoch_loss, epoch_loss1, epoch_loss2))        
        
        self.old_model = copy.deepcopy(self.model)

        self.train_time_cost['num_rounds'] += 1
        self.train_time_cost['total_cost'] += time.time() - start_time

    def set_parameters(self, model):
        global_w = model.state_dict()
        self.model.load_state_dict(global_w)
        self.global_model.load_state_dict(global_w)