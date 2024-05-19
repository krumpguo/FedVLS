import torch
import torch.nn as nn
import numpy as np
import time
import os
import copy
import torch.nn.functional as F
from torch.nn.utils import parameters_to_vector

class clientED(object):
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
        self.weight_decay = 0.0
        
        self.train_time_cost = {'num_rounds': 0, 'total_cost': 0.0}
        self.send_time_cost = {'num_rounds': 0, 'total_cost': 0.0}
        
        self.num_per_class = None
        self.prior = None
        self.no_exist_label = None
        self.exist_label = None
        self.teacher_model = None
        self.beta_y = None
        self.beta = args.beta
        self.lamda = args.lamda
        self.global_epoch = 0

        self.loss = nn.CrossEntropyLoss()
        
    def train(self, data_this_client, round):
        self.optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, self.model.parameters()), lr=self.learning_rate, momentum=0.9, weight_decay=self.weight_decay)
        start_time = time.time()
        trainloader = data_this_client
        self.global_epoch = round
        if self.num_per_class == None :
            self.num_per_class = [0. for i in range(self.num_classes)]
            for _, labels in trainloader:
             # labels是一个包含标签的Tensor
                for label in labels:
                    label = label.item()  # 将标签转换为普通的Python整数
                    self.num_per_class[label] += 1         
            self.num_per_class = torch.Tensor(self.num_per_class).float().cuda()
            self.prior = self.num_per_class / self.num_per_class.sum()
            self.beta_y = (self.num_per_class / self.num_per_class.max()).pow(0.05)
            self.no_exist_label = torch.where(self.prior == 0)[0]
            self.exist_label = torch.where(self.prior != 0)[0]
            self.no_exist_label = torch.Tensor(self.no_exist_label).int().cuda()
            self.exist_label = torch.Tensor(self.exist_label).int().cuda()
            self.exist_prior = self.prior[self.exist_label]
        
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
                output = self.model(x)                
                LADE_loss = self.new_LADEloss(output, y)
                Prior_CELoss = self.PriorCELoss(output, y)
                
                teach_output = self.teacher_model(x).detach()
                NED_loss = self.NEDloss(output, teach_output, self.no_exist_label, self.exist_label, self.exist_prior, y)  
                loss = Prior_CELoss + LADE_loss * 0.005 + NED_loss * self.lamda
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
        self.teacher_model = copy.deepcopy(model)      
    
    def new_LADEloss(self, y_pred, target): 
        cls_weight = (self.num_per_class.float() / torch.sum(self.num_per_class.float())).cuda()
        balanced_prior = torch.tensor(1. / self.num_classes).float().cuda()
        pred_spread = (y_pred - torch.log(self.prior + 1e-9) + torch.log(balanced_prior + 1e-9)).T * (target != torch.arange(0, self.num_classes).view(-1, 1).type_as(target))# C x N
        N = pred_spread.size(-1)
        second_term = torch.logsumexp(pred_spread, -1) - np.log(N)
        loss = - torch.sum( (- second_term ) * cls_weight)
        return loss
    
    def PriorCELoss(self, output, y):
        logits = output + torch.log(self.prior + 1e-9)
        loss = self.loss(logits, y)
        return loss

    def NEDloss(self, output, teach_output, no_exist_label, exist_label, exist_prior, y):
        output_no_exist_log_soft = torch.nn.functional.log_softmax(output[:,no_exist_label], dim=None, _stacklevel=3, dtype=None)
        output_no_exist_teacher_soft = torch.nn.functional.softmax(teach_output[:,no_exist_label], dim=None, _stacklevel=3, dtype=None)
        kl = nn.KLDivLoss(reduction='batchmean')
        no_exist_label_loss = kl(output_no_exist_log_soft, output_no_exist_teacher_soft)
        return no_exist_label_loss    
            
    def LADEloss(self, y_pred, target, remine_lambda=0.1):
        cls_weight = (self.num_per_class.float() / torch.sum(self.num_per_class.float())).cuda()
        balanced_prior = torch.tensor(1. / self.num_classes).float().cuda()
        # my = self.prior[target]
        per_cls_pred_spread = y_pred.T * (target == torch.arange(0, self.num_classes).view(-1, 1).type_as(target))  # C x N
        pred_spread = (y_pred - torch.log(self.prior + 1e-9) + torch.log(balanced_prior + 1e-9)).T # C x N

        num_samples_per_cls = torch.sum(target == torch.arange(0, self.num_classes).view(-1, 1).type_as(target), -1).float()  # C
        estim_loss, first_term, second_term = self.remine_lower_bound(per_cls_pred_spread, pred_spread, num_samples_per_cls, remine_lambda)
        
        loss = -torch.sum(second_term *cls_weight)
        return loss
    
    def remine_lower_bound(self, x_p, x_q, num_samples_per_cls, remine_lambda):
        loss, first_term, second_term = self.mine_lower_bound(x_p, x_q, num_samples_per_cls)
        reg = (second_term ** 2) * remine_lambda
        return loss - reg, first_term, - second_term 
    
    def mine_lower_bound(self, x_p, x_q, num_samples_per_cls):
        N = x_p.size(-1)
        first_term = torch.sum(x_p, -1) / (num_samples_per_cls + 1e-8)
        second_term = torch.logsumexp(x_q, -1) - np.log(N)

        return first_term - second_term, first_term, second_term
