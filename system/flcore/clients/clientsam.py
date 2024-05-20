import torch
import torch.nn as nn
import numpy as np
import time
import os
import copy
import torch.nn.functional as F
from argparse import ArgumentParser


class clientSAM(object):
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
        
        self.rho = args.rho
        self.momentum = args.momentum
        
        # rebuild
        self.base_optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate, momentum=self.momentum, weight_decay=self.weight_decay)
        self.optimizer = ESAM(self.model.parameters(), self.base_optimizer, rho=self.rho, num_classes=self.num_classes)
        self.loss = nn.CrossEntropyLoss()
        
        self.max_norm = 5
        self.epoch_loss_collector = []
        
        self.num_per_class = None
        self.prior = None
        self.no_exist_label = None
        self.teacher_model = None
        self.beta = args.beta
        self.lamda = args.lamda
        
    def train(self, data_this_client):
        start_time = time.time()
        trainloader = data_this_client
        if self.num_per_class == None :
            self.num_per_class = [0. for i in range(self.num_classes)]
            for _, labels in trainloader:

                for label in labels:
                    label = label.item()  
                    self.num_per_class[label] += 1         
            self.num_per_class = torch.Tensor(self.num_per_class).float().cuda()
            self.prior = self.num_per_class / self.num_per_class.sum()
            self.no_exist_label = torch.where(self.prior == 0)[0]
            self.no_exist_label = torch.Tensor(self.no_exist_label).int().cuda()
            
        self.model.train()
        
        print(f"\n-------------clinet: {self.id}-------------")        

        max_local_epochs = self.local_epochs

        for step in range(max_local_epochs):
            # self.epoch_loss_collector = []
            for i, (x, y) in enumerate(trainloader):
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                y = y.long()
                
                # first forward-backward step
                
                self.optimizer.paras = [x, y, self.model, self.teacher_model, self.num_per_class, self.prior, self.no_exist_label, self.beta, self.lamda]
                # self.optimizer.paras = [x, y, self.loss, self.model]
                self.optimizer.step()

                # Clip gradients to prevent exploding
                torch.nn.utils.clip_grad_norm_(parameters=self.model.parameters(), max_norm=self.max_norm) 
                
                self.base_optimizer.step()
                
                # epoch_loss_collector.append(loss.item())
                
            # epoch_loss = sum(epoch_loss_collector) / len(epoch_loss_collector)
            # print('Epoch: %d Loss: %f' % (step, epoch_loss))

        self.train_time_cost['num_rounds'] += 1
        self.train_time_cost['total_cost'] += time.time() - start_time

    def set_parameters(self, model):
        global_w = model.state_dict()
        self.model.load_state_dict(global_w)
        self.teacher_model = copy.deepcopy(model)

class ESAM(torch.optim.Optimizer):
    def __init__(self, params, base_optimizer, rho, num_classes, adaptive=False, **kwargs):
        assert rho >= 0.0, f"Invalid perturbation rate, should be non-negative: {rho}"
        self.max_norm = 10

        defaults = dict(rho=rho, adaptive=adaptive, **kwargs)
        super(ESAM, self).__init__(params, defaults)
        self.num_classes = num_classes
        self.base_optimizer = base_optimizer
        self.param_groups = self.base_optimizer.param_groups
        self.loss = nn.CrossEntropyLoss()
        for group in self.param_groups:
            group["rho"] = rho
            group["adaptive"] = adaptive
        self.paras = None

    @torch.no_grad()
    def first_step(self):
        #first order sum 
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-7)
            for p in group["params"]:
                p.requires_grad = True 
                if p.grad is None: 
                    continue
                # original SAM 
                # e_w = p.grad * scale.to(p)
                # ASAM 
                e_w = (torch.pow(p, 2) if group["adaptive"] else 1.0) * p.grad * scale.to(p)
                # climb to the local maximum "w + e(w)"
                p.add_(e_w * 1)  
                self.state[p]["e_w"] = e_w
                

    @torch.no_grad()
    def second_step(self):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None or not self.state[p]:
                    continue
                # go back to "w" from "w + e(w)"
                p.sub_(self.state[p]["e_w"])  
                self.state[p]["e_w"] = 0


    def step(self, alpha=1.):
        # model.require_backward_grad_sync = False
        # model.require_forward_param_sync = True
        
        inputs, labels, model, teacher_model, num_per_class, prior, no_exist_label, beta, lamda = self.paras

        output = model(inputs)
        teach_output = teacher_model(inputs).detach()
        LADE_loss = self.new_LADEloss(output, labels, num_per_class, prior, remine_lambda=0.1)
        Prior_CELoss = self.PriorCELoss(output, labels, prior)
        NED_loss = self.NEDloss(output, teach_output, no_exist_label)
        loss = Prior_CELoss + LADE_loss * beta + NED_loss * lamda
        self.zero_grad()
        loss.backward()
        
        self.first_step()
        # model.require_backward_grad_sync = True
        # model.require_forward_param_sync = False
        
        output = model(inputs)
        teach_output = teacher_model(inputs).detach()
        LADE_loss = self.new_LADEloss(output, labels, num_per_class, prior, remine_lambda=0.1)
        Prior_CELoss = self.PriorCELoss(output, labels, prior)
        NED_loss = self.NEDloss(output, teach_output, no_exist_label)
        loss = Prior_CELoss + LADE_loss * beta + NED_loss * lamda
        self.zero_grad()
        loss.backward()
        self.second_step()
        
        
    def _grad_norm(self):
        norm = torch.norm(torch.stack([
                        # original SAM
                        # p.grad.norm(p=2).to(shared_device)
                        # ASAM 
                        ((torch.abs(p) if group["adaptive"] else 1.0) * p.grad).norm(p=2)
                        for group in self.param_groups for p in group["params"]
                        if p.grad is not None]), p=2)
        return norm
    
    def new_LADEloss(self, y_pred, target, num_per_class, prior, remine_lambda=0.1,):
        cls_weight = (num_per_class.float() / torch.sum(num_per_class.float())).cuda()
        balanced_prior = torch.tensor(1. / self.num_classes).float().cuda()
        
        pred_spread = (y_pred - torch.log(prior + 1e-9) + torch.log(balanced_prior + 1e-9)).T * (target != torch.arange(0, self.num_classes).view(-1, 1).type_as(target))# C x N

        N = pred_spread.size(-1)
        second_term = torch.logsumexp(pred_spread, -1) - np.log(N)

        loss = - torch.sum( (- second_term ) * cls_weight)
        return loss
    
    def PriorCELoss(self, output, y, prior):
        logits = output + torch.log(prior + 1e-9)
        loss = self.loss(logits, y)
        return loss
    
    def NEDloss(self, output, teach_output, no_exist_label):
        output_log_soft = torch.nn.functional.log_softmax(output[:,no_exist_label], dim=None, _stacklevel=3, dtype=None)
        output_teacher_soft = torch.nn.functional.softmax(teach_output[:,no_exist_label], dim=None, _stacklevel=3, dtype=None)
        kl = nn.KLDivLoss(reduction='batchmean')
        loss = kl(output_log_soft, output_teacher_soft)  
        return loss


