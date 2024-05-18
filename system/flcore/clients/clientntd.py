import torch
import torch.nn as nn
import numpy as np
import time
import os
import copy
import torch.nn.functional as F
import random

class clientNTD(object):
    def __init__(self, args, id, train_samples, **kwargs):
        
        self.model = copy.deepcopy(args.model)
        self.global_model = copy.deepcopy(args.model)
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
        
        self.tau = args.tau
        self.beta = args.beta
        
        self.train_time_cost = {'num_rounds': 0, 'total_cost': 0.0}
        self.send_time_cost = {'num_rounds': 0, 'total_cost': 0.0}
        self.CE_loss = nn.CrossEntropyLoss()
        self.MSE_loss = nn.MSELoss()
        self.KL_loss = nn.KLDivLoss(reduction="batchmean")
        self.optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, self.model.parameters()), lr=self.learning_rate, momentum=0.9, weight_decay=self.weight_decay)
 
        
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
                output, global_output = self.model(x), self.global_model(x).detach()
                loss = self.NTD_loss(output, global_output, y)
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
        self.global_model = copy.deepcopy(model)
        for params in self.global_model.parameters():
            params.requires_grad = False

    def NTD_loss(self, logits, dg_logits, targets):
        ce_loss = self.CE_loss(logits, targets)
        ntd_loss = self._ntd_loss(logits, dg_logits, targets)

        loss = ce_loss + self.beta * ntd_loss

        return loss


    def _ntd_loss(self, logits, dg_logits, targets):
        """Not-tue Distillation Loss"""

        # Get smoothed local model prediction
        logits = self.refine_as_not_true(logits, targets, self.num_classes)
        pred_probs = F.log_softmax(logits / self.tau, dim=1)

        # Get smoothed global model prediction
        with torch.no_grad():
            dg_logits = self.refine_as_not_true(dg_logits, targets, self.num_classes)
            dg_probs = torch.softmax(dg_logits / self.tau, dim=1)

        loss = (self.tau ** 2) * self.KL_loss(pred_probs, dg_probs)

        return loss
    
    def refine_as_not_true(self, logits, targets, num_classes):
        nt_positions = torch.arange(0, num_classes).to(logits.device)
        nt_positions = nt_positions.repeat(logits.size(0), 1)
        nt_positions = nt_positions[nt_positions[:, :] != targets.view(-1, 1)]
        nt_positions = nt_positions.view(-1, num_classes - 1)

        logits = torch.gather(logits, 1, nt_positions)

        return logits


