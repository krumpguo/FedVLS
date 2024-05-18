import time
from flcore.clients.clientlogitcal import clientLogitCal
from threading import Thread
import torch.nn as nn
import torch
import copy
import numpy as np

class FedLogitCal(object):
    def __init__(self, args, times, party2loaders, global_train_dl, test_dl):
 
        self.args = args
        self.device = args.device
        self.dataset = args.dataset
        self.num_classes = args.num_classes
        self.global_rounds = args.global_rounds
        self.local_epochs = args.local_epochs
        self.batch_size = args.batch_size
        self.learning_rate = args.local_learning_rate
        self.global_model = copy.deepcopy(args.model)
        self.teacher_model = None

        self.num_clients = args.num_clients
        self.join_ratio = args.join_ratio
        self.random_join_ratio = args.random_join_ratio
        self.num_join_clients = int(self.num_clients * self.join_ratio)
        self.current_num_join_clients = self.num_join_clients
        self.algorithm = args.algorithm
        self.goal = args.goal
        self.time_threthold = args.time_threthold
        self.top_cnt = 100
        self.auto_break = args.auto_break

        self.clients = []
        self.selected_clients = []
        
        self.uploaded_weights = []
        self.uploaded_ids = []
        self.uploaded_models = []
    
        self.rs_test_acc = []
        self.rs_test_auc = []
        self.rs_train_loss = []

        self.times = times


        self.party2loaders_train = party2loaders
 
        self.party2loaders_test = test_dl
        self.set_clients(clientLogitCal, party2loaders)
        
        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print("Finished creating server and clients.")

        self.Budget = []

    def train(self):
        for round in range(self.global_rounds):
            s_t = time.time()
            self.selected_clients = self.select_clients()

            self.send_models()

            print(f"\n-------------Round number: {round}-------------")
            current_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time()))
            print(f"-------------{current_time}-------------")
                
            for client in self.selected_clients:
                client.train(self.party2loaders[client.id])
                
            self.receive_models()
            self.aggregate_parameters()

            print("\nEvaluate aggregated global model")
            test_acc, test_loss, test_acc_per_class  = self.compute_accuracy(self.global_model, self.party2loaders_test)
            print('>> Aggregated global Model Test Accuracy : %f' % test_acc)
        
            self.Budget.append(time.time() - s_t)
            print('-'*25, 'time cost', '-'*25, self.Budget[-1])

            if self.auto_break and self.check_done(acc_lss=[self.rs_test_acc], top_cnt=self.top_cnt):
                break

        print("\nBest accuracy.")
        print(max(self.rs_test_acc))
        print("\nAverage time cost per round.")
        print(sum(self.Budget[1:])/len(self.Budget[1:]))

    def compute_accuracy(self, model, dataloader):
        was_training = False
        if model.training:
            model.eval()
            was_training = True

        correct, total = 0, 0
        criterion = nn.CrossEntropyLoss()
        loss_collector = []
        with torch.no_grad():
            for batch_idx, (x, target) in enumerate(dataloader):
                x, target = x.to(self.device), target.to(dtype=torch.int64).to(self.device)
                out = model(x)

                loss = criterion(out, target)
                _, pred_label = torch.max(out.data, 1)
                loss_collector.append(loss.item())
                total += x.data.size()[0]
                correct += (pred_label == target.data).sum().item()

            avg_loss = sum(loss_collector) / len(loss_collector)

        if was_training:
            model.train()

        return correct / float(total), avg_loss
    
    def set_clients(self, clientObj, party2loaders):
        for i in range(self.num_clients):
            dataload =party2loaders[i]
            client = clientObj(self.args, 
                            id=i, 
                            train_samples=len(dataload.dataset), 
                            )
            self.clients.append(client)
            
    def select_clients(self):
        if self.random_join_ratio:
            self.current_num_join_clients = np.random.choice(range(self.num_join_clients, self.num_clients+1), 1, replace=False)[0]
        else:
            self.current_num_join_clients = self.num_join_clients
        selected_clients = list(np.random.choice(self.clients, self.current_num_join_clients, replace=False))

        return selected_clients
    
    def send_models(self):
        assert (len(self.clients) > 0)

        for client in self.selected_clients:
            start_time = time.time()
            
            client.set_parameters(self.global_model)

            client.send_time_cost['num_rounds'] += 1
            client.send_time_cost['total_cost'] += 2 * (time.time() - start_time)
            
    def receive_models(self):
        assert (len(self.selected_clients) > 0)

        self.uploaded_ids = []
        self.uploaded_weights = []
        self.uploaded_models = []
        tot_samples = 0 
        for client in self.selected_clients:
                tot_samples += client.train_samples
                
                self.uploaded_ids.append(client.id)
                self.uploaded_weights.append(client.train_samples)
                self.uploaded_models.append(client.model)
        for i, w in enumerate(self.uploaded_weights):
            self.uploaded_weights[i] = w / tot_samples
            
    def aggregate_parameters(self):
        assert (len(self.uploaded_models) > 0)
            
        global_model_w = self.global_model.state_dict()
            
        temp = True
        for w, client_model in zip(self.uploaded_weights, self.uploaded_models):
            client_model_w = client_model.state_dict()
            if temp:
                for key in client_model_w:
                    global_model_w[key] = client_model_w[key] * w
                temp = False
            else:
                for key in client_model_w:
                    global_model_w[key] += client_model_w[key] * w

        self.global_model.load_state_dict(global_model_w)