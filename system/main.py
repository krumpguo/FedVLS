#!/usr/bin/env python
import copy
import torch
import argparse
import os
import time
import warnings
import numpy as np
import torchvision
import logging
import time

from flcore.servers.serveravg import FedAvg
from flcore.servers.serverexe import FedEXE
from flcore.servers.servermr import FedMR
from flcore.servers.servergela import FedGELA
from flcore.servers.serverntd import FedNTD
from flcore.servers.serversam import FedSAM
from flcore.servers.serverlogitcal import FedLogitCal
from flcore.servers.serverrs import FedRS
from flcore.servers.serverexp import FedEXP
from flcore.servers.serverprox import FedProx
from flcore.servers.servermoon import MOON

from flcore.trainmodel.models import *
from flcore.trainmodel.resnetcifar import *
from flcore.trainmodel.mobilenetv2 import *

from utils.result_utils import average_data
from utils.mem_utils import MemReporter
from data.pacs_dataset import *
from data.meta_dataset import *
from data.generate_mnist import *
from dataset_utils import partition_data, get_dataloader

logger = logging.getLogger()
logger.setLevel(logging.ERROR)

warnings.simplefilter("ignore")
torch.manual_seed(10)
# yuan lai seed shi 10 

# hyper-params for Text tasks
vocab_size = 98635
max_len=200
emb_dim=32

def run(args):

    time_list = []
    reporter = MemReporter()
    model_str = args.model
    args.model_name = args.model

    for i in range(args.prev, args.times):
        print(f"\n============= Running time: {i}th =============")
        print("Creating server and clients ...")
        start = time.time()

        # Generate args.model  
        if model_str == "dnn": # non-convex
            if "mnist" in args.dataset:
                args.model = DNN(1*28*28, 100, num_classes=args.num_classes).to(args.device)
            elif "cifar10" in args.dataset:
                args.model = DNN(3*32*32, 100, num_classes=args.num_classes).to(args.device)
            else:
                args.model = DNN(60, 20, num_classes=args.num_classes).to(args.device)
        
        elif model_str == "resnet18":
            args.model = torchvision.models.resnet18(pretrained=False, num_classes=args.num_classes).to(args.device)
        
        elif model_str == "resnet32":
            args.model = resnet32(num_classes=args.num_classes).to(args.device)

        elif model_str == "mobilenetv2":
            args.model = mobilenetv2(num_classes=args.num_classes).to(args.device)           
        else:
            raise NotImplementedError

        print(args.model)

        if args.dataset == 'mnist':
            party2loaders, global_train_dl, test_dl  = generate_mnist(args.datadir, args.num_classes, args.num_clients, niid=True, balance=False, partition=args.partition, alpha=args.alpha )
        else:    
            # mapping from individual client to its local training data loader
            party2dataidx = partition_data(
            args.dataset, args.datadir, args.partition, args.num_clients, alpha=args.alpha )
            
            party2loaders = {}
            party2loaders_ds = {}
            datadistribution = np.zeros((args.num_clients, args.num_classes, 2))

            for party_id in range(args.num_clients):
                train_dl_local, _, train_ds_local, _ = get_dataloader(args, args.dataset, args.datadir,
                    args.batch_size, args.batch_size, party2dataidx[party_id])
                party2loaders[party_id] = train_dl_local
                party2loaders_ds[party_id] = train_ds_local
                for i in range(args.num_classes):
                    datadistribution[party_id][i][0] = i
                all_labels = np.empty((0,), dtype=np.int64)
                for data, targets in party2loaders[party_id]:
                # targets为每个batch标签
                    labels = targets.numpy()  
                # 拼接到all_labels数组
                    all_labels = np.concatenate((all_labels, labels), axis=0)
                uniq_val, uniq_count = np.unique(all_labels, return_counts=True)
                for j, c in enumerate(uniq_val.tolist()):
                    datadistribution[party_id][c][1] = uniq_count[j]
            
            np.set_printoptions(threshold=np.inf)
            # the data distribution of clients
            print(datadistribution)
            # these loaders are used for evaluating accuracy of global model
            global_train_dl, test_dl, _, _ = get_dataloader(args, args.dataset, args.datadir,
                                train_bs=args.batch_size, test_bs=args.batch_size)

            # select algorithm             
        if args.algorithm == "FedAvg":
            args.head = copy.deepcopy(args.model.fc)
            args.model.fc = nn.Identity()
            args.model = BaseHeadSplit(args.model, args.head)
            server = FedAvg(args, i, party2loaders, global_train_dl, test_dl)
  
        elif args.algorithm == "FedEXE":
            args.head = copy.deepcopy(args.model.fc)
            args.model.fc = nn.Identity()
            args.model = BaseHeadSplit(args.model, args.head)
            server = FedEXE(args, i, party2loaders, global_train_dl, test_dl)
            
        elif args.algorithm == "FedMR":
            args.head = copy.deepcopy(args.model.fc)
            args.model.fc = nn.Identity()
            args.model = BaseHeadSplit(args.model, args.head)
            server = FedMR(args, i, party2loaders, global_train_dl, test_dl)
        
        elif args.algorithm == "FedGELA":
            args.head = nn.Identity()
            args.model.fc = nn.Identity()
            args.model = BaseHeadSplit(args.model, args.head)
            server = FedGELA(args, i, party2loaders, global_train_dl, test_dl, party2loaders_ds)
            
        elif args.algorithm == "FedNTD":
            args.head = copy.deepcopy(args.model.fc)
            args.model.fc = nn.Identity()
            args.model = BaseHeadSplit(args.model, args.head)
            server = FedNTD(args, i, party2loaders, global_train_dl, test_dl)

        elif args.algorithm == "FedLogitCal":
            args.head = copy.deepcopy(args.model.fc)
            args.model.fc = nn.Identity()
            args.model = BaseHeadSplit(args.model, args.head)
            server = FedLogitCal(args, i, party2loaders, global_train_dl, test_dl)
        
        elif args.algorithm == "FedSAM":
            server = FedSAM(args, i, party2loaders, global_train_dl, test_dl)
            
        elif args.algorithm == "FedRS":
            server = FedRS(args, i, party2loaders, global_train_dl, test_dl)
            
        elif args.algorithm == "FedEXP":
            server = FedEXP(args, i, party2loaders, global_train_dl, test_dl)
            
        elif args.algorithm == "FedProx":
            server = FedProx(args, i, party2loaders, global_train_dl, test_dl)
            
        elif args.algorithm == "MOON":
            args.head = copy.deepcopy(args.model.fc)
            args.model.fc = nn.Identity()
            args.model = BaseHeadSplit(args.model, args.head)
            server = MOON(args, i, party2loaders, global_train_dl, test_dl)
            
        else:
            raise NotImplementedError

        server.train()

        time_list.append(time.time()-start)

    print(f"\nAverage time cost: {round(np.average(time_list), 2)}s.")

    print("All done!")

    reporter.report()


if __name__ == "__main__":
    total_start = time.time()

    parser = argparse.ArgumentParser()
    # general
    parser.add_argument('-go', "--goal", type=str, default="test", 
                        help="The goal for this experiment")
    parser.add_argument('-dev', "--device", type=str, default="cuda",
                        choices=["cpu", "cuda"])
    parser.add_argument('-did', "--device_id", type=str, default="0")
    parser.add_argument('-data', "--dataset", type=str, default="mnist")
    parser.add_argument('-nb', "--num_classes", type=int, default=10)
    parser.add_argument('-m', "--model", type=str, default="cnn")
    parser.add_argument('-lbs', "--batch_size", type=int, default=10)
    parser.add_argument('-lr', "--local_learning_rate", type=float, default=0.005,
                        help="Local learning rate")
    parser.add_argument('-ed', "--weight_decay", type=float, default=1e-5,help="weight decay during local training")
    parser.add_argument('-gr', "--global_rounds", type=int, default=2000)
    parser.add_argument('-ls', "--local_epochs", type=int, default=1, 
                        help="Multiple update steps in one local epoch.")
    parser.add_argument('-algo', "--algorithm", type=str, default="FedAvg")
    parser.add_argument('-jr', "--join_ratio", type=float, default=1.0,
                        help="Ratio of clients per round")
    parser.add_argument('-rjr', "--random_join_ratio", type=bool, default=False,
                        help="Random ratio of clients per round")
    parser.add_argument('-nc', "--num_clients", type=int, default=2,
                        help="Total number of clients")
    parser.add_argument('-t', "--times", type=int, default=1,
                        help="Running times")
    parser.add_argument('-ab', "--auto_break", type=bool, default=False)
    parser.add_argument('-dlg', "--dlg_eval", type=bool, default=False)
    parser.add_argument('-dlgg', "--dlg_gap", type=int, default=100)
    parser.add_argument('-bnpc', "--batch_num_per_client", type=int, default=2)
    parser.add_argument('-pv', "--prev", type=int, default=0,
                    help="Previous Running times")
    parser.add_argument('-dd','--datadir', type=str, required=False, default="./data/",
                        help="Data directory")

    # practical
    parser.add_argument('-tth', "--time_threthold", type=float, default=10000,
                        help="The threthold for droping slow clients")
    # pFedMe / PerAvg / FedProx / FedAMP / FedPHP /MOON / MT
    parser.add_argument('-bt', "--beta", type=float, default=0.005,
                        help="Average moving parameter for pFedMe, Second learning rate of Per-FedAvg, \
                        or L1 regularization weight of FedTransfer")
    parser.add_argument('-lam', "--lamda", type=float, default=1.0,
                        help="Regularization weight")
    parser.add_argument('-mu', "--mu", type=float, default=0.001,
                        help="Proximal rate for FedProx")

    # MOON
    parser.add_argument('-pro_d',"--proj_dim", type=int, default=256,
                            help='projection dimension of the projector')
    parser.add_argument('-tem',"--temperature", type=float, default=0.5,
                            help='the temperature parameter for contrastive loss')
    parser.add_argument('-use_prod',"--use_proj_head", type=bool, default=True,
                            help='whether to use projection head')

    #the non-iid level
    parser.add_argument('-al', "--alpha", type=float, default=1.0)
    parser.add_argument('-partition','--partition', type=str, default='noniid',
                        help='the data partitioning strategy')
    parser.add_argument('-aug', '--auto_aug', type=bool, default=True,
                        help='whether to apply auto augmentation')

    parser.add_argument('-tau', "--tau", type=float, default=0.001,
                            help='tau introduced in FedAdam paper. \
                            Essentially, this hyper-parameter provides \
                            numeric protection for second-order momentum')
    #fedasm
    parser.add_argument('-rho', "--rho", type=float, default=1.0, help="rho hyper-parameter for sam")
    #fedlogitcal
    parser.add_argument('-cal_tem',"--calibration_temp", type=float, default=0.1, help='calibration temperature')
    #fedrs
    parser.add_argument('-rs',"--restricted_strength", type=float, default=0.5,
                            help='hyper-parameter for restricted strength')
    # FedExp
    parser.add_argument('-eps',"--eps", type=float, default=1e-3,
                            help='epsilon of the FedExp algorithm')
 
    args = parser.parse_args()
    
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device_id

    if args.device == "cuda" and not torch.cuda.is_available():
        print("\ncuda is not avaiable.\n")
        args.device = "cpu"
        
    print("当前时间：", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time())))
    
    print("=" * 50)

    print("Algorithm: {}".format(args.algorithm))
    print("Local batch size: {}".format(args.batch_size))
    print("Local steps: {}".format(args.local_epochs))
    print("Local learing rate: {}".format(args.local_learning_rate))
    print("Weight decay: {}".format(args.weight_decay))
    print("Total number of clients: {}".format(args.num_clients))
    print("Clients join in each round: {}".format(args.join_ratio))
    print("Running times: {}".format(args.times))
    print("Dataset: {}".format(args.dataset))
    print("Number of classes: {}".format(args.num_classes))
    print("Backbone: {}".format(args.model))
    print("Using device: {}".format(args.device))
    print("Auto break: {}".format(args.auto_break))
    if not args.auto_break:
        print("Global rounds: {}".format(args.global_rounds))
    if args.device == "cuda":
        print("Cuda device id: {}".format(os.environ["CUDA_VISIBLE_DEVICES"]))
    print("weight_decay: {}".format(args.weight_decay))
    print("noniid level: {}".format(args.alpha))
    print("auto_aug or not : {}".format(args.auto_aug))
    if args.algorithm == "FedProx":
        print("the coefficient of prox loss : {}".format(args.mu))
    elif args.algorithm == "MOON":
        print("the coefficient of moon loss : {}".format(args.mu))
        print("the projection dimension of the projector : {}".format(args.proj_dim))
        print("the temperature parameter for contrastive loss : {}".format(args.temperature))
        print("whether to use projection head : {}".format(args.use_proj_head))
    elif args.algorithm == "FedSAM":
        print("momentum : {}".format(args.momentum))
        print("rho : {}".format(args.rho))
    elif args.algorithm == "FedLogitCal":
        print("calibration_temp : {}".format(args.calibration_temp))
    elif args.algorithm == "FedRS":
        print("restricted_strength : {}".format(args.restricted_strength))
    elif args.algorithm == "FedEXP":
        print("eps : {}".format(args.eps))

    elif args.algorithm == "FedNTD":
        print("the coefficient of NTD loss : {}".format(args.beta))
    elif args.algorithm == "FedEXE":
        print("the coefficient of NED loss : {}".format(args.lamda))

        print("the l2_gre : {}".format(args.weight_decay))
    elif args.algorithm == "FedMR":
        print("the coefficient of deco loss : {}".format(args.mu))

    print("=" * 50)

    run(args)

    current_struct_time1 = time.localtime(time.time())
    formatted_time1 = time.strftime("%Y-%m-%d %H:%M:%S", current_struct_time1)
    print("当前时间：", formatted_time1)
    

