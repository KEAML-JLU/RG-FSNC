import argparse
import os.path as osp
import random
from time import perf_counter as t
import yaml
import json
from yaml import SafeLoader

import torch
import torch_geometric.transforms as T
import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.datasets import Planetoid, CitationFull
from torch_geometric.utils import dropout_edge
from torch_geometric.nn import GCNConv, SGConv

from collections import defaultdict
from model import Encoder, Model, drop_feature, SupConLoss, Criteria
from validation import fs_dev, fs_test
from data_split import *
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from copy import deepcopy
from math import sqrt
import warnings
warnings.filterwarnings("ignore")


def relabeling(labels, train_class, dev_class, test_class, id_by_class):
    print("Start relabeling...")
    labels = labels.tolist()
    contrast_labels = deepcopy(labels)
    masked_class = dev_class + test_class
    masked_idx = []
    for cla in masked_class:
        masked_idx.extend(id_by_class[cla])

    train_class.sort()
    train_class_map = {i: train_class.index(i) for i in train_class}

    tmp_class = len(train_class)
    for cla, idx_list in id_by_class.items():
        if cla in train_class:
            for idx in idx_list:
                contrast_labels[idx] = train_class_map[cla]
        else:
            for idx in idx_list:
                contrast_labels[idx] = tmp_class
                tmp_class += 1
    print("Relabeling finished!")
    return contrast_labels

def train(model: Model, x, contrast_labels, edge_index, optimizer):
    model.train()
    optimizer.zero_grad()
    edge_index_1 = dropout_edge(edge_index, p=drop_edge_rate_1)[0]
    edge_index_2 = dropout_edge(edge_index, p=drop_edge_rate_2)[0]
    x_1 = drop_feature(x, drop_feature_rate_1)
    x_2 = drop_feature(x, drop_feature_rate_2)
    z1 = model(x_1, edge_index_1)
    z2 = model(x_2, edge_index_2)


    if args.sup == False:
        print("unsupervised cl loss")
        loss = model.loss(z1, z2, batch_size=args.batch_size)
    else:
        print("supervised cl loss")
        loss_sup = SupConLoss()
        loss = loss_sup(torch.cat([z1.unsqueeze(1), z2.unsqueeze(1)], dim=1), contrast_labels)

    loss.backward()
    optimizer.step()

    return loss.item()


def train_eval():
    dataset, train_idx, id_by_class, train_class, dev_class, test_class = split(args.dataset) if args.dataset in ['CoraFull', 'Coauthor-CS', 'Amazon-Computer', 'Cora', 'WikiCS', 'Cora_ML'] else load_data(args.dataset)
    if args.dataset in ['CoraFull', 'Coauthor-CS', 'Cora', 'WikiCS', 'Cora_ML']:
        data = dataset[0]
    else:
        data = dataset

    device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() else 'cpu')
    print("device is ", device)
    data = data.to(device)
    
    
    contrast_labels = relabeling(data.y, train_class, dev_class, test_class, id_by_class)
    contrast_labels = torch.LongTensor(contrast_labels).to(device)


    encoder = Encoder(dataset.num_features, num_hidden, activation,
                    base_model=base_model, k=num_layers)#.to(device)
    model = Model(encoder, num_hidden, num_proj_hidden, tau).to(device)

    train_criterion = Criteria(temperature=args.temperature, con_weight=args.con_weight).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay)


    cnt_wait = 0
    best_acc = 0
    for epoch in range(1, num_epochs + 1):
        _ = train(model, data.x, contrast_labels, data.edge_index, optimizer)
        if (epoch - 1) % 10:
            final_mean, final_std = fs_dev(model, data.x, data.edge_index, data.y, setting['test_num'], id_by_class, dev_class, setting['n_way'], setting['k_shot'], setting['m_qry'])
            print("===="*20)
            print("novel_dev_acc: " + str(final_mean))
            print("novel_dev_std: " + str(final_std))
            if best_acc < final_mean:
                best_acc = final_mean
                cnt_wait = 0
                torch.save(model.state_dict(), './savepoint/'+args.dataset+'_model.pkl')
            else:
                cnt_wait += 1

        if cnt_wait == setting['patience']:
            print('Early stopping!')
            break


    print("=== Final Test ===")
    path = './savepoint/'
    model.load_state_dict(torch.load(path+args.dataset+'_model.pkl'))
    print("model load success!")
    final_mean, final_std, final_mean_after, final_std_after = fs_test(model, data.x, data.edge_index, data.y, train_idx, setting['test_num'], id_by_class, test_class, setting['n_way'], setting['k_shot'], setting['m_qry'], train_criterion, device, args)
    print("novel_test_acc: " + str(final_mean_after))
    print("novel_test_std: " + str(final_std_after))

    return final_mean, final_std, final_mean_after, final_std_after


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='Cora')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--sup', type=bool, default=False)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--way', type=int, default=5)
    parser.add_argument('--shot', type=int, default=5)
    parser.add_argument('--query', type=int, default=10)
    parser.add_argument('--test_num', type=int, default=50)
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--n_pos', type=int, default=50)
    parser.add_argument('--n_neg', type=int, default=500)
    parser.add_argument('--k_pos', type=int, default=20)
    parser.add_argument('--k_neg', type=int, default=20)
    parser.add_argument('--alpha', type=float, default=0.5)
    parser.add_argument('--ratio_h', type=float, default=1)
    parser.add_argument('--temperature', type=float, default=0.5)
    parser.add_argument('--con_weight', type=float, default=1)
    parser.add_argument('--beta', type=float, default=0.5)
    parser.add_argument('--gamma', type=float, default=1)
    parser.add_argument('--eta', type=float, default=0.1)
    parser.add_argument('--wd', type=float, default=1e-9)
    parser.add_argument('--epoch', type=int, default=50)
    parser.add_argument('--encoder', type=str, default='gcn')
    parser.add_argument('--config', type=str, default='fs_config.yaml')
    args = parser.parse_args()

    print(args)

    setting = {'n_way': args.way, 'k_shot': args.shot, 'm_qry': args.query, 'test_num': args.test_num, 'patience': args.patience}
    config = yaml.load(open(args.config), Loader=SafeLoader)[args.dataset]

    torch.manual_seed(config['seed'])
    random.seed(config['seed'])
    np.random.seed(config['seed'])
    if torch.cuda.is_available():
        torch.cuda.manual_seed(config['seed'])
        torch.cuda.manual_seed_all(config['seed'])
    

    learning_rate = config['learning_rate']
    num_hidden = config['num_hidden']
    num_proj_hidden = config['num_proj_hidden']
    activation = ({'relu': F.relu, 'prelu': nn.PReLU()})[config['activation']]
    if args.encoder == 'gcn':
        base_model = GCNConv
    elif args.encoder == 'sgc':
        base_model = SGConv
    num_layers = config['num_layers']

    drop_edge_rate_1 = config['drop_edge_rate_1']
    drop_edge_rate_2 = config['drop_edge_rate_2']
    drop_feature_rate_1 = config['drop_feature_rate_1']
    drop_feature_rate_2 = config['drop_feature_rate_2']
    tau = config['tau']
    num_epochs = config['num_epochs']
    weight_decay = config['weight_decay']

    acc_mean = []
    acc_std = []
    acc_mean_after = []
    acc_std_after = []
    for __ in range(5):
        m, s, mf, sf  = train_eval()
        acc_mean.append(m)
        acc_std.append(s)
        acc_mean_after.append(mf)
        acc_std_after.append(sf)
    print("======"*10)
    print("acc mean: " + str(np.mean(acc_mean)))
    print("acc std: " + str(np.mean(acc_std)))
    print("Final acc mean: " + str(np.mean(acc_mean_after)))
    print("Final acc std: " + str(np.mean(acc_std_after)))
    
    result=defaultdict(list)
    result[tuple([np.mean(acc_mean_after),np.mean(acc_std_after)])] = {
    'way': args.way,
    'shot': args.shot,
    'n_pos': args.n_pos,
    'n_neg': args.n_neg,
    'temperature': args.temperature,
    'beta': args.beta,
    'gamma': args.gamma,
    'eta': args.eta,
    'epoch': args.epoch,
    'weight_decay': args.wd}
    with open("./res/" + args.dataset+"_res.txt", "w+") as f:
        f.write(json.dumps({str(k): result[k] for k in result}, indent=4))
        f.close()
