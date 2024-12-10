import torch
import torch.nn.functional as F
from ogb.nodeproppred import PygNodePropPredDataset
from torch_geometric.datasets import CoraFull, Reddit2, Coauthor, Planetoid, Amazon, EmailEUCore, Reddit, WikiCS, CitationFull
import random
import numpy as np
import torch_geometric.transforms as T
from torch_geometric.data import Data
from sklearn import preprocessing
import scipy.sparse as sp
import scipy.io as sio

class_split = {
    "CoraFull": {"train": 40, 'dev': 15, 'test': 15},  # Sufficient number of base classes
    "Coauthor-CS": {"train": 5, 'dev': 5, 'test': 5},
    "Cora": {"train": 3, 'dev': 2, 'test': 2},
    "Amazon_clothing": {"train": 40, 'dev': 17, 'test': 20},
    "WikiCS": {"train": 4, 'dev': 3, 'test': 3},
    "Cora_ML": {"train": 3, 'dev': 2, 'test': 2}
    
}

valid_num_dic = {'Amazon_clothing': 17}

def split(dataset_name):
    
    if dataset_name == 'Cora':
        dataset = Planetoid(root='./dataset/' + dataset_name, name="Cora")
        num_nodes = dataset.data.num_nodes
    elif dataset_name == 'Cora_ML':
        dataset = CitationFull(root='./dataset/' + dataset_name, name="Cora_ML")
        num_nodes = dataset.data.num_nodes
    elif dataset_name == 'WikiCS':
        dataset = WikiCS(root='./dataset/' + dataset_name)
        num_nodes = dataset.data.num_nodes
    elif dataset_name == 'Coauthor-CS':
        dataset = Coauthor(root='./dataset/' + dataset_name, name="CS")
        num_nodes = dataset.data.num_nodes
    elif dataset_name == 'CoraFull':
        dataset = CoraFull(root='./dataset/' + dataset_name)
        num_nodes = dataset.data.num_nodes
    else:
        print("Dataset not support!")
        exit(0)
    if dataset_name != 'Email':
        data = dataset.data
        class_list = [i for i in range(dataset.num_classes)]

    else:
        data = dataset
        class_list = [i for i in range(len(set(data.y.numpy())))]
    print("********" * 10)

    train_num = class_split[dataset_name]["train"]
    dev_num = class_split[dataset_name]["dev"]
    test_num = class_split[dataset_name]["test"]

    random.shuffle(class_list)
    train_class = class_list[: train_num]
    dev_class = class_list[train_num : train_num + dev_num]
    test_class = class_list[train_num + dev_num :]

    print("train_num: {}; dev_num: {}; test_num: {}".format(train_num, dev_num, test_num))

    id_by_class = {}
    for i in class_list:
        id_by_class[i] = []
    for id, cla in enumerate(torch.squeeze(data.y).tolist()):
        id_by_class[cla].append(id)

    train_idx = []
    for cla in (train_class + dev_class):
        train_idx.extend(id_by_class[cla])

    return dataset, np.array(train_idx), id_by_class, train_class, dev_class, test_class#, degree_inv


def load_data(dataset_source):
    n1s = []
    n2s = []
    datapath = "./dataset/"
    for line in open(datapath + "few_shot_data/{}_network".format(dataset_source)):
        n1, n2 = line.strip().split('\t')
        n1s.append(int(n1))
        n2s.append(int(n2))

    num_nodes = max(max(n1s),max(n2s)) + 1
    adj = sp.coo_matrix((np.ones(len(n1s)), (n1s, n2s)),
                                 shape=(num_nodes, num_nodes))

    data_train = sio.loadmat(datapath + "few_shot_data/{}_train.mat".format(dataset_source))
    train_class = list(set(data_train["Label"].reshape((1,len(data_train["Label"])))[0]))

    data_test = sio.loadmat(datapath + "few_shot_data/{}_test.mat".format(dataset_source))
    class_list_test = list(set(data_test["Label"].reshape((1,len(data_test["Label"])))[0]))

    labels = np.zeros((num_nodes,1))
    labels[data_train['Index']] = data_train["Label"]
    labels[data_test['Index']] = data_test["Label"]

    features = np.zeros((num_nodes,data_train["Attributes"].shape[1]))
    features[data_train['Index']] = data_train["Attributes"].toarray()
    features[data_test['Index']] = data_test["Attributes"].toarray()

    class_list = []
    for cla in labels:
        if cla[0] not in class_list:
            class_list.append(cla[0])  # unsorted

    id_by_class = {}
    for i in class_list:
        id_by_class[i] = []
    for id, cla in enumerate(labels):
        id_by_class[cla[0]].append(id)

    lb = preprocessing.LabelBinarizer()
    labels = lb.fit_transform(labels)

    #adj = normalize_adj(adj + sp.eye(adj.shape[0]))
    #adj = sparse_mx_to_torch_sparse_tensor(adj)#.to(device)
    features = torch.FloatTensor(features)#.to(device)
    labels = torch.LongTensor(np.where(labels)[1])#.to(device)
    
    
    edge_index = torch.LongTensor(np.vstack([n1s, n2s]))
    dataset = Data(x=features, edge_index=edge_index, y=labels)
    
    class_list_valid = random.sample(train_class, valid_num_dic[dataset_source])
    class_list_train = list(set(train_class).difference(set(class_list_valid)))
    
    train_idx = []
    for cla in (class_list_train+class_list_valid):
        train_idx.extend(id_by_class[cla])
    

    return dataset, np.array(train_idx), id_by_class, class_list_train, class_list_valid, class_list_test #adj, features, labels, class_list_train, class_list_valid, class_list_test, id_by_class
    


def test_task_generator(id_by_class, class_list, n_way, k_shot, m_query):

    # sample class indices
    class_selected = random.sample(class_list, n_way)
    id_support = []
    id_query = []
    for cla in class_selected:
        temp = random.sample(id_by_class[cla], k_shot + m_query)
        id_support.extend(temp[:k_shot])
        id_query.extend(temp[k_shot:])

    return np.array(id_support), np.array(id_query), class_selected


def sample_generator(class_index, feature_1, protos, targets, ratio_h, n_gen, k, alpha=0.5, mode=0):
    
    device = (torch.device('cuda')
                  if feature_1.is_cuda
                  else torch.device('cpu'))
    # mode == 1:pos, mode==0, neg
    
    num_classes = protos.shape[0]
    proto_targets = torch.arange(num_classes).to(device)
    # print(targets.shape, proto_targets.shape)
    labels = torch.cat([targets, proto_targets], dim=0)
    feats = torch.cat([feature_1, protos], dim=0)
    feats = F.normalize(feats, dim=1)
    protos = F.normalize(protos, dim=1)
    
    if(mode == 1):
        cls_ind = torch.where(labels == class_index)[0]
    else:
        cls_ind = torch.where(labels != class_index)[0]
        cls_lab = labels[cls_ind]
        
    cls_feat = feats[cls_ind]
    cls_sim = torch.matmul(protos[class_index], cls_feat.T)
    
    if(mode == 1):
        cls_sim = -cls_sim
    
    n_hard = min(k ,len(cls_ind))
    _, idx_hard = torch.topk(cls_sim, k=n_hard, dim=-1, sorted=False)
    
    hard_num = int(ratio_h*n_gen)

    idx_1, idx_2 = torch.randint(n_hard, size=(2, hard_num)).to(device)

    candidate_1 = cls_feat[torch.gather(idx_hard, dim=0, index=idx_1)]
    candidate_2 = cls_feat[torch.gather(idx_hard, dim=0, index=idx_2)]
    
    if mode == 0:
        target_1 = cls_lab[torch.gather(idx_hard, dim=0, index=idx_1)]
        target_2 = cls_lab[torch.gather(idx_hard, dim=0, index=idx_2)]

    beta_lamda = torch.tensor(np.random.beta(a=alpha, b=alpha, size=(len(candidate_1), 1)).astype(np.float32)).to(device)
    hard_samples = beta_lamda*candidate_1 + (1-beta_lamda)*candidate_2
    hard_samples = F.normalize(hard_samples, dim=1)
    
    easy_num = n_gen - hard_num

    eazy_idx = torch.randint(len(cls_ind), size=(1, easy_num))[0].to(device)

    # easy_samples = feats[torch.gather(torch.arange(len(cls_ind)), dim=0, index=eazy_idx)]
    easy_samples = feats[torch.gather(cls_ind, dim=0, index=eazy_idx)]
    
    # return hard_samples, easy_samples
    if mode == 1:
        return torch.cat([hard_samples, easy_samples])
    else:
        return beta_lamda, torch.cat([hard_samples, easy_samples]), target_1, target_2
