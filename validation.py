import torch
import numpy as np
import pickle as pkl
import torch.nn as nn
import torch.nn.functional as F
from model import hn_scl, Classifier
from data_split import test_task_generator, sample_generator
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler
import torch.optim as optim


class DistillKL(nn.Module):
    """KL divergence for distillation"""
    def __init__(self, T):
        super(DistillKL, self).__init__()
        self.T = T

    def forward(self, y_s, y_t):
        p_s = F.log_softmax(y_s/self.T, dim=1)
        p_t = F.softmax(y_t/self.T, dim=1)
        loss = F.kl_div(p_s, p_t, reduction='sum') * (self.T**2) / y_s.shape[0]
        return loss


def fs_dev(model, x, edge_index, y, test_num, id_by_class, test_class, n_way, k_shot, m_qry):
    model.eval()
    z = model(x, edge_index)
    z = z.detach().cpu().numpy()
    scaler = MinMaxScaler()
    scaler.fit(z)
    z = scaler.transform(z)

    test_acc_all = []
    for i in range(test_num):
        test_id_support, test_id_query, test_class_selected = \
            test_task_generator(id_by_class, test_class, n_way, k_shot, m_qry)

        train_z = z[test_id_support]
        test_z = z[test_id_query]

        train_y = np.array([test_class_selected.index(i) for i in torch.squeeze(y)[test_id_support]])
        test_y = np.array([test_class_selected.index(i) for i in torch.squeeze(y)[test_id_query]])

        clf = LogisticRegression(solver='lbfgs', max_iter=1000,
                                     multi_class='auto').fit(train_z, train_y) 

        test_acc = clf.score(test_z, test_y)
        test_acc_all.append(test_acc)
        
    final_mean = np.mean(test_acc_all)
    final_std = np.std(test_acc_all)

    return final_mean, final_std
    
    
def fs_test(model, x, edge_index, y, train_idx, test_num, id_by_class, test_class, n_way, k_shot, m_qry, criteria, device, args):
    model.eval()
    z = model(x, edge_index)
    z = F.normalize(z)
    z = z.detach().cpu().numpy()

    test_acc_all = []
    test_acc_all_after = []
    for i in range(test_num):
        test_id_support, test_id_query, test_class_selected = \
            test_task_generator(id_by_class, test_class, n_way, k_shot, m_qry)

        train_z = z[test_id_support]
        test_z = z[test_id_query]

        train_y = np.array([test_class_selected.index(i) for i in torch.squeeze(y)[test_id_support]])
        test_y = np.array([test_class_selected.index(i) for i in torch.squeeze(y)[test_id_query]])

        clf = LogisticRegression(solver='lbfgs', max_iter=1000,
                                     multi_class='auto').fit(train_z, train_y) 
        
        test_acc = clf.score(test_z, test_y)
        test_acc_all.append(test_acc)
        
        if n_way != 2:
            pseudo_y = clf.decision_function(z[train_idx]) # before softmax
            pseudo_y_tensor = torch.FloatTensor(pseudo_y).to(device)
        else:
            pseudo_prob = clf.predict_proba(z[train_idx])
            log_probs = np.log(pseudo_prob)
            pseudo_y = log_probs - np.log(np.sum(np.exp(log_probs)))
            pseudo_y_tensor = torch.FloatTensor(pseudo_y).to(device)
        
        # this is used for synthetic samples
        prob = clf.predict_proba(z[train_idx])
        prob = torch.FloatTensor(prob)
        log_prob = clf.predict_log_proba(z[train_idx])
        log_prob = torch.FloatTensor(log_prob)

        entropy = -torch.sum(prob * log_prob, dim=1)
        threshold  = 1.55 if n_way == 5 else 0.67
        
        index_sel = np.where(entropy<threshold)
        prob_sel = prob[index_sel]
        prob_sel_lab = torch.argmax(prob_sel, dim=1)

        proto_feats = torch.from_numpy(np.stack([np.mean(feature, axis=0) for feature in \
        np.split(train_z, n_way, axis=0)]))
        composi_fea = torch.from_numpy(np.vstack((z[train_idx][index_sel], train_z)))
        composi_lab = torch.cat([prob_sel_lab, torch.LongTensor(train_y)], dim=0)
        ori_fea = torch.from_numpy(train_z)
        ori_lab = torch.LongTensor(train_y)

        syn_samples = []
        pos_samples = []
        neg_samples = [] # neg
        pos_lab = []
        neg_lab_1 = [] # neg
        neg_lab_2 = [] # neg
        beta_lamda_list = [] # neg
        for class_index in range(n_way):
            pos_samples_c = sample_generator(class_index=class_index, feature_1=composi_fea,
                                                      protos=proto_feats, targets=composi_lab, ratio_h=args.ratio_h,
                                                      n_gen=args.n_pos, k=args.k_pos, alpha=args.alpha, mode=1)
            beta_lamda, neg_samples_c, target_1, target_2 = sample_generator(class_index=class_index, feature_1=composi_fea,
                                                      protos=proto_feats, targets=composi_lab, ratio_h=args.ratio_h,
                                                      n_gen=args.n_neg, k=args.k_neg, alpha=args.alpha, mode=0)

            syn_samples.append(torch.cat([pos_samples_c, neg_samples_c], dim=0).unsqueeze(0))
            pos_samples.append(pos_samples_c)
            neg_samples.append(neg_samples_c)
            pos_lab.extend([class_index for i in range(args.n_pos)])
            neg_lab_1.append(target_1)
            neg_lab_2.append(target_2)
            beta_lamda_list.append(beta_lamda)

        syn_samples = torch.cat(syn_samples, dim=0)
        pos_samples = torch.cat(pos_samples, dim=0)
        neg_samples = torch.cat(neg_samples, dim=0)
        beta_lamda_list = torch.cat(beta_lamda_list, dim=0)
        neg_lab_1 = torch.cat(neg_lab_1, dim=0)
        neg_lab_2 = torch.cat(neg_lab_2, dim=0)

        sim_con, labels_con = hn_scl(syn_samples=syn_samples, feats_1=ori_fea, targets=ori_lab, 
                                 num_classes=n_way, n_pos=args.n_pos, n_neg=args.n_neg)#.to(device)
        
        #print('device is ', device)
        if device != 'cpu':
            sim_con = sim_con.to(device)
            labels_con = labels_con.to(device)
            log_prob = log_prob.to(device)
            composi_lab = composi_lab.to(device)
            
        classifier = Classifier(ori_fea.shape[1], n_way).to(device)
        kd_criterion = DistillKL(T=4).to(device)
        

        optimizer = optim.Adam([{'params':model.parameters()},
        {'params': classifier.parameters(), 'lr': 0.05}], lr=0.001, weight_decay=args.wd)

#         train_z_tensor = torch.FloatTensor(train_z).to(device)
        train_y_tensor = torch.LongTensor(train_y).to(device)
        # hard positive samples
        index = torch.randperm(len(pos_samples))
        pos_z = pos_samples[index].to(device)
        pos_y = torch.LongTensor(pos_lab).to(device)
        
        # hard negative samples
        neg_z = neg_samples.to(device)
        neg_y_1 = neg_lab_1.to(device)
        neg_y_2 = neg_lab_2.to(device)
        beta_lamda_list = beta_lamda_list.flatten().to(device)

        
        model.train()
        classifier.train()
        for i in range(args.epoch):
            z_mid = model(x, edge_index)
            train_base = z_mid[train_idx]
            logits = classifier(train_base)
            loss_kd = kd_criterion(logits, pseudo_y_tensor)
            logits_y = classifier(z_mid[test_id_support])

            loss_cls, loss_con, loss = criteria(sim_con, labels_con, logits_y, train_y_tensor)
            
            # hard positive sample
            logit_pos = classifier(pos_z)
            loss_pos = F.cross_entropy(logit_pos, pos_y)
            
            # hard negative sample
            logit_neg = classifier(neg_z)

            loss_neg = (beta_lamda_list * F.cross_entropy(logit_neg, neg_y_1, reduction='none') + (1-beta_lamda_list) * F.cross_entropy(logit_neg, neg_y_2, reduction='none')).mean()
            

            final_loss = args.beta * loss_kd + args.gamma * loss + args.eta * (loss_pos + loss_neg) #args.beta * loss_cls + args.gamma * loss_con
            optimizer.zero_grad()
            final_loss.backward()
            optimizer.step()

        
        # eval
        model.eval()
        classifier.eval()
        z_fin = model(x, edge_index)
        test_z = z_fin[test_id_query]
        test_y = torch.LongTensor(test_y).to(device)
        query_pred = classifier(test_z)

        _, query_pred = torch.max(query_pred, dim=1)
        query_correct = (query_pred == test_y).sum().item()
        query_accuracy = query_correct / test_y.size(0)
        test_acc_all_after.append(query_accuracy)
        
        
    final_mean_clf = np.mean(test_acc_all)
    final_std_clf = np.std(test_acc_all)
    final_mean_query = np.mean(test_acc_all_after)
    final_std_query = np.std(test_acc_all_after)

    return final_mean_clf, final_std_clf, final_mean_query, final_std_query
