import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SGConv


class Encoder(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, activation,
                 base_model=GCNConv, k: int = 2):
        super(Encoder, self).__init__()
        self.base_model = base_model

        assert k >= 2
        self.k = k
        self.conv = [base_model(in_channels, 2 * out_channels, cached=True)]
        for _ in range(1, k-1):
            self.conv.append(base_model(2 * out_channels, 2 * out_channels, cached=True))
        self.conv.append(base_model(2 * out_channels, out_channels))
        self.conv = nn.ModuleList(self.conv)

        self.activation = activation

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor):
        for i in range(self.k):
            x = self.activation(self.conv[i](x, edge_index))
        return x


class Model(nn.Module):
    def __init__(self, encoder: Encoder, num_hidden: int, num_proj_hidden: int,
                 tau: float = 0.5):
        super(Model, self).__init__()
        self.encoder: Encoder = encoder
        self.tau: float = tau

        self.fc1 = torch.nn.Linear(num_hidden, num_proj_hidden)
        self.fc2 = torch.nn.Linear(num_proj_hidden, num_hidden)

    def forward(self, x: torch.Tensor,
                edge_index: torch.Tensor) -> torch.Tensor:
        return self.encoder(x, edge_index)

    def projection(self, z: torch.Tensor) -> torch.Tensor:
        z = F.elu(self.fc1(z))
        return self.fc2(z)

    def sim(self, z1: torch.Tensor, z2: torch.Tensor):
        z1 = F.normalize(z1)
        z2 = F.normalize(z2)
        return torch.mm(z1, z2.t())

    def semi_loss(self, z1: torch.Tensor, z2: torch.Tensor):
        f = lambda x: torch.exp(x / self.tau)
        refl_sim = f(self.sim(z1, z1))
        between_sim = f(self.sim(z1, z2))

        return -torch.log(
            between_sim.diag()
            / (refl_sim.sum(1) + between_sim.sum(1) - refl_sim.diag()))

    def batched_semi_loss(self, z1: torch.Tensor, z2: torch.Tensor,
                          batch_size: int):
        # Space complexity: O(BN) (semi_loss: O(N^2))
        device = z1.device
        num_nodes = z1.size(0)
        num_batches = (num_nodes - 1) // batch_size + 1
        f = lambda x: torch.exp(x / self.tau)
        indices = torch.arange(0, num_nodes).to(device)
        losses = []

        for i in range(num_batches):
            mask = indices[i * batch_size:(i + 1) * batch_size]
            refl_sim = f(self.sim(z1[mask], z1))  # [B, N]
            between_sim = f(self.sim(z1[mask], z2))  # [B, N]

            losses.append(-torch.log(
                between_sim[:, i * batch_size:(i + 1) * batch_size].diag()
                / (refl_sim.sum(1) + between_sim.sum(1)
                   - refl_sim[:, i * batch_size:(i + 1) * batch_size].diag())))

        return torch.cat(losses)

    def loss(self, z1: torch.Tensor, z2: torch.Tensor,
             mean: bool = True, batch_size: int = 0):
        h1 = self.projection(z1)
        h2 = self.projection(z2)

        if batch_size == 0:
            l1 = self.semi_loss(h1, h2)
            l2 = self.semi_loss(h2, h1)
        else:
            l1 = self.batched_semi_loss(h1, h2, batch_size)
            l2 = self.batched_semi_loss(h2, h1, batch_size)

        ret = (l1 + l2) * 0.5
        ret = ret.mean() if mean else ret.sum()

        return ret
    
    
def hn_scl(syn_samples, feats_1, targets, num_classes, n_pos, n_neg):
        
        device = (torch.device('cuda')
                      if feats_1.is_cuda
                      else torch.device('cpu'))

        syn_samples = torch.cat(torch.unbind(syn_samples, dim=0), dim=0)
        feats = feats_1
        pos_idx = torch.zeros(num_classes, n_pos).long().to(device)
        neg_idx = torch.zeros(num_classes, n_neg).long().to(device)
        q_cls = n_pos + n_neg
        for i in range(num_classes):
            pos_idx[i,:] = torch.arange(n_pos) + i*q_cls
            neg_idx[i,:] = torch.arange(n_pos, q_cls) + i*q_cls

        sim_con_queue = torch.einsum('ik,jk->ij',[feats, syn_samples])
        labels = targets

        sim_con_pos = torch.gather(sim_con_queue, 1, pos_idx[labels, :])
        sim_con_neg = torch.gather(sim_con_queue, 1, neg_idx[labels, :])

        sim_con_batch = feats @ feats.T
        mask = torch.ones_like(sim_con_batch).scatter_(1, torch.arange(len(feats)).unsqueeze(1).to(device), 0.)
        sim_con_batch = sim_con_batch[mask.bool()].view(sim_con_batch.shape[0], sim_con_batch.shape[1] - 1)
        sim_con = torch.cat([sim_con_batch, sim_con_pos, sim_con_neg], dim=1)

        labels_con_batch = torch.eq(labels[:, None], labels[None, :]).float()
        labels_con_batch = labels_con_batch[mask.bool()].view(labels_con_batch.shape[0], labels_con_batch.shape[1] - 1)
        labels_con = torch.cat([labels_con_batch, torch.ones_like(sim_con_pos), torch.zeros_like(sim_con_neg)], dim=1)

        return sim_con, labels_con


def drop_feature(x, drop_prob):
    drop_mask = torch.empty(
        (x.size(1), ),
        dtype=torch.float32,
        device=x.device).uniform_(0, 1) < drop_prob
    x = x.clone()
    x[:, drop_mask] = 0

    return x


class Criteria(nn.Module):
    def __init__(self, temperature=1, con_weight=1.0):
        super(Criteria, self).__init__()
        self.temperature = temperature
        #self.queue_size_per_cls = queue_size_per_cls
        self.con_weight = con_weight

        self.criterion_cls = nn.CrossEntropyLoss()#nn.NLLLoss()
        self.log_sfx = nn.LogSoftmax(dim=1)
        self.criterion_con = torch.nn.KLDivLoss(reduction='none')

    # Supervised contrastvie style
    def forward(self, sim_con, labels_con, logits_cls, labels):
        #print(sim_con.device, labels_con.device, logits_cls.device, labels.device)
        loss_cls = self.criterion_cls(logits_cls, labels)
        
        
        sim_con = self.log_sfx(sim_con / self.temperature)
        
        loss_con = self.criterion_con(sim_con, labels_con)
        loss_con = loss_con.sum(dim=1) / (labels_con.sum(dim=1) + 1e-9)
        loss_con = loss_con.mean()

#         instance_weight = self.class_weight.squeeze()[labels.squeeze()]
#         loss_con = (instance_weight * loss_con).mean()

        # Total loss
        loss = loss_cls + self.con_weight * loss_con
        
        return loss_cls, loss_con, loss 
    
    
class SupConLoss(nn.Module):
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = torch.mean(loss.view(anchor_count, batch_size))

        return loss


class Classifier(nn.Module):
    def __init__(self, n_in, n_out):
        super(Classifier, self).__init__()
        self.bn = nn.BatchNorm1d(n_in)
        self.w = nn.Linear(n_in, n_out)
    
    def forward(self, x):
        x = F.relu(self.bn(x))
        #x = F.relu(x)
        x = self.w(x)
        return x
