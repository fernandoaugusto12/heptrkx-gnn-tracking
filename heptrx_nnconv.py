import os.path as osp
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets.hitgraphs import HitGraphDatasetG
import torch_geometric.transforms as T
from torch_geometric.data import DataLoader
from torch_geometric.utils import normalized_cut
from torch_geometric.nn import (NNConv, graclus, max_pool, max_pool_x,
                                global_mean_pool)

directed = False
sig_weight = 1.0
bkg_weight = 0.15
batch_size = 16

path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'training_data', 'single_mu')
full_dataset = HitGraphDatasetG(path, directed=directed)
fulllen = len(full_dataset)
tv_frac = 0.10
tv_num = math.ceil(fulllen*tv_frac)
splits = np.cumsum([fulllen-2*tv_num,tv_num,tv_num])

train_dataset = torch.utils.data.Subset(full_dataset,np.arange(start=0,stop=splits[0]))
test_dataset = torch.utils.data.Subset(full_dataset,np.arange(start=splits[0],stop=splits[1]))
valid_dataset = torch.utils.data.Subset(full_dataset,np.arange(start=splits[1],stop=splits[2]))
train_loader = DataLoader(train_dataset, batch_size=batch_size, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)

d = full_dataset
num_features = d.num_features
num_classes = d[0].y.max().item() + 1 if d[0].y.dim() == 1 else d[0].y.size(1)

from models.gnn_geometric import GNNSegmentClassifierG as Net

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Net(input_dim=5,hidden_dim=64,n_iters=6).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr = 0.01)

def train(epoch):
    model.train()

    if epoch == 30:
        for param_group in optimizer.param_groups:
            param_group['lr'] = 0.001
    
    #if epoch == 20:
    #    for param_group in optimizer.param_groups:
    #        param_group['lr'] = 0.0001
    
    sum_loss = 0.
    for i,data in enumerate(train_loader):
        data = data.to(device)
        batch_target = data.y
        batch_weights_real = batch_target*sig_weight
        batch_weights_fake = (1 - batch_target)*bkg_weight
        batch_weights = batch_weights_real + batch_weights_fake
        optimizer.zero_grad()
        batch_output = model(data)
        batch_loss = F.binary_cross_entropy(batch_output, batch_target, weight=batch_weights)
        batch_loss.backward()
        sum_loss += batch_loss.item()
        optimizer.step()

    return sum_loss/(i+1)


@torch.no_grad()
def test():
    model.eval()
    correct = 0

    sum_loss = 0
    sum_correct = 0
    sum_truepos = 0
    sum_trueneg = 0
    sum_falsepos = 0
    sum_falseneg = 0
    sum_true = 0
    sum_false = 0
    sum_total = 0
    for i,data in enumerate(test_loader):
        data = data.to(device)
        batch_target = data.y
        batch_output = model(data)
        sum_loss += F.binary_cross_entropy(batch_output, batch_target).item()
        matches = ((batch_output > 0.5) == (batch_target > 0.5))
        true_pos = ((batch_output > 0.5) & (batch_target > 0.5))
        true_neg = ((batch_output < 0.5) & (batch_target < 0.5))
        false_pos = ((batch_output > 0.5) & (batch_target < 0.5))
        false_neg = ((batch_output < 0.5) & (batch_target > 0.5))
        sum_truepos += true_pos.sum().item()
        sum_trueneg += true_neg.sum().item()
        sum_falsepos += false_pos.sum().item()
        sum_falseneg += false_neg.sum().item()
        sum_correct += matches.sum().item()
        sum_true += batch_target.sum().item()
        sum_false += (batch_target < 0.5).sum().item()
        sum_total += matches.numel()

    print('scor', sum_correct,
          'stru', sum_true,
          'stp', sum_truepos,
          'stn', sum_trueneg,
          'sfp', sum_falsepos,
          'sfn', sum_falseneg,
          'stot', sum_total)
    return sum_correct / sum_total, sum_truepos/sum_true, sum_falsepos / sum_false, sum_falseneg / sum_true, sum_truepos/(sum_truepos+sum_falsepos)


for epoch in range(1, 60):
    epoch_loss = train(epoch)
    test_acc, test_eff, test_fp, test_fn, test_pur = test()
    print('Epoch: {:02d}, Epoch Loss: {:.4f}, Eff.: {:.4f}, FalsePos: {:.4f}, FalseNeg: {:.4f}, Purity: {:,.4f}'.format(epoch, epoch_loss, test_eff, test_fp, test_fn, test_pur))
