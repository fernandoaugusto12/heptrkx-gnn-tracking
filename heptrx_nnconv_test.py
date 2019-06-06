import os
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
import tqdm
import argparse

from models.gnn_geometric import GNNSegmentClassifierG as Net
from EdgeNet import EdgeNet

from heptrx_nnconv import test

batch_size = 1
hidden_dim = 64
n_iters = 6

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('using device %s'%device)

def main(args):
    
    directed = False
    path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'training_data', 'single_mu')
    full_dataset = HitGraphDatasetG(path, directed=directed)
    fulllen = len(full_dataset)
    tv_frac = 0.10
    tv_num = math.ceil(fulllen*tv_frac)
    splits = np.cumsum([fulllen-2*tv_num,tv_num,tv_num])
    
    test_dataset = torch.utils.data.Subset(full_dataset,np.arange(start=splits[0],stop=splits[1]))
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    test_samples = len(test_dataset)

    d = full_dataset
    num_features = d.num_features
    num_classes = d[0].y.max().item() + 1 if d[0].y.dim() == 1 else d[0].y.size(1)
    
    model = EdgeNet(input_dim=num_features,hidden_dim=hidden_dim,n_iters=n_iters).to(device)
    model_fname = args.model
    print('Model: \n%s\nParameters: %i' %
          (model, sum(p.numel()
                      for p in model.parameters())))
    print('Testing with %s samples'%test_samples)
    
    model.load_state_dict(torch.load(model_fname))

    #test_loss, test_acc, test_eff, test_fp, test_fn, test_pur = test(model, test_loader, test_samples)
    #print('Testing: Loss: {:.4f}, Eff.: {:.4f}, FalsePos: {:.4f}, FalseNeg: {:.4f}, Purity: {:,.4f}'.format(test_loss, test_eff,
    #                                                                                                        test_fp, test_fn, test_pur))


    t = tqdm.tqdm(enumerate(test_loader),total=test_samples/batch_size)
    for i,data in t:
        data = data.to(device)
        batch_target = data.y
        batch_output = model(data)
        X = data.x
        row,col = data.edge_index
        print(X.shape)
        print(X)
        print(row.shape)
        print(row)
        print(col.shape)
        print(col)
        print(batch_target.shape)
        print(batch_target)
        print(batch_output.shape)
        print(batch_output)
        break

    
if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    # Required positional arguments
    parser.add_argument("model", help="model PyTorch state dict file [*.pth]")
    args = parser.parse_args()
    main(args)
