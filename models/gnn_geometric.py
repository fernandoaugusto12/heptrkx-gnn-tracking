"""
This module implements the PyTorch modules that define the
message-passing graph neural networks for hit or segment classification.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Sequential, Linear
from torch_geometric.nn import NNConv
from torch_scatter import scatter_add

class EdgeNetworkG(nn.Module):
    def __init__(self, input_dim, hidden_dim=8, hidden_activation=nn.Tanh):
        super(EdgeNetworkG, self).__init__()
        self.edgec = nn.Sequential(
            nn.Linear(input_dim * 2, hidden_dim),
            hidden_activation(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid())
        self.sigm_out = nn.Sequential(nn.Sigmoid())
        self.network = NNConv(input_dim * 2, 1, self.edgec, aggr='add')

    def forward(self, data):
        row,col = data.edge_index.detach()
        bi = data.x[col]
        bo = data.x[row]
        # the original network constantly updates the edge network
        # the neural networks used are actually edge attributes
        #data.edge_attr = torch.cat([bo,bi],dim=-1)
        #print('EdgeNetworkG forward:',data.edge_attr.shape)
        B = torch.cat([bi,bo],dim=-1).detach()
        return self.edgec(B) #self.network(data.x, data.edge_index, data.edge_attr)

class NodeNetworkG(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_activation=nn.Tanh):
        super(NodeNetworkG, self).__init__()
        self.nodec = nn.Sequential(
            nn.Linear(input_dim * 3, output_dim),
            hidden_activation(),
            nn.Linear(output_dim, output_dim),
            hidden_activation())
        self.network = NNConv(input_dim * 3, output_dim, self.nodec, aggr='add')

    def forward(self, data):
        row,col = data.edge_index.detach()
        e = data.edge_attr.detach()
        bi = data.x[col]*e
        bo = data.x[row]*e
        mi = data.x.new_zeros(data.x.shape)
        mo = data.x.new_zeros(data.x.shape)

        mi = scatter_add(bo,col,dim=0,out=mi)
        mo = scatter_add(bi,row,dim=0,out=mo)
        
        M = torch.cat([mi,mo,data.x],dim=-1).detach()
        
        return self.nodec(M)

class GNNSegmentClassifierG(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=8, n_iters=3, hidden_activation=nn.Tanh):
        super(GNNSegmentClassifierG, self).__init__()
        self.n_iters = n_iters
        # Setup the input network
        self.input_network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            hidden_activation())
        # Setup the edge network
        self.edge_network = EdgeNetworkG(input_dim + hidden_dim, hidden_dim,
                                         hidden_activation)
        # Setup the node layers
        self.node_network = NodeNetworkG(input_dim + hidden_dim, hidden_dim,
                                         hidden_activation)

    def forward(self, data):
        """Apply forward pass of the model"""
        X = data.x
        # Apply input network to get hidden representation
        H = self.input_network(X)
        # Shortcut connect the inputs onto the hidden representation
        data.x = torch.cat([H, X], dim=-1)
        # Loop over iterations of edge and node networks
        for i in range(self.n_iters):
            # Apply edge network, update edge_attrs
            data.edge_attr = self.edge_network(data)
            # Apply node network
            H = self.node_network(data)
            # Shortcut connect the inputs onto the hidden representation
            data.x = torch.cat([H, X], dim=-1)
            # Apply final edge network
        return self.edge_network(data).squeeze(-1)
        
    
from torch_sparse import spmm
from torch_sparse import spspmm

from sparse_tensor import SpTensor

import scipy

def spBmm(listR, X):
    res = list()
    for i in range(len(listR)):
         res.append(spmm(listR[i].idxs, listR[i].vals, listR[i].shape[0], X[i]).unsqueeze(0))
    ret = torch.cat(res)
    return ret


def scaleCols(listR, sp_diag):
    ret = list()
    for i, R in enumerate(listR):
        R.idxs.requires_grad_(False)
        sp_diag[i].idxs.requires_grad_(False)
        (idxs, vals) = spspmm(R.idxs, R.vals, sp_diag[i].idxs, sp_diag[i].vals, R.shape[0], R.shape[1], sp_diag[i].shape[1])
        #idxs = idxs.clone().detach()
        ret.append(SpTensor(idxs.detach(), vals, (R.shape[0], sp_diag[i].shape[1])))
    return ret


class EdgeNetwork(nn.Module):
    """
    A module which computes weights for edges of the graph.
    For each edge, it selects the associated nodes' features
    and applies some fully-connected network layers with a final
    sigmoid activation.
    """

    def __init__(self, input_dim, hidden_dim=8, hidden_activation=nn.Tanh):
        super(EdgeNetwork, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim * 2, hidden_dim),
            hidden_activation(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid())

    # @profile
    def forward(self, X, Ri, Ro):
        # Select the features of the associated nodes
        Ri = [ri.transpose() for ri in Ri]
        Ro = [ro.transpose() for ro in Ro]
        bi = spBmm(Ri, X)
        bo = spBmm(Ro, X)

        B = torch.cat([bo, bi], dim=2)
        # Apply the network to each edge
        return self.network(B).squeeze(-1)


class NodeNetwork(nn.Module):
    """
    A module which computes new node features on the graph.
    For each node, it aggregates the neighbor node features
    (separately on the input and output side), and combines
    them with the node's previous features in a fully-connected
    network to compute the new features.
    """

    def __init__(self, input_dim, output_dim, hidden_activation=nn.Tanh):
        super(NodeNetwork, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim * 3, output_dim),
            hidden_activation(),
            nn.Linear(output_dim, output_dim),
            hidden_activation())

    # @profile
    def forward(self, X, e, Ri, Ro):
        Ri_t = [ri.transpose() for ri in Ri]
        Ro_t = [ro.transpose() for ro in Ro]

        bi = spBmm(Ri_t, X)
        bo = spBmm(Ro_t, X)

        # Creating diag matrix from e
        idxs = torch.stack((torch.arange(e.shape[1]), torch.arange(e.shape[1])))
        idxs.requires_grad_(False)
        e_diag = []
        for i in range(len(Ri)):
            if X.is_cuda:
                e_diag.append(SpTensor(idxs.to('cuda'), torch.squeeze(e, 0)[i], (e.shape[1], e.shape[1])))
            else:
                e_diag.append(SpTensor(idxs, torch.squeeze(e, 0)[i], (e.shape[1], e.shape[1])))

        print('NodeNetwork scaleCols')
        print('Rwi')
        Rwi = scaleCols(Ri, e_diag)
        print('Rwo')
        Rwo = scaleCols(Ro, e_diag)
        print('NodeNetwork Rwi/Rwo')

        mi = spBmm(Rwi, bo)
        mo = spBmm(Rwo, bi)
        
        M = torch.cat([mi, mo, X], dim=2)
        return self.network(M)


class GNNSegmentClassifier(nn.Module):
    """
    Segment classification graph neural network model.
    Consists of an input network, an edge network, and a node network.
    """

    def __init__(self, input_dim=2, hidden_dim=8, n_iters=3, hidden_activation=nn.Tanh):
        super(GNNSegmentClassifier, self).__init__()
        self.n_iters = n_iters
        # Setup the input network
        self.input_network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            hidden_activation())
        # Setup the edge network
        self.edge_network = EdgeNetwork(input_dim + hidden_dim, hidden_dim,
                                        hidden_activation)
        # Setup the node layers
        self.node_network = NodeNetwork(input_dim + hidden_dim, hidden_dim,
                                        hidden_activation)

    # @profile
    def forward(self, inputs):
        """Apply forward pass of the model"""
        X, Ri, Ro = inputs
        # Apply input network to get hidden representation
        H = self.input_network(X)
        # Shortcut connect the inputs onto the hidden representation
        H = torch.cat([H, X], dim=-1)
        # Loop over iterations of edge and node networks
        for i in range(self.n_iters):
            # Apply edge network
            e = self.edge_network(H, Ri, Ro)
            # Apply node network
            H = self.node_network(H, e, Ri, Ro)
            # Shortcut connect the inputs onto the hidden representation
            H = torch.cat([H, X], dim=-1)
        # Apply final edge network
        return self.edge_network(H, Ri, Ro)
