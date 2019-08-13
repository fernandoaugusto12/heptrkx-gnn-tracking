"""
This module contains code for interacting with hit graphs.
A Graph is a namedtuple of matrices X, Ri, Ro, y.
"""

from collections import namedtuple

import numpy as np
import torch

# A Graph is a namedtuple of matrices (X, Ri, Ro, y)
# Graph = namedtuple('Graph', ['X', 'Ri', 'Ro', 'y'])
from sparse_tensor import SpTensor

Graph = namedtuple('Graph', ['X', 'spRi', 'spRo', 'y'])

SparseGraph = namedtuple('SparseGraph',
        ['X', 'Ri_rows', 'Ri_cols', 'Ro_rows', 'Ro_cols', 'y'])


def make_sparse_graph(X, Ri, Ro, y):
    Ri_rows, Ri_cols = Ri.nonzero()
    Ro_rows, Ro_cols = Ro.nonzero()
    return SparseGraph(X, Ri_rows, Ri_cols, Ro_rows, Ro_cols, y)

def graph_to_sparse(graph):
    Ri_rows, Ri_cols = graph.spRi.nonzero()
    Ro_rows, Ro_cols = graph.spRo.nonzero()
    return SparseGraph(graph.X, Ri_rows, Ri_cols, Ro_rows, Ro_cols, graph.y)

def sparse_to_graph(X, Ri_rows, Ri_cols, Ro_rows, Ro_cols, y, dtype=np.float32):
    n_nodes, n_edges = X.shape[0], Ri_rows.shape[0]
    spRi_idxs = np.stack([Ri_rows.astype(np.int64), Ri_cols.astype(np.int64)])
    # Ri_rows and Ri_cols have the same shape
    spRi_vals = np.ones((Ri_rows.shape[0],), dtype=dtype)
    spRi = (spRi_idxs,spRi_vals,n_nodes,n_edges)#SpTensor(spRi_idxs, spRi_vals, (n_nodes, n_edges))

    spRo_idxs = np.stack([Ro_rows.astype(np.int64), Ro_cols.astype(np.int64)])
    # Ro_rows and Ro_cols have the same shape
    spRo_vals = np.ones((Ro_rows.shape[0],), dtype=dtype)
    spRo = (spRo_idxs,spRo_vals,n_nodes,n_edges)#SpTensor(spRo_idxs, spRo_vals, (n_nodes, n_edges))

    if y.dtype != np.uint8:
        y = y.astype(np.uint8)

    return Graph(X, spRi, spRo, y)


def save_graph(graph, filename):
    """Write a single graph to an NPZ file archive"""
    np.savez(filename, **graph_to_sparse(graph)._asdict())


def save_graphs(graphs, filenames):
    for graph, filename in zip(graphs, filenames):
        save_graph(graph, filename)


def load_graph(filename):
    """Reade a single graph NPZ"""
    with np.load(filename) as f:
        return sparse_to_graph(**dict(f.items()))


def load_graphs(filenames, graph_type=Graph):
    return [load_graph(f, graph_type) for f in filenames]


#thanks Steve :-)
def draw_sample(X, Ri, Ro, y, out,
                cmap='bwr_r', 
                skip_false_edges=True,
                alpha_labels=False, 
                sim_list=None): 
    # Select the i/o node features for each segment    
    feats_o = X[Ro]
    feats_i = X[Ri]    
    # Prepare the figure
    import matplotlib.pyplot as plt
    fig, (ax0,ax1) = plt.subplots(1, 2, figsize=(20,12))
    cmap = plt.get_cmap(cmap)
    
    if sim_list is None:    
        # Draw the hits (r, phi, z)
        ax0.scatter(X[:,0], X[:,1], c='k')
        ax1.scatter(X[:,0], X[:,2], c='k')
    else:        
        ax0.scatter(X[:,0], X[:,1], c='k')
        ax1.scatter(X[:,0], X[:,2], c='k')
        ax0.scatter(X[sim_list,0], X[sim_list,2], c='b')
        ax1.scatter(X[sim_list,1], X[sim_list,2], c='b')
    
    # Draw the segments
    for j in range(y.shape[0]):
        if not y[j] and skip_false_edges: continue
        if alpha_labels:
            seg_args = dict(c='k', alpha=float(y[j]))
        else:
            seg_args = dict(c=cmap(float(y[j])))
        ax0.plot([feats_o[j,0], feats_i[j,0]],
                 [feats_o[j,1], feats_i[j,1]], '-', **seg_args)
        ax1.plot([feats_o[j,0], feats_i[j,0]],
                 [feats_o[j,2], feats_i[j,2]], '-', **seg_args)
        
     
    if out is not None:
        for j in range(out.shape[0]):
            #if out[j]<0.5 : continue
            seg_args = dict(c='r',alpha=out[j])        
            ax0.plot([feats_o[j,0], feats_i[j,0]],
                     [feats_o[j,1], feats_i[j,1]], '-', **seg_args)
            ax1.plot([feats_o[j,0], feats_i[j,0]],
                     [feats_o[j,2], feats_i[j,2]], '-', **seg_args)
        
        
    # Adjust axes
    ax0.set_xlabel('$r$ [m]')
    ax1.set_xlabel('$r$ [m]')
    ax0.set_ylabel('$\phi$ [pi/8]')
    ax1.set_ylabel('$z$ [m]')
    plt.tight_layout()
    return fig;
