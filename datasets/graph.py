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

def graph_to_sparse(graph):
    Ri_rows, Ri_cols = graph.Ri.nonzero()
    Ro_rows, Ro_cols = graph.Ro.nonzero()
    return dict(X=graph.X, y=graph.y,
                Ri_rows=Ri_rows, Ri_cols=Ri_cols,
                Ro_rows=Ro_rows, Ro_cols=Ro_cols)

def sparse_to_graph(X, Ri_rows, Ri_cols, Ro_rows, Ro_cols, y, simmatched, dtype=np.float32):
    n_nodes, n_edges = X.shape[0], Ri_rows.shape[0]
    spRi_idxs = np.stack([Ri_rows.astype(np.int64), Ri_cols.astype(np.int64)])
    # Ri_rows and Ri_cols have the same shape
    spRi_vals = np.ones((Ri_rows.shape[0],), dtype=dtype)
    spRi = (spRi_idxs,spRi_vals,n_nodes,n_edges)#SpTensor(spRi_idxs, spRi_vals, (n_nodes, n_edges))

    spRo_idxs = np.stack([Ro_rows.astype(np.int64), Ro_cols.astype(np.int64)])
    # Ro_rows and Ro_cols have the same shape
    spRo_vals = np.ones((Ro_rows.shape[0],), dtype=dtype)
    spRo = (spRi_idxs,spRi_vals,n_nodes,n_edges)#SpTensor(spRo_idxs, spRo_vals, (n_nodes, n_edges))

    if y.dtype != np.uint8:
        y = y.astype(np.uint8)

    return Graph(X, spRi, spRo, y)


def save_graph(graph, filename):
    """Write a single graph to an NPZ file archive"""
    np.savez(filename, **graph_to_sparse(graph))


def save_graphs(graphs, filenames):
    for graph, filename in zip(graphs, filenames):
        save_graph(graph, filename)


def load_graph(filename):
    """Reade a single graph NPZ"""
    with np.load(filename) as f:
        return sparse_to_graph(**dict(f.items()))


def load_graphs(filenames, graph_type=Graph):
    return [load_graph(f, graph_type) for f in filenames]
