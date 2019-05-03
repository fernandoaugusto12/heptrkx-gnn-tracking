"""
PyTorch specification for the hit graph dataset.
"""

# System imports
import os
import glob
import os.path as osp

# External imports
import numpy as np
import torch
from tqdm import tqdm
from torch.utils.data import Dataset, random_split
from torch_geometric.utils import is_undirected, to_undirected
from torch_geometric.data import Dataset as DatasetG
from torch_geometric.data import Data as DataG

# Local imports
from sparse_tensor import SpTensor
from datasets.graph import load_graph

class HitGraphDatasetG(DatasetG):
    """PyTorch geometric dataset from processed hit information"""
    
    def __init__(self, root,
                 directed = True,
                 transform = None,
                 pre_transform = None):
        self._directed = directed
        super(HitGraphDatasetG, self).__init__(root, transform, pre_transform)

    def download(self):
        pass #download from xrootd or something later
        
    @property
    def raw_file_names(self):
        if not hasattr(self,'input_files'):
            self.input_files = glob.glob(self.raw_dir+'/*.npz')        
        return [f.split('/')[-1] for f in self.input_files]
    
    @property
    def processed_file_names(self):
        if not hasattr(self,'processed_files'):
            proc_names = ['data_{}.pt'.format(idx) for idx in range(len(self.raw_file_names))]
            self.processed_files = [osp.join(self.processed_dir,name) for name in proc_names]
        return self.processed_files
    
    def __len__(self):
        return len(self.processed_file_names)
    
    def get(self, idx):
        data = torch.load(self.processed_files[idx])
        return data
    
    def process(self):
        #convert the npz into pytorch tensors and save them
        path = self.processed_dir
        for idx,raw_path in enumerate(tqdm(self.raw_paths)):
            g = load_graph(raw_path)

            Ro = g.spRo[0].T.astype(np.int64)
            Ri = g.spRi[0].T.astype(np.int64)

            i_out = Ro[Ro[:,1].argsort()][:,0]
            i_in  = Ri[Ri[:,1].argsort()][:,0]

            x = g.X.astype(np.float32)
            edge_index = np.stack((i_out,i_in))            
            y = g.y.astype(np.float32)
            outdata = DataG(x=torch.from_numpy(x),
                            edge_index=torch.from_numpy(edge_index),
                            y=torch.from_numpy(y))
            if not self._directed and not outdata.is_undirected():
                rows,cols = outdata.edge_index
                temp = torch.stack((cols,rows))
                outdata.edge_index = torch.cat([outdata.edge_index,temp],dim=-1)
                outdata.y = torch.cat([outdata.y,outdata.y])
            
            torch.save(outdata, osp.join(self.processed_dir, 'data_{}.pt'.format(idx)))

class HitGraphDataset(Dataset):
    """PyTorch dataset specification for hit graphs"""

    def __init__(self, input_dir, n_samples=None):
        input_dir = os.path.expandvars(input_dir)
        filenames = [os.path.join(input_dir, f) for f in os.listdir(input_dir)
                     if f.startswith('partGun') and f.endswith('.npz')]
        self.filenames = (
            filenames[:n_samples] if n_samples is not None else filenames)

    def __getitem__(self, index):
        return load_graph(self.filenames[index])

    def __len__(self):
        return len(self.filenames)


def get_datasets(input_dir, n_train, n_valid):
    data = HitGraphDataset(input_dir, n_train + n_valid)
    # deterministic splitting ensures all workers split the same way
    torch.manual_seed(1)
    # Split into train and validation
    train_data, valid_data = random_split(data, [n_train, n_valid])
    return train_data, valid_data


def collate_fn(graphs):
    """
    Collate function for building mini-batches from a list of hit-graphs.
    This function should be passed to the pytorch DataLoader.
    It will stack the hit graph matrices sized according to the maximum
    sizes in the batch and padded with zeros.

    This implementation could probably be optimized further.
    """
    batch_size = len(graphs)
    
    n_features = graphs[0].X.shape[1]
    n_nodes = np.array([g.X.shape[0] for g in graphs])
    n_edges = np.array([g.y.shape[0] for g in graphs])
    max_nodes = n_nodes.max()
    max_edges = n_edges.max()

    batch_X = np.zeros((batch_size, max_nodes, n_features), dtype=np.float32)
    # spRi_vals = torch.from_numpy(np.ones((Ri_rows.shape[0],), dtype=dtype))
    # spRi = SpTensor(spRi_idxs, spRi_vals, (n_nodes, n_edges))
    
    # spRo_idxs = torch.tensor([Ro_rows.astype(np.int64), Ro_cols.astype(np.int64)])
    # Ro_rows and Ro_cols have the same shape
    # spRo_vals = torch.from_numpy(np.ones((Ro_rows.shape[0],), dtype=dtype))
    # spRo = SpTensor(spRo_idxs, spRo_vals, (n_nodes, n_edges))
    batch_spRi = []
    batch_spRo = []
    batch_y = np.zeros((batch_size, max_edges), dtype=np.float32)

    for i, g in enumerate(graphs):
        batch_X[i, :n_nodes[i]] = g.X
        
        ispRi = SpTensor(torch.from_numpy(g.spRi[0]), torch.from_numpy(g.spRi[1]), (max_nodes, max_edges))
        ispRo = SpTensor(torch.from_numpy(g.spRo[0]), torch.from_numpy(g.spRo[1]), (max_nodes, max_edges))
        batch_spRi.append(ispRi)
        batch_spRo.append(ispRo)

        batch_y[i, :n_edges[i]] = g.y

    batch_X = torch.from_numpy(batch_X)
    batch_y = torch.from_numpy(batch_y)
    
    batch_inputs = [batch_X, batch_spRi, batch_spRo]
    batch_target = batch_y
    
    return batch_inputs, batch_target

    # Special handling of batch size 1
    # if batch_size == 1:
    #     g = graphs[0]
    #     # Prepend singleton batch dimension, convert inputs and target to torch
    #     batch_inputs = [torch.from_numpy(m[None]).float() for m in [g.X, g.Ri, g.Ro]]
    #     batch_target = torch.from_numpy(g.y[None]).float()
    #     return batch_inputs, batch_target
    #
    # # Get the matrix sizes in this batch
    # n_features = graphs[0].X.shape[1]
    # n_nodes = np.array([g.X.shape[0] for g in graphs])
    # n_edges = np.array([g.y.shape[0] for g in graphs])
    # max_nodes = n_nodes.max()
    # max_edges = n_edges.max()
    #
    # # Allocate the tensors for this batch
    # batch_X = np.zeros((batch_size, max_nodes, n_features), dtype=np.float32)
    # batch_Ri = np.zeros((batch_size, max_nodes, max_edges), dtype=np.float32)
    # batch_Ro = np.zeros((batch_size, max_nodes, max_edges), dtype=np.float32)
    # batch_y = np.zeros((batch_size, max_edges), dtype=np.float32)
    #
    # # Loop over samples and fill the tensors
    # for i, g in enumerate(graphs):
    #     batch_X[i, :n_nodes[i]] = g.X
    #     batch_Ri[i, :n_nodes[i], :n_edges[i]] = g.Ri
    #     batch_Ro[i, :n_nodes[i], :n_edges[i]] = g.Ro
    #     batch_y[i, :n_edges[i]] = g.y
    #
    # batch_inputs = [torch.from_numpy(bm) for bm in [batch_X, batch_Ri, batch_Ro]]
    # batch_target = torch.from_numpy(batch_y)
    #
    # return batch_inputs, batch_target
