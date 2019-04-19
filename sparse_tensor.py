from torch_sparse import transpose

class SpTensor:
    def __init__(self, idxs, vals, shape):
        self.idxs = idxs        
        self.vals = vals
        self.shape = shape

    def to(self, device):
        return SpTensor(self.idxs.to(device), self.vals.to(device), self.shape)

    def transpose(self):
        (tidxs, tvals) = transpose(self.idxs, self.vals, self.shape[0], self.shape[1])
        return SpTensor(tidxs, tvals, (self.shape[1], self.shape[0]))
