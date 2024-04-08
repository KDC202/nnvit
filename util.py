import os
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

class MyDataset(Dataset):
    def __init__(self,data):
        self.data = data
    
    def __len__(self):
        return len(self.data)
    def __getitem__(self,idx):
        sample = self.data[idx]
        return idx, sample


def ivecs_read(fname):
    a = np.fromfile(fname, dtype='int32')
    d = a[0]
    return a.reshape(-1, d+1)[:, 1:].copy()

def mmap_bvecs(fname):
    x = np.memmap(fname, dtype='uint8', mode='r')
    d = x[:4].view('int32')[0]
    return x.reshape(-1, d + 4)[:, 4:]

def mmap_fvecs(fname):
    x = np.memmap(fname, dtype='int32', mode='r')
    d = x[0]
    return x.view('float32').reshape(-1, d + 1)[:, 1:]

def data_loder(data, batch_size, shuffle, num_workers):
    return DataLoader(data, batch_size=batch_size, shuffle=shuffle,num_workers=num_workers)


def get_dist(data):
    distances = torch.cdist(data,data)
    for i in range(len(distances)):
        distances[i, i] = float('inf')
    return distances

def euclidean_dist(x, y):
    dist = torch.cdist(x, y, p=2)
    return dist

def pairwise_weighted_fusion(tensor,weight):
    N, D = tensor.shape
    combined_tensor = torch.zeros(N, N, D)
    for i in range(N):
        for j in range(N):
            if i != j:  # 排除自己和自己融合
                # 计算加权平均值
                combined_tensor[i, j] = tensor[i] * weight + tensor[j] * weight
    return combined_tensor

# def create_optimizer(cfg, model, re_lr=0):
#     opt_lower = cfg.SOLVER.OPTIM
#     parameters = model.parameters()

#     if re_lr == 0:
#         init_lr = cfg.SOLVER.INIT_LR
#     else:
#         init_lr = re_lr

#     if opt_lower == 'sgd' or opt_lower == 'nesterov':
#         optimizer = optim.SGD(parameters, lr=init_lr)
#     elif opt_lower == 'adam':
#         optimizer = optim.Adam(parameters, lr=init_lr)
#     elif opt_lower == 'adamw':
#         optimizer = optim.AdamW(parameters, lr=init_lr)
#     elif opt_lower == 'adadelta':
#         optimizer = optim.Adadelta(parameters)
#     else:
#         assert False and "Invalid optimizer"
#         raise ValueError

#     return optimizer