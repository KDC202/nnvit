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