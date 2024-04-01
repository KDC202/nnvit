import numpy as np
import torch
import random
import os.path as osp

class Dataset:
    def __init__(self, dataset, data_pase='./data', normalize=False, random_state=50, **kwargs):
        np.random.seed(random_state)
        torch.manual_seed(random_state)
        random.seed(random_state)

def fetch_SIFT_small(path, train_size=None, test_size=None):
    base_path = osp      
        
DATASETS = {
    'sift_small': fetch_SIFT_small
}