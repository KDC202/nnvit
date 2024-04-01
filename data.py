import numpy as np
import torch
import random
import os.path as osp
import warnings
from util import ivecs_read, mmap_bvecs, mmap_fvecs

class Dataset:
    def __init__(self, dataset, data_path='./data', normalize=False, random_state=50, **kwargs):
        np.random.seed(random_state)
        torch.manual_seed(random_state)
        random.seed(random_state)
        if dataset in DATASETS:
            data_dict = DATASETS[dataset](osp.join(data_path,dataset), **kwargs)
        else:
            print('error')
        self.train_vectors = torch.tensor(data_dict['train_vectors']).type(torch.float32)
        self.test_vectors = torch.tensor(data_dict['test_vectors']).type(torch.float32)
        self.query_vectors = torch.tensor(data_dict['query_vectors']).type(torch.float32)
        self.ground_vectors = data_dict['ground_vectors']
        assert self.train_vectors.shape[1] == self.test_vectors.shape[1] == self.query_vectors.shape[1]
        self.vector_dim = self.train_vectors.shape[1]
        
        print(self.train_vectors.shape, self.test_vectors.shape, self.query_vectors.shape)
        
        mean_norm = self.train_vectors.norm(p=2, dim=-1).mean().item()
        if normalize:
            self.train_vectors = self.train_vectors / mean_norm
            self.test_vectors = self.test_vectors / mean_norm
            self.query_vectors = self.query_vectors / mean_norm
        else:
            if mean_norm < 0.1 or mean_norm > 10.0:
                warnings.warn("Mean train_vectors norm is {}, consider normalizing")

def fetch_SIFT_small(path, train_size=None, test_size=None):
    base_path = osp.join(path, 'sift_base.fvecs')
    learn_path = osp.join(path, 'sift_learn.fvecs')
    query_path = osp.join(path, 'sift_query.fvecs')
    ground_path = osp.join(path, 'sift_groundtruth.ivecs')
    
    return dict(
        train_vectors=mmap_fvecs(learn_path)[:train_size],
        test_vectors=mmap_fvecs(base_path)[:test_size],
        query_vectors=mmap_fvecs(query_path),
        ground_vectors=ivecs_read(ground_path)
    )
        
DATASETS = {
    'sift_small': fetch_SIFT_small
}



if __name__ == '__main__':
    dataset_name = 'sift_small'
    data_path = '/home/sfy/study/data'
    dataset = Dataset(dataset_name, data_path=data_path, normalize=True)
    print(dataset.ground_vectors)