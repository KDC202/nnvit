import torch
import argparse
import torch
import numpy as np
import os, sys
import data
import time
import util



def main():
    parser = argparse.ArgumentParser(description="FeCT training and evaluation")
    parser.add_argument("--dataset_name", type=str,required=True)
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=1000)
    parser.add_argument("--device", type=str, default='1')
    
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.device
    # os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    # device = torch.device("cuda")
    # device_ids = list(range(torch.cuda.device_count()))
    
    experiment_name = '{}_{}.{:0>2d}.{:0>2d}_{:0>2d}:{:0>2d}:{:0>2d}'.format(args.dataset_name, *time.localtime()[:6])
    print("experiment:", experiment_name)
    
    dataset = data.Dataset(args.dataset_name,data_path=args.data_path,normalize=True)
    
    test_base = dataset.test_vectors
    train_base = dataset.train_vectors
    # print(train_base)
    
    
    train_dist = util.get_dist(train_base)
    test_dist = util.get_dist(test_base)
    
    test_dataset = util.MyDataset(test_base)
    train_dataset = util.MyDataset(train_base)
    
    
    train_loader = util.data_loder(train_dataset,args.batch_size,shuffle=True,num_workers=20)
    print(train_loader)
    
    
    print(train_dist.shape)
    
    # print(test_base.shape)

if __name__ == '__main__':
    main()