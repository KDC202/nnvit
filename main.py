import torch
import argparse
import torch
import numpy as np
import os, sys
import data
import time

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

def main():
    parser = argparse.ArgumentParser(description="FeCT training and evaluation")
    parser.add_argument("--dataset_name", type=str,required=True)
    parser.add_argument("--data_path", type=str, required=True)
    
    args = parser.parse_args()
    device_ids = list(range(torch.cuda.device_count()))
    experiment_name = '{}_{}.{:0>2d}.{:0>2d}_{:0>2d}:{:0>2d}:{:0>2d}'.format(args.dataset_name, *time.localtime()[:6])
    print("experiment:", experiment_name)
    
    dataset = data.Dataset(args.dataset_name,data_path=args.data_path,normalize=True)
    
    test_base = dataset.test_vectors.cuda()
    train_base = dataset.train_vectors.cuda()
    print(test_base.shape)

if __name__ == '__main__':
    main()