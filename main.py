import torch
import argparse
import torch
import numpy as np
import os, sys
import data
import time
import util
import torch.optim as optim
from model import build_model
from loss import Loss
from train import trainer, val


def main():
    parser = argparse.ArgumentParser(description="FeCT training and evaluation")
    parser.add_argument("--dataset_name", type=str,required=True)
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=1000)
    parser.add_argument("--num_head", type=int, default=4)
    parser.add_argument("--node_feat", type=int, default=96)
    parser.add_argument("--token_dim", type=int, default=32)
    parser.add_argument("--deep", type=int, default=2)
    parser.add_argument("--epoch", type=int, default=1000)
    parser.add_argument("--device", type=str, default='1')
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.device
    # os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    # device = torch.device("cuda")
    # device_ids = list(range(torch.cuda.device_count()))
    experiment_name = '{}_{}.{:0>2d}.{:0>2d}_{:0>2d}:{:0>2d}:{:0>2d}'.format(args.dataset_name, *time.localtime()[:6])
    print("experiment:", experiment_name)
    
    model = build_model(args)
    
    # 构建优化器
    optimizer = optim.SGD(model.parameters(),lr=0.1)
    lr_scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.998)
    
    criterion = Loss(0.5, model=model)
    
    dataset = data.Dataset(args.dataset_name,data_path=args.data_path,normalize=True)
    
    test_base = dataset.test_vectors
    train_base = dataset.train_vectors
    # print(train_base)
    
    
    # train_dist = util.get_dist(train_base)
    # test_dist = util.get_dist(test_base)
    
    test_dataset = util.MyDataset(test_base)
    train_dataset = util.MyDataset(train_base)
    
    
    train_loader = util.data_loder(train_dataset,args.batch_size,shuffle=True,num_workers=20)
    val_loader = util.data_loder(test_dataset,args.batch_size,shuffle=True,num_workers=20)
    # print(train_loader.shape)
    for epoch in range(args.epoch):
        nb_loss = trainer(optimizer=optimizer, train_loader=train_loader, model=model, epoch=epoch, criterion=criterion, dataset=train_dataset)
        if epoch % 10 == 0:
            val_loss = val(model, val_loader, criterion)
            print(val_loss)
            # writer.add_scalars('loss/loss_w', {'train_loss': train_loss_w, 'val_loss': val_loss_w}, epoch)
            # writer.add_scalars('loss/loss', {'train_loss': train_loss, 'val_loss': val_loss}, epoch)
            # writer.add_scalars('learning_rate', {'learning_rate': optimizer.param_groups[0]['lr']}, epoch)
        lr_scheduler.step()
    
    # print(train_dist.shape)
    
    # print(test_base.shape)

if __name__ == '__main__':
    main()