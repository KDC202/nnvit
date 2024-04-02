import torch
import numpy as np


def trainer(model, optimizer, train_loader,epoch):
    nb_loss = 0
    ct_loss = 0
    sum_loss = 0
    for batch_idx, (indices, feats) in enumerate(train_loader):
        # 在这里对每个批次的数据进行处理
        # print(batch_idx)
        # print("Batch indices:", indices)
        # print("Batch data shape:", feats.shape)
        feats =feats.cuda()
        out_feats = model(feats) #(b,out_dim)
        nb_loss = 
        