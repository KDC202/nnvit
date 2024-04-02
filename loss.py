import torch
import torch.nn as nn
import numpy as np
import util


# 定义三元组损失函数
def triplet_loss(anchor, positive, negative, margin=1.0):
    distance_positive = torch.norm(anchor - positive, p=2, dim=1)
    distance_negative = torch.norm(anchor - negative, p=2, dim=1)
    loss = torch.clamp(distance_positive - distance_negative + margin, min=0.0)
    return loss.mean()

# def neighbor_loss(original_distance, feat)

class loss(nn.Modukes):
    def __init__(self, dist, graph, weight):
        super(loss, self)
        self.dist = dist
        self.graph = graph
        self.weight = weight
        
    def forward(self,x,y):
        xx = util.euclidean_dist(x, x)
        yy = util.euclidean_dist(y, y)
        nb_loss = torch.sqrt(torch.pow((xx - yy), 2).clamp(min=1e-12))
        combine_y = util.pairwise_weighted_fusion(y, self.weight) 
        