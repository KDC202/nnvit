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

class Loss(nn.Module):
    def __init__(self, weight, model, dist=None,graph=None):
        super(Loss, self).__init__()
        self.dist = dist
        self.graph = graph
        self.weight = weight
        self.model = model
        
    def forward(self,x,y,dataset):
        n = y.shape[0]
        xx = util.euclidean_dist(x, x)
        yy = util.euclidean_dist(y, y)
        nb_loss = torch.sqrt(torch.pow((xx - yy), 2).clamp(min=1e-12))
        ct_loss = 0
        # return nb_loss.mean()
        combine_x = util.pairwise_weighted_fusion(x, self.weight)
        combine_y = util.pairwise_weighted_fusion(y, self.weight)
        for i in range(n):
            for j in range (n):
                if i!= j:
                    neighbor = self.graph[j]
                    neighbor_feat = dataset[neighbor]
                    out_neighbor=[]
                    with torch.no_grad():
                        out_neighbor = self.model(neighbor_feat)
                    combine_x_neighbor = util.euclidean_dist(combine_x[i,j], neighbor_feat)
                    combine_x_neighbor = util.euclidean_dist(combine_y[i,j],out_neighbor)
                    ct_loss += torch.sqrt(torch.pow((combine_x_neighbor - combine_x_neighbor), 2).clamp(min=1e-12)).mean()
        ct_loss /= n*n
        sum_loss = (nb_loss+ct_loss) / 2
        return nb_loss.mean(), ct_loss, sum_loss
        #             num = len(neighbor)
        #             sorted_indices = torch.argsort(self.dist[i, neighbor,2])
        #             sorted_neighbors = neighbor[sorted_indices]
        #             pos = sorted_neighbors[:num//2]
        #             nei = sorted_neighbors[num//2:]
                    
        