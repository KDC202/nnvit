import torch
import torch.nn as nn
import numpy as np
import util


# ������Ԫ����ʧ����
def triplet_loss(anchor, positive, negative, margin=1.0):
    distance_positive = torch.norm(anchor - positive, p=2, dim=1)
    distance_negative = torch.norm(anchor - negative, p=2, dim=1)
    loss = torch.clamp(distance_positive - distance_negative + margin, min=0.0)
    return loss.mean()

# def neighbor_loss(original_distance, feat)

class Loss(nn.Module):
    def __init__(self, weight, model, dist=None):
        super(Loss, self).__init__()
        self.dist = dist
        # self.graph = graph
        self.weight = weight
        self.model = model
        
    def forward(self,x,y,dataset,graph):
        n = y.shape[0]
        xx = util.euclidean_dist(x, x)
        yy = util.euclidean_dist(y, y)
        nb_loss = torch.sqrt(torch.pow((xx - yy), 2).clamp(min=1e-12))
        ct_loss = 0
        # dataset = torch.tensor(dataset)
        # return nb_loss.mean()
        # combine_x = util.pairwise_weighted_fusion(x, self.weight).cuda()
        # combine_y = util.pairwise_weighted_fusion(y, self.weight).cuda()
        # for i in range(n):
        #     for j in range (n):
        #         if i!= j:
        #             neighbor = graph[j]
        #             # neighbor = torch.tensor(neighbor)
        #             # print(dataset[1])
        #             neighbor_feat = [dataset[idx][1] for idx in neighbor]
        #             neighbor_feat = torch.stack(neighbor_feat).cuda()
        #             # neighbor_feat = [item[1] for item in neighbor_feat]
        #             # neighbor_feat = torch.index_select(dataset, 0, neighbor)
        #             # neighbor_feat = torch.tensor(neighbor_feat)
                    
        #             out_neighbor=[]
        #             with torch.no_grad():
        #                 out_neighbor = self.model(neighbor_feat)
        #             combine_x_neighbor = util.euclidean_dist2(combine_x[i,j], neighbor_feat)
        #             combine_x_neighbor = util.euclidean_dist2(combine_y[i,j],out_neighbor)
        #             ct_loss += torch.sqrt(torch.pow((combine_x_neighbor - combine_x_neighbor), 2).clamp(min=1e-12)).mean()
        # ct_loss /= n*n
        # sum_loss = (nb_loss+ct_loss) / 2
        # return nb_loss.mean(), ct_loss, sum_loss.mean()
        return nb_loss.mean()
        #             num = len(neighbor)
        
        #             sorted_indices = torch.argsort(self.dist[i, neighbor,2])
        #             sorted_neighbors = neighbor[sorted_indices]
        #             pos = sorted_neighbors[:num//2]
        #             nei = sorted_neighbors[num//2:]
                    
        