import torch

# 定义三元组损失函数
def triplet_loss(anchor, positive, negative, margin=1.0):
    distance_positive = torch.norm(anchor - positive, p=2, dim=1)
    distance_negative = torch.norm(anchor - negative, p=2, dim=1)
    loss = torch.clamp(distance_positive - distance_negative + margin, min=0.0)
    return loss.mean()