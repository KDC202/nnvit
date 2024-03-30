import os
import torch
import torch.nn as nn

class Attention(nn.module):
    def __init__(self,
                 dim,
                 num_heads = 8,
                 qkv_bias = False,
                 attn_drop = 0,
                 proj_drop = 0):
        super().__init()
        # self.pos_bias = nn.Parameter(torch.randn(num_head, ))
        self.num_head = num_heads
        # qkv向量长度
        head_dim = dim // num_heads
        self.scale = head_dim ** -o.5
        



class nnvit(nn.module):
    def __init__(self,
                 Dim = 128,
                 num_head = 8,
                 out_dim = 32):
        
    
    
