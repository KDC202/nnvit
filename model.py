import os
import torch
import torch.nn as nn
import numpy as np

class Residual(torch.nn.Module):
    def __init__(self, func, drop=0):
        super().__init__()
        self.func = func
        self.drop = drop

    def forward(self, x):
        if self.training and self.drop > 0:
            return x + self.func(x) * torch.rand(x.size(0), 1, 1,
                                              device=x.device).ge_(self.drop).div(1 - self.drop).detach()
        else:
            return x + self.func(x)


class Attention(nn.Module):
    def __init__(self,
                 dim,
                 token_num,
                 num_heads = 8,
                 qkv_bias = False,
                 attn_drop = 0,
                 proj_drop = 0):
        super().__init__()
        self.pos_bias = nn.Parameter(torch.randn(num_heads, token_num, token_num))
        self.num_heads = num_heads
        # qkv向量长度
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim*3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
    
    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C//self.num_heads).permute(2,0,3,1,4)
        q, k, v = qkv.unbind(0)
        
        attn = (q @ k.transpose(-2,-1)) * self.scale + self.pos_bias
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        
        x = (attn @ v).transpose(1,2).reshape(B, N, C)
        
        x = self.proj(x)
        x = self.proj_drop(x)
        
        return x
 

class nnvit(nn.Module):
    def __init__(self,
                 Dim = 128,
                 num_head = 8,
                 token_dim = 32,
                 batch_size = 1024,
                 deep = 2):
        
        super().__init__()
        assert Dim % token_dim == 0, "Dim is not divisible by token_dim"
        self.num_tokens = Dim // token_dim
        self.dim = Dim
        self.batch_size = batch_size
        self.token_dim = token_dim
        self.num_head = num_head
        self.learnable_token = nn.Parameter(torch.randn(1, 1, self.token_dim))
        # self.pos_embedding = nn.Parameter(torch.randn(1, self.num_tokens + 1, dim))
        
        self.trans = nn.ModuleList()
        for i in range(deep):
            stage_i = []
            stage_i.append(
                Residual(
                    Attention(dim=self.token_dim, token_num=self.num_tokens+1, num_heads=self.num_head,
                              qkv_bias=True)
                )
            )
            self.trans.append(nn.Sequential(*stage_i))
        
    def forward(self, x):
        x_split = x.view(self.batch_size, self.num_tokens, self.dim // self.num_tokens)
        # print(x)
        print("x_split",x_split.shape)
        expanded_token = self.learnable_token.expand(self.batch_size, -1, -1)
        # print(expanded_token)
        y = torch.cat([expanded_token, x_split], dim=1)
        print("y",y.shape)
        
        for stage_i in self.trans:
            y = stage_i(y)
        y = y[:,0,:]
        return y
        # print(y)
    
    
    
    
if __name__ == '__main__':
    batch_size = 10
    dim = 128
    model = nnvit(Dim=dim,num_head=4,token_dim=32,batch_size=batch_size)
    tensor = torch.randn(batch_size, dim)
    out = model(tensor)
    print(out.shape)