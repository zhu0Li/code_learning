import torch
from torch import nn


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads  #head的数目
        head_dim = dim // num_heads  #根据head的数目计算每个head的通道数
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5
        #or的用法为当qk_scale有数值时，scale = qk_scale，否则scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape  #C:num_channels, N:num_pixels, B:Batch_size
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads,
                C // self.num_heads).permute(2, 0, 3, 1, 4)# 3,B,num_heads,N,CH:channels of head
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)
        print(q.shape,k.shape,v.shape)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        print(attn.shape)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        # x: (B, N, C)
        return x

if __name__ =="__main__":
    a = torch.rand((5,128,64))
    att = Attention(dim=64, num_heads=8)
    b = att(a)
    print(b.shape)