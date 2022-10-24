#窗口注意力
# from typing import Tuple
import torch
from torch import nn
from timm.models.layers import  trunc_normal_

def window_partition(x, window_size):
    """
    将feature map按照window_size划分成一个个没有重叠的window
    Args:
        x: (B, C, H, W)
        window_size (int): window size(M)

    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    if isinstance(window_size,int):
        window_size = [window_size, window_size]
    x = x.permute(0,2,3,1) #BCHW->BHWC
    B, H, W, C = x.shape
    window_size_H,window_size_W = window_size[0],window_size[1]
    x = x.view(B, H // window_size_H, window_size_H, W // window_size_W, window_size_W, C)
    # permute: [B, H//Mh, Mh, W//Mw, Mw, C] -> [B, H//Mh, W//Mh, Mw, Mw, C]
    # view: [B, H//Mh, W//Mw, Mh, Mw, C] -> [B*num_windows, Mh, Mw, C]
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, C, window_size_H, window_size_W)

    return windows

class WindowAttention(nn.Module):
    r""" Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.

    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):

        super().__init__()
        self.dim = dim#96*(2^layer_index 0,1,2,3...)
        self.window_size = window_size  # Wh, Ww (7,7)
        # self.num_heads = num_heads#[3, 6, 12, 24]
        self.num_heads = num_heads if (dim // num_heads) % 2 == 0 else dim // num_heads
        head_dim = dim // self.num_heads#(96//3=32,96*2^1 // 6=32,...)
        self.scale = qk_scale or head_dim ** -0.5#default：head_dim ** -0.5

        # define a parameter table of relative position bias
        #定义相对位置偏置表格
        #[(2*7-1)*(2*7-1),3]
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH

        # get pair-wise relative position index for each token inside the window
        #得到一对在窗口中的相对位置索引
        coords_h = torch.arange(self.window_size[0])#[0,1,2,3,4,5,6]
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        #让相对坐标从0开始
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        #relative_coords[:, :, 0] * (2*7-1)
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        #为位置偏置表中索引值，位置偏移表(13*13,nHeads)索引0-168
        #索引值为 (49,49) 值在0-168对应位置偏移表的索引
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)
        #dim*(dim*3)
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        #attn_drop=0.0
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        #初始化相对位置偏置值表(截断正态分布)
        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    #输入划分为一个个window
    def window_partition(self, x):
        """
        将feature map按照window_size划分成一个个没有重叠的window
        Args:
            x: (B, C, H, W)
            window_size (int): window size(M)

        Returns:
            windows: (num_windows*B, window_size, window_size, C)
        """
        window_size = self.window_size
        if isinstance(window_size, int):
            window_size = [window_size, window_size]
        x = x.permute(0, 2, 3, 1)  # BCHW->BHWC
        B, H, W, C = x.shape
        window_size_H, window_size_W = window_size[0], window_size[1]
        x = x.view(B, H // window_size_H, window_size_H, W // window_size_W, window_size_W, C)
        # permute: [B, H//Mh, Mh, W//Mw, Mw, C] -> [B, H//Mh, W//Mh, Mw, Mw, C]
        # view: [B, H//Mh, W//Mw, Mh, Mw, C] -> [B*num_windows, Mh, Mw, C]
        windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, C, window_size_H, window_size_W)

        return windows

    #模块的前向传播
    def forward(self, x, mask=None):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        B_o, C_o, H_o, W_o = x.shape  # BCHW
        x = self.window_partition(x)
        B, C_, H, W = x.shape  # BCHW
        x = x.permute(0,2,3,1).reshape(B,H*W,C_) # BCHW->BHWC->B,H*W,C
        B_, N, C = x.shape#输入特征的尺寸
        #(3, B_, num_heads, N, C // num_heads)

        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        # q/k/v: [B_, num_heads, N, C // num_heads]
        # print("qkv:", qkv.shape)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)
        # print('q:',q.shape,'k',k.shape,'v',v.shape)
        # q*head_dim ** -0.5
        q = q * self.scale
        # attn:B_, num_heads,N,N
        attn = (q @ k.transpose(-2, -1))
        # 在 随机在relative_position_bias_table中的第一维(169)选择position_index对应的值，共49*49个
        #由于relative_position_bias_table第二维为 nHeads所以最终表变为了 49*49*nHead 的随机表
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        # print(relative_position_bias.shape)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        #attn每一个批次，加上随机的相对位置偏移 说明attn.shape=B_,num_heads,Wh*Ww, Wh*Ww
        # print(attn.shape,relative_position_bias.shape)
        attn = attn + relative_position_bias.unsqueeze(0)
        #mask 在某阶段的奇数层为None 偶数层才存在
        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)
        #进行 dropout
        attn = self.attn_drop(attn)
        #attn @ v：B_, num_heads, N, C/num_heads
        #x: B_, N, C 其中
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        #经过一层全连接
        x = self.proj(x)
        #进行drop out
        x = self.proj_drop(x)
        x = x.reshape(B_o, C_o, H_o, W_o)
        return x


if __name__ == '__main__':
    use_cuda = True
    device = torch.device("cuda" if use_cuda else "cpu")
    B,C,H,W = 1,8,3072,5
    # input = torch.randn((B,C,H,W)).to(device)
    x = torch.randn((B,C,H,W)).to(device)
    # input = torch.randn((4,640,128)).to(device)
    win_p = window_partition(x,window_size=[H//32,W])
    print(win_p.shape)
    model =WindowAttention(C,window_size=[H//32,W],num_heads=4,).to(device)
    # model_ = HRViTAxialBlock(in_dim=8,dim=8, H=32, W=5).to(device)
    # model1 = MixCFN(in_features=128).to(device)
    # summary(model_, input_size=(8,128,5))
    # summary(model1, input_size=[(128,128,5),128,5])
    output1 = model(x)
    print('output1',output1.shape)
    # output = model1(output1)
    # summary(model1, input_size=[(640, 128)])
    # print(output.shape)