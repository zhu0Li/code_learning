import math
# import summary
import torch
from torch import nn
from torch import Tensor
import torch.nn.functional as F
from timm.models.layers import DropPath
from typing import List, Optional, Tuple
from timm.models.layers import trunc_normal_
from torchsummary import summary
class DES(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        act_func: nn.Module = nn.GELU,
    ) -> None:
        super().__init__()
        _, self.p = self._decompose(min(in_features, out_features))
        self.k_out = out_features // self.p
        self.k_in = in_features // self.p
        self.proj_right = nn.Linear(self.p, self.p, bias=bias)
        self.act = act_func()
        self.proj_left = nn.Linear(self.k_in, self.k_out, bias=bias)

    def _decompose(self, n: int) -> List[int]:
        # assert n % 2 == 0, f"Feature dimension has to be a multiple of 2, but got {n}"
        e = int(math.log2(n))
        e1 = e // 2
        e2 = e - e // 2
        return 2 ** e1, 2 ** e2

    def forward(self, x: Tensor) -> Tensor:
        # print('-----DES-----')
        # print(x.shape)
        B,N,C=x.size()
        x = x.reshape(x.shape[0],-1,self.k_in *self.p)
        B = x.shape[:-1]
        # print(B,self.k_in,self.p)
        x = x.view(*B, self.k_in, self.p)
        x = self.proj_right(x).transpose(-1, -2)
        if self.act is not None:
            x = self.act(x)
        x = self.proj_left(x).transpose(-1, -2).flatten(-2, -1)
        # x = x.reshape(B,N,C)
        return x

class Axial_Layer(nn.Module):
    def __init__(self, in_channels, num_heads=8, kernel_size=56, stride=1, height_dim=True, inference=False,
                 attn_drop=0., proj_drop=0.):
        super(Axial_Layer, self).__init__()
        self.depth = in_channels
        self.num_heads = num_heads if (in_channels // num_heads)%2==0 else in_channels // num_heads
        self.kernel_size = kernel_size    ## Maximum local field on which Axial-Attention is applied
        self.stride = stride
        self.height_dim = height_dim
        self.dh = self.depth // self.num_heads
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)
        assert self.depth % self.num_heads == 0, "depth should be divided by num_heads. (example: depth: 32, num_heads: 8)"

        # self.kqv_conv = nn.Conv1d(in_channels, self.depth * 2, kernel_size=1, bias=False).to(device)
        # self.kqv_bn = nn.BatchNorm1d(self.depth * 2).to(device)
        # self.logits_bn = nn.BatchNorm2d(num_heads * 3).to(device)
        self.kqv_conv = nn.Conv1d(in_channels, self.depth * 2, kernel_size=1, bias=False)
        self.kqv_bn = nn.BatchNorm1d(self.depth * 2)
        self.logits_bn = nn.BatchNorm2d(self.num_heads * 3)
        # Positional encodings
        self.rel_encoding = nn.Parameter(torch.randn(self.dh * 2, kernel_size * 2 - 1), requires_grad=True)
        # key_index = torch.arange(kernel_size).to(device)
        # query_index = torch.arange(kernel_size).to(device)
        key_index = torch.arange(kernel_size)
        query_index = torch.arange(kernel_size)
        # Shift the distance_matrix so that it is >= 0. Each entry of the
        # distance_matrix distance will index a relative positional embedding.
        distance_matrix = (key_index[None, :] - query_index[:, None]) + kernel_size - 1
        self.register_buffer('distance_matrix', distance_matrix.reshape(kernel_size * kernel_size))

        # later access attention weights
        self.inference = inference
        if self.inference:
            self.register_parameter('weights', None)

    def forward(self, x):
        if self.height_dim:
            #   depth:num_channels   height:X    width:Y
            x = x.permute(0, 3, 1, 2)  # batch_size, width, depth, height
        else:
            x = x.permute(0, 2, 1, 3)  # batch_size, height, depth, width

        batch_size, width, depth, height = x.size()
        # print(width)
        x = x.reshape(batch_size * width, depth, height)

        # Compute q, k, v
        # kqv = self.kqv_conv(x).to(device)
        # print('x',x.shape)
        kqv = self.kqv_conv(x)
        # print('kqv:',kqv.shape)
        kqv = self.kqv_bn(kqv)  # apply batch normalization on k, q, v
        # print(self.depth)
        # print(self.num_heads)
        # print(self.dh)
        if self.dh  == 3:
            self.num_heads = 3
            self.dh = self.depth // self.num_heads
            k, q, v = torch.split(kqv.reshape(batch_size * width, self.num_heads, self.dh * 2, height),
                                                             [self.dh // 2, self.dh // 2, self.dh], dim=2)
        else:
            k, q, v = torch.split(kqv.reshape(batch_size * width, self.num_heads, self.dh * 2, height),
                                  [self.dh // 2, self.dh // 2, self.dh], dim=2)
        # else:
        #
        # print('k:{},q:{},v:{}'.format(k.shape,q.shape,v.shape))
        # Positional encodings
        rel_encodings = torch.index_select(self.rel_encoding, 1, self.distance_matrix).reshape(self.dh * 2,
                                                                                               self.kernel_size,
                                                                                               self.kernel_size)
        # print('rel_encodings: ',rel_encodings.shape)
        if self.dh  == 3:
            self.num_heads = 3
            self.dh = self.depth // self.num_heads
            q_encoding, k_encoding, v_encoding = torch.split(rel_encodings, [self.dh // 2, self.dh // 2, self.dh], dim=0)
        else:
            q_encoding, k_encoding, v_encoding = torch.split(rel_encodings, [self.dh // 2, self.dh // 2, self.dh],
                                                             dim=0)

        # qk + qr + kr
        qk = torch.matmul(q.transpose(2, 3), k)
        # print('q',q.shape)
        # print('q_encoding:',q_encoding.shape)
        # print('qk:',qk.shape)
        qr = torch.einsum('bhdx,dxy->bhxy', q, q_encoding)    ## qr：乘了positional encoding之后
        kr = torch.einsum('bhdx,dxy->bhxy', k, k_encoding).transpose(2, 3)
        # print('qr: ',qr.shape,'kr: ',kr.shape)
        logits = torch.cat([qk, qr, kr], dim=1)
        # print('logits: ',logits.shape)
        logits = self.logits_bn(logits)  # apply batch normalization on qk, qr, kr
        # print('logits: ',logits.shape)
        logits = logits.reshape(batch_size * width, 3, self.num_heads, height, height).sum(dim=1)
        # print('logits: ', logits.shape)

        weights = F.softmax(logits, dim=3)  ## weights：q，k之后与v相乘的权重，即attention权重
        weights = self.attn_drop(weights)
        if self.inference:
            self.weights = nn.Parameter(weights)

        attn = torch.matmul(weights, v.transpose(2, 3)).transpose(2, 3)
        attn_encoding = torch.einsum('bhxy,dxy->bhdx', weights, v_encoding)
        # print('attn',attn.shape)
        attn_out = torch.cat([attn, attn_encoding], dim=-1)
        # print('attn_out: ', attn_out.shape)
        attn_out = attn_out.reshape(batch_size * width, self.depth * 2, height)
        # print('attn_out: ', attn_out.shape)
        attn_out = self.proj_drop(attn_out)
        # print('attn_out: ', attn_out.shape)
        # # output = attn_out.reshape(batch_size, width, self.depth, 2, height).sum(dim=-2)
        # output = attn_out.reshape(batch_size, -1, 2, width).sum(dim=-2)
        #
        output = attn_out.reshape(batch_size, width, self.depth, 2, height).sum(dim=-2)
        if self.height_dim:
            output = output.permute(0, 2, 3, 1)
        else:
            output = output.permute(0, 2, 1, 3)

        return output


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
        self.dim = dim  # 96*(2^layer_index 0,1,2,3...)
        self.window_size = window_size  # Wh, Ww (7,7)
        self.num_heads = num_heads  # [3, 6, 12, 24]
        head_dim = dim // num_heads  # (96//3=32,96*2^1 // 6=32,...)
        self.scale = qk_scale or head_dim ** -0.5  # default：head_dim ** -0.5

        # define a parameter table of relative position bias
        # 定义相对位置偏置表格
        # [(2*7-1)*(2*7-1),3]
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH

        # get pair-wise relative position index for each token inside the window
        # 得到一对在窗口中的相对位置索引
        coords_h = torch.arange(self.window_size[0])  # [0,1,2,3,4,5,6]
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        # 让相对坐标从0开始
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        # relative_coords[:, :, 0] * (2*7-1)
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        # 为位置偏置表中索引值，位置偏移表(13*13,nHeads)索引0-168
        # 索引值为 (49,49) 值在0-168对应位置偏移表的索引
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)
        # dim*(dim*3)
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        # attn_drop=0.0
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        # 初始化相对位置偏置值表(截断正态分布)
        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    # 输入划分为一个个window
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

    # 模块的前向传播
    def forward(self, x, mask=None):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        x = self.window_partition(x)
        B, C_, H, W = x.shape  # BCHW
        x = x.permute(0, 2, 3, 1).reshape(B, H * W, C_)  # BCHW->BHWC->B,H*W,C
        B_, N, C = x.shape  # 输入特征的尺寸
        # (3, B_, num_heads, N, C // num_heads)
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
        # 由于relative_position_bias_table第二维为 nHeads所以最终表变为了 49*49*nHead 的随机表
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        # print(relative_position_bias.shape)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        # attn每一个批次，加上随机的相对位置偏移 说明attn.shape=B_,num_heads,Wh*Ww, Wh*Ww
        # print(attn.shape,relative_position_bias.shape)
        attn = attn + relative_position_bias.unsqueeze(0)
        # mask 在某阶段的奇数层为None 偶数层才存在
        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)
        # 进行 dropout
        attn = self.attn_drop(attn)
        # attn @ v：B_, num_heads, N, C/num_heads
        # x: B_, N, C 其中
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        # 经过一层全连接
        x = self.proj(x)
        # 进行drop out
        x = self.proj_drop(x)
        x = x.reshape(B, C_, H, W)
        return x

class MixConv2d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.conv_3 = nn.Conv2d(
            in_channels // 2,
            out_channels // 2,
            kernel_size,
            stride=1,
            padding=padding,
            dilation=dilation,
            groups=groups // 2,
            bias=bias,
        )
        self.conv_5 = nn.Conv2d(
            in_channels - in_channels // 2,
            out_channels - out_channels // 2,
            kernel_size + 2,
            stride=stride,
            padding=padding + 1,
            dilation=dilation,
            groups=groups - groups // 2,
            bias=bias,
        )

    def forward(self, x: Tensor) -> Tensor:
        x1, x2 = x.chunk(2, dim=1)
        x1 = self.conv_3(x1)
        x2 = self.conv_5(x2)
        x = torch.cat([x1, x2], dim=1)
        return x

class MixCFN(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        act_func: nn.Module = nn.GELU,
        with_cp: bool = False,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.with_cp = with_cp
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.conv = MixConv2d(
            hidden_features,
            hidden_features,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=hidden_features,
            dilation=1,
            bias=True,
        )
        self.act = act_func()
        self.fc2 = nn.Linear(hidden_features, out_features)

    def forward(self, x: Tensor, H: int = 128, W: int = 5) -> Tensor:
        # def _inner_forward(x: Tensor) -> Tensor:
        x = self.fc1(x)
        # print(x.shape)
        B, N, C = x.shape
        x = self.conv(x.transpose(1, 2).view(B, C, H, W))
        x = self.act(x)
        x = self.fc2(x.flatten(2).transpose(-1, -2))
        return x

class HRViTAxialBlock(nn.Module):
    def __init__(
        self,
        in_dim: int = 64,
        dim: int = 64,
        H: int = 64,
        W: int = 5,
        heads: int = 2,
        window_size: [int,int]=[5,5],
        # proj_dropout: float = 0.0,
        attn_drop: float=0.,
        proj_drop: float=0.,
        mlp_ratio: float = 4.0,
        drop_path: float = 0.1,
        ws: int = 1,
        with_cp: bool = False,
    ) -> None:
        super().__init__()
        self.with_cp = with_cp

        # build layer normalization
        self.attn_norm = nn.LayerNorm(in_dim)
        self.attn_norm_2 = nn.LayerNorm(in_dim//2)

        # build attention layer
        # self.attn = HRViTAttention(
        #     in_dim=in_dim,
        #     dim=dim,
        #     heads=heads,
        #     ws=ws,
        #     proj_drop=proj_dropout,
        #     with_cp=with_cp,
        # )
        self.attnh = Axial_Layer(
            in_channels=in_dim//2,
            kernel_size=H,
            num_heads=heads//2,
            # ws=ws,
            height_dim=True,
            attn_drop=attn_drop,
            proj_drop=proj_drop
            # with_cp=with_cp,
        )
        self.attnw = Axial_Layer(
            in_channels=in_dim//2,
            kernel_size=W,
            num_heads=heads//2,
            # ws=ws,
            height_dim=False,
            attn_drop=attn_drop,
            proj_drop=proj_drop
            # with_cp=with_cp,
        )
        self.attn_win = WindowAttention(
            dim = in_dim//2,
            window_size = window_size,
            num_heads = heads//2,
            attn_drop=attn_drop,
            proj_drop=proj_drop
        )

        # build diversity-enhanced shortcut DES
        self.des = DES(
            in_features=in_dim,
            out_features=dim,
            bias=True,
            act_func=nn.GELU,
        )
        # build drop path
        self.attn_drop_path = DropPath(
            drop_path) if drop_path > 0.0 else nn.Identity()

        # build layer normalization
        self.ffn_norm = nn.LayerNorm(in_dim)

        # build FFN
        self.ffn = MixCFN(
            in_features=in_dim,
            hidden_features=int(dim * mlp_ratio),
            out_features=dim,
            act_func=nn.GELU,
            with_cp=with_cp,
        )

        # build drop path
        self.ffn_drop_path = DropPath(
            drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        # attention block
        # print('1',x.shape)
        B,C,H,W = x.size()
        res = x.reshape(B,H*W,C)
        x = x.permute(0,2,3,1)
        x = self.attn_norm(x)
        x = x.permute(0, 3, 1,2)
        # print('2',x.shape)

        # * 实现并行的Axial_Attention
        x1, x2 = x[:, 0:C // 2, :, :], x[:, C // 2:, :, :]
        # print(x1.shape,x2.shape)
        x1 = self.attnh(x1)
        # print('3',x.shape)
        x1 = x1.permute(0, 2, 3, 1)
        # print('4',x.shape)
        x1 = self.attn_norm_2(x1)
        x1 = x1.permute(0, 3, 1, 2)
        # print('5',x.shape)
        x2 = self.attnw(x2)
        x2 = x2.permute(0, 2, 3, 1)
        # print('4',x.shape)
        x2 = self.attn_norm_2(x2)
        x2 = x2.permute(0, 3, 1, 2)
        # print('6',x1.shape,x2.shape)
        # print('res',res.shape)
        x = torch.cat((x1, x2), dim=1)
        # *

        # print('6',x.shape)
        # print('res',res.shape)
        x_des = self.des(res)
        # print('x_des',x_des.shape)
        x = x.reshape(B,H*W,C)
        x = self.attn_drop_path(x.add(x_des)).add(res)
        # print('7', x.shape)
        # ffn block
        res = x
        x = self.ffn_norm(x)
        # print('8', x.shape)
        x = self.ffn(x,H,W)
        # print('9', x.shape)
        x = self.ffn_drop_path(x).add(res)
        x = x.reshape(B,C,H,W)
        return x



if __name__ == '__main__':
    use_cuda = True
    device = torch.device("cuda" if use_cuda else "cpu")
    input = torch.randn((1,8,32,5)).to(device)
    # input = torch.randn((4,640,128)).to(device)
    model =Axial_Layer(128,kernel_size=5,height_dim=False).to(device)
    model_ = HRViTAxialBlock(in_dim=8,dim=8, H=32, W=5).to(device)
    # model1 = MixCFN(in_features=128).to(device)
    # summary(model_, input_size=(8,128,5))
    # summary(model1, input_size=[(128,128,5),128,5])
    output1 = model_(input)
    print('output1',output1.shape)
    # output = model1(output1)
    # summary(model1, input_size=[(640, 128)])
    # print(output.shape)