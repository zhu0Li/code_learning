import math
# import summary
import torch
from torch import nn
from torch import Tensor
import torch.nn.functional as F
from timm.models.layers import DropPath
from typing import List, Optional, Tuple
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
    def __init__(self, in_channels, num_heads=8, kernel_size=56, stride=1, height_dim=True, inference=False):
        super(Axial_Layer, self).__init__()
        self.depth = in_channels
        self.num_heads = num_heads
        self.kernel_size = kernel_size    ## Maximum local field on which Axial-Attention is applied
        self.stride = stride
        self.height_dim = height_dim
        self.dh = self.depth // self.num_heads

        assert self.depth % self.num_heads == 0, "depth should be divided by num_heads. (example: depth: 32, num_heads: 8)"

        # self.kqv_conv = nn.Conv1d(in_channels, self.depth * 2, kernel_size=1, bias=False).to(device)
        # self.kqv_bn = nn.BatchNorm1d(self.depth * 2).to(device)
        # self.logits_bn = nn.BatchNorm2d(num_heads * 3).to(device)
        self.kqv_conv = nn.Conv1d(in_channels, self.depth * 2, kernel_size=1, bias=False)
        self.kqv_bn = nn.BatchNorm1d(self.depth * 2)
        self.logits_bn = nn.BatchNorm2d(num_heads * 3)
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
        kqv = self.kqv_conv(x)
        # print('kqv:',kqv.shape)
        kqv = self.kqv_bn(kqv)  # apply batch normalization on k, q, v
        k, q, v = torch.split(kqv.reshape(batch_size * width, self.num_heads, self.dh * 2, height),
                              [self.dh // 2, self.dh // 2, self.dh], dim=2)
        # print('k:{},q:{},v:{}'.format(k.shape,q.shape,v.shape))
        # Positional encodings
        rel_encodings = torch.index_select(self.rel_encoding, 1, self.distance_matrix).reshape(self.dh * 2,
                                                                                               self.kernel_size,
                                                                                               self.kernel_size)
        # print('rel_encodings: ',rel_encodings.shape)
        q_encoding, k_encoding, v_encoding = torch.split(rel_encodings, [self.dh // 2, self.dh // 2, self.dh], dim=0)

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

        if self.inference:
            self.weights = nn.Parameter(weights)

        attn = torch.matmul(weights, v.transpose(2, 3)).transpose(2, 3)
        attn_encoding = torch.einsum('bhxy,dxy->bhdx', weights, v_encoding)
        attn_out = torch.cat([attn, attn_encoding], dim=-1).reshape(batch_size * width, self.depth * 2, height)
        # print('attn_out: ', attn_out.shape)
        # # output = attn_out.reshape(batch_size, width, self.depth, 2, height).sum(dim=-2)
        # output = attn_out.reshape(batch_size, -1, 2, width).sum(dim=-2)
        #
        # # if self.height_dim:
        # #     output = output.permute(0, 2, 3, 1)
        # # else:
        # #     output = output.permute(0, 2, 1, 3)

        if self.height_dim:
            output = attn_out.reshape(batch_size, width, self.depth, 2, height).sum(dim=-2)
            output = output.permute(0, 2, 3, 1)
        else:
            output = attn_out.reshape(batch_size, -1, 2, height).sum(dim=-2)

        return output

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
        proj_dropout: float = 0.0,
        mlp_ratio: float = 4.0,
        drop_path: float = 0.1,
        ws: int = 1,
        with_cp: bool = False,
    ) -> None:
        super().__init__()
        self.with_cp = with_cp

        # build layer normalization
        self.attn_norm = nn.LayerNorm(in_dim)

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
            in_channels=in_dim,
            kernel_size=H,
            num_heads=heads,
            # ws=ws,
            height_dim=True,
            # with_cp=with_cp,
        )
        self.attnw = Axial_Layer(
            in_channels=in_dim,
            kernel_size=W,
            num_heads=heads,
            # ws=ws,
            height_dim=False,
            # with_cp=with_cp,
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
        x = self.attnh(x)
        # print('3',x.shape)
        x = x.permute(0, 2, 3, 1)
        # print('4',x.shape)
        x = self.attn_norm(x)
        x = x.permute(0, 3, 1, 2)
        # print('5',x.shape)
        x = self.attnw(x)
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
    input = torch.randn((1,8,32,3)).to(device)
    # input = torch.randn((4,640,128)).to(device)
    model =Axial_Layer(128,kernel_size=5,height_dim=False).to(device)
    model_ = HRViTAxialBlock(in_dim=8,dim=8, H=32, W=3).to(device)
    # model1 = MixCFN(in_features=128).to(device)
    # summary(model_, input_size=(8,128,5))
    # summary(model1, input_size=[(128,128,5),128,5])
    output1 = model_(input)
    print('output1',output1.shape)
    # output = model1(output1)
    # summary(model1, input_size=[(640, 128)])
    # print(output.shape)