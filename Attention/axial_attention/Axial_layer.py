import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from torch import Tensor
import torch.utils.checkpoint as cp
from torchsummary import summary
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
# device = torch.device("cpu")

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
        # output = attn_out.reshape(batch_size, width, self.depth, 2, height).sum(dim=-2)
        output = attn_out.reshape(batch_size, -1, 2, width).sum(dim=-2)

        # if self.height_dim:
        #     output = output.permute(0, 2, 3, 1)
        # else:
        #     output = output.permute(0, 2, 1, 3)

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
        print(x.shape)
        B, N, C = x.shape
        x = self.conv(x.transpose(1, 2).view(B, C, H, W))
        x = self.act(x)
        x = self.fc2(x.flatten(2).transpose(-1, -2))
        return x

        # if self.with_cp and x.requires_grad:
        #     x = cp.checkpoint(_inner_forward, x)
        # else:
        #     x = _inner_forward(x)
        # return x

if __name__ == '__main__':
    input = torch.randn((4,128,128,5)).to(device)
    model =Axial_Layer(128,kernel_size=5,height_dim=False).to(device)
    model1 = MixCFN(in_features=128).to(device)
    # summary(model, input_size=(128,128,5))
    # summary(model1, input_size=[(128,128,5),128,5])
    output1 = model(input)
    print('output1',output1.shape)
    output = model1(output1)
    # summary(model1, input_size=[(640, 128)])
    print(output.shape)