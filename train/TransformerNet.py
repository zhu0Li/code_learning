import torch
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from torch import nn

'''
直接分割即把图像直接分成多块。在代码实现上需要使用einops这个库，完成的操作是将（B，C，H，W）的shape调整为（B，(H/P *W/P)，P*P*C）。
'''
self.to_patch_embedding = nn.Sequential(
           Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_height, p2 = patch_width),
           nn.Linear(patch_dim, dim),
      )
'''
Rearrange用于对张量的维度进行重新变换排序，可用于替换pytorch中的reshape，view，transpose和permute等操作。举几个例子
'''
#假设images的shape为[32,200,400,3]
#实现view和reshape的功能
images = torch.randn((16,3,128,128))
Rearrange(images,'b h w c -> (b h) w c')#shape变为（32*200, 400, 3）
#实现permute的功能
Rearrange(images, 'b h w c -> b c h w')#shape变为（32, 3, 200, 400）
#实现这几个都很难实现的功能
Rearrange(images, 'b h w c -> (b c w) h')#shape变为（32*3*400, 200）