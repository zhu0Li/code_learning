import torch
import torch.nn as nn
from einops import repeat
# from ..common import expand_to_batch
def expand_to_batch(tensor, desired_size):
    tile = desired_size // tensor.shape[0]
    return repeat(tensor, 'b ... -> (b tile) ...', tile=tile)

class AbsPositionalEncoding1D(nn.Module):
    def __init__(self, tokens, dim):
        super(AbsPositionalEncoding1D, self).__init__()
        self.abs_pos_enc = nn.Parameter(torch.randn(1,tokens, dim))

    def forward(self, x):
        batch = x.size()[0]
        return x + expand_to_batch(self.abs_pos_enc, desired_size=batch)