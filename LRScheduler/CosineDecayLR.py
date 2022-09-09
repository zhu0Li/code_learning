import math
from torch import nn

class CosineDecayLR(nn.Module):
    def __init__(
        self,
        optimizer,
        decay_steps: int,
        alpha: float = 0.0,
    ):
        assert (
            decay_steps > 0
        ), f"decay_steps must greater than zero, but got {decay_steps}"
        self.decay_steps = decay_steps
        self.alpha = alpha
        self.base_lr = optimizer.param_groups[0]['lr']
        super(CosineDecayLR, self).__init__()

    def forward(self, optimizer, step):
        base_lr = self.base_lr
        if step < self.decay_steps:
            cos_decay = 0.5 * (1 + math.cos(math.pi * step / self.decay_steps))
            decay_factor = (1 - self.alpha) * cos_decay + self.alpha
        else:
            decay_factor = self.alpha
        learning_rate = base_lr * decay_factor
        optimizer.param_groups[0]['lr'] = learning_rate
        return learning_rate
