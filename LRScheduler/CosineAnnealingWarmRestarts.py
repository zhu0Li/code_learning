import math
from torch import nn

class CosineAnnealingWarmRestarts(nn.Module):

    def __init__(
            self,
            optimizer,
            T_0: int,
            T_mult: int = 1,
            eta_min: float = 0.0,
            decay_rate: float = 1.0,
            restart_limit: int = 0,
    ):
        if T_0 <= 0 or not isinstance(T_0, int):
            raise ValueError(f"Expected positive integer T_0, but got {T_0}")

        if T_mult < 1 or not isinstance(T_mult, int):
            raise ValueError(f"Expected integer T_mult >= 1, but got {T_mult}")
        self.base_lr = optimizer.param_groups[0]['lr']
        self.T_0 = T_0
        self.T_mult = T_mult
        self.eta_min = eta_min
        self.decay_rate = decay_rate
        self.restart_limit = restart_limit

        super(CosineAnnealingWarmRestarts, self).__init__()

    def forward(self, optimizer, step):
        base_lr = self.base_lr
        if self.T_mult > 1:
            epoch = math.floor(
                math.log(1 - step / self.T_0 * (1 - self.T_mult), self.T_mult)
            )
            epoch_steps = self.T_mult ** epoch * self.T_0
            step_in_epoch = (
                    step - (1 - self.T_mult ** epoch) / (1 - self.T_mult) * self.T_0
            )
        else:
            epoch = step // self.T_0
            epoch_steps = self.T_0
            step_in_epoch = step - (epoch_steps * epoch)

        gamma = self.decay_rate ** epoch
        if self.restart_limit == 0 or (
                self.restart_limit > 0 and epoch < self.restart_limit
        ):
            cos_decay = 0.5 * (1 + math.cos(math.pi * step_in_epoch / epoch_steps))
            learning_rate = self.eta_min + (base_lr * gamma - self.eta_min) * cos_decay
            optimizer.param_groups[0]['lr'] = learning_rate
            return learning_rate
        learning_rate = self.eta_min
        optimizer.param_groups[0]['lr'] = learning_rate
        return learning_rate