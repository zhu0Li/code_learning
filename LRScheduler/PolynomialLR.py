import math

from torch import nn


class PolynomialLR(nn.Module):
    def __init__(
            self,
            optimizer,
            steps: int,
            end_learning_rate: float = 0.0001,
            power: float = 1.0,
            cycle: bool = False,
    ):
        assert steps > 0, f"steps must greater than zero, but got {steps}"
        self.max_decay_steps = steps
        self.end_learning_rate = end_learning_rate
        self.power = power
        self.cycle = cycle
        self.base_lr = optimizer.param_groups[0]['lr']
        super(PolynomialLR, self).__init__()

    def forward(self, optimizer, step):
        base_lr = self.base_lr
        decay_batch = self.max_decay_steps
        cur_batch = step
        if self.cycle:
            if cur_batch == 0:
                cur_batch = 1
            decay_batch = decay_batch * math.ceil(cur_batch / decay_batch)
        else:
            cur_batch = min(cur_batch, decay_batch)

        factor = (1 - cur_batch / decay_batch) ** (self.power)
        learning_rate = (base_lr - self.end_learning_rate) * factor + self.end_learning_rate
        optimizer.param_groups[0]['lr'] = learning_rate
        return learning_rate

# def PolynomialLR1(
#             optimizer,
#             steps: int,
#             end_learning_rate: float = 0.0001,
#             power: float = 1.0,
#             cycle: bool = False
#                   ):
#     assert steps > 0, f"steps must greater than zero, but got {steps}"
#     def f(base_lr):
#         decay_batch = steps
#         cur_batch = (x - warmup_epochs * num_step)
#         if cycle:
#             if cur_batch == 0:
#                 cur_batch = 1
#             decay_batch = decay_batch * math.ceil(cur_batch / decay_batch)
#         else:
#             cur_batch = min(cur_batch, decay_batch)
#
#         factor = (1 - cur_batch / decay_batch) ** (power)
#         learning_rate = (base_lr - end_learning_rate) * factor + end_learning_rate
#         optimizer.param_groups[0]['lr'] = learning_rate
#         return learning_rate
#     return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=f)

