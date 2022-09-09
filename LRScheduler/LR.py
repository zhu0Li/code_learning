import math

import numpy as np
import torch
from torch import nn

from LRScheduler.CosineAnnealingWarmRestarts import CosineAnnealingWarmRestarts
from LRScheduler.warmup import  WarmupcosLR
from PolynomialLR import PolynomialLR
from CosineDecayLR import CosineDecayLR

model = nn.Sequential(
    nn.Linear(10,10),
    nn.Linear(10,10),
)
learning_rate = 0.001

optimizer = torch.optim.Adam(lr=0.06, params=model.parameters())
optimizer.param_groups[0]['lr'] = learning_rate
torch.optim.lr_scheduler.ConstantLR(optimizer, factor = 1.0/3,
                                    total_iters = 8, last_epoch = -1)

optimizer = torch.optim.Adam(lr=0.1, params=model.parameters())
torch.optim.lr_scheduler.LinearLR( optimizer ,start_factor=0.3,
                                   end_factor=1, total_iters = 5, last_epoch = -1)

optimizer = torch.optim.Adam(lr=0.1, params=model.parameters())
torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.3)

optimizer = torch.optim.Adam(lr=0.1, params=model.parameters())
sch1 =torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.7)
# sch1 =torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, step_size=3, gamma=0.7)

optimizer = torch.optim.Adam(lr=0.1, params=model.parameters())
scheduler = PolynomialLR(optimizer=optimizer, cycle=False,
                         steps=5, end_learning_rate=0.01, power=2)

optimizer = torch.optim.Adam(lr=0.1, params=model.parameters())
scheduler = PolynomialLR(optimizer=optimizer, cycle=True,
                             steps=2, end_learning_rate=0.07, power=1)

optimizer = torch.optim.Adam(lr=0.1, params=model.parameters())
scheduler = CosineDecayLR(optimizer=optimizer, decay_steps=5, alpha=0.03)

optimizer = torch.optim.Adam(lr=0.1, params=model.parameters())
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                           T_max=20, eta_min=0.02)

optimizer = torch.optim.Adam(lr=0.1, params=model.parameters())
scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer
                        , T_0=4, T_mult=1, eta_min=0.03)

optimizer = torch.optim.Adam(lr=0.1, params=model.parameters())
scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=4,
                    T_mult=1, eta_min=0.03, decay_rate=0.1, restart_limit=0)

optimizer = torch.optim.Adam(lr=0.1, params=model.parameters())
scheduler = WarmupcosLR(optimizer, num_step=1, epochs=20,
                        warmup=True, warmup_epochs=3,
                        warmup_factor=0.1, end_factor=0.001)
# scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer
#                         , T_0=4, T_mult=1, eta_min=0.03)
optimizer = torch.optim.Adam(lr=0.1, params=model.parameters())
# scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer
#                         , schedulers=[torch.optim.lr_scheduler.LinearLR,
#                                       ], milestones=, eta_min=0.03)

optimizer = torch.optim.Adam(lr=0.1, params=model.parameters())
scheduler1 = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.3,
                                               end_factor=1, total_iters=5, last_epoch=-1)
scheduler2 = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer
                                                                  , T_0=4, T_mult=1, eta_min=0.03)
scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer
                                                  , schedulers=[scheduler1, scheduler2]
                                                  , milestones=[4],
                                                  )

optimizer = torch.optim.Adam(lr=0.1, params=model.parameters())
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor = 0.1,
            patience = 3, threshold = 1e-4, threshold_mode = "rel", cooldown = 0,
            min_lr = 0, eps = 1e-8, verbose = False)
import matplotlib.pyplot as plt
if __name__ == "__main__":
    epochs = 15
    optimizer = torch.optim.Adam(lr=0.1, params=model.parameters())
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.1,
                                                           patience=2,threshold=1, cooldown=2,
                                                           min_lr=0, eps=1e-8, verbose=True)
    y = np.zeros(epochs)
    x = np.arange(epochs) + 1
    val_loss = 20
    for i in range(epochs):
        val_loss = val_loss-1
        print('step{}'.format(i), optimizer.param_groups[0]["lr"])
        scheduler.step(val_loss)

        y[i] = optimizer.param_groups[0]["lr"]
    plt.plot(x,y)
    plt.xlim([0,epochs])
    plt.show()
        # print(lr)