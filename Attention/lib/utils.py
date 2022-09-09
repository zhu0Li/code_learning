import json
import math

import numpy as np
import sys
from pdb import set_trace as T
import time

import torch
import torch as t
import torch.nn.functional as F
from torch import nn
from torch.autograd import Variable
import torch.nn.init as init


# Generic
def invertDict(x):
    return {v: k for k, v in x.items()}


def loadDict(fName):
    with open(fName) as f:
        s = eval(f.read())
    return s


def norm(x, n=2):
    return (np.sum(np.abs(x) ** n) ** (1.0 / n)) / np.prod(x.shape)


# Tracks inds of a permutation
class Perm():
    def __init__(self, n):
        self.inds = np.random.permutation(np.arange(n))
        self.m = n
        self.pos = 0

    def next(self, n):
        assert (self.pos + n < self.m)
        ret = self.inds[self.pos:(self.pos + n)]
        self.pos += n
        return ret


# Continuous moving average
class CMA():
    def __init__(self):
        self.t = 0.0
        self.cma = 0.0

    def update(self, x):
        self.cma = (x + self.t * self.cma) / (self.t + 1)
        self.t += 1.0


# Exponentially decaying average
class EDA():
    def __init__(self, k=0.99):
        self.k = k
        self.eda = 0.0

    def update(self, x):
        # self.eda = self.eda * k / (x * (1-k))
        self.eda = (1 - self.k) * x + self.k * self.eda


# Print model size
def modelSize(net):
    params = 0
    for e in net.parameters():
        params += np.prod(e.size())
    params = int(params / 1000)
    print("Network has ", params, "K params")


# Same padded (odd k)
def Conv2d(fIn, fOut, k):
    pad = int((k - 1) / 2)
    return nn.Conv2d(fIn, fOut, k, padding=pad)


# ModuleList wrapper
# def list(module, *args, n=1):
#     return nn.ModuleList([module(*args) for i in range(n)])


# Variable wrapper
def var(xNp, volatile=False, cuda=False):
    x = Variable(t.from_numpy(xNp), volatile=volatile)
    if cuda:
        x = x.cuda()
    return x


# Full-network initialization wrapper
def initWeights(net, scheme='orthogonal'):
    print('Initializing weights. Warning: may overwrite sensitive bias parameters (e.g. batchnorm)')
    for e in net.parameters():
        if scheme == 'orthogonal':
            if len(e.size()) >= 2:
                init.orthogonal(e)
        elif scheme == 'normal':
            init.normal(e, std=1e-2)
        elif scheme == 'xavier':
            init.xavier_normal(e)


class SaveManager():
    def __init__(self, root):
        self.tl, self.ta, self.vl, self.va = [], [], [], []
        self.root = root
        self.stateDict = None
        self.lock = False

    def update(self, net, tl, ta, vl, va):
        nan = np.isnan(sum([t.sum(e) for e in net.state_dict().values()]))
        if nan or self.lock:
            self.lock = True
            print('NaN in update. Locking. Call refresh() to reset')
            return

        if self.epoch() == 1 or va > np.max(self.va):
            self.stateDict = net.state_dict().copy()
            t.save(net.state_dict(), self.root + 'weights')

        self.tl += [tl];
        self.ta += [ta]
        self.vl += [vl];
        self.va += [va]

        np.save(self.root + 'tl.npy', self.tl)
        np.save(self.root + 'ta.npy', self.ta)
        np.save(self.root + 'vl.npy', self.vl)
        np.save(self.root + 'va.npy', self.va)

    def load(self, net, raw=False):
        stateDict = t.load(self.root + 'weights')
        self.stateDict = stateDict
        if not raw:
            net.load_state_dict(stateDict)
        self.tl = np.load(self.root + 'tl.npy').tolist()
        self.ta = np.load(self.root + 'ta.npy').tolist()
        self.vl = np.load(self.root + 'vl.npy').tolist()
        self.va = np.load(self.root + 'va.npy').tolist()

    def refresh(self, net):
        self.lock = False
        net.load_state_dict(self.stateDict)

    def epoch(self):
        return len(self.tl) + 1


# From Github user jihunchoi
def _sequence_mask(sequence_length, max_len=None):
    if max_len is None:
        max_len = sequence_length.data.max()
    batch_size = sequence_length.size(0)
    seq_range = t.range(0, max_len - 1).long()
    seq_range_expand = seq_range.unsqueeze(0).expand(batch_size, max_len)
    seq_range_expand = Variable(seq_range_expand)
    if sequence_length.is_cuda:
        seq_range_expand = seq_range_expand.cuda()
    seq_length_expand = (sequence_length.unsqueeze(1)
                         .expand_as(seq_range_expand))
    return seq_range_expand < seq_length_expand


# From Github user jihunchoi
def maskedCE(logits, target, length):
    """
    Args:
        logits: A Variable containing a FloatTensor of size
            (batch, max_len, num_classes) which contains the
            unnormalized probability for each class.
        target: A Variable containing a LongTensor of size
            (batch, max_len) which contains the index of the true
            class for each corresponding step.
        length: A Variable containing a LongTensor of size (batch,)
            which contains the length of each data in a batch.

    Returns:
        loss: An average loss value masked by the length.
    """

    # logits_flat: (batch * max_len, num_classes)
    logits_flat = logits.view(-1, logits.size(-1))
    # log_probs_flat: (batch * max_len, num_classes)
    log_probs_flat = F.log_softmax(logits_flat)
    # target_flat: (batch * max_len, 1)
    target_flat = target.view(-1, 1)
    # losses_flat: (batch * max_len, 1)
    losses_flat = -t.gather(log_probs_flat, dim=1, index=target_flat)
    # losses: (batch, max_len)
    losses = losses_flat.view(*target.size())
    # mask: (batch, max_len)
    mask = _sequence_mask(sequence_length=length, max_len=target.size(1))
    losses = losses * mask.float()
    loss = losses.sum() / length.float().sum()
    return loss


def runMinibatch(net, batcher, cuda=True, volatile=False, trainable=False):
    x, y, mask = batcher.next()
    x = [var(e, volatile=volatile, cuda=cuda) for e in x]
    y = [var(e, volatile=volatile, cuda=cuda) for e in y]
    if mask is not None:
        mask = var(mask, volatile=volatile, cuda=cuda)

    if len(x) == 1:
        x = x[0]
    if len(y) == 1:
        y = y[0]

    a = net(x, trainable)
    return a, y, mask


def runData(net, opt, batcher, criterion=maskedCE,
            trainable=False, verbose=False, cuda=True,
            gradClip=10.0, minContext=0, numPrints=10):
    iters = batcher.batches
    meanAcc = CMA()
    meanLoss = CMA()

    for i in range(iters):
        try:
            if verbose and i % int(iters / numPrints) == 0:
                sys.stdout.write('#')
                sys.stdout.flush()
        except:
            pass

        # Always returns mask. None if unused
        a, y, mask = runMinibatch(net, batcher, trainable=trainable, cuda=cuda, volatile=not trainable)

        # Compute loss/acc with proper criterion/masking
        loss, acc = stats(criterion, a, y, mask)

        if trainable:
            opt.zero_grad()
            loss.backward()
            if gradClip is not None:
                t.nn.utils.clip_grad_norm(net.parameters(),
                                          gradClip, norm_type=1)
            opt.step()

        # Accumulate average
        meanLoss.update(loss.data[0])
        meanAcc.update(acc)

    return meanLoss.cma, meanAcc.cma


def stats(criterion, a, y, mask):
    if mask is not None:
        _, preds = t.max(a.data, 2)
        batch, sLen, c = a.size()
        loss = criterion(a.view(-1, c), y.view(-1))

        m = t.sum(mask)
        mask = _sequence_mask(mask, sLen)
        acc = t.sum(mask.data.float() * (y.data == preds).float()) / float(m.data[0])
        # loss = criterion(a.view(-1, c), y.view(-1))
    else:
        _, preds = t.max(a.data, 1)
        loss = criterion(a, y)
        acc = t.mean((y.data == preds).float())

    return loss, acc

class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def get_params_groups(model: torch.nn.Module, weight_decay: float = 1e-5):
    # 记录optimize要训练的权重参数
    parameter_group_vars = {"decay": {"params": [], "weight_decay": weight_decay},
                            "no_decay": {"params": [], "weight_decay": 0.}}

    # 记录对应的权重名称
    parameter_group_names = {"decay": {"params": [], "weight_decay": weight_decay},
                             "no_decay": {"params": [], "weight_decay": 0.}}

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # frozen weights

        if len(param.shape) == 1 or name.endswith(".bias"):
            group_name = "no_decay"
        else:
            group_name = "decay"

        parameter_group_vars[group_name]["params"].append(param)
        parameter_group_names[group_name]["params"].append(name)

    # print("Param groups = %s" % json.dumps(parameter_group_names, indent=2))
    return list(parameter_group_vars.values())

def create_lr_scheduler(optimizer,
                        num_step: int,
                        epochs: int,
                        warmup=True,
                        warmup_epochs=1,
                        warmup_factor=1e-3,
                        end_factor=1e-6):
    assert num_step > 0 and epochs > 0
    if warmup is False:
        warmup_epochs = 0

    def f(x):
        """
        根据step数返回一个学习率倍率因子，
        注意在训练开始之前，pytorch会提前调用一次lr_scheduler.step()方法
        """
        if warmup is True and x <= (warmup_epochs * num_step):
            alpha = float(x) / (warmup_epochs * num_step)
            # warmup过程中lr倍率因子从warmup_factor -> 1
            return warmup_factor * (1 - alpha) + alpha
        else:
            current_step = (x - warmup_epochs * num_step)
            cosine_steps = (epochs - warmup_epochs) * num_step
            # warmup后lr倍率因子从1 -> end_factor
            return ((1 + math.cos(current_step * math.pi / cosine_steps)) / 2) * (1 - end_factor) + end_factor

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=f)