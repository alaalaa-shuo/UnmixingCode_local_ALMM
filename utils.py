import torch
import torch.nn as nn
import numpy as np
import itertools


def parameters(index=0, time_print=False):

    lr_range = np.linspace(5e-3, 5e-3, 1)
    epoch_range = np.linspace(300, 300, 1, dtype=int)
    re_range = np.linspace(1e2, 1e3, 2)
    sad_range = [1e-1, 1, 1e2]
    abu_range = np.linspace(1e-3, 1e-2, 10)
    sv_a_range = np.linspace(1e-3, 1e-2, 10)
    orth_range = np.linspace(1e-3, 1e-2, 10)
    reg_range = np.linspace(1e-3, 1e-2, 10)
    sv_L_range = np.linspace(90, 90, 1, dtype=int)
    minvol_range = np.linspace(2e-1, 1, 5)
    #
    # # re_range = np.arange(0, 100, 10)
    # sad_range = np.concatenate((np.arange(0, 1, 0.05), np.arange(0, 1100, 100)), axis=0)
    # # abu_range = np.concatenate((np.arange(0, 1, 0.05), np.arange(0, 1100, 100)), axis=0)
    # # reg_range = np.arange(0, 1, 0.005)
    # minvol_range = np.arange(0, 1e-2, 0.002)

    # re_range = np.concatenate((np.arange(0, 1, 0.05), np.arange(0, 1100, 100), np.arange(1000, 6000, 1000)), axis=0)
    # sad_range = np.concatenate((np.arange(0, 1, 0.05), np.arange(0, 1100, 100)), axis=0)
    # abu_range = np.linspace(1e-3, 2e-2, 20)
    # sv_a_range = np.linspace(8e-3, 8e-3, 1)
    # orth_range = np.linspace(1e-3, 1e-3, 1)
    # reg_range = np.linspace(5e-3, 5e-3, 1)
    # sv_L_range = np.linspace(90, 90, 1, dtype=int)
    # minvol_range = np.linspace(4e-1, 4e-1, 1)

    hyperparams_list = list(itertools.product(lr_range, epoch_range, re_range, sad_range, abu_range, sv_a_range, orth_range, reg_range, sv_L_range, minvol_range))
    if time_print:
        times = len(hyperparams_list)
        return times
    else:
        return hyperparams_list[index]


def compute_rmse(x_true, x_pre):
    w, h, c = x_true.shape
    class_rmse = [0] * c
    for i in range(c):
        class_rmse[i] = np.sqrt(((x_true[:, :, i] - x_pre[:, :, i]) ** 2).sum() / (w * h))
    mean_rmse = np.sqrt(((x_true - x_pre) ** 2).sum() / (w * h * c))
    return class_rmse, mean_rmse


def compute_re(x_true, x_pred):
    img_w, img_h, img_c = x_true.shape
    return np.sqrt(((x_true - x_pred) ** 2).sum() / (img_w * img_h * img_c))


def compute_sad(inp, target):
    p = inp.shape[-1]
    sad_err = [0] * p
    for i in range(p):
        inp_norm = np.linalg.norm(inp[:, i], 2)
        tar_norm = np.linalg.norm(target[:, i], 2)
        summation = np.matmul(inp[:, i].T, target[:, i])
        sad_err[i] = np.arccos(summation / (inp_norm * tar_norm))
    mean_sad = np.mean(sad_err)
    return sad_err, mean_sad


def Nuclear_norm(inputs):
    _, band, h, w = inputs.shape
    inp = torch.reshape(inputs, (band, h * w))
    out = torch.norm(inp, p='nuc')
    return out


class SparseKLloss(nn.Module):
    def __init__(self):
        super(SparseKLloss, self).__init__()

    def __call__(self, inp, decay):
        inp = torch.sum(inp, 0, keepdim=True)
        loss = Nuclear_norm(inp)
        return decay * loss


class SumToOneLoss(nn.Module):
    def __init__(self, device):
        super(SumToOneLoss, self).__init__()
        self.register_buffer('one', torch.tensor(1, dtype=torch.float, device=device))
        self.loss = nn.L1Loss(reduction='sum')

    def get_target_tensor(self, inp):
        target_tensor = self.one
        return target_tensor.expand_as(inp)

    def __call__(self, inp, gamma_reg):
        inp = torch.sum(inp, 1)
        target_tensor = self.get_target_tensor(inp)
        loss = self.loss(inp, target_tensor)
        return gamma_reg * loss


class SAD(nn.Module):
    def __init__(self, num_bands):
        super(SAD, self).__init__()
        self.num_bands = num_bands

    def forward(self, inp, target):
        try:
            input_norm = torch.sqrt(torch.bmm(inp.view(-1, 1, self.num_bands),
                                              inp.view(-1, self.num_bands, 1)))
            target_norm = torch.sqrt(torch.bmm(target.view(-1, 1, self.num_bands),
                                               target.view(-1, self.num_bands, 1)))

            summation = torch.bmm(inp.view(-1, 1, self.num_bands), target.view(-1, self.num_bands, 1))
            angle = torch.acos(summation / (input_norm * target_norm))

        except ValueError:
            return 0.0

        return angle


class SID(nn.Module):
    def __init__(self, epsilon: float = 1e5):
        super(SID, self).__init__()
        self.eps = epsilon

    def forward(self, inp, target):
        normalize_inp = (inp / torch.sum(inp, dim=0)) + self.eps
        normalize_tar = (target / torch.sum(target, dim=0)) + self.eps
        sid = torch.sum(normalize_inp * torch.log(normalize_inp / normalize_tar) +
                        normalize_tar * torch.log(normalize_tar / normalize_inp))

        return sid
