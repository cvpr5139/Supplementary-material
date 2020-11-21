import torch
from torch import nn
import torch.nn.functional as F
from math import exp
import numpy as np
from utils import AverageMeter

class LogReconLoss(nn.Module):
    def __init__(self):
        super(LogReconLoss, self).__init__()

    def forward(self, x, x_r, fg):
        b, _, _, _, _ = x.size()
        L = torch.abs((((x + 1) / 2.) - ((x_r + 1) / 2.)))
        if fg is not None:
            fg1 = fg.sum(1)
            L = L.mean(1)
            L[np.where(fg1.detach().numpy() == -3.)] = 0
            batch_errors = (torch.log10(1 - (torch.mean(L)))) * (-1)
        else:
            batch_errors = (torch.log10(1 - (torch.mean(L)))) * (-1)

        return torch.mean(batch_errors)

class PatchLoss(nn.Module):

    def __init__(self):
        super(PatchLoss, self).__init__()

    def forward(self, x, x_r):
        _, c, t, h, w = x.size()
        if w ==512:
            size = 64
        elif (w < 512) and (w >= 256):
            size = 32
        else:
            print("error size")
        tc = 1
        hc = size//2
        wc = size//2

        L = torch.abs((((x + 1) / 2.) - ((x_r + 1) / 2.)))
        Diff = L.clone().detach()
        Diff = Diff.squeeze(0)

        max_patches = []

        for k in range(0, t, tc):
            max_patch = torch.Tensor([0.])
            for i in range(0, h, hc):
                for j in range(0, w, wc):
                    if (i + hc <= h) and (j + wc <= w):
                            patch_val = torch.mean(Diff[:, k:k + tc, i:i + hc, j:j + wc])
                            if max_patch < patch_val:
                                max_patch = patch_val
            max_patches.append(max_patch)

        max_patches = torch.stack(max_patches, dim=0)

        return torch.log(torch.mean(max_patches))

class NLL(nn.Module):
    def __init__(self):
        super(NLL, self).__init__()
    def forward(self, nll):
        return nll.mean()

class TestMethod(nn.Module):
    def __init__(self, dataset, DEVICE):
        super(TestMethod, self).__init__()

        self.reconstruction_loss_patch = None
        self.reconstruction_loss_logrecon = None

        if dataset == 'Ped2':
            self.reconstruction_loss_patch = PatchLoss().to(DEVICE)
            self.reconstruction_loss_patch.eval()
        else:
            self.reconstruction_loss_logrecon = LogReconLoss().to(DEVICE)
            self.reconstruction_loss_logrecon.eval()

        self.total_loss = None

    def forward(self, x, fg, x_r):
        self.reconstruction_loss = 0.
        if self.reconstruction_loss_logrecon is not None:
            rec_loss =self.reconstruction_loss_logrecon(x, x_r, fg)
            self.reconstruction_loss += rec_loss.item()
        if self.reconstruction_loss_patch is not None:
            rec_loss =self.reconstruction_loss_patch(x, x_r)
            self.reconstruction_loss += rec_loss.item()

        return self.reconstruction_loss