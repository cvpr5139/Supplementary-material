import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage.morphology import binary_dilation
from torchvision import transforms

class ToFloatTensor3D(object):

    def __call__(self, sample):
        X, Y = sample

        X = X.transpose(3, 0, 1, 2)
        Y = Y.transpose(3, 0, 1, 2)

        X = np.float32(X)
        Y = np.float32(Y)
        return torch.from_numpy(X), torch.from_numpy(Y)

class RemoveBackground:

    def __init__(self, threshold, iter = 8):
        self.threshold = threshold
        self.iter = iter

    def __call__(self, sample: tuple) -> tuple:
        X, background = sample

        mask = np.uint8(np.sum(np.abs(np.int32(X) - background), axis=-1) > self.threshold)
        mask = np.expand_dims(mask, axis=-1)

        mask = np.stack([binary_dilation(mask_frame, iterations=self.iter) for mask_frame in mask])

        Y = X*mask

        return X, Y

class ToCrops(object):

    def __init__(self, raw_shape, crop_shape, train = True):
        self.raw_shape = raw_shape
        self.crop_shape = crop_shape
        self.train = train

    def __call__(self, sample: tuple):
        X, Y = sample

        c, t, h, w = self.raw_shape
        cc, tc, hc, wc = self.crop_shape

        crops_X = []
        crops_Y = []

        for k in range(0, t, tc):
            for i in range(0, h, hc // 2):
                for j in range(0, w, wc // 2):
                    if (i + hc <= h) and (j + wc <= w):
                        crops_X.append(X[:, k:k + tc, i:i + hc, j:j + wc])
                        crops_Y.append(Y[:, k:k + tc, i:i + hc, j:j + wc])

        X = torch.stack(crops_X, dim=0)
        Y = torch.stack(crops_Y, dim=0)
        return X, Y

class ToRandomCrops(object):

    def __init__(self, raw_shape, crop_shape):
        self.raw_shape = raw_shape
        self.crop_shape = crop_shape

    def __call__(self, sample: tuple):
        X, Y = sample

        c, t, h, w = self.raw_shape
        cc, tc, hc, wc = self.crop_shape

        crops_X = []
        crops_Y = []

        for k in range(0, t, tc):
            for i in range(0, h, hc // 2):
                for j in range(0, w, wc // 2):
                    rd_t = np.random.randint(0, t - tc)
                    rd_h = np.random.randint(0, h - hc)
                    rd_w = np.random.randint(0, w - wc)

                    crops_X.append(X[:, rd_t:rd_t + tc, rd_h:rd_h + hc, rd_w:rd_w + wc])
                    crops_Y.append(Y[:, rd_t:rd_t + tc, rd_h:rd_h + hc, rd_w:rd_w + wc])

        X = torch.stack(crops_X, dim=0)
        Y = torch.stack(crops_Y, dim=0)

        return X, Y

class ToSpatialCrops(object):

    def __init__(self, raw_shape, crop_shape):
        self.raw_shape = raw_shape
        self.crop_shape = crop_shape

    def __call__(self, sample: tuple):
        X, Y = sample

        c, t, h, w = self.raw_shape
        _, _, hc, wc = self.crop_shape

        if (h-hc)>0:
            rd_h = np.random.randint(0, h - hc)
            rd_w = np.random.randint(0, w - wc)

            crops_X = X[:, :, rd_h:rd_h + hc, rd_w:rd_w + wc]
            crops_Y = Y[:, :, rd_h:rd_h + hc, rd_w:rd_w + wc]
        else:
            crops_X = X
            crops_Y = Y

        return crops_X, crops_Y

class ToCenterCrops(object):

    def __init__(self, raw_shape, crop_shape):
        self.raw_shape = raw_shape
        self.crop_shape = crop_shape

    def __call__(self, sample: tuple):
        X, Y = sample

        c, t, h, w = self.raw_shape
        _, _, hc, wc = self.crop_shape

        if (h-hc)>0:
            rd_h = int((h - hc) / 2)
            rd_w = int((w - wc) / 2)

            crops_X = X[:, :, rd_h:rd_h + hc, rd_w:rd_w + wc]
            crops_Y = Y[:, :, rd_h:rd_h + hc, rd_w:rd_w + wc]
        else:
            crops_X = X
            crops_Y = Y

        return crops_X, crops_Y

class DropoutNoise(object):
    def __init__(self, p):
        self._p = p

    def __call__(self, sample):
        X, X = sample

        X_noise = F.dropout(X, p=self._p, training=True)

        return X_noise, X

class Normalize(object):
    def __call__(self, sample):
        X, Y = sample

        X = X / 255.
        Y = Y / 255.
        if X.size(0) == 1:
            mean = torch.as_tensor([0.5], dtype=X.dtype, device=X.device)
            std = torch.as_tensor([0.5], dtype=X.dtype, device=X.device)
        else:
            mean = torch.as_tensor([0.5, 0.5, 0.5], dtype=X.dtype, device=X.device)
            std = torch.as_tensor([0.5, 0.5, 0.5], dtype=X.dtype, device=X.device)

        if X.ndim == 3:
            X.sub_(mean[:, None, None]).div_(std[:, None, None])
            Y.sub_(mean[:, None, None]).div_(std[:, None, None])
        elif X.ndim == 4:
            X.sub_(mean[:, None, None, None]).div_(std[:, None, None, None])
            Y.sub_(mean[:, None, None, None]).div_(std[:, None, None, None])
        elif X.ndim == 5:
            X.sub_(mean[None, :, None, None, None]).div_(std[None, :, None, None, None])
            Y.sub_(mean[None, :, None, None, None]).div_(std[None, :, None, None, None])
        else:
            print("Dim ERROR")

        return X, Y

class ConcatFlowInput_r(object):
    def __init__(self, DEVICE):
        self.cuda = DEVICE

    def __call__(self, flow, x, x_r):
        assert not x.requires_grad

        x_s = (x[:, :, ::4, :, :].clone().detach()+1)*0.5
        b, c, t, h, w = x_s.size()
        app = []
        for i in range(t):
            images = [transforms.ToPILImage()(im) for im in x_s[:, :, i, :, :].cpu()]
            images = [transforms.Resize((flow[0].size(-2), flow[0].size(-1)))(im) for im in images]
            images = [transforms.Grayscale()(im) for im in images]
            flow_grd = [transforms.ToTensor()(im) for im in images]
            flow_grd = torch.stack(flow_grd, dim=0)
            app.append((flow_grd).unsqueeze(2))
        flow_app = torch.cat(app, dim=2).to(self.cuda)

        b, c, t, h, w = flow_app.size()
        tt = flow_app.permute(0, 2, 1, 3, 4)
        tensor = tt.contiguous().view(b * t, c, h, w)
        return torch.cat((flow[0], tensor), 1)

class ConcatFlowInput_e(object):
    def __init__(self, DEVICE):
        self.cuda = DEVICE

    def __call__(self, flow, x, x_r):
        assert not x.requires_grad
        x_s = torch.abs(x[:, :, ::4, :, :].clone().detach()-x_r[:, :, ::4, :, :].cpu().clone().detach())
        x_s = torch.clamp(x_s, 0, 1)

        b, c, t, h, w = x_s.size()
        app = []
        for i in range(t):
            images = [transforms.ToPILImage()(im) for im in x_s[:, :, i, :, :].cpu()]
            images = [transforms.Resize((flow[0].size(-2), flow[0].size(-1)))(im) for im in images]
            images = [transforms.Grayscale()(im) for im in images]
            flow_grd = [transforms.ToTensor()(im) for im in images]
            flow_grd = torch.stack(flow_grd, dim=0)
            app.append((flow_grd).unsqueeze(2))
        flow_app = torch.cat(app, dim=2).to(self.cuda)

        b, c, t, h, w = flow_app.size()
        tt = flow_app.permute(0, 2, 1, 3, 4)
        tensor = tt.contiguous().view(b * t, c, h, w)

        return torch.cat((flow[0], tensor), 1)

class Static_intensity(object):
    def __init__(self, DEVICE):
        self.cuda = DEVICE

    def __call__(self, x):
        assert not x.requires_grad

        if x.size(-1) ==256:
            scaled = 8
        else:
            scaled = 4
        resize_h = int(x.size(-2) / scaled)
        resize_w = int(x.size(-1) / scaled)

        x_s = (x[:, :, ::4, :, :].clone().detach()+1)*0.5
        b, c, t, h, w = x_s.size()
        app = []
        for i in range(t):
            images = [transforms.ToPILImage()(im) for im in x_s[:, :, i, :, :].cpu()]
            images = [transforms.Resize((resize_h, resize_w))(im) for im in images]
            images = [transforms.Grayscale()(im) for im in images]
            flow_grd = [transforms.ToTensor()(im) for im in images]
            flow_grd = torch.stack(flow_grd, dim=0)
            app.append((flow_grd).unsqueeze(2))
        flow_app = torch.cat(app, dim=2).to(self.cuda)

        b, c, t, h, w = flow_app.size()
        tt = flow_app.permute(0, 2, 1, 3, 4)
        tensor = tt.contiguous().view(b * t, c, h, w)
        return tensor

