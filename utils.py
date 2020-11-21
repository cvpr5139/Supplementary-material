import collections
import random
import re
import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.data.dataloader import string_classes
from torch.optim.lr_scheduler import _LRScheduler
from torch._six import int_classes
_use_shared_memory = False
numpy_type_map = {
    'float64': torch.DoubleTensor,
    'float32': torch.FloatTensor,
    'float16': torch.HalfTensor,
    'int64': torch.LongTensor,
    'int32': torch.IntTensor,
    'int16': torch.ShortTensor,
    'int8': torch.CharTensor,
    'uint8': torch.ByteTensor,
}
class GradualWarmupScheduler(_LRScheduler):

    def __init__(self, optimizer, multiplier, total_epoch, after_scheduler=None):
        self.multiplier = multiplier
        if self.multiplier < 1.:
            raise ValueError('multiplier should be greater thant or equal to 1.')
        self.total_epoch = total_epoch
        self.after_scheduler = after_scheduler
        self.finished = False
        super(GradualWarmupScheduler, self).__init__(optimizer)

    def get_lr(self):
        if self.last_epoch > self.total_epoch:
            if self.after_scheduler:
                if not self.finished:
                    self.after_scheduler.base_lrs = [base_lr * self.multiplier for base_lr in self.base_lrs]
                    self.finished = True
                return self.after_scheduler.get_last_lr()
            return [base_lr * self.multiplier for base_lr in self.base_lrs]

        if self.multiplier == 1.0:
            return [base_lr * (float(self.last_epoch) / self.total_epoch) for base_lr in self.base_lrs]
        else:
            return [base_lr * ((self.multiplier - 1.) * self.last_epoch / self.total_epoch + 1.) for base_lr in self.base_lrs]

    def step(self, epoch=None, metrics=None):
        if self.finished and self.after_scheduler:
            if epoch is None:
                self.after_scheduler.step(None)
            else:
                self.after_scheduler.step(epoch - self.total_epoch)
            self._last_lr = self.after_scheduler.get_last_lr()
        else:
            return super(GradualWarmupScheduler, self).step(epoch)

def concat_collate(batch):
    error_msg = "batch must contain tensors, numbers, dicts or lists; found {}"
    elem_type = type(batch[0])
    if isinstance(batch[0], torch.Tensor):
        out = None
        if _use_shared_memory:
            numel = sum([x.numel() for x in batch])
            storage = batch[0].storage()._new_shared(numel)
            out = batch[0].new(storage)
        return torch.cat(batch, 0, out=out)
    elif elem_type.__module__ == 'numpy' and elem_type.__name__ != 'str_' \
            and elem_type.__name__ != 'string_':
        elem = batch[0]
        if elem_type.__name__ == 'ndarray':
            if re.search('[SaUO]', elem.dtype.str) is not None:
                raise TypeError(error_msg.format(elem.dtype))

            return torch.cat([torch.from_numpy(b) for b in batch], 0)
        if elem.shape == ():  # scalars
            py_type = float if elem.dtype.name.startswith('float') else int
            return numpy_type_map[elem.dtype.name](list(map(py_type, batch)))
    elif isinstance(batch[0], int_classes):
        return torch.LongTensor(batch)
    elif isinstance(batch[0], float):
        return torch.DoubleTensor(batch)
    elif isinstance(batch[0], string_classes):
        return batch
    elif isinstance(batch[0], collections.Mapping):
        return {key: concat_collate([d[key] for d in batch]) for key in batch[0]}
    elif isinstance(batch[0], collections.Sequence):
        transposed = zip(*batch)
        return [concat_collate(samples) for samples in transposed]

    raise TypeError((error_msg.format(type(batch[0]))))

def novelty_score(sample_llk_norm, sample_rec_norm, lamb =1):
    ns = (sample_llk_norm * lamb + sample_rec_norm)/(lamb+1)*2
    return ns

def plot_score_curves(dir, gt, label, score, name = ''):

    x = np.arange(0, len(score))
    fig, ax1 = plt.subplots(figsize=(20,5))
    ax1.set_xlabel('Frames', fontsize=50)
    ax1.set_ylabel('Score', fontsize=50)
    ln1 = ax1.plot(x, score, color='tab:green', linewidth=3.0)
    ax1.grid()
    lns = ln1

    normal = True
    for i in range(len(gt)):
        if gt[i] == 1:
            if normal:
                abnormal1 = i
                normal = False
        else:
            if not normal:
                abnormal2 = i
                normal = True
                ax1.axvspan(abnormal1, abnormal2, alpha=0.2, color='red')
    if not normal:
        ax1.axvspan(abnormal1, len(gt) - 1, alpha=0.2, color='red')

    plt.xticks(fontsize=30)
    plt.yticks(fontsize=30)
    plt.legend(lns, [label], fontsize=30)
    plt.tight_layout()
    plt.savefig(dir + '/learning_curve_' + name + '.png', dpi=1000)
    plt.close()

def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def normalize(samples, min, max):
    return (samples - min) / (max - min)

class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val   = 0
        self.avg   = 0
        self.sum   = 0
        self.count = 0

    def update(self, val, n = 1):
        self.val   = val
        self.sum   += val * n
        self.count += n
        self.avg   = self.sum / self.count

def save_txt(filename, txt):
    configtxt = open(filename, 'a+')
    configtxt.write(txt)
    configtxt.close()

def histogram(normal, abnormal, modelsave, name = ''):
    if len(normal) == 0:
        thetype = type(abnormal[0])
        if thetype == torch.Tensor:
            normal = torch.Tensor([])
        else:
            normal = np.array([])
    else:
        thetype = type(normal[0])

    if len(abnormal)==0:
        thetype = type(normal[0])
        if thetype == torch.Tensor:
            abnormal = torch.Tensor([])
        else:
            abnormal = np.array([])

    if thetype == torch.Tensor:
        plt.figure()
        plt.title(name)
        plt.xlabel("Negative bits per dimension")
        if len(normal) != 0:
            normal_nll = torch.cat(normal).cpu()
        else:
            normal_nll = normal
        if len(abnormal) != 0:
            abnormal_nll = torch.cat(abnormal).cpu()
        else:
            abnormal_nll = abnormal
        plt.hist(-normal_nll.numpy(), label="Normal", density=True, bins=20, alpha=0.6)
        plt.hist(-abnormal_nll.numpy(), label="Abnormal", density=True, bins=20, alpha=0.6)
        # plt.legend()
        plt.legend(prop={"size": 20})
        plt.xticks(fontsize=15)
        plt.yticks(fontsize = 15)
        plt.savefig(modelsave + '/histogram1_'+name+'.png', dpi=1000)
    else:
        plt.figure()
        plt.title(name)
        plt.xlabel("Negative bits per dimension")
        if len(normal) != 0:
            normal_nll = np.concatenate(normal)
        else:
            normal_nll = normal
        if len(abnormal) != 0:
            abnormal_nll = np.concatenate(abnormal)
        else:
            abnormal_nll = abnormal
        plt.hist(-normal_nll, label="Normal", density=True, bins=20, alpha=0.6)
        plt.hist(-abnormal_nll, label="Abnormal", density=True, bins=20, alpha=0.6)
        # plt.legend()
        plt.legend(prop={"size": 20})
        plt.xticks(fontsize=15)
        plt.yticks(fontsize=15)
        plt.savefig(modelsave + '/histogram_'+name+'.png', dpi=1000)
    plt.close()

def save_checkpoint(path, model, optimizer=None, scheduler= None, batch = None):
    if optimizer is not None:
        torch.save({
            'batch': batch + 1,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
        }, path)
    else:
        torch.save({
            'state_dict': model.state_dict()
        }, path)

def call_checkpoint(path, model, optimizer=None, scheduler= None):
    if optimizer is not None:
        checkpoint = torch.load(path)
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        check_batch = checkpoint['batch'] - 1
        return model, optimizer, scheduler, check_batch
    else:
        checkpoint = torch.load(path)
        model.load_state_dict(checkpoint['state_dict'])
        return model
