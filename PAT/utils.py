import os
import torch
import numpy as np
from skimage import metrics
from torch.nn import init

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

class L1Loss(object):
    def __call__(self, input, target):
        return torch.abs(input - target).mean()

def cal_psnr(img1, img2):
    img1_np = np.array(img1.cpu())
    img2_np = np.array(img2.cpu())
    return metrics.peak_signal_noise_ratio(img1_np, img2_np, data_range=1.0)

def cal_ssim(img1, img2):
    img1_np = np.array(img1.squeeze().cpu()).transpose(1,2,0)
    img2_np = np.array(img2.squeeze().cpu()).transpose(1,2,0)
    return metrics.structural_similarity(img1_np, img2_np, multichannel=True, data_range=1.0)

def save_ckpt(state, save_path='./log', filename='checkpoint.pth.tar'):
    torch.save(state, os.path.join(save_path,filename))

def weights_init_xavier(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1:
        init.xavier_normal_(m.weight.data)
