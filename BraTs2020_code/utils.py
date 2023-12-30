
import numpy as np
import torch
import torch.nn as nn
# import torch.nn.functional as F
import torchvision.utils as vutils
from PIL import Image
import pandas as pd


def softmax_helper(x): return torch.softmax(x, 1)
def sigmoid_helper(x): return torch.sigmoid(x)


class InitWeights_He(object):
    def __init__(self, neg_slope=1e-2):
        self.neg_slope = neg_slope

    def __call__(self, module):
        if isinstance(module, nn.Conv3d) or isinstance(module, nn.Conv2d) or isinstance(module, nn.ConvTranspose2d) or isinstance(module, nn.ConvTranspose3d):
            module.weight = nn.init.kaiming_normal_(
                module.weight, a=self.neg_slope)
            if module.bias is not None:
                module.bias = nn.init.constant_(module.bias, 0)


def maybe_to_torch(d):
    if isinstance(d, list):
        d = [maybe_to_torch(i) if not isinstance(
            i, torch.Tensor) else i for i in d]
    elif not isinstance(d, torch.Tensor):
        d = torch.from_numpy(d).float()
    return d


def to_cuda(data, non_blocking=True, gpu_id=0):
    if isinstance(data, list):
        data = [i.cuda(gpu_id, non_blocking=non_blocking) for i in data]
    else:
        data = data.cuda(gpu_id, non_blocking=non_blocking)
    return data


class no_op(object):
    def __enter__(self):
        pass

    def __exit__(self, *args):
        pass


def staple(a):
    # a: n,c,h,w detach tensor
    mvres = mv(a)
    gap = 0.4
    if gap > 0.02:
        for i, s in enumerate(a):
            r = s * mvres
            res = r if i == 0 else torch.cat((res, r), 0)
        nres = mv(res)
        gap = torch.mean(torch.abs(mvres - nres))
        mvres = nres
        a = res
    return mvres


def allone(disc, cup):
    disc = np.array(disc) / 255
    cup = np.array(cup) / 255
    res = np.clip(disc * 0.5 + cup, 0, 1) * 255
    res = 255 - res
    res = Image.fromarray(np.uint8(res))
    return res


def dice_score(pred, targs):
    pred = (pred > 0).float()
    return 2. * (pred*targs).sum() / (pred+targs).sum()


def mv(a):
    # res = Image.fromarray(np.uint8(img_list[0] / 2 + img_list[1] / 2 ))
    # res.show()
    b = a.size(0)
    return torch.sum(a, 0, keepdim=True) / b


def tensor_to_img_array(tensor):
    image = tensor.cpu().detach().numpy()
    image = np.transpose(image, [0, 2, 3, 1])
    return image


def export(tar, img_path=None):
    # image_name = image_name or "image.jpg"
    c = tar.size(1)
    if c == 3:
        vutils.save_image(tar, fp=img_path)
    else:
        s = torch.tensor(tar)[:, -1, :, :].unsqueeze(1)
        s = torch.cat((s, s, s), 1)
        vutils.save_image(s, fp=img_path)


def norm(t):
    m, s, v = torch.mean(t), torch.std(t), torch.var(t)
    return (t - m) / s


def iou_score(output, target):
    smooth = 1e-5

    if torch.is_tensor(output):
        output = torch.sigmoid(output).data.cpu().numpy()
    if torch.is_tensor(target):
        target = target.data.cpu().numpy()
    output_ = output > 0.5
    target_ = target > 0.5
    intersection = (output_ & target_).sum()
    union = (output_ | target_).sum()
    iou = (intersection + smooth) / (union + smooth)
    dice = (2 * iou) / (iou+1)
    return iou, dice


def as_num(x):
    y = '{:.10f}'.format(x)  # .10f 保留10位小数
    return y


def diceCoeff(pred, gt, smooth=1e-5, activation='sigmoid'):
    r""" computational formula：
        dice = (2 * (pred ∩ gt)) / (pred ∪ gt)
    """

    if activation is None or activation == "none":
        def activation_fn(x): return x
    elif activation == "sigmoid":
        activation_fn = nn.Sigmoid()
    elif activation == "softmax2d":
        activation_fn = nn.Softmax2d()
    else:
        raise NotImplementedError(
            "Activation implemented for sigmoid and softmax2d 激活函数的操作")

    pred = activation_fn(pred)

    N = gt.size(0)
    pred_flat = pred.view(N, -1)
    gt_flat = gt.view(N, -1)

    intersection = (pred_flat * gt_flat).sum(1)
    unionset = pred_flat.sum(1) + gt_flat.sum(1)
    loss = (2 * intersection + smooth) / (unionset + smooth)

    return loss.sum() / N
