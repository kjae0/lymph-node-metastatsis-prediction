import torch
import torch.nn as nn
import numpy as np
import random
import os


def set_seeds(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
def dice_loss(pred, target, smooth = 1e-5, coef=0.01):
    # binary cross entropy loss
    bce = nn.BCEWithLogitsLoss(reduction='mean')(pred, target)
    
    pred = torch.sigmoid(pred)
    intersection = (pred * target).sum(dim=(2,3))
    union = pred.sum(dim=(2,3)) + target.sum(dim=(2,3))
    
    # dice coefficient
    dice = 2.0 * (intersection + smooth) / (union + smooth)
    
    # dice loss
    dice_loss = 1.0 - dice
    
    # total loss
    loss = bce + dice_loss * coef
    return loss.sum()
    
def segmentation_metric(pred, true):
    union = (pred != 0) | (true != 0)
    intersection = (pred != 0) & (true != 0)
    wrong = (pred != 0) & (true == 0)
    miss = (pred == 0) & (true != 0)
    intersection = intersection.sum()
    wrong = wrong.sum()
    miss = miss.sum()
    union = union.sum()
    # 맞게, 잘못, 못 예측한
    return union, intersection, wrong, miss    
    