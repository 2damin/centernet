
import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import _transpose_and_gather_feat

def _neg_loss(pred, gt):

    pos_inds = gt.eq(1).float()
    neg_inds = gt.lt(1).float()

    neg_weight = torch.pow(1 - gt, 4)

    loss = 0

    pos_loss = torch.pow(1 - pred, 2) * torch.log(pred) * pos_inds
    neg_loss = neg_weight * torch.pow(pred, 2) * torch.log(1 - pred) * neg_inds

    num_pos = pos_inds.float().sum()
    pos_loss = pos_loss.sum()
    neg_loss = neg_loss.sum()

    if num_pos == 0:
        loss = loss - neg_loss
    else:
        loss = loss - (pos_loss + neg_loss) / num_pos

    return loss

class FocalLoss():
    def __init__(self):
        super(FocalLoss, self).__init__()
        self.neg_loss = _neg_loss
    
    def forward(self, out, target):
        return self.neg_loss(out, target)

class RegL1Loss(nn.Module):
    def __init__(self):
        super(RegL1Loss, self).__init__()

    def forward(self, out, mask, ind, target):
        pred = _transpose_and_gather_feat(out, ind)
        mask = mask.unsqueeze(2).expand_as(pred).float()
        loss = F.l1_loss(pred * mask, target * mask, size_average=False)
        loss = loss / (mask.sum() + 1e-4)
        return loss

class NormRegL1Loss(nn.Module):
  def __init__(self):
    super(NormRegL1Loss, self).__init__()
  
  def forward(self, output, mask, ind, target):
    pred = _transpose_and_gather_feat(output, ind)
    mask = mask.unsqueeze(2).expand_as(pred).float()
    # loss = F.l1_loss(pred * mask, target * mask, reduction='elementwise_mean')
    pred = pred / (target + 1e-4)
    target = target * 0 + 1
    loss = F.l1_loss(pred * mask, target * mask, size_average=False)
    loss = loss / (mask.sum() + 1e-4)
    return loss