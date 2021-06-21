import torch
import numpy as np

from model.losses import FocalLoss

from .base_trainer import BaseTrainer

class CtdetLoss(torch.nn.Module):
    def __init__(self, opt):
        super(CtdetLoss, self).__init__()
        self.crit = FocalLoss()
        self.crit_reg = RegL1Loss()
        self.crit_wh = torch.nn.L1Loss(reduction='sum')

