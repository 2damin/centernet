import os
import sys
import torch
import numpy as np

from model.losses import FocalLoss, RegL1Loss, NormRegL1Loss

from .base_trainer import BaseTrainer

from utils.utils import _sigmoid

class CtdetLoss(torch.nn.Module):
    def __init__(self, opt):
        super(CtdetLoss, self).__init__()
        self.crit = FocalLoss()
        self.crit_reg = RegL1Loss()
        self.crit_wh = NormRegL1Loss()
        self.opt = opt
    
    def forward(self, outputs, batch):
        opt = self.opt
        hm_loss, wh_loss, off_loss = 0, 0, 0

        output = outputs[0]

        output['hm'] = _sigmoid(output['hm'])

        batch['hm'] = batch['hm'].to(device=torch.device("cuda"), dtype=torch.float32)
        batch['wh'] = batch['wh'].to(device=torch.device("cuda"), dtype=torch.float32)
        batch['reg_mask'] = batch['reg_mask'].to(device=torch.device("cuda"), dtype=torch.float32)
        batch['reg'] = batch['reg'].to(device=torch.device("cuda"), dtype=torch.float32)

        hm_loss += self.crit.forward(output['hm'], batch['hm'])
        wh_loss += self.crit_wh.forward(output['wh'], batch['reg_mask'], batch['ind'], batch['wh'])
        off_loss += self.crit_reg.forward(output['reg'], batch['reg_mask'], batch['ind'], batch['reg'])


        loss = float(opt["hm_weight"]) * hm_loss + float(opt["wh_weight"]) * wh_loss + float(opt["off_weight"]) * off_loss
        loss_stats = {'loss': loss, 'hm_loss':hm_loss, 'wh_loss' : wh_loss, 'off_loss' : off_loss}
        
        return loss, loss_stats


class CtdetTrainer(BaseTrainer):
    def __init__(self, opt, model, optimizer=None):
        super(CtdetTrainer, self).__init__(opt, model, optimizer=optimizer)
    
    def _get_losses(self, opt):
        loss_states = ['loss', 'hm_loss', 'wh_loss', 'off_loss']
        loss = CtdetLoss(opt)
        return loss_states, loss
