import time
import torch
from utils.utils import AverageMeter
from progress.bar import Bar


class ModelWithLoss(torch.nn.Module):
    def __init__(self, model, loss):
        super(ModelWithLoss, self).__init__()
        self.model = model
        self.loss = loss
    
    def forward(self, batch):
        #batch['image'] = batch['image'].to(device=torch.device("cuda"), dtype=torch.float32)
        outputs = self.model(batch['image'])
        loss, loss_stats = self.loss(outputs, batch)
        return outputs[-1], loss, loss_stats
    
class BaseTrainer(object):
    def __init__(self, opt, model, optimizer=None):
        self.opt = opt
        self.model = model
        self.optimizer = optimizer
        self.loss_stats, self.loss = self._get_losses(opt)
        self.model_with_loss = ModelWithLoss(model, self.loss)

    def set_device(self, gpus, chunk_sizes, device):
        self.model_with_loss = self.model_with_loss.to(device)

        for state in self.optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device=device, non_blocking=True)

    def run_epoch(self, phase, epoch, data_loader):
        model_with_loss = self.model_with_loss
        if(phase == 'train'):
            model_with_loss.train()
        else:
            model_with_loss.eval()
            torch.cuda.empty_cache()
        
        opt = self.opt 
        results = {}
        data_time, batch_time = AverageMeter(), AverageMeter()
        avg_loss_stats = {l: AverageMeter() for l in self.loss_stats}
        num_iters = len(data_loader)
        bar = Bar('{}/{}'.format('ctdet', 'default'), max=num_iters)
        end = time.time()

        print("num_iters : ", num_iters)
        
        for iter_id, batch in enumerate(data_loader):
            if iter_id >= num_iters:
                break
            data_time.update(time.time() - end)

            for k in batch:
                if k != "gt_det":
                    batch[k] = batch[k].to(device=torch.device("cuda"), non_blocking=True, dtype=torch.float32)

            output, loss, loss_stats = model_with_loss(batch)
            loss = loss.mean()
            if phase == 'train':
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            batch_time.update(time.time() - end)
            end = time.time()

            Bar.suffix = '{phase}: [{0}][{1}/{2}]|Tot: {total:} |ETA: {eta:} '.format(
                epoch, iter_id, num_iters, phase=phase, total=bar.elapsed_td, eta=bar.eta_td)
            for l in avg_loss_stats:
                avg_loss_stats[l].update(
                    loss_stats[l].mean().item(), batch['image'].size(0))
                Bar.suffix = Bar.suffix + '|{} {:.4f} '.format(l, avg_loss_stats[l].avg)
            bar.next()

        bar.finish()
        ret = {k: v.avg for k, v in avg_loss_stats.items()}
        ret['time'] = bar.elapsed_td.total_seconds() / 60.     
        return ret, results

    def _get_losses(self, opt):
        raise NotImplementedError

    def train(self, epoch, data_loader):
        return self.run_epoch('train', epoch, data_loader)
    
    def validate(self, epoch, data_loader):
        return self.run_epoch('val', epoch, data_loader)