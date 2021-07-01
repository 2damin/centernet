import argparse
import torch
import cv2
import os

from dataloader import dataloader
from model import model
from torch.utils.data import DataLoader
from trains.ctdet_trainer import CtdetTrainer

def read_data_file(path):
    options = dict()
    with open(path, 'r') as f:
        lines = f.readlines()
    for line in lines:
        line = line.strip()
        if line == "":
            continue
        key, value = line.split('=')
        options[key] = value
    return options

def save_model(path, epoch, model, optimizer=None):
    if isinstance(model, torch.nn.DataParallel):
        state_dict = model.module.state_dict()
    else:
        state_dict = model.state_dict()
    data = {'epoch': epoch, 'state_dict': state_dict}

    if not (optimizer is None):
        data['optimizer'] = optimizer.state_dict()
    torch.save(data, path)

def train(trainDataset,
          validDataset,
          model,
          batch_size: int = 1,
          shuffle: bool = True,
          num_workers: int = 0,
          opt = dict()):

    train_loader = DataLoader(trainDataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=1)
    valid_loader = DataLoader(validDataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=1)

    lr_steps_str = opt['lr_step'].split(",")
    lr_steps = []
    for lr in lr_steps_str:
        lr_steps.append(int(lr))

    print("lr : ", lr_steps)

    device = torch.device("cuda")

    model.to(device=device)
    model.train()

    optimizer = torch.optim.Adam(model.parameters(), float(opt["lr"]))

    train = CtdetTrainer(opt, model, optimizer)

    for epoch in range(int(opt['num_epoch'])):
        train.train(epoch, data_loader=train_loader)
        #for batch in train_loader:
            #x = batch['image']
            #x = x.to(device=device, dtype=torch.float32)
            #pred = model(x)
            #CtdetLoss.forward(pred, )
        if epoch % int(opt['val_interval']) == 0 and epoch != 0:
            print("Validate")
            save_model(os.path.join(opt['save_dir'], 'model_best.pth'),epoch,model,optimizer)
            model.eval()
            with torch.no_grad():
                log_dict_val, preds = train.validate(epoch, data_loader=valid_loader)

        if epoch in lr_steps:
            lr = opt.lr * ( 0.1 ** (opt.lr_step.index(epoch) + 1))
            print("Drop LR to", lr)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
    return model

if __name__ == "__main__":

    print("run train")
    data_config = read_data_file("./centernet.data")
    print(" read data_config file ")

    trainDataset = dataloader.YOLODataset(data_config['train'],
                                  img_w = 608,
                                  img_h = 608,
                                  classes = int(data_config['classes']),
                                  use_augmentation = True)

    validDataset = dataloader.YOLODataset(data_config['valid'],
                                  img_w = 608,
                                  img_h = 608,
                                  classes = int(data_config['classes']),
                                  use_augmentation = False)

    print("Train dataset : ", trainDataset.__len__())

    print( "load dataset ")

    heads = {'hm' : int(data_config['classes']),
            'wh' : 2,
            'reg' : 2}

    model = model.CenterNet_ResNet(3, 2, heads )

    train(trainDataset, validDataset, model, batch_size = 8, shuffle = True, num_workers = 1, opt = data_config)

    print("done")
