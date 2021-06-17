import argparse
import torch
import cv2

from dataloader import dataloader
from model import model
from torch.utils.data import DataLoader

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


def train(trainDataset,
          validDataset,
          model,
          batch_size: int = 1,
          shuffle: bool = True,
          num_workers: int = 0):

    train_loader = DataLoader(trainDataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=1)

    device = torch.device("cuda")

    model.to(device=device)
    model.eval()

    print(len(train_loader))

    for batch in train_loader:
        x = batch['image']
        x = x.to(device=device, dtype=torch.float32)
        pred = model(x)
        #print(x)
        #cv2.imwrite("x.jpg",x.numpy())
        #pred = model()
    
    return model

if __name__ == "__main__":

    print("run train")
    data_config = read_data_file("./centernet.data")
    print(" read data_config file ")

    trainDataset = dataloader.YOLODataset(data_config['train'],
                                  img_w = 608,
                                  img_h = 608,
                                  use_augmentation = True)

    validDataset = dataloader.YOLODataset(data_config['valid'],
                                  img_w = 608,
                                  img_h = 608,
                                  use_augmentation = False)

    print( "load dataset ")

    heads = {'hm' : int(data_config['classes']),
            'wh' : 2}

    model = model.CenterNet_ResNet(3, 2, heads )

    train(trainDataset, validDataset, model, batch_size = 1, shuffle = True, num_workers = 1)

    print("done")
