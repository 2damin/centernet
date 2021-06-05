import argparse
from dataloader import dataloader
from model import model

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


def train(dataset, model):
    return model

    



if __name__ == "__main__":

    print("run train")
    data_config = read_data_file("./centernet.data")
    print(" read data_config file ")

    dataset = dataloader.YOLODataset(data_config['valid'],
                                  img_w = 608,
                                  img_h = 608,
                                  use_augmentation = True)

    print( "load dataset ")

    model = model.ResNet(3, 2)

    # for idx in range(len(dataset.imgs_path)):
    #     dataset.__getitem__(idx)

    train(dataset, model)

    print("done")