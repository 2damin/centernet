import argparse
from dataloader import dataloader

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


def train():
    args = argparse.ArgumentParser()



if __name__ == "__main__":
    data_config = read_data_file("./centernet.data")

    dataset = dataloader.YOLODataset(data_config['train'],
                                  img_w = 608,
                                  img_h = 608,
                                  use_augmentation = True)

    for idx in range(len(dataset.imgs_path)):
        dataset.__getitem__(idx)