import os
import sys
import cv2
import numpy as np
import torch
import torchvision
from torchvision import transforms
import numpy as np

from PIL import Image

from torch.utils.data import Dataset
import torch.nn
from pathlib import Path

from . import augmentation

class Data(Dataset):

    IMAGE_FOLDER = "JPEGImages"
    LABEL_FOLDER = "Annotations"
    IMG_EXTENSIONS = ".jpg"

    def __init__(self, root, train=True, transform=None, target_transform=None, resize=608, class_path="./data.names"):
        self.root = root
        self.transform = transform
        self.target_transform = target_transform
        self.train = train 
        self.resize_factor = resize
        self.class_path = class_path

        with open(class_path) as f:
            self.classes = f.read().splitlines()
        
        if not self._check_exists():
            raise RuntimeError("Dataset not found.")

        self.data = self.cvtData()
    
    def _check_exists(self):
        print("Image Folder : {}".format(os.path.join(self.root, self.IMAGE_FOLDER)))
        print("Label Folder : {}".format(os.path.join(self.root, self.LABEL_FOLDER)))

        return os.path.exists(os.path.join(self.root, self.IMAGE_FOLDER)) and \
               os.path.exists(os.path.join(self.root, self.LABEL_FOLDER))

    def cvtData(self):
        print(self.classes)
        
        result = []

        result.append({os.path.join(self.root, self.IMAGE_FOLDER, "".join([self.classes, self.IMG_EXTENSIONS])): target})
        print(result)
        return result

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):

        key = list(self.data[index].keys())[0]
        img = Image.open(key).convert('RGB')
        current_shape = img.size
        img = img.resize((self.resize_factor, self.resize_factor))

        target = self.data[index][key]

        if self.transform is not None:
            img, aug_target = self.transform([img, target])
            img = torchvision.transforms.ToTensor()(img)
        
        if self.target_transform is not None:
            pass
    
        return img, aug_target, current_shape

def read_data(root):
    imgs_path = []
    labels_path = []

    for r, d, f in os.walk(root):
        for file in f:
            if file.lower().endswith((".png", ".jpg", ".bmp", ".jpeg")):
                img_path = os.path.join(r, file).replace(os.sep, '/')
                label_path = Path(img_path).with_suffix('.txt')
                
                if not os.path.isfile(img_path):
                    continue
                
                if not os.path.isfile(label_path):
                    continue
                
                imgs_path.append(img_path)
                labels_path.append(label_path)
    return imgs_path, labels_path

class YOLODataset(Dataset):

    def __init__(self,
                path,
                img_w=416,
                img_h=416,
                use_augmentation=True):

                self.imgs_path, self.labels_path = read_data(path)

                self.labels = [np.loadtxt(label_path,
                                          dtype=np.float32,
                                          delimiter =' ').reshape(-1,5) for label_path in self.labels_path]
                
                self.img_w = img_w
                self.img_h = img_h

                assert len(self.imgs_path) == len(self.labels_path), "not equal the number of imgs and the number of labels"

                self.use_augmentation = use_augmentation
    
    def __getitem__(self, idx):
        assert Path(self.imgs_path[idx]).stem == Path(self.labels_path[idx]).stem, "not equal img name and label name."
        
        img = cv2.imread(self.imgs_path[idx],cv2.IMREAD_COLOR)

        label = self.labels[idx].copy()
        
        np.random.shuffle(label)

        bboxes_class = label[:, 0].astype(np.long).reshape(-1,1)
        bboxes_xywh = label[:, 1:].reshape(-1,4)


        img_copy = img.copy()
        for box in bboxes_xywh:
            cv2.rectangle(img_copy, (int(img.shape[0] * (box[0] - box[2]/2.)), int(img.shape[1] * (box[1] - box[3]/2.))), (int(img.shape[0] * (box[0] + box[2]/2.)), int( img.shape[1] * (box[1] + box[3]/2.))), (255,0,0), 1)
        
        #cv2.imwrite("test_copy.jpg",img_copy)

        if self.use_augmentation:
            img, bboxes_xywh, bboxes_class = augmentation.random_crop(img, bboxes_xywh, bboxes_class, p=1.0)
            img, bboxes_xywh, bboxes_class = augmentation.horizontal_flip(img, bboxes_xywh, bboxes_class)
            img, bboxes_xywh, bboxes_class = augmentation.random_scale(img, bboxes_xywh, bboxes_class)

        img = cv2.resize(img, dsize=(self.img_w, self.img_h))

        img_h, img_w = img.shape[0:2]

        img = img[..., ::-1].transpose(2,0,1)

        img = np.ascontiguousarray(img, dtype=np.uint8)
        img = torch.tensor(img, dtype=torch.float32)/255.

        label = np.concatenate([bboxes_class, bboxes_xywh], axis=1)
        label = torch.tensor(label, dtype=torch.float32)

        #for box in bboxes_xywh:
        #    cv2.rectangle(img, (int(img_w * (box[0] - box[2]/2.)), int(img_h * (box[1] - box[3]/2.))), (int(img_w * (box[0] + box[2]/2.)), int( img_h * (box[1] + box[3]/2.))), (255,0,0), 1)
        
        return {'image' : img, 'mask': label}
    
    def __len__(self):
        return len(self.imgs_path)

            
            

if __name__ == "__main__":
    data = Data(root="./datasets", class_path="./datasets/data.names")
    
