import os
import sys
import cv2
import torch
import torchvision
from torchvision import transforms
import numpy as np
import math

from PIL import Image

from torch.utils.data import Dataset
import torch.nn
from pathlib import Path

from . import augmentation
from utils.image import affine_transform, gaussian_radius, gaussian2D, draw_gaussian, get_affine_transform

def _covertCxcy2Ltrb(bboxes_xywh: torch.tensor):
    bboxes_ltrb = bboxes_xywh
    bboxes_ltrb[1] = bboxes_xywh[1] - bboxes_xywh[3] / 2.
    bboxes_ltrb[2] = bboxes_xywh[2] - bboxes_xywh[4] / 2.
    bboxes_ltrb[3] = bboxes_xywh[1] + bboxes_xywh[3] / 2.
    bboxes_ltrb[4] = bboxes_xywh[2] + bboxes_xywh[4] / 2.
    return bboxes_ltrb

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

    mean = np.array([0.40789654, 0.44719302, 0.47026115],
                   dtype=np.float32).reshape(1, 1, 3)
    std  = np.array([0.28863828, 0.27408164, 0.27809835],
                   dtype=np.float32).reshape(1, 1, 3)

    def __init__(self,
                path,
                img_w=416,
                img_h=416,
                classes = 2,
                use_augmentation=True):

                self.imgs_path, self.labels_path = read_data(path)

                self.labels = [np.loadtxt(label_path,
                                          dtype=np.float32,
                                          delimiter =' ').reshape(-1,5) for label_path in self.labels_path]
                
                self.img_w = img_w
                self.img_h = img_h
                self.num_classes = classes

                assert len(self.imgs_path) == len(self.labels_path), "not equal the number of imgs and the number of labels"

                self.use_augmentation = use_augmentation

                self.max_objs = 100
    
    def __getitem__(self, idx):
        assert Path(self.imgs_path[idx]).stem == Path(self.labels_path[idx]).stem, "not equal img name and label name."
        
        img = cv2.imread(self.imgs_path[idx],cv2.IMREAD_COLOR)
    

        center = np.array([img.shape[1]/2., img.shape[0]/2.], dtype = np.float32)
        s = np.array([self.img_w, self.img_h], dtype=np.float32)

        trans_input = get_affine_transform(center, s, 0, [self.img_w, self.img_h])

        inp = cv2.warpAffine(img, trans_input, (self.img_w, self.img_h), flags=cv2.INTER_LINEAR)

        #cv2.imwrite("warp.jpg",inp)

        label = self.labels[idx].copy()

        np.random.shuffle(label)

        bboxes_class = label[:, 0].astype(np.long).reshape(-1,1)
        bboxes_xywh = label[:, 1:].reshape(-1,4)

        if self.use_augmentation:
            inp, bboxes_xywh, bboxes_class = augmentation.random_crop(inp, bboxes_xywh, bboxes_class, p=1.0)
            inp, bboxes_xywh, bboxes_class = augmentation.horizontal_flip(inp, bboxes_xywh, bboxes_class)
            #inp, bboxes_xywh, bboxes_class = augmentation.random_scale(inp, bboxes_xywh, bboxes_class)

        #cv2.imwrite("result.jpg",inp)
        #img = cv2.resize(img, dsize=(self.img_w, self.img_h))

        inp = (inp.astype(np.float32) / 255.)
        inp = (inp - self.mean) / self.std
        inp = inp.transpose(2,0,1)

        img_h, img_w = img.shape[0:2]

        #img = img[..., ::-1].transpose(2,0,1)
        #img = np.ascontiguousarray(img, dtype=np.uint8)
        #img = torch.tensor(img, dtype=torch.float32)/255.

        label = np.concatenate([bboxes_class, bboxes_xywh], axis=1)
        label = torch.tensor(label, dtype=torch.float32)

        num_objs = min(len(label), self.max_objs)

        output_h = img_h // 4
        output_w = img_w // 4
        num_classes = self.num_classes

        trans_output = get_affine_transform(center, s, 0, [output_w, output_h])

        c = np.array([self.img_w / 2., self.img_h / 2.], dtype=np.float32)

        hm = np.zeros([num_classes, output_h, output_w], dtype=np.float32)
        wh = np.zeros((self.max_objs,2), dtype=np.float32)
        dense_wh = np.zeros((2,output_h,output_w), dtype=np.float32)
        reg = np.zeros((self.max_objs,2),dtype=np.float32)
        ind = np.zeros((self.max_objs), dtype=np.int64)
        reg_mask = np.zeros((self.max_objs), dtype=np.uint8)
        cat_spec_wh = np.zeros((self.max_objs, num_classes * 2), dtype=np.float32)
        cat_spec_mask = np.zeros((self.max_objs, num_classes * 2), dtype=np.uint8)

        gt_det = []

        for k in range(0, num_objs):
            anno = label[k]
            bbox = _covertCxcy2Ltrb(anno)
            bbox[1], bbox[3] = bbox[1] * output_w, bbox[3] * output_w
            bbox[2], bbox[4] = bbox[2] * output_h, bbox[4] * output_h
            bbox = bbox.numpy() #torch to numpy
            bbox[1:3] = affine_transform(bbox[1:3],trans_output)
            bbox[3:] = affine_transform(bbox[3:], trans_output)
            bbox[[1,3]] = np.clip(bbox[[1,3]], 0, output_w - 1)
            bbox[[2,4]] = np.clip(bbox[[2,4]], 0, output_h - 1)
            h, w = bbox[4] - bbox[2], bbox[3] - bbox[1]
            class_id = int(bbox[0])
            radius = gaussian_radius((math.ceil(h), math.ceil(w)))
            radius = max(0,int(radius))
            ct = np.array( [(bbox[1] + bbox[3])/2, (bbox[2] + bbox[4])/2], dtype=np.float32 )
            ct_int = ct.astype(np.int32)

            draw_gaussian(hm[class_id], ct_int, radius)
            wh[k] = 1. * w, 1. * h
            ind[k] = ct_int[1] * output_w + ct_int[0]
            reg[k] = ct - ct_int
            reg_mask[k] = 1
            gt_det.append([ct[0] - w/2, ct[1] - h/2, ct[0] + w/2, ct[1] + h/2, 1, class_id])
        
        ret = {'image' : inp, 'hm' : hm, 'reg_mask' : reg_mask, 'ind' : ind, 'wh' : wh, 'gt_det' : gt_det}

        return ret
    
    def __len__(self):
        return len(self.imgs_path)

            
            

if __name__ == "__main__":
    data = Data(root="./datasets", class_path="./datasets/data.names")
    
