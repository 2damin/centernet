import torchvision
import numpy as np
import random
import cv2

import sys

def LINE():
    return sys._getframe(1).f_lineno


def covertCxcy2Ltrb(bboxes_xywh: np.ndarray):
    bboxes_ltrb = bboxes_xywh.copy()
    bboxes_ltrb[:,0] = bboxes_xywh[:,0] - bboxes_xywh[:,2] / 2.
    bboxes_ltrb[:,1] = bboxes_xywh[:,1] - bboxes_xywh[:,3] / 2.
    bboxes_ltrb[:,2] = bboxes_xywh[:,0] + bboxes_xywh[:,2] / 2.
    bboxes_ltrb[:,3] = bboxes_xywh[:,1] + bboxes_xywh[:,3] / 2.
    return bboxes_ltrb


def covertLtrb2Cxcy(bboxes_ltrb: np.ndarray):
    bboxes_cxcy = bboxes_ltrb.copy()
    bboxes_cxcy[:,0] = (bboxes_ltrb[:,2] + bboxes_ltrb[:,0]) / 2.
    bboxes_cxcy[:,1] = (bboxes_ltrb[:,3] + bboxes_ltrb[:,1]) / 2.
    bboxes_cxcy[:,2] = bboxes_ltrb[:,2] - bboxes_ltrb[:,0]
    bboxes_cxcy[:,3] = bboxes_ltrb[:,3] - bboxes_ltrb[:,1]
    return bboxes_cxcy

def random_resize(img, bboxes_xywh, bboxes_class, p = 0.5):
    if random.random() > p:
        scale_ratio = random.randrange(70, 100) / 100
        img_w = img.shape[1]
        img_h = img.shape[0]
        img = cv2.resize(img, dsize=(0,0), fx = scale_ratio, fy=scale_ratio, interpolation=cv2.INTER_LINEAR )
        return img, bboxes_xywh, bboxes_class
    return img, bboxes_xywh, bboxes_class

def random_scale(img, bboxes_xywh, bboxes_class, p = 0.1):
    if random.random() > p:
        bboxes_ltrb = covertCxcy2Ltrb(bboxes_xywh)
        img_h, img_w = img.shape[0:2]
        if bboxes_ltrb.shape[0] == 0:
            bboxes_xmin = 0
            bboxes_ymin = 0
            bboxes_xmax = img_w - 1
            bboxes_ymax = img_h - 1
            gap_x = random.randrange(0, img_w/10)
            gap_y = random.randrange(0, img_h/10)
        else:
            bboxes_xmin = int(np.min(bboxes_ltrb[:,0]) * img_w)
            bboxes_ymin = int(np.min(bboxes_ltrb[:,1]) * img_h)
            bboxes_xmax = int(np.max(bboxes_ltrb[:,2]) * img_w)
            bboxes_ymax = int(np.max(bboxes_ltrb[:,3]) * img_h)
            gap_x = random.randrange(0,int(min(bboxes_xmin, img_w - bboxes_xmax ) - 1))
            gap_y = random.randrange(0,int(min(bboxes_ymin, img_h - bboxes_ymax) - 1))

        scale_size = max(bboxes_xmax - bboxes_xmin, bboxes_ymax - bboxes_ymin) + 1

        img_crop = img[max(0,bboxes_ymin - gap_y) : min(img_h - 1, bboxes_ymin + scale_size + gap_y) , max(0,bboxes_xmin - gap_x) : min(img_w - 1, bboxes_xmin + scale_size + gap_x)]

        crop_img_h, crop_img_w = img_crop.shape[0:2]

        bboxes_ltrb[:,[0,2]] *= img_w
        bboxes_ltrb[:,[1,3]] *= img_h

        bboxes_ltrb[:,[0,2]] -= bboxes_xmin - gap_x
        bboxes_ltrb[:,[1,3]] -= bboxes_ymin - gap_y

        bboxes_ltrb[:,[0,2]] /= crop_img_w
        bboxes_ltrb[:,[1,3]] /= crop_img_h

        bboxes_xywh = covertLtrb2Cxcy(bboxes_ltrb)

        return img_crop, bboxes_xywh, bboxes_class
    return img, bboxes_xywh, bboxes_class

def horizontal_flip(img, bboxes_xywh, bboxes_class, p = 0.5):
    if random.random() > p:
        img = np.fliplr(img)
        bboxes_xywh[:,0] = 1. - bboxes_xywh[:,0]
        return img, bboxes_xywh, bboxes_class
    return img, bboxes_xywh, bboxes_class


def random_crop(img, bboxes_xywh, bboxes_class, trial=50, p=1.0, border_value = 127):
    if random.random() < p:
        img_h, img_w = img.shape[0:2]
        
        bboxes_ltrb = covertCxcy2Ltrb(bboxes_xywh)

        bboxes_ltrb[:,[0,2]] *= img_w
        bboxes_ltrb[:,[1,3]] *= img_h

        bboxes_w = bboxes_ltrb[:,2] - bboxes_ltrb[:,0]
        bboxes_h = bboxes_ltrb[:,3] - bboxes_ltrb[:,1]

        bboxes_area = bboxes_w * bboxes_h

        for _ in range(trial):

            crop_w = random.randint(img_w // 8, img_w)
            crop_h = random.randint(img_h // 8, img_h)

            crop_xmin = random.randint(0, max(img_w - crop_w - 1, 1))
            crop_ymin = random.randint(0, max(img_h - crop_h - 1, 1))

            crop_xmax = crop_xmin + crop_w
            crop_ymax = crop_ymin + crop_h

            bboxes_selected_ltrb = bboxes_ltrb.copy()
            
            bboxes_selected_ltrb[:, [0,2]] = np.clip(bboxes_selected_ltrb[:,[0,2]], a_min=crop_xmin, a_max=crop_xmax)
            bboxes_selected_ltrb[:, [1,3]] = np.clip(bboxes_selected_ltrb[:,[1,3]], a_min=crop_ymin, a_max=crop_ymax)

            bboxes_selected_w = bboxes_selected_ltrb[:,2] - bboxes_selected_ltrb[:,0]
            bboxes_selected_h = bboxes_selected_ltrb[:,3] - bboxes_selected_ltrb[:,1]
            bboxes_selected_area = bboxes_selected_w * bboxes_selected_h

            iou = bboxes_selected_area / bboxes_area

            size_constraint = (iou > 0.1) & (bboxes_selected_area > 0) & (bboxes_selected_w > 3) & (bboxes_selected_h > 3)

            bboxes_selected_ltrb = bboxes_selected_ltrb[size_constraint]

            if len(bboxes_selected_ltrb) == 0: continue

            iou = iou[size_constraint]

            if np.count_nonzero(iou < 0.9) > 0: continue

            mask = np.zeros_like(img)

            mask[ crop_ymin : crop_ymax ,crop_xmin : crop_xmax] = 1

            img[mask==0] = border_value

            bboxes_class = bboxes_class[size_constraint]

            bboxes_selected_ltrb[:,[0,2]] /= img_w
            bboxes_selected_ltrb[:,[1,3]] /= img_h

            bboxes_selected_xywh = covertLtrb2Cxcy(bboxes_selected_ltrb)

            return img, bboxes_selected_xywh, bboxes_class

    return img, bboxes_xywh, bboxes_class

        
