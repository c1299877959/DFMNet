# precesso RGBT, save the GT points as .npy

import numpy as np
import os
from glob import glob
import cv2
import json


def generate_data(label_path):
    rgb_path = label_path.replace('GT', 'RGB').replace('json', 'jpg')
    t_path = label_path.replace('GT', 'T').replace('json', 'jpg')

    rgb = cv2.imread(rgb_path)[..., ::-1].copy()#(480,640,3)
    t = cv2.imread(t_path)[..., ::-1].copy()#(480,640,3)
    im_h, im_w, _ = rgb.shape
    # print('rgb and t shape', rgb.shape, t.shape)

    # Load labels
    with open(label_path, 'r') as f:
        label_file = json.load(f)
    points = np.asarray(label_file['points'])
    # print('points', points.shape) #(count,2)

    # Keep points inside rgb.shape
    idx_mask = (points[:, 0] >= 0) * (points[:, 0] <= im_w) * (points[:, 1] >= 0) * (points[:, 1] <= im_h)
    points = points[idx_mask]

    return rgb, t, points


if __name__ == '__main__':

    root_path = '/home/hlj/YYB/RGBT-datasetes/RGBTCCV3'
    save_dir = '/home/hlj/YYB/RGBT-datasetes/rgbt_light'

    #light_path = '/home/hlj/CYJ/RGBTCC/bright_list.json'
    light_path = '/home/hlj/YYB/RGBT-datasetes/bright_list.json'
    with open(light_path, 'r') as f:
        light_list = json.load(f)

    for phase in ['train', 'val', 'test']:
        sub_dir = os.path.join(root_path, phase)
        sub_save_dir = os.path.join(save_dir, phase)
        if not os.path.exists(sub_save_dir):
            os.makedirs(sub_save_dir)
        gt_list = glob(os.path.join(sub_dir, '*json'))

        gt_list = [gt_path for gt_path in gt_list if os.path.basename(gt_path) in light_list]
        for gt_path in gt_list:
            name = os.path.basename(gt_path)
            # print('name', name)
            rgb, t, points = generate_data(gt_path)# rgb, t，head pisition

            im_save_path = os.path.join(sub_save_dir, name)
            rgb_save_path = im_save_path.replace('GT', 'RGB').replace('json', 'jpg')
            t_save_path = im_save_path.replace('GT', 'T').replace('json', 'jpg')
            cv2.imwrite(rgb_save_path, rgb)
            cv2.imwrite(t_save_path, t)
            gd_save_path = im_save_path.replace('json', 'npy')
            np.save(gd_save_path, points)# save points as .npy