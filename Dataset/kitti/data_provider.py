# -*- coding: utf-8 -*-
"""
@Project: Perception
@File   : data_provider.py
@Author : Zhang P.H
@Date   : 2022/3/19
@Desc   :
"""
import os
import numpy as np
from Utils.io import *

CURRENT_PATH = os.path.dirname(__file__)

# class
CLASS_KITTI = ['Pedestrian', 'Car', 'Cyclist', 'Van', 'Truck', 'Person_sitting', 'Tram', 'Misc', 'DontCare']
CLASS_KITTI_ID = {cat: i + 1 for i, cat in enumerate(CLASS_KITTI)}


def get_dataset_path():
    info_file = os.path.join(CURRENT_PATH, "dataset_info.json")
    data_info = read_json(info_file)
    return data_info["path"]


def bbox_to_coco_bbox(bbox):
    return [(bbox[0]), (bbox[1]),
            (bbox[2] - bbox[0]), (bbox[3] - bbox[1])]


def read_clib(calib_path):
    f = open(calib_path, 'r')
    for i, line in enumerate(f):
        if i == 2:
            calib = np.array(line[:-1].split(' ')[1:], dtype=np.float32)
            calib = calib.reshape(3, 4)
            return calib


if __name__ == '__main__':
    print(get_dataset_path())
