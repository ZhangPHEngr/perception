

import os
import torch
import numpy as np
import torch.utils.data as data
from Utils.io import *

CURRENT_PATH = os.path.dirname(__file__)


def get_dataset_path():
    info_file = os.path.join(CURRENT_PATH, "dataset_info.json")
    data_info = read_json(info_file)
    return data_info["path"]


class ModelNetDataset(data.Dataset):
    def __init__(self, root, npoints=2500, split='train', data_augmentation=True):
        self.npoints = npoints
        self.root = root
        self.split = split
        self.data_augmentation = data_augmentation
        self.fns = []
        with open(os.path.join(root, 'modelnet40_{}.txt'.format(self.split)), 'r') as f:
            for line in f:
                self.fns.append(line.strip())

        self.cat = {}
        cnt = 0
        with open(os.path.join(self.root, 'modelnet40_shape_names.txt'), 'r') as f:
            for line in f:
                cls_name = line.strip()
                self.cat[cls_name] = cnt
                cnt += 1

        # print(self.cat)
        self.classes = list(self.cat.keys())

    def __getitem__(self, index):
        fn = self.fns[index]  # airplane_0001
        cls_name = fn[:-5]
        cls = self.cat[cls_name]
        pts = np.loadtxt(os.path.join(self.root, cls_name, fn + ".txt"), delimiter=',')

        choice = np.random.choice(len(pts), self.npoints, replace=True)
        point_set = pts[choice, :3]

        point_set = point_set - np.expand_dims(np.mean(point_set, axis=0), 0)  # center
        dist = np.max(np.sqrt(np.sum(point_set ** 2, axis=1)), 0)
        point_set = point_set / dist  # scale

        if self.data_augmentation:
            theta = np.random.uniform(0, np.pi * 2)
            rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
            point_set[:, [0, 2]] = point_set[:, [0, 2]].dot(rotation_matrix)  # random rotation
            point_set += np.random.normal(0, 0.02, size=point_set.shape)  # random jitter

        point_set = torch.from_numpy(point_set.astype(np.float32))
        cls = torch.from_numpy(np.array([cls]).astype(np.int64))
        return point_set, cls

    def __len__(self):
        return len(self.fns)


if __name__ == '__main__':
    datapath = get_dataset_path()
    d = ModelNetDataset(root=datapath)
    print(len(d))
    print(d[0])
