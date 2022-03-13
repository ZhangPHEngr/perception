from __future__ import print_function
import torch.utils.data as data
import os
import torch
import numpy as np
from Utils.io import *

CURRENT_PATH = os.path.dirname(__file__)
# def get_segmentation_classes(root):
#     catfile = os.path.join(root, 'synsetoffset2category.txt')
#     cat = {}
#     meta = {}
#
#     with open(catfile, 'r') as f:
#         for line in f:
#             ls = line.strip().split()
#             cat[ls[0]] = ls[1]
#
#     for item in cat:
#         dir_seg = os.path.join(root, cat[item], 'points_label')
#         dir_point = os.path.join(root, cat[item], 'points')
#         fns = sorted(os.listdir(dir_point))
#         meta[item] = []
#         for fn in fns:
#             token = (os.path.splitext(os.path.basename(fn))[0])
#             meta[item].append((os.path.join(dir_point, token + '.pts'), os.path.join(dir_seg, token + '.seg')))
#
#     with open(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../misc/num_seg_classes.txt'), 'w') as f:
#         for item in cat:
#             datapath = []
#             num_seg_classes = 0
#             for fn in meta[item]:
#                 datapath.append((item, fn[0], fn[1]))
#
#             for i in tqdm(range(len(datapath))):
#                 l = len(np.unique(np.loadtxt(datapath[i][-1]).astype(np.uint8)))
#                 if l > num_seg_classes:
#                     num_seg_classes = l
#
#             print("category {} num segmentation classes {}".format(item, num_seg_classes))
#             f.write("{}\t{}\n".format(item, num_seg_classes))
#
#
# def gen_modelnet_id(root):
#     classes = []
#     with open(os.path.join(root, 'train.txt'), 'r') as f:
#         for line in f:
#             classes.append(line.strip().split('/')[0])
#     classes = np.unique(classes)
#     with open(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../misc/modelnet_id.txt'), 'w') as f:
#         for i in range(len(classes)):
#             f.write('{}\t{}\n'.format(classes[i], i))

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
