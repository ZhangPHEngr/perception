# -*- coding: utf-8 -*-
"""
@Project: Perception
@File   : show_data.py
@Author : Zhang P.H
@Date   : 2022/3/14
@Desc   :
"""
from data_provider import ModelNetDataset, get_dataset_path
from Utils.show import draw_pc

if __name__ == '__main__':
    data_path = get_dataset_path()
    d = ModelNetDataset(root=data_path)
    pc = d[100][0].numpy()
    print(pc[:, 1])
    draw_pc(pc)
