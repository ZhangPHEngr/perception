# -*- coding: utf-8 -*-
"""
@Project: Perception
@File   : coco_annos_files.py
@Author : Zhang P.H
@Date   : 2022/8/6
@Desc   :
"""
import json

instance_file = "I:\\dataset\\coco\\annotations\\instances_val2017.json"

json_labels = json.load(open(instance_file, "r"))
print(json_labels["info"])

"""
1.instances annotation:
    info
    license
    images(5k list)
        0000(8 dict)
            license filename url height width date_captured flickr_url id
        0001
        ...
    annotations(36.7k list) 标注数目 一个图里可能有多个标注
        0000(7 dict)
            segmentation(polygons) area iscrowd(0单个对象 1对象集合) image_id bbox(左上x,左上y,w,h) category_id id
    categories(80)
        supercategory id name
    
2.person keypoint annotation:


3.stuff annotation:

4.
"""

if __name__ == '__main__':
    pass
