# -*- coding: utf-8 -*-
"""
@Project: Perception
@File   : coco_api_demo.py
@Author : Zhang P.H
@Date   : 2022/8/6
@Desc   :
"""
import os

import cv2
import matplotlib.pyplot as plt
from pycocotools.coco import COCO

"""
COCO类:
    COCO(anno_file)         使用标注json初始化coco对象
    常用属性：
        cats                类别信息 字典(cat_id: cat_info)
        anns                标注信息 字典(anno_id: anno_info)
        imgs                图像信息 字典(image_id: image_info)
        imgToAnns           图像和标注的映射 字典(image_id: [anno_id_1, anno_id_2]) 肯定都是有标注的图像
        catToImgs           类别和图像的映射 字典(cat_id: [image_id_1, image_id_2])
    常用方法：
        getCatIds           指定类别筛选条件，筛选类别id list
        getAnnIds           指定img范围，指定类别范围，指定area范围，获取对应的标注id list
        getImgIds           指定img范围，指定类别范围， 获取img id list
        loadCats            根据id 获取cat info list 
        loadAnns            根据id 获取anno info list
        loadImgs            根据id 获取img info list
        showAnns            显示真值显示
        
    
    id加载            属性间映射           信息加载
    getCatIds       catToImgs           loadCats       
    getAnnIds                           loadAnns
    getImgIds       imgToAnns           loadImgs
"""


instance_file = "I:\\dataset\\coco\\annotations\\instances_val2017.json"
coco = COCO(instance_file)

img_id = coco.getImgIds()[0]

# 加载图像并绘制
img_info = coco.loadImgs(img_id)[0]
img = cv2.imread(os.path.join("I:\\dataset\\coco\\val2017\\", img_info["file_name"]))
plt.imshow(img)

# 加载标注并绘制
anno = coco.imgToAnns[img_id]
coco.showAnns(anno, draw_bbox=True)
plt.show()


if __name__ == '__main__':
    pass
