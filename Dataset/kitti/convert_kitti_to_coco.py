import json
import os.path
from tqdm import tqdm
import cv2
import numpy as np

from data_provider import *
from Utils.ddd_utils import compute_box_3d, project_to_image, alpha2rot_y
from Utils.ddd_utils import draw_box_3d, unproject_2d_to_3d

'''
#Values    Name      Description
----------------------------------------------------------------------------
   1    type         Describes the type of object: 'Car', 'Van', 'Truck',
                     'Pedestrian', 'Person_sitting', 'Cyclist', 'Tram',
                     'Misc' or 'DontCare'
   1    truncated    Float from 0 (non-truncated) to 1 (truncated), where
                     truncated refers to the object leaving image boundaries
   1    occluded     Integer (0,1,2,3) indicating occlusion state:
                     0 = fully visible, 1 = partly occluded
                     2 = largely occluded, 3 = unknown
   1    alpha        Observation angle of object, ranging [-pi..pi]
   4    bbox         2D bounding box of object in the image (0-based index):
                     contains left, top, right, bottom pixel coordinates
   3    dimensions   3D object dimensions: height, width, length (in meters)
   3    location     3D object location x,y,z in camera coordinates (in meters)
   1    rotation_y   Rotation ry around Y-axis in camera coordinates [-pi..pi]
   1    score        Only for results: Float, indicating confidence in
                     detection, needed for p/r curves, higher is better.
'''
DEBUG = False
DATA_PATH = get_dataset_path()
OUTPUT_DIR = os.path.join(DATA_PATH, "annotations")
os.makedirs(OUTPUT_DIR, exist_ok=True)
SPLITS = ['3dop', 'subcnn']

F = 721
H = 384  # 375
W = 1248  # 1242
EXT = [45.75, -0.34, 0.005]
CALIB = np.array([[F, 0, W / 2, EXT[0]], [0, F, H / 2, EXT[1]], [0, 0, 1, EXT[2]]], dtype=np.float32)

cat_info = []
for i, cat in enumerate(CLASS_KITTI):
    cat_info.append({'name': cat, 'id': i + 1})

calib_type = {'train': 'training', 'val': 'training', 'trainval': 'training', 'test': 'testing'}
splits = ['train', 'val']

# 针对多种不同的拆法
for SPLIT in SPLITS:
    image_set_path = os.path.join(DATA_PATH, "ImageSets_{}".format(SPLIT))
    # 针对训练集和验证集
    for split in splits:
        # 指向指定目录
        image_dir = os.path.join(DATA_PATH, calib_type[split], "image_2")
        ann_dir = os.path.join(DATA_PATH, calib_type[split], "label_2")
        calib_dir = os.path.join(DATA_PATH, calib_type[split], "calib")
        # 保存中间结果
        ret = {'images': [], 'annotations': [], "categories": cat_info}
        # 遍历所有图像
        image_set = np.loadtxt(os.path.join(image_set_path, '{}.txt'.format(split)), dtype = str)
        bar = tqdm(image_set)
        for image_id in bar:
            if image_id[-1] == '\n':
                image_id = image_id[:-1]
            bar.set_description(image_id)
            # 获取原图
            if DEBUG:
                image = cv2.imread(os.path.join(image_dir, '{}.png'.format(image_id)))
            # 获取标定信息
            calib = read_clib(os.path.join(calib_dir, '{}.txt'.format(image_id)))
            # 写入数据信息
            image_info = {'file_name': '{}.png'.format(image_id),
                          'id': int(image_id),
                          'calib': calib.tolist()}
            ret['images'].append(image_info)
            if split == 'test':
                continue

            # 获取标注信息
            anns = open(os.path.join(ann_dir, '{}.txt'.format(image_id)), 'r')
            for ann_ind, txt in enumerate(anns):
                tmp = txt[:-1].split(' ')
                cat_id = CLASS_KITTI_ID[tmp[0]]
                truncated = int(float(tmp[1]))
                occluded = int(tmp[2])
                alpha = float(tmp[3])
                bbox = [float(tmp[4]), float(tmp[5]), float(tmp[6]), float(tmp[7])]
                dim = [float(tmp[8]), float(tmp[9]), float(tmp[10])]
                location = [float(tmp[11]), float(tmp[12]), float(tmp[13])]
                rotation_y = float(tmp[14])

                ann = {'image_id': image_id,
                       'id': int(len(ret['annotations']) + 1),
                       'category_id': cat_id,
                       'dim': dim,
                       'bbox': bbox_to_coco_bbox(bbox),
                       'depth': location[2],
                       'alpha': alpha,
                       'truncated': truncated,
                       'occluded': occluded,
                       'location': location,
                       'rotation_y': rotation_y}
                ret['annotations'].append(ann)
                if DEBUG and tmp[0] != 'DontCare':
                    box_3d = compute_box_3d(dim, location, rotation_y)
                    box_2d = project_to_image(box_3d, calib)
                    # print('box_2d', box_2d)
                    image = draw_box_3d(image, box_2d)
                    x = (bbox[0] + bbox[2]) / 2
                    '''
                    print('rot_y, alpha2rot_y, dlt', tmp[0], 
                          rotation_y, alpha2rot_y(alpha, x, calib[0, 2], calib[0, 0]),
                          np.cos(
                            rotation_y - alpha2rot_y(alpha, x, calib[0, 2], calib[0, 0])))
                    '''
                    depth = np.array([location[2]], dtype=np.float32)
                    pt_2d = np.array([(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2], dtype=np.float32)
                    pt_3d = unproject_2d_to_3d(pt_2d, depth, calib)
                    pt_3d[1] += dim[0] / 2
                    print('pt_3d', pt_3d)
                    print('location', location)
            if DEBUG:
                cv2.imshow('image', image)
                cv2.waitKey()

        print("# images: ", len(ret['images']))
        print("# annotations: ", len(ret['annotations']))
        write_json(os.path.join(OUTPUT_DIR, 'kitti_{}_{}.json'.format(SPLIT, split)), ret)
