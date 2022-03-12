# -*- coding: utf-8 -*-
"""
@Project: 车载感知
@File   : nuscenes_vis.py
@Author : Zhang P.H
@Date   : 2022/3/12
@Desc   :
"""
from nuscenes.nuscenes import NuScenes
import matplotlib.pyplot as plt

# 加载所有数据
nusc = NuScenes(version='v1.0-mini', dataroot='I:/dataset/nuscenes/mini', verbose=True)


my_sample = nusc.sample[10]
# 1.渲染当前场景所有信息
nusc.render_sample(my_sample['token'])

# 2.渲染指定传感器
nusc.render_sample_data(my_sample['data']['CAM_FRONT']) #指定传感器
# 聚合帧数为5
nusc.render_sample_data(my_sample['data']['LIDAR_TOP'], nsweeps=5, underlay_map=True)
nusc.render_sample_data(my_sample['data']['RADAR_FRONT'], nsweeps=5, underlay_map=True)


# 2.渲染点云
# 将该sample的lidar点云绘制在图片上，图片是摄像机拍摄的
nusc.render_pointcloud_in_image(sample_token=my_sample['token'], pointsensor_channel='LIDAR_TOP')
# 将激光雷达点的反射强度体现在图片上
nusc.render_pointcloud_in_image(my_sample['token'], pointsensor_channel='LIDAR_TOP',  render_intensity=True)
# 更换传感器为车辆前方的毫米波雷达
nusc.render_pointcloud_in_image(my_sample['token'], pointsensor_channel='RADAR_FRONT')


# 3.渲染标注信息
nusc.render_annotation(my_sample['anns'][22])


# 4.渲染场景
my_scene_token = nusc.field2token('scene', 'name', 'scene-0061')[0] # 找到所有满足name==scene-0061的scene，并返回所有的token
# The rendering command below is commented out because it may crash in notebooks
nusc.render_scene_channel(my_scene_token, 'CAM_FRONT')
# 将前面的相机拍摄的场景渲染成video
nusc.render_egoposes_on_map(log_location='singapore-onenorth')


if __name__ == '__main__':
    pass
