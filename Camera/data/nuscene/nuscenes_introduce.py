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

str_demo = "\033[;32;47m{}\033[0m"

# 加载所有数据
nusc = NuScenes(version='v1.0-mini', dataroot='I:/dataset/nuscenes/mini', verbose=True)

# ---------------------------------------------------scene相关----------------------------------------------------------
print(str_demo.format("scene list:"))
nusc.list_scenes()  # 列出当前场景所有帧信息

print(str_demo.format("scene 0:"), nusc.scene[0])  # 第一帧场景信息
print(str_demo.format("first_scene info"), nusc.get('scene', nusc.scene[0]["token"]))  # 效果同上， get方法要提供对应token
# scene {
#    "token":                   //<str> -- Unique record identifier.
#    "name":                    //<str> -- Short string identifier.
#    "description":             //<str> -- Longer description of the scene.
#    "log_token":               //<str> -- 外键，指向一个log，scene中的data都是从该log提取出来的.
#    "nbr_samples":             //<int> -- scene中的sample的数量.
#    "first_sample_token":      //<str> -- 外键，指向场景中第一个sample.
#    "last_sample_token":       //<str> -- 外键，指向场景中最后一个sample.
# }

# ---------------------------------------------------sample相关----------------------------------------------------------
# sample也即是每数据帧信息， 其包含两个部分，sample_data(自车位姿参数，观传感器测结果，传感器标定参数)， sample_anns(对他车的标注结果)
first_sample_token = nusc.scene[0]['first_sample_token']
print(str_demo.format("first_sample_token:"), first_sample_token)

print(str_demo.format("sample list:"))
nusc.list_sample(first_sample_token)  # 列出当前帧所有自车和标注信息

print(str_demo.format("first_sample info"), nusc.get('sample', first_sample_token))  # 效果同上. 如下数据格式
# sample {
#     'token': 'ca9a282c9e77460f8360f564131a8af5',
#     'timestamp': 1532402927647951,
#     'prev': '',
#     'next': '39586f9d59004284a7114a68825e8eec',
#     'scene_token': 'cc8c0bf57f984915a77078b10eb33198',
#     'data': {
#         'RADAR_FRONT': '37091c75b9704e0daa829ba56dfa0906',
#         'RADAR_FRONT_LEFT': '11946c1461d14016a322916157da3c7d',
#         'RADAR_FRONT_RIGHT': '491209956ee3435a9ec173dad3aaf58b',
#         'RADAR_BACK_LEFT': '312aa38d0e3e4f01b3124c523e6f9776',
#         'RADAR_BACK_RIGHT': '07b30d5eb6104e79be58eadf94382bc1',
#         'LIDAR_TOP': '9d9bf11fb0e144c8b446d54a8a00184f',
#         'CAM_FRONT': 'e3d495d4ac534d54b321f50006683844',
#         'CAM_FRONT_RIGHT': 'aac7867ebf4f446395d29fbd60b63b3b',
#         'CAM_BACK_RIGHT': '79dbb4460a6b40f49f9c150cb118247e',
#         'CAM_BACK': '03bea5763f0f4722933508d5999c5fd8',
#         'CAM_BACK_LEFT': '43893a033f9c46d4a51b5e08a67a1eb7',
#         'CAM_FRONT_LEFT': 'fe5422747a7d4268a4b07fc396707b23'},
#     'anns': [
#         'ef63a697930c4b20a6b9791f423351da',
#         '6b89da9bf1f84fd6a5fbe1c3b236f809',
#         '70af124fceeb433ea73a79537e4bea9e']
#     }

# ---------------------------------------------------sample_data相关-----------------------------------------------------
sample0 = nusc.get('sample', nusc.scene[0]['first_sample_token'])
data = sample0["data"]
# sample_data {
#    "token":                   //<str> -- Unique record identifier.
#    "sample_token":            //<str> -- 外键，指向该sample_data关联的sample.
#    "ego_pose_token":          //<str> -- Foreign key.
#    "calibrated_sensor_token": //<str> -- Foreign key.
#    "filename":                //<str> -- 文件存放的相对路径.
#    "fileformat":              //<str> -- data文件的格式.
#    "width":                   //<int> -- 如果sample data是图片，则表示宽度（像素）.
#    "height":                  //<int> -- 如果sample data是图片，则表示高度（像素）.
#    "timestamp":               //<int> -- Unix 时间戳.
#    "is_key_frame":            //<bool> -- 标识是否为keyframes的部分
#    "next":                    //<str> -- 外键，时间上，相同传感器下该sample data的下一个sample data. Empty if end of scene.
#    "prev":                    //<str> -- 外键，时间上，相同传感器下该sample data的前一个sample data. Empty if start of scene.
# }
print(str_demo.format("cur sample data:"), data)  # 是个dict注意
# 渲染sample_data
nusc.render_sample_data(data["CAM_FRONT"])  # 传入参数为sample_data某个的token

# ---------------------------------------------------sample_anns相关-----------------------------------------------------
anns = sample0["anns"]  # 是个list注意, 表示每个标注物体可以取其中一个就是下面dict结构， 也就是说每帧有一个sample_data表示自车信息及观测，有多个anns表示其他车标注结果
anns0 = nusc.get("sample_annotation", anns[0])
# sample_annotation {
#    "token":                   //<str> -- Unique record identifier.
#    "sample_token":            //<str> -- 外键，指向其所属的sample，而非sample_data。
#    "instance_token":          //<str> -- 外键，指向该annotation的instance. 一个instance可以有多个annotations.
#    "attribute_tokens":        //<str> [n] -- 一个list，存放着该annotation的attributes.
#    "visibility_token":        //<str> -- Foreign key. Visibility may also change over time. If no visibility is annotated, the token is an empty string.
#    "translation":             //<float> [3] -- Bounding box location in meters as center_x, center_y, center_z.
#    "size":                    //<float> [3] -- Bounding box size in meters as width, length, height.
#    "rotation":                //<float> [4] -- 旋转角？不懂咋用。Bounding box orientation as quaternion: w, x, y, z.
#    "num_lidar_pts":           //<int> -- 在box里的激光雷达点的数量。
#    "num_radar_pts":           //<int> -- 在box里的毫米波雷达点的数量。This number is summed across all radar sensors without any invalid point filtering.
#    "next":                    //<str> -- Foreign key. Sample annotation from the same object instance that follows this in time. Empty if this is the last annotation for this object.
#    "prev":                    //<str> -- Foreign key. Sample annotation from the same object instance that precedes this in time. Empty if this is the first annotation for this object.
# }
print(str_demo.format("cur sample anns 0:"), anns0)
nusc.render_annotation(anns0["token"])

# ---------------------------------------------------instance相关-----------------------------------------------------
# 具体的某个被标注目标
my_instance = nusc.instance[599]
# instance {
#    "token":                   //<str> -- Unique record identifier.
#    "category_token":          //<str> -- 指向物体所属类别的token
#    "nbr_annotations":         //<int> -- 对该物体的标注的数量。
#    "first_annotation_token":  //<str> -- 对该物体的第一个annotation的token.
#    "last_annotation_token":   //<str> -- 最后一个annotation的token.
# }
print(str_demo.format("某个标注目标信息："), my_instance)
nusc.render_instance(my_instance["token"])

# ---------------------------------------------------category相关-----------------------------------------------------
# 当前数据中共有多少类别
nusc.list_categories()
print(str_demo.format("第9类目标详细信息："), nusc.category[9])

# ---------------------------------------------------attributes相关-----------------------------------------------------
# 当前数据中共有多少类别
nusc.list_attributes()

# ---------------------------------------------------visibility相关-----------------------------------------------------
# 目标可见性定义
print(str_demo.format("目标可见性分类：", nusc.visibility))
anntoken = 'a7d0722bce164f88adf03ada491ea0ba'
visibility_token = nusc.get('sample_annotation', anntoken)['visibility_token']

print("Visibility: {}".format(nusc.get('visibility', visibility_token)))
nusc.render_annotation(anntoken)
# ---------------------------------------------------sensor信息-----------------------------------------------------
print(str_demo.format("传感器信息："), nusc.sensor)

# ---------------------------------------------------calibrated_sensor信息----------------------------------------------
# translation应该是矫正后的sensor相对于车辆的xyz偏移。rotation则是一组四元数，表示sensor的旋转。
# calibrated_sensor {
#    "token":                   //<str> -- Unique record identifier.
#    "sensor_token":            //<str> -- Foreign key pointing to the sensor type.
#    "translation":             //<float> [3] -- Coordinate system origin in meters: x, y, z.
#    "rotation":                //<float> [4] -- Coordinate system orientation as quaternion: w, x, y, z.
#    "camera_intrinsic":        //<float> [3, 3] -- Intrinsic camera calibration. Empty for sensors that are not cameras.
# }

# ---------------------------------------------------ego_pose信息----------------------------------------------
# ego_pose的数量跟sample_data的数量是一直的，这两者是一对一对应的关系。
print(str_demo.format("ego_pose："), nusc.ego_pose[0])
# ego_pose {
#    "token":                   <str> -- Unique record identifier.
#    "translation":             <float> [3] -- Coordinate system origin in meters: x, y, z. Note that z is always 0.
#    "rotation":                <float> [4] -- Coordinate system orientation as quaternion: w, x, y, z.
#    "timestamp":               <int> -- Unix time stamp.
# }


# ---------------------------------------------------calibrated_sensor信息----------------------------------------------
print(str_demo.format("log信息："), nusc.log[0])
# log {
#    "token":                   //<str> -- Unique record identifier.
#    "logfile":                 //<str> -- Log file name.
#    "vehicle":                 //<str> -- Vehicle name.
#    "date_captured":           //<str> -- Date (YYYY-MM-DD).
#    "location":                //<str> -- Area where log was captured, e.g. singapore-onenorth.
# }


# ---------------------------------------------------calibrated_sensor信息----------------------------------------------
print(str_demo.format("map信息："), nusc.map[0])


plt.show()

if __name__ == '__main__':
    pass
