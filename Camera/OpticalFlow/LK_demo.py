# -*- coding: utf-8 -*-
"""
@Project: Perception
@File   : LK_demo.py
@Author : Zhang P.H
@Date   : 2022/8/20
@Desc   :
"""
"""
LK光流法详细解释：https://blog.csdn.net/leviopku/article/details/121773298
光流Opencv实现：https://blog.csdn.net/tengfei461807914/article/details/80978947

"""
import numpy as np
import cv2


# -------------------------> 数据读入
# 获取两帧图像，转换原始灰度图
frame_t0 = cv2.imread("source/t0.jpg")
frame_t0 = cv2.resize(frame_t0, (1920, 1500))
frame_t0_gray = cv2.cvtColor(frame_t0, cv2.COLOR_BGR2GRAY)
frame_t1 = cv2.imread("source/t1.jpg")
frame_t1 = cv2.resize(frame_t1, (1920, 1500))
frame_t1_gray = cv2.cvtColor(frame_t1, cv2.COLOR_BGR2GRAY)
# print(frame_t0_gray.shape)
# print(frame_t1_gray.shape)

# -------------------------> 初帧角点计算
# ShiTomasi 角点检测参数
feature_params = dict(maxCorners=1000,  # 角点数量
                      qualityLevel=0.1,  # 角点质量阈值
                      minDistance=100,  # 任意两角点之间的距离阈值
                      blockSize=30  # 角点计算参考邻域范围
                      )
# 获取t0图像中的角点，返回到p0中
p0 = cv2.goodFeaturesToTrack(frame_t0_gray, mask=None, **feature_params)
# # 根据图像分辨率调整好角点
# for item in p0:
#     a, b = item.ravel()
#     frame_t0 = cv2.circle(frame_t0, (int(a), int(b)), 10, (0, 255, 0), -1)
# cv2.imshow("", frame_t0)
# cv2.waitKey(-1)

# -----------------------> 计算光流
# lucas kanade光流法参数
lk_params = dict(winSize=(100, 100),
                 maxLevel=4,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
p1, st, err = cv2.calcOpticalFlowPyrLK(frame_t0_gray, frame_t1_gray, p0, None, **lk_params)

# 选取好的跟踪点
good_p1 = p1[st == 1]
good_p0 = p0[st == 1]

# ------------------> 可视化
# 创建随机颜色
color = np.random.randint(0, 255, (1000, 3))

for i, (new, old) in enumerate(zip(good_p1, good_p0)):
    a, b = new
    c, d = old
    frame_t1 = cv2.line(frame_t1, (int(a), int(b)), (int(c), int(d)), color[i].tolist(), 2)
    frame_t1 = cv2.circle(frame_t1, (int(a), int(b)), 10, color[i].tolist(), -1)
frame_t1 = cv2.resize(frame_t1, (960, 750))
cv2.imshow("", frame_t1)
cv2.waitKey(-1)


