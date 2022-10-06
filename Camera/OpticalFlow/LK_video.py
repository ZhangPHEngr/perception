# -*- coding: utf-8 -*-
"""
@Project: Perception
@File   : LK_demo.py
@Author : Zhang P.H
@Date   : 2022/8/20
@Desc   :
"""
import numpy as np
import cv2

# -------------------> 全局参数设置
# ShiTomasi 角点检测参数
feature_params = dict(maxCorners=100,  # 角点数量
                      qualityLevel=0.1,  # 角点质量阈值
                      minDistance=10,  # 任意两角点之间的距离阈值
                      blockSize=3  # 角点计算参考邻域范围
                      )
# lucas kanade光流法参数
lk_params = dict(winSize=(100, 100),
                 maxLevel=2,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
# 创建随机颜色
color = np.random.randint(0, 255, (1000, 3))

# 视频处理
cap = cv2.VideoCapture("../source/person.mp4")
frame_last_gray = None
frame_cur_gray = None
frame_cnt = 1
while cap.isOpened():
    ret, frame = cap.read()  # 若获取成功，ret为True，否则为False；frame是图像
    if ret:
        # 读取对应的label
        rois = np.loadtxt("../source/labels/testvideo1_{}.txt".format(frame_cnt)).astype(np.int)
        frame_cnt += 1  # TODO 不规范，不能保证label和图像是对应的
        # 读取图像
        frame_cur_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        print(frame.shape)
        if frame_last_gray is None:
            frame_last_gray = frame_cur_gray.copy()
            continue

        # 选取感兴趣的区域
        mask = np.zeros_like(frame_cur_gray)
        for roi in rois:
            print(roi[1], roi[2], roi[3], roi[4])
            mask[roi[2]:roi[4], roi[1]:roi[3]] = 255
            frame = cv2.rectangle(frame, (roi[1], roi[2]), (roi[3], roi[4]), color=(0, 255, 0))
        # cv2.imshow("", mask)
        # cv2.waitKey(100)

        # 计算角点和光流
        p0 = cv2.goodFeaturesToTrack(frame_last_gray, mask=mask, **feature_params)
        p1, st, err = cv2.calcOpticalFlowPyrLK(frame_last_gray, frame_cur_gray, p0, None, **lk_params)
        # 选取好的跟踪点
        good_p1 = p1[st == 1]
        good_p0 = p0[st == 1]

        # 可视化
        for i, (new, old) in enumerate(zip(good_p1, good_p0)):
            a, b = new
            c, d = old
            frame = cv2.line(frame, (int(a), int(b)), (int(c), int(d)), color[i].tolist(), 2)
            frame = cv2.circle(frame, (int(a), int(b)), 5, color[i].tolist(), -1)
        cv2.imshow('frame', frame)  # 两个参数，一个是展示画面的名字，一个是像素内容
        key = cv2.waitKey(100)  # 停留25ms，当为0的时候则堵塞在第一帧不会继续下去

        frame_last_gray = frame_cur_gray

cap.release()  # 释放视频
cv2.destroyAllWindows()  # 释放所有显示图像的窗口
