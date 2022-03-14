# -*- coding: utf-8 -*-
"""
@Project: Perception
@File   : show.py
@Author : Zhang P.H
@Date   : 2022/3/14
@Desc   :
"""
import open3d as o3d
# open3d使用详细见 http://www.open3d.org/docs/release/tutorial/geometry/pointcloud.html

def draw_pc(pc):
    """
    绘制单个样本的三维点图
    pc N*3 numpy
    """
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pc)
    o3d.visualization.draw_geometries([pcd])


if __name__ == '__main__':
    pass
