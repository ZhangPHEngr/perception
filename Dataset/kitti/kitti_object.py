""" Helper class and functions for loading KITTI objects

Author: Charles R. Qi
Date: September 2017
"""
import os
import sys
import numpy as np
import cv2

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR, "mayavi"))
import kitti_util

# data path
KITTI_OBJ_DET_PATH = "I:\\dataset\\kitti\\object_detection"

# class settings
KITTI_CLS = ['Pedestrian', 'Car', 'Cyclist', 'Van', 'Truck', 'Person_sitting', 'Tram', 'Misc', 'DontCare']
KITTI_CLS_2_ID = {cat: i + 1 for i, cat in enumerate(KITTI_CLS)}
KITTI_ID_2_CLS = {id: cls for cls, id in KITTI_CLS_2_ID.items()}
KITTI_CLS_COLOR = {
    'Pedestrian': (0, 255, 255),
    'Car': (0, 255, 255),
    'Cyclist': (0, 255, 255),
    'Van': (0, 255, 255),
    'Truck': (0, 255, 255),
    'Person_sitting': (0, 255, 255),
    'Tram': (0, 255, 255),
    'Misc': (0, 255, 255),
    'DontCare': (0, 255, 255),
}
cbox = np.array([[0, 70.4], [-40, 40], [-3, 1]])


class KittiObject(object):
    """Load and parse object data into a usable format."""

    def __init__(self, root_dir, split="training", verbose=False):
        """root_dir contains training and testing folders"""
        self.verbose = verbose
        self.root_dir = root_dir
        self.split = split
        self.split_dir = os.path.join(root_dir, split)
        if self.verbose:
            print(self.split_dir)

        if split == "training":
            self.num_samples = 7481
        elif split == "testing":
            self.num_samples = 7518
        else:
            print("Unknown split: %s" % (split))
            exit(-1)

        self.image_dir = os.path.join(self.split_dir, "image_2")
        self.label_dir = os.path.join(self.split_dir, "label_2")
        self.calib_dir = os.path.join(self.split_dir, "calib")
        self.lidar_dir = os.path.join(self.split_dir, "velodyne")
        self.depthpc_dir = os.path.join(self.split_dir, "depth_pc")
        self.depth_dir = os.path.join(self.split_dir, "depth")
        self.pred_dir = os.path.join(self.split_dir, "pred")

    def __len__(self):
        return self.num_samples

    def get_image(self, idx):
        """
        :param idx:
        :return: ndarray (h,w,c)
        """
        assert idx < self.num_samples
        img_filename = os.path.join(self.image_dir, "%06d.png" % (idx))
        return cv2.imread(img_filename)

    def get_lidar(self, idx, dtype=np.float32, n_vec=4):
        """
        :param idx:
        :param dtype:
        :param n_vec:
        :return: ndarray (n, 4)
        """
        assert idx < self.num_samples
        lidar_filename = os.path.join(self.lidar_dir, "%06d.bin" % (idx))
        pc = np.fromfile(lidar_filename, dtype=dtype).reshape((-1, n_vec))
        return pc

    def get_calibration(self, idx):
        """
        :param idx:
        :return: Calib class
        """
        assert idx < self.num_samples
        calib_filename = os.path.join(self.calib_dir, "%06d.txt" % (idx))
        return Calibration(calib_filename)

    def get_label_objects(self, idx):
        """
        :param idx:
        :return: obj 3d class
        """
        assert idx < self.num_samples and self.split == "training"
        label_filename = os.path.join(self.label_dir, "%06d.txt" % (idx))
        lines = [line.rstrip() for line in open(label_filename)]
        objects = [Object3d(line) for line in lines]
        return objects

    # def get_pred_objects(self, idx):
    #     assert idx < self.num_samples
    #     pred_filename = os.path.join(self.pred_dir, "%06d.txt" % (idx))
    #     is_exist = os.path.exists(pred_filename)
    #     if is_exist:
    #         return kitti_util.read_label(pred_filename)
    #     else:
    #         return None
    #
    # def get_depth(self, idx):
    #     assert idx < self.num_samples
    #     img_filename = os.path.join(self.depth_dir, "%06d.png" % (idx))
    #     return kitti_util.load_depth(img_filename)
    #
    # def get_depth_image(self, idx):
    #     assert idx < self.num_samples
    #     img_filename = os.path.join(self.depth_dir, "%06d.png" % (idx))
    #     return kitti_util.load_depth(img_filename)
    #
    # def get_depth_pc(self, idx):
    #     assert idx < self.num_samples
    #     lidar_filename = os.path.join(self.depthpc_dir, "%06d.bin" % (idx))
    #     is_exist = os.path.exists(lidar_filename)
    #     if is_exist:
    #         return kitti_util.load_velo_scan(lidar_filename), is_exist
    #     else:
    #         return None, is_exist
    #
    # def get_top_down(self, idx):
    #     pass
    #
    # def isexist_pred_objects(self, idx):
    #     assert idx < self.num_samples and self.split == "training"
    #     pred_filename = os.path.join(self.pred_dir, "%06d.txt" % (idx))
    #     return os.path.exists(pred_filename)
    #
    # def isexist_depth(self, idx):
    #     assert idx < self.num_samples and self.split == "training"
    #     depth_filename = os.path.join(self.depth_dir, "%06d.txt" % (idx))
    #     return os.path.exists(depth_filename)


class Object2d(object):
    """ 2d object label """

    def __init__(self, label_file_line):
        data = label_file_line.split(" ")

        # extract label, truncation, occlusion
        self.img_name = int(data[0])  # 'Car', 'Pedestrian', ...
        self.typeid = int(data[1])  # truncated pixel ratio [0..1]
        self.prob = float(data[2])
        self.box2d = np.array([int(data[3]), int(data[4]), int(data[5]), int(data[6])])

    def print_object(self):
        print(
            "img_name, typeid, prob: %s, %d, %f"
            % (self.img_name, self.typeid, self.prob)
        )
        print(
            "2d bbox (x0,y0,x1,y1): %d, %d, %d, %d"
            % (self.box2d[0], self.box2d[1], self.box2d[2], self.box2d[3])
        )


class Object3d(object):
    """ 3d object label """

    def __init__(self, label_file_line):
        data = label_file_line.split(" ")
        data[1:] = [float(x) for x in data[1:]]

        # extract label, truncation, occlusion
        self.type = data[0]  # 'Car', 'Pedestrian', ...
        self.truncation = data[1]  # truncated pixel ratio [0..1]
        self.occlusion = int(data[2])  # 0=visible, 1=partly occluded, 2=fully occluded, 3=unknown
        self.alpha = data[3]  # object observation angle [-pi..pi]

        # extract 2d bounding box in 0-based coordinates
        self.xmin = data[4]  # left
        self.ymin = data[5]  # top
        self.xmax = data[6]  # right
        self.ymax = data[7]  # bottom
        self.box2d = np.array([self.xmin, self.ymin, self.xmax, self.ymax])

        # extract 3d bounding box information
        self.h = data[8]  # box height
        self.w = data[9]  # box width
        self.l = data[10]  # box length (in meters)
        self.t = (data[11], data[12], data[13])  # location (x,y,z) in camera coord.
        self.ry = data[14]  # yaw angle (around Y-axis in camera coordinates) [-pi..pi]

    def estimate_difficulty(self):
        """ Function that estimate difficulty to detect the object as defined in kitti website"""
        # height of the bounding box
        bb_height = np.abs(self.xmax - self.xmin)

        if bb_height >= 40 and self.occlusion == 0 and self.truncation <= 0.15:
            return "Easy"
        elif bb_height >= 25 and self.occlusion in [0, 1] and self.truncation <= 0.30:
            return "Moderate"
        elif (
                bb_height >= 25 and self.occlusion in [0, 1, 2] and self.truncation <= 0.50
        ):
            return "Hard"
        else:
            return "Unknown"

    def print_object(self):
        print(
            "Type, truncation, occlusion, alpha: %s, %d, %d, %f"
            % (self.type, self.truncation, self.occlusion, self.alpha)
        )
        print(
            "2d bbox (x0,y0,x1,y1): %f, %f, %f, %f"
            % (self.xmin, self.ymin, self.xmax, self.ymax)
        )
        print("3d bbox h,w,l: %f, %f, %f" % (self.h, self.w, self.l))
        print(
            "3d bbox location, ry: (%f, %f, %f), %f"
            % (self.t[0], self.t[1], self.t[2], self.ry)
        )
        print("Difficulty of estimation: {}".format(self.estimate_diffculty()))


class Calibration(object):
    """ Calibration matrices and kitti_util
        3d XYZ in <label>.txt are in rect camera coord.
        2d box xy are in image2 coord
        Points in <lidar>.bin are in Velodyne coord.

        y_image2 = P^2_rect * x_rect
        y_image2 = P^2_rect * R0_rect * Tr_velo_to_cam * x_velo
        x_ref = Tr_velo_to_cam * x_velo
        x_rect = R0_rect * x_ref

        P^2_rect = [f^2_u,  0,      c^2_u,  -f^2_u b^2_x;
                    0,      f^2_v,  c^2_v,  -f^2_v b^2_y;
                    0,      0,      1,      0]
                 = K * [1|t]

        image2 coord:
         ----> x-axis (u)
        |
        |
        v y-axis (v)

        velodyne coord:
        front x, left y, up z

        rect/ref camera coord:
        right x, down y, front z

        Ref (KITTI paper): http://www.cvlibs.net/publications/Geiger2013IJRR.pdf

        TODO(rqi): do matrix multiplication only once for each projection.
    """

    def __init__(self, calib_filepath, from_video=False):
        if from_video:
            calibs = self.read_calib_from_video(calib_filepath)
        else:
            calibs = self.read_calib_file(calib_filepath)
        # Projection matrix from rect camera coord to image2 coord
        self.P = calibs["P2"]
        self.P = np.reshape(self.P, [3, 4])
        # Rigid transform from Velodyne coord to reference camera coord
        self.V2C = calibs["Tr_velo_to_cam"]
        self.V2C = np.reshape(self.V2C, [3, 4])
        self.C2V = self.inverse_rigid_trans(self.V2C)
        # Rotation from reference camera coord to rect camera coord
        self.R0 = calibs["R0_rect"]
        self.R0 = np.reshape(self.R0, [3, 3])

        # Camera intrinsics and extrinsics
        self.c_u = self.P[0, 2]
        self.c_v = self.P[1, 2]
        self.f_u = self.P[0, 0]
        self.f_v = self.P[1, 1]
        self.b_x = self.P[0, 3] / (-self.f_u)  # relative
        self.b_y = self.P[1, 3] / (-self.f_v)

    @staticmethod
    def read_calib_file(filepath):
        """ Read in a calibration file and parse into a dictionary.
        Ref: https://github.com/utiasSTARS/pykitti/blob/master/pykitti/kitti_util.py
        """
        data = {}
        with open(filepath, "r") as f:
            for line in f.readlines():
                line = line.rstrip()
                if len(line) == 0:
                    continue
                key, value = line.split(":", 1)
                # The only non-float values in these files are dates, which
                # we don't care about anyway
                try:
                    data[key] = np.array([float(x) for x in value.split()])
                except ValueError:
                    pass

        return data

    @staticmethod
    def cart2hom(pts_3d):
        """ Input: nx3 points in Cartesian
            Oupput: nx4 points in Homogeneous by pending 1
        """
        n = pts_3d.shape[0]
        pts_3d_hom = np.hstack((pts_3d, np.ones((n, 1))))
        return pts_3d_hom

    @staticmethod
    def inverse_rigid_trans(Tr):
        """ Inverse a rigid body transform matrix (3x4 as [R|t])
            [R'|-R't; 0|1]
        """
        inv_Tr = np.zeros_like(Tr)  # 3x4
        inv_Tr[0:3, 0:3] = np.transpose(Tr[0:3, 0:3])
        inv_Tr[0:3, 3] = np.dot(-np.transpose(Tr[0:3, 0:3]), Tr[0:3, 3])
        return inv_Tr

    # ===========================
    # ------- 3d to 3d ----------
    # ===========================
    def project_velo_to_ref(self, pts_3d_velo):
        pts_3d_velo = self.cart2hom(pts_3d_velo)  # nx4
        return np.dot(pts_3d_velo, np.transpose(self.V2C))

    def project_ref_to_velo(self, pts_3d_ref):
        pts_3d_ref = self.cart2hom(pts_3d_ref)  # nx4
        return np.dot(pts_3d_ref, np.transpose(self.C2V))

    def project_rect_to_ref(self, pts_3d_rect):
        """ Input and Output are nx3 points """
        return np.transpose(np.dot(np.linalg.inv(self.R0), np.transpose(pts_3d_rect)))

    def project_ref_to_rect(self, pts_3d_ref):
        """ Input and Output are nx3 points """
        return np.transpose(np.dot(self.R0, np.transpose(pts_3d_ref)))

    def project_rect_to_velo(self, pts_3d_rect):
        """ Input: nx3 points in rect camera coord.
            Output: nx3 points in velodyne coord.
        """
        pts_3d_ref = self.project_rect_to_ref(pts_3d_rect)
        return self.project_ref_to_velo(pts_3d_ref)

    def project_velo_to_rect(self, pts_3d_velo):
        pts_3d_ref = self.project_velo_to_ref(pts_3d_velo)
        return self.project_ref_to_rect(pts_3d_ref)

    # ===========================
    # ------- 3d to 2d ----------
    # ===========================
    def project_rect_to_image(self, pts_3d_rect):
        """ Input: nx3 points in rect camera coord.
            Output: nx2 points in image2 coord.
        """
        pts_3d_rect = self.cart2hom(pts_3d_rect)
        pts_2d = np.dot(pts_3d_rect, np.transpose(self.P))  # nx3
        pts_2d[:, 0] /= pts_2d[:, 2]
        pts_2d[:, 1] /= pts_2d[:, 2]
        return pts_2d[:, 0:2]

    def project_velo_to_image(self, pts_3d_velo):
        """ Input: nx3 points in velodyne coord.
            Output: nx2 points in image2 coord.
        """
        pts_3d_rect = self.project_velo_to_rect(pts_3d_velo)
        return self.project_rect_to_image(pts_3d_rect)

    def project_velo_to_4p(self, pts_3d_velo):
        """ Input: nx3 points in velodyne coord.
            Output: 4 points in image2 coord.
        """
        pts_2d_velo = self.project_velo_to_image(pts_3d_velo)
        return self.project_8p_to_4p(pts_2d_velo)
    
    @staticmethod
    def project_8p_to_4p(pts_2d):
        x0 = np.min(pts_2d[:, 0])
        x1 = np.max(pts_2d[:, 0])
        y0 = np.min(pts_2d[:, 1])
        y1 = np.max(pts_2d[:, 1])
        x0 = max(0, x0)
        # x1 = min(x1, proj.image_width)
        y0 = max(0, y0)
        # y1 = min(y1, proj.image_height)
        return np.array([x0, y0, x1, y1])
    
    # ===========================
    # ------- 2d to 3d ----------
    # ===========================
    def project_image_to_rect(self, uv_depth):
        """ Input: nx3 first two channels are uv, 3rd channel
                   is depth in rect camera coord.
            Output: nx3 points in rect camera coord.
        """
        n = uv_depth.shape[0]
        x = ((uv_depth[:, 0] - self.c_u) * uv_depth[:, 2]) / self.f_u + self.b_x
        y = ((uv_depth[:, 1] - self.c_v) * uv_depth[:, 2]) / self.f_v + self.b_y
        pts_3d_rect = np.zeros((n, 3))
        pts_3d_rect[:, 0] = x
        pts_3d_rect[:, 1] = y
        pts_3d_rect[:, 2] = uv_depth[:, 2]
        return pts_3d_rect

    def project_image_to_velo(self, uv_depth):
        pts_3d_rect = self.project_image_to_rect(uv_depth)
        return self.project_rect_to_velo(pts_3d_rect)

    def project_depth_to_velo(self, depth, constraint_box=True):
        depth_pt3d = kitti_util.get_depth_pt3d(depth)
        depth_UVDepth = np.zeros_like(depth_pt3d)
        depth_UVDepth[:, 0] = depth_pt3d[:, 1]
        depth_UVDepth[:, 1] = depth_pt3d[:, 0]
        depth_UVDepth[:, 2] = depth_pt3d[:, 2]
        # print("depth_pt3d:",depth_UVDepth.shape)
        depth_pc_velo = self.project_image_to_velo(depth_UVDepth)
        # print("dep_pc_velo:",depth_pc_velo.shape)
        if constraint_box:
            depth_box_fov_inds = (
                    (depth_pc_velo[:, 0] < cbox[0][1])
                    & (depth_pc_velo[:, 0] >= cbox[0][0])
                    & (depth_pc_velo[:, 1] < cbox[1][1])
                    & (depth_pc_velo[:, 1] >= cbox[1][0])
                    & (depth_pc_velo[:, 2] < cbox[2][1])
                    & (depth_pc_velo[:, 2] >= cbox[2][0])
            )
            depth_pc_velo = depth_pc_velo[depth_box_fov_inds]
        return depth_pc_velo


if __name__ == "__main__":
    pass

