# -*- coding: utf-8 -*-
"""
@Project: Perception
@File   : vis_demo.py
@Author : Zhang P.H
@Date   : 2022/8/22
@Desc   :
"""

from kitti_object import *
import kitti_util


def show_lidar_on_image(pc_velo, img, calib, img_width, img_height, vis=True, waite=-1):
    """ Project LiDAR points to image """
    img = np.copy(img)
    imgfov_pc_velo, pts_2d, fov_inds = kitti_util.get_lidar_in_image_fov(pc_velo, calib, 0, 0, img_width, img_height, True)
    imgfov_pts_2d = pts_2d[fov_inds, :]
    imgfov_pc_rect = calib.project_velo_to_rect(imgfov_pc_velo)

    import matplotlib.pyplot as plt

    cmap = plt.cm.get_cmap("hsv", 256)
    cmap = np.array([cmap(i) for i in range(256)])[:, :3] * 255

    for i in range(imgfov_pts_2d.shape[0]):
        depth = imgfov_pc_rect[i, 2]
        color = cmap[int(640.0 / depth), :]
        cv2.circle(
            img,
            (int(np.round(imgfov_pts_2d[i, 0])), int(np.round(imgfov_pts_2d[i, 1]))),
            2,
            color=tuple(color),
            thickness=-1,
        )
    if vis:
        cv2.imshow("projection", img)
        cv2.waitKey(waite)
    return img


def show_image_with_boxes(img, objects, calib, show2d=True, show3d=True, waite=-1):
    """ Show image with 2D bounding boxes """
    img1 = np.copy(img)  # for 2d bbox
    img2 = np.copy(img)  # for 3d bbox
    # img3 = np.copy(img)  # for 3d bbox
    # TODO: change the color of boxes
    for obj in objects:
        # for 2d bbox
        if obj.type == "DontCare":
            continue
        cv2.rectangle(img1, (int(obj.xmin), int(obj.ymin)), (int(obj.xmax), int(obj.ymax)), KITTI_CLS_COLOR[obj.type], 2)
        # for 3d bbox
        box3d_pts_2d, _ = kitti_util.compute_box_3d(obj, calib.P)
        if box3d_pts_2d is None:
            print("something wrong in the 3D box.")
            continue
        img2 = kitti_util.draw_projected_box3d(img2, box3d_pts_2d, color=KITTI_CLS_COLOR[obj.type])

        # project
        # box3d_pts_3d_velo = calib.project_rect_to_velo(box3d_pts_3d)
        # box3d_pts_32d = kitti_util.box3d_to_rgb_box00(box3d_pts_3d_velo)
        # box3d_pts_32d = calib.project_velo_to_image(box3d_pts_3d_velo)
        # img3 = kitti_util.draw_projected_box3d(img3, box3d_pts_32d)
    # print("img1:", img1.shape)
    # cv2.imshow("2dbox", img1)
    # print("img3:",img3.shape)
    # Image.fromarray(img3).show()

    if show2d:
        cv2.imshow("2dbox", img1)
        cv2.waitKey(waite)
    if show3d:
        cv2.imshow("3dbox", img2)
        cv2.waitKey(waite)

    return img1, img2


def show_lidar_with_depth(
        pc_velo,
        objects,
        calib,
        fig,
        img_fov=False,
        img_width=None,
        img_height=None,
        objects_pred=None,
        depth=None,
        cam_img=None,
        constraint_box=False,
        pc_label=False,
        save=False,
):
    """ Show all LiDAR points.
        Draw 3d box in LiDAR point cloud (in velo coord system) """
    if "mlab" not in sys.modules:
        import mayavi.mlab as mlab
    from viz_util import draw_lidar_simple, draw_lidar, draw_gt_boxes3d

    print(("All point num: ", pc_velo.shape[0]))
    if img_fov:
        pc_velo_index = kitti_util.get_lidar_index_in_image_fov(
            pc_velo[:, :3], calib, 0, 0, img_width, img_height
        )
        pc_velo = pc_velo[pc_velo_index, :]
        print(("FOV point num: ", pc_velo.shape))
    print("pc_velo", pc_velo.shape)
    draw_lidar(pc_velo, fig=fig, pc_label=pc_label)

    # Draw depth
    if depth is not None:
        depth_pc_velo = calib.project_depth_to_velo(depth, constraint_box)

        indensity = np.ones((depth_pc_velo.shape[0], 1)) * 0.5
        depth_pc_velo = np.hstack((depth_pc_velo, indensity))
        print("depth_pc_velo:", depth_pc_velo.shape)
        print("depth_pc_velo:", type(depth_pc_velo))
        print(depth_pc_velo[:5])
        draw_lidar(depth_pc_velo, fig=fig, pts_color=(1, 1, 1))

        if save:
            data_idx = 0
            vely_dir = "data/object/training/depth_pc"
            save_filename = os.path.join(vely_dir, "%06d.bin" % (data_idx))
            print(save_filename)
            # np.save(save_filename+".npy", np.array(depth_pc_velo))
            depth_pc_velo = depth_pc_velo.astype(np.float32)
            depth_pc_velo.tofile(save_filename)

    # color = (0, 1, 0)
    for obj in objects:
        if obj.type == "DontCare":
            continue
        # Draw 3d bounding box
        _, box3d_pts_3d = kitti_util.compute_box_3d(obj, calib.P)
        box3d_pts_3d_velo = calib.project_rect_to_velo(box3d_pts_3d)
        print("box3d_pts_3d_velo:")
        print(box3d_pts_3d_velo)

        # TODO: change the color of boxes
        if obj.type == "Car":
            draw_gt_boxes3d([box3d_pts_3d_velo], fig=fig, color=(0, 1, 0), label=obj.type)
        elif obj.type == "Pedestrian":
            draw_gt_boxes3d([box3d_pts_3d_velo], fig=fig, color=(0, 1, 1), label=obj.type)
        elif obj.type == "Cyclist":
            draw_gt_boxes3d([box3d_pts_3d_velo], fig=fig, color=(1, 1, 0), label=obj.type)

    if objects_pred is not None:
        color = (1, 0, 0)
        for obj in objects_pred:
            if obj.type == "DontCare":
                continue
            # Draw 3d bounding box
            _, box3d_pts_3d = kitti_util.compute_box_3d(obj, calib.P)
            box3d_pts_3d_velo = calib.project_rect_to_velo(box3d_pts_3d)
            print("box3d_pts_3d_velo:")
            print(box3d_pts_3d_velo)
            draw_gt_boxes3d([box3d_pts_3d_velo], fig=fig, color=color)
            # Draw heading arrow
            _, ori3d_pts_3d = kitti_util.compute_orientation_3d(obj, calib.P)
            ori3d_pts_3d_velo = calib.project_rect_to_velo(ori3d_pts_3d)
            x1, y1, z1 = ori3d_pts_3d_velo[0, :]
            x2, y2, z2 = ori3d_pts_3d_velo[1, :]
            mlab.plot3d(
                [x1, x2],
                [y1, y2],
                [z1, z2],
                color=color,
                tube_radius=None,
                line_width=1,
                figure=fig,
            )
    mlab.show(1)


def show_lidar_with_boxes(
        pc_velo,
        objects,
        calib,
        img_fov=False,
        img_width=None,
        img_height=None,
        objects_pred=None,
        depth=None,
        cam_img=None,
):
    """ Show all LiDAR points.
        Draw 3d box in LiDAR point cloud (in velo coord system) """
    if "mlab" not in sys.modules:
        import mayavi.mlab as mlab
    from viz_util import draw_lidar_simple, draw_lidar, draw_gt_boxes3d

    print(("All point num: ", pc_velo.shape[0]))
    fig = mlab.figure(
        figure=None, bgcolor=(0, 0, 0), fgcolor=None, engine=None, size=(1000, 500)
    )
    if img_fov:
        pc_velo = kitti_util.get_lidar_in_image_fov(
            pc_velo[:, 0:3], calib, 0, 0, img_width, img_height
        )
        print(("FOV point num: ", pc_velo.shape[0]))
    print("pc_velo", pc_velo.shape)
    draw_lidar(pc_velo, fig=fig)
    # pc_velo=pc_velo[:,0:3]

    color = (0, 1, 0)
    for obj in objects:
        if obj.type == "DontCare":
            continue
        # Draw 3d bounding box
        _, box3d_pts_3d = kitti_util.compute_box_3d(obj, calib.P)
        box3d_pts_3d_velo = calib.project_rect_to_velo(box3d_pts_3d)
        print("box3d_pts_3d_velo:")
        print(box3d_pts_3d_velo)

        draw_gt_boxes3d([box3d_pts_3d_velo], fig=fig, color=color)

        # Draw depth
        if depth is not None:
            # import pdb; pdb.set_trace()
            depth_pt3d = kitti_util.depth_region_pt3d(depth, obj)
            depth_UVDepth = np.zeros_like(depth_pt3d)
            depth_UVDepth[:, 0] = depth_pt3d[:, 1]
            depth_UVDepth[:, 1] = depth_pt3d[:, 0]
            depth_UVDepth[:, 2] = depth_pt3d[:, 2]
            print("depth_pt3d:", depth_UVDepth)
            dep_pc_velo = calib.project_image_to_velo(depth_UVDepth)
            print("dep_pc_velo:", dep_pc_velo)

            draw_lidar(dep_pc_velo, fig=fig, pts_color=(1, 1, 1))

        # Draw heading arrow
        _, ori3d_pts_3d = kitti_util.compute_orientation_3d(obj, calib.P)
        ori3d_pts_3d_velo = calib.project_rect_to_velo(ori3d_pts_3d)
        x1, y1, z1 = ori3d_pts_3d_velo[0, :]
        x2, y2, z2 = ori3d_pts_3d_velo[1, :]
        mlab.plot3d(
            [x1, x2],
            [y1, y2],
            [z1, z2],
            color=color,
            tube_radius=None,
            line_width=1,
            figure=fig,
        )
    if objects_pred is not None:
        color = (1, 0, 0)
        for obj in objects_pred:
            if obj.type == "DontCare":
                continue
            # Draw 3d bounding box
            _, box3d_pts_3d = kitti_util.compute_box_3d(obj, calib.P)
            box3d_pts_3d_velo = calib.project_rect_to_velo(box3d_pts_3d)
            print("box3d_pts_3d_velo:")
            print(box3d_pts_3d_velo)
            draw_gt_boxes3d([box3d_pts_3d_velo], fig=fig, color=color)
            # Draw heading arrow
            _, ori3d_pts_3d = kitti_util.compute_orientation_3d(obj, calib.P)
            ori3d_pts_3d_velo = calib.project_rect_to_velo(ori3d_pts_3d)
            x1, y1, z1 = ori3d_pts_3d_velo[0, :]
            x2, y2, z2 = ori3d_pts_3d_velo[1, :]
            mlab.plot3d(
                [x1, x2],
                [y1, y2],
                [z1, z2],
                color=color,
                tube_radius=None,
                line_width=1,
                figure=fig,
            )
    mlab.show(1)


def stat_lidar_with_boxes(pc_velo, objects, calib):
    """ Show all LiDAR points.
        Draw 3d box in LiDAR point cloud (in velo coord system) """

    # print(('All point num: ', pc_velo.shape[0]))

    # draw_lidar(pc_velo, fig=fig)
    # color=(0,1,0)
    for obj in objects:
        if obj.type == "DontCare":
            continue
        _, box3d_pts_3d = kitti_util.compute_box_3d(obj, calib.P)
        box3d_pts_3d_velo = calib.project_rect_to_velo(box3d_pts_3d)
        v_l, v_w, v_h, _ = kitti_util.get_velo_whl(box3d_pts_3d_velo, pc_velo)
        print("%.4f %.4f %.4f %s" % (v_w, v_h, v_l, obj.type))


def show_lidar_topview_with_boxes(pc_velo, objects, calib, objects_pred=None):
    """ top_view image"""
    # print('pc_velo shape: ',pc_velo.shape)
    top_view = kitti_util.lidar_to_top(pc_velo)
    top_image = kitti_util.draw_top_image(top_view)
    print("top_image:", top_image.shape)

    # gt

    def bbox3d(obj):
        _, box3d_pts_3d = kitti_util.compute_box_3d(obj, calib.P)
        box3d_pts_3d_velo = calib.project_rect_to_velo(box3d_pts_3d)
        return box3d_pts_3d_velo

    boxes3d = [bbox3d(obj) for obj in objects if obj.type != "DontCare"]
    gt = np.array(boxes3d)
    # print("box2d BV:",boxes3d)
    lines = [obj.type for obj in objects if obj.type != "DontCare"]
    top_image = kitti_util.draw_box3d_on_top(
        top_image, gt, text_lables=lines, scores=None, thickness=1, is_gt=True
    )
    # pred
    if objects_pred is not None:
        boxes3d = [bbox3d(obj) for obj in objects_pred if obj.type != "DontCare"]
        gt = np.array(boxes3d)
        lines = [obj.type for obj in objects_pred if obj.type != "DontCare"]
        top_image = kitti_util.draw_box3d_on_top(
            top_image, gt, text_lables=lines, scores=None, thickness=1, is_gt=False
        )

    cv2.imshow("top_image", top_image)
    return top_image


if __name__ == '__main__':
    dataset = KittiObject(KITTI_OBJ_DET_PATH, split="training")
    ## load 2d detection results
    # objects2ds = read_det_file("box2d.list")

    # if args.show_lidar_with_depth:
    #     # import mayavi.mlab as mlab
    #
    #     fig = mlab.figure(
    #         figure=None, bgcolor=(0, 0, 0), fgcolor=None, engine=None, size=(1000, 500)
    #     )
    for data_idx in range(len(dataset)):
        data_idx = 0
        # Load data from dataset
        objects = dataset.get_label_objects(data_idx)
        pc_velo = dataset.get_lidar(data_idx)
        calib = dataset.get_calibration(data_idx)
        img = dataset.get_image(data_idx)
        img_height, img_width, _ = img.shape
        print(data_idx, "image shape: ", img.shape)
        print(data_idx, "velo  shape: ", pc_velo.shape)

        # cv2.imshow("", img)
        # cv2.waitKey(-1)
        # depth = cv2.cvtColor(depth, cv2.COLOR_BGR2RGB)
        # depth_height, depth_width, depth_channel = img.shape

        # print(('Image shape: ', img.shape))

        # if args.stat:
        #     stat_lidar_with_boxes(pc_velo, objects, calib)
        #     continue
        # print("======== Objects in Ground Truth ========")
        # n_obj = 0
        # for obj in objects:
        #     if obj.type != "DontCare":
        #         print("=== {} object ===".format(n_obj + 1))
        #         obj.print_object()
        #         n_obj += 1
        #
        # # Draw 3d box in LiDAR point cloud
        # if args.show_lidar_topview_with_boxes:
        #     # Draw lidar top view
        #     show_lidar_topview_with_boxes(pc_velo, objects, calib, objects_pred)
        #
        # # show_image_with_boxes_3type(img, objects, calib, objects2d, data_idx, objects_pred)

        # Draw 2d and 3d boxes on image
        show_image_with_boxes(img, objects, calib)

        # Show LiDAR points on image.
        show_lidar_on_image(pc_velo[:, 0:3], img, calib, img_width, img_height)

        # if args.show_lidar_with_depth:
        #     # Draw 3d box in LiDAR point cloud
        #     show_lidar_with_depth(
        #         pc_velo,
        #         objects,
        #         calib,
        #         fig,
        #         args.img_fov,
        #         img_width,
        #         img_height,
        #         objects_pred,
        #         depth,
        #         img,
        #         constraint_box=args.const_box,
        #         save=args.save_depth,
        #         pc_label=args.pc_label,
        #     )
        #     # show_lidar_with_boxes(pc_velo, objects, calib, True, img_width, img_height, \
        #     #    objects_pred, depth, img)

        # input_str = raw_input()
        #
        # mlab.clf()
        # if input_str == "killall":
        #     break
