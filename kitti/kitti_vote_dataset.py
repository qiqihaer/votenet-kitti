import numpy as np
import os
import pickle
import torch
from torch.utils.data import DataLoader
from kitti.kitti_dataset import KittiDataset
import kitti.kitti_utils as kitti_utils
from kitti.model_util_kitti import KittiDatasetConfig
# import lib.utils.roipool3d.roipool3d_utils as roipool3d_utils
# from lib.config import cfg


class Config():
    def __init__(self):
        self.GT_AUG_HARD_RATIO = 0.6  # TODO: make a cfg
        self.INCLUDE_SIMILAR_TYPE = True
        self.PC_REDUCE_BY_RANGE = True
        self.PC_AREA_SCOPE = [[-40, 40], [-1, 3], [0, 70.4]]
        self.GT_AUG_ENABLED = True
        self.GT_AUG_APPLY_PROB = 1
        self.USE_INTENSITY = False
        self.GT_AUG_RAND_NUM = True
        self.GT_EXTRA_NUM = 15
        self.AUG_DATA = True
        self.AUG_METHOD_LIST = ['rotation', 'scaling', 'flip']
        self.AUG_METHOD_PROB = [1.0, 1.0, 0.5]
        self.AUG_ROT_RANGE = 18

cfg = Config()
# TODO: make a log
MAX_NUM_OBJ = 64
DC = KittiDatasetConfig()


class KittiVoteDataset(KittiDataset):

    def __init__(self, root_dir, npoints=16384, split='train', classes='Car', mode='TRAIN', random_select=True,
                 gt_database_dir=None):
        super().__init__(root_dir=root_dir, split=split)
        if classes == 'Car':
            self.classes = ('Background', 'Car')
            aug_scene_root_dir = os.path.join(root_dir, 'KITTI', 'aug_scene')
        elif classes == 'People':
            self.classes = ('Background', 'Pedestrian', 'Cyclist')
        elif classes == 'Pedestrian':
            self.classes = ('Background', 'Pedestrian')
            aug_scene_root_dir = os.path.join(root_dir, 'KITTI', 'aug_scene_ped')
        elif classes == 'Cyclist':
            self.classes = ('Background', 'Cyclist')
            aug_scene_root_dir = os.path.join(root_dir, 'KITTI', 'aug_scene_cyclist')
        else:
            assert False, "Invalid classes: %s" % classes

        self.num_class = self.classes.__len__()

        self.npoints = npoints
        self.sample_id_list = []
        self.random_select = random_select

        if split == 'train_aug':
            self.aug_label_dir = os.path.join(aug_scene_root_dir, 'training', 'aug_label')
            self.aug_pts_dir = os.path.join(aug_scene_root_dir, 'training', 'rectified_data')
        else:
            self.aug_label_dir = os.path.join(aug_scene_root_dir, 'training', 'aug_label')
            self.aug_pts_dir = os.path.join(aug_scene_root_dir, 'training', 'rectified_data')

        self.gt_database = None
        assert mode in ['TRAIN', 'EVAL', 'TEST'], 'Invalid mode: %s' % mode
        self.mode = mode

        if gt_database_dir is not None:
            self.gt_database = pickle.load(open(gt_database_dir, 'rb'))

            if cfg.GT_AUG_HARD_RATIO > 0:
                easy_list, hard_list = [], []
                for k in range(self.gt_database.__len__()):
                    obj = self.gt_database[k]
                    if obj['points'].shape[0] > 100:
                        easy_list.append(obj)
                    else:
                        hard_list.append(obj)
                self.gt_database = [easy_list, hard_list]
                print('Loading gt_database(easy(pt_num>100): %d, hard(pt_num<=100): %d) from %s'
                      % (len(easy_list), len(hard_list), gt_database_dir))
            else:
                print('Loading gt_database(%d) from %s' % (len(self.gt_database), gt_database_dir))

        if mode == 'TRAIN':
            self.preprocess_rpn_training_data()
        else:
            self.sample_id_list = [int(sample_id) for sample_id in self.image_idx_list]
            print('Load testing samples from %s' % self.imageset_dir)
            print('Done: total test samples %d' % len(self.sample_id_list))

        # self.sample_id_list = self.sample_id_list[0:10]

    def preprocess_rpn_training_data(self):
        """
        Discard samples which don't have current classes, which will not be used for training.
        Valid sample_id is stored in self.sample_id_list
        """
        print('Loading %s samples from %s ...' % (self.mode, self.label_dir))
        for idx in range(0, self.num_sample):
            sample_id = int(self.image_idx_list[idx])
            obj_list = self.filtrate_objects(self.get_label(sample_id))
            if len(obj_list) == 0:
                # self.logger.info('No gt classes: %06d' % sample_id)
                continue
            self.sample_id_list.append(sample_id)

        print('Done: filter %s results: %d / %d\n' % (self.mode, len(self.sample_id_list), len(self.image_idx_list)))

    def filtrate_objects(self, obj_list):
        """
        Discard objects which are not in self.classes (or its similar classes)
        :param obj_list: list
        :return: list
        """
        type_whitelist = self.classes
        if self.mode == 'TRAIN' and cfg.INCLUDE_SIMILAR_TYPE:
            type_whitelist = list(self.classes)
            if 'Car' in self.classes:
                type_whitelist.append('Van')
            if 'Pedestrian' in self.classes:  # or 'Cyclist' in self.classes:
                type_whitelist.append('Person_sitting')

        valid_obj_list = []
        for obj in obj_list:
            if obj.cls_type not in type_whitelist:  # rm Van, 20180928
                continue
            if self.mode == 'TRAIN' and cfg.PC_REDUCE_BY_RANGE and (self.check_pc_range(obj.pos) is False):
                continue
            valid_obj_list.append(obj)
        return valid_obj_list

    def check_pc_range(self, xyz):
        """
        :param xyz: [x, y, z]
        :return:
        """
        x_range, y_range, z_range = cfg.PC_AREA_SCOPE
        if (x_range[0] <= xyz[0] <= x_range[1]) and (y_range[0] <= xyz[1] <= y_range[1]) and \
                (z_range[0] <= xyz[2] <= z_range[1]):
            return True
        return False

    def __len__(self):
        return len(self.sample_id_list)

    def __getitem__(self, index):
        # load points
        sample_id = int(self.sample_id_list[index])
        if sample_id < 10000:
            calib = self.get_calib(sample_id)
            # img = self.get_image(sample_id)
            img_shape = self.get_image_shape(sample_id)
            pts_lidar = self.get_lidar(sample_id)

            # get valid point (projected points should be in image)
            pts_rect = calib.lidar_to_rect(pts_lidar[:, 0:3])
            pts_intensity = pts_lidar[:, 3]
        else:
            calib = self.get_calib(sample_id % 10000)
            # img = self.get_image(sample_id % 10000)
            img_shape = self.get_image_shape(sample_id % 10000)

            pts_file = os.path.join(self.aug_pts_dir, '%06d.bin' % sample_id)
            assert os.path.exists(pts_file), '%s' % pts_file
            aug_pts = np.fromfile(pts_file, dtype=np.float32).reshape(-1, 4)
            pts_rect, pts_intensity = aug_pts[:, 0:3], aug_pts[:, 3]

        # transform lidar coord to camera coord
        pts_img, pts_rect_depth = calib.rect_to_img(pts_rect)
        pts_valid_flag = self.get_valid_flag(pts_rect, pts_img, pts_rect_depth, img_shape)

        pts_rect = pts_rect[pts_valid_flag][:, 0:3]
        pts_intensity = pts_intensity[pts_valid_flag]

        # gt augmentation
        if cfg.GT_AUG_ENABLED and self.mode == 'TRAIN':
            # all labels for checking overlapping
            all_gt_obj_list = self.filtrate_dc_objects(self.get_label(sample_id))
            all_gt_boxes3d = kitti_utils.objs_to_boxes3d(all_gt_obj_list)

            # kitti_utils.data_viz(objects=all_gt_boxes3d, pts=pts_rect)  # viz raw points and labels

            gt_aug_flag = False
            if np.random.rand() < cfg.GT_AUG_APPLY_PROB:
                # augment one scene
                gt_aug_flag, pts_rect, pts_intensity, extra_gt_boxes3d, extra_gt_obj_list = \
                    self.apply_gt_aug_to_one_scene(sample_id, pts_rect, pts_intensity, all_gt_boxes3d)
                # print(len(extra_gt_obj_list))
        # generate inputs
        if self.mode == 'TRAIN' or self.random_select:
            if self.npoints < len(pts_rect):
                pts_depth = pts_rect[:, 2]
                pts_near_flag = pts_depth < 40.0
                far_idxs_choice = np.where(pts_near_flag == 0)[0]
                near_idxs = np.where(pts_near_flag == 1)[0]
                near_idxs_choice = np.random.choice(near_idxs, self.npoints - len(far_idxs_choice), replace=False)

                choice = np.concatenate((near_idxs_choice, far_idxs_choice), axis=0) \
                    if len(far_idxs_choice) > 0 else near_idxs_choice
                np.random.shuffle(choice)
            else:
                choice = np.arange(0, len(pts_rect), dtype=np.int32)
                if self.npoints > len(pts_rect):
                    extra_choice = np.random.choice(choice, self.npoints - len(pts_rect), replace=False)
                    choice = np.concatenate((choice, extra_choice), axis=0)
                np.random.shuffle(choice)

            ret_pts_rect = pts_rect[choice, :]
            ret_pts_intensity = pts_intensity[choice] - 0.5  # translate intensity to [-0.5, 0.5]
        else:
            ret_pts_rect = pts_rect
            ret_pts_intensity = pts_intensity - 0.5

        pts_features = [ret_pts_intensity.reshape(-1, 1)]
        ret_pts_features = np.concatenate(pts_features, axis=1) if pts_features.__len__() > 1 else pts_features[0]



        gt_obj_list = self.filtrate_objects(self.get_label(sample_id))

        if cfg.GT_AUG_ENABLED and self.mode == 'TRAIN' and gt_aug_flag:
            gt_obj_list.extend(extra_gt_obj_list)
        gt_boxes3d = kitti_utils.objs_to_boxes3d(gt_obj_list)

        gt_alpha = np.zeros((gt_obj_list.__len__()), dtype=np.float32)
        for k, obj in enumerate(gt_obj_list):
            gt_alpha[k] = obj.alpha

        # data augmentation
        aug_pts_rect = ret_pts_rect.copy()
        aug_gt_boxes3d = gt_boxes3d.copy()
        if cfg.AUG_DATA and self.mode == 'TRAIN':
            aug_pts_rect, aug_gt_boxes3d, aug_method = self.data_augmentation(aug_pts_rect, aug_gt_boxes3d, gt_alpha,
                                                                              sample_id)

        # prepare input
        if cfg.USE_INTENSITY:
            pts_input = np.concatenate((aug_pts_rect, ret_pts_features), axis=1)  # (N, C)
        else:
            pts_input = aug_pts_rect

        # kitti_utils.data_viz(objects=aug_gt_boxes3d, pts=pts_input, name='demo_{}'.format(index))  # viz

        # ------------------------------- LABELS ------------------------------
        box3d_centers = np.zeros((MAX_NUM_OBJ, 3))
        box3d_sizes = np.zeros((MAX_NUM_OBJ, 3))
        angle_classes = np.zeros((MAX_NUM_OBJ,))
        angle_residuals = np.zeros((MAX_NUM_OBJ,))
        size_classes = np.zeros((MAX_NUM_OBJ,))
        size_residuals = np.zeros((MAX_NUM_OBJ, 3))
        label_mask = np.zeros((MAX_NUM_OBJ))
        label_mask[0:aug_gt_boxes3d.shape[0]] = 1
        max_bboxes = np.zeros((MAX_NUM_OBJ, 8))
        # max_bboxes[0:aug_gt_boxes3d.shape[0], :] = bboxes

        for i in range(aug_gt_boxes3d.shape[0]):
            bbox = aug_gt_boxes3d[i]
            semantic_class = DC.type2class[gt_obj_list[i].cls_type]
            box3d_size = bbox[3:6].copy()  # h, w, l
            box3d_size = box3d_size[::-1]  # hwl to lwh
            size_class, size_residual = DC.size2class(box3d_size, DC.class2type[semantic_class])
            box3d_center = bbox[0:3].copy()  # y is on the bottom surface of the box
            box3d_center[1] = box3d_center[1] - box3d_size[2] / 2
            angle = bbox[6]
            angle_class, angle_residual = DC.angle2class(angle)

            box3d_centers[i,:] = box3d_center
            angle_classes[i] = angle_class
            angle_residuals[i] = angle_residual
            size_classes[i] = size_class
            size_residuals[i] = size_residual
            box3d_sizes[i,:] = box3d_size  # lwh
            max_bboxes[i, 0:3] = box3d_center
            max_bboxes[i, 3:6] = box3d_size
            max_bboxes[i, 6] = angle
            max_bboxes[i, 7] = semantic_class

        ret_dict = {}
        ret_dict['point_clouds'] = pts_input.astype(np.float32)
        ret_dict['center_label'] = box3d_centers.astype(np.float32)
        ret_dict['heading_class_label'] = angle_classes.astype(np.int64)
        ret_dict['heading_residual_label'] = angle_residuals.astype(np.float32)
        ret_dict['size_class_label'] = size_classes.astype(np.int64)
        ret_dict['size_residual_label'] = size_residuals.astype(np.float32)
        target_bboxes_semcls = np.zeros((MAX_NUM_OBJ))
        target_bboxes_semcls[0:aug_gt_boxes3d.shape[0]] = max_bboxes[0:aug_gt_boxes3d.shape[0], -1]  # from 0 to 1
        ret_dict['sem_cls_label'] = target_bboxes_semcls.astype(np.int64)
        target_bboxes_mask = label_mask
        ret_dict['box_label_mask'] = target_bboxes_mask.astype(np.float32)
        ret_dict['scan_idx'] = np.array(sample_id).astype(np.int64)
        ret_dict['max_gt_bboxes'] = max_bboxes

        # compute votes
        N = aug_pts_rect.shape[0]
        point_votes = np.zeros((N, 10))  # 3 votes and 1 vote mask
        point_vote_idx = np.zeros((N)).astype(np.int32)  # in the range of [0,2]
        indices = np.arange(N)
        for box_ind in range(aug_gt_boxes3d.shape[0]):
            try:
                obj = aug_gt_boxes3d[box_ind]
                # Find all points in this object's OBB
                box3d_pts_3d = kitti_utils.single_box3d_to_corner3d(obj)
                pc_in_box3d, inds = kitti_utils.extract_pc_in_box3d(aug_pts_rect, box3d_pts_3d)
                # Assign first dimension to indicate it is in an object box
                point_votes[inds, 0] = 1
                cur_center = box3d_centers[box_ind]
                # Add the votes (all 0 if the point is not in any object's OBB)
                votes = np.expand_dims(cur_center, 0) - pc_in_box3d[:, 0:3]
                sparse_inds = indices[inds]  # turn dense True,False inds to sparse number-wise inds
                for i in range(len(sparse_inds)):
                    j = sparse_inds[i]
                    point_votes[j, int(point_vote_idx[j] * 3 + 1):int((point_vote_idx[j] + 1) * 3 + 1)] = votes[i, :]
                    # Populate votes with the fisrt vote
                    if point_vote_idx[j] == 0:
                        point_votes[j, 4:7] = votes[i, :]
                        point_votes[j, 7:10] = votes[i, :]
                point_vote_idx[inds] = np.minimum(2, point_vote_idx[inds] + 1)
            except:
                print('ERROR ----', sample_id)
        point_votes_mask = point_votes[:, 0]
        point_votes = point_votes[:, 1:]
        ret_dict['vote_label'] = point_votes.astype(np.float32)
        ret_dict['vote_label_mask'] = point_votes_mask.astype(np.int64)

        # kitti_utils.data_viz(objects=aug_gt_boxes3d, pts=pts_input, name='demo_{}'.format(index))  # viz
        # kitti_utils.point_viz(pts_input, name='all_points_{}'.format(index))
        # kitti_utils.box_viz(aug_gt_boxes3d, name='all_boxes_{}'.format(index))
        # points_in_boxes = pts_input[point_votes_mask == 1, :]
        # kitti_utils.point_viz(points_in_boxes, name='point_in_boxes_{}'.format(index))
        # kitti_utils.point_viz(box3d_centers, name='centers_{}'.format(index))
        # point_in_box_votes = point_votes[point_votes_mask == 1, 0:3]
        # kitti_utils.point_viz(point_in_box_votes + points_in_boxes, name='recovered_center_{}'.format(index))

        # corner3d = kitti_utils.single_box3d_to_corner3d(aug_gt_boxes3d[0])
        # box3d = kitti_utils.corer3d_to_box3d(corner3d)

        return ret_dict

    @staticmethod
    def get_valid_flag(pts_rect, pts_img, pts_rect_depth, img_shape):
        """
        Valid point should be in the image (and in the PC_AREA_SCOPE)
        :param pts_rect:
        :param pts_img:
        :param pts_rect_depth:
        :param img_shape:
        :return:
        """
        val_flag_1 = np.logical_and(pts_img[:, 0] >= 0, pts_img[:, 0] < img_shape[1])
        val_flag_2 = np.logical_and(pts_img[:, 1] >= 0, pts_img[:, 1] < img_shape[0])
        val_flag_merge = np.logical_and(val_flag_1, val_flag_2)
        pts_valid_flag = np.logical_and(val_flag_merge, pts_rect_depth >= 0)

        if cfg.PC_REDUCE_BY_RANGE:
            x_range, y_range, z_range = cfg.PC_AREA_SCOPE
            pts_x, pts_y, pts_z = pts_rect[:, 0], pts_rect[:, 1], pts_rect[:, 2]
            range_flag = (pts_x >= x_range[0]) & (pts_x <= x_range[1]) \
                         & (pts_y >= y_range[0]) & (pts_y <= y_range[1]) \
                         & (pts_z >= z_range[0]) & (pts_z <= z_range[1])
            pts_valid_flag = pts_valid_flag & range_flag
        return pts_valid_flag

    @staticmethod
    def filtrate_dc_objects(obj_list):
        valid_obj_list = []
        for obj in obj_list:
            if obj.cls_type in ['DontCare']:
                continue
            valid_obj_list.append(obj)

        return valid_obj_list

    def apply_gt_aug_to_one_scene(self, sample_id, pts_rect, pts_intensity, all_gt_boxes3d):
        """
        :param pts_rect: (N, 3)
        :param all_gt_boxex3d: (M2, 7)
        :return:
        """
        assert self.gt_database is not None
        # extra_gt_num = np.random.randint(10, 15)
        # try_times = 50
        if cfg.GT_AUG_RAND_NUM:
            extra_gt_num = np.random.randint(10, cfg.GT_EXTRA_NUM)
        else:
            extra_gt_num = cfg.GT_EXTRA_NUM
        try_times = 100
        cnt = 0
        cur_gt_boxes3d = all_gt_boxes3d.copy()
        cur_gt_boxes3d[:, 4] += 0.5  # TODO: consider different objects
        cur_gt_boxes3d[:, 5] += 0.5  # enlarge new added box to avoid too nearby boxes
        cur_gt_corners = kitti_utils.boxes3d_to_corners3d(cur_gt_boxes3d)

        extra_gt_obj_list = []
        extra_gt_boxes3d_list = []
        new_pts_list, new_pts_intensity_list = [], []
        src_pts_flag = np.ones(pts_rect.shape[0], dtype=np.int32)

        road_plane = self.get_road_plane(sample_id)
        a, b, c, d = road_plane

        while try_times > 0:
            if cnt > extra_gt_num:
                break

            try_times -= 1
            if cfg.GT_AUG_HARD_RATIO > 0:
                p = np.random.rand()
                if p > cfg.GT_AUG_HARD_RATIO:
                    # use easy sample
                    rand_idx = np.random.randint(0, len(self.gt_database[0]))
                    new_gt_dict = self.gt_database[0][rand_idx]
                else:
                    # use hard sample
                    rand_idx = np.random.randint(0, len(self.gt_database[1]))
                    new_gt_dict = self.gt_database[1][rand_idx]
            else:
                rand_idx = np.random.randint(0, self.gt_database.__len__())
                new_gt_dict = self.gt_database[rand_idx]

            new_gt_box3d = new_gt_dict['gt_box3d'].copy()
            new_gt_points = new_gt_dict['points'].copy()
            new_gt_intensity = new_gt_dict['intensity'].copy()
            new_gt_obj = new_gt_dict['obj']
            center = new_gt_box3d[0:3]
            if cfg.PC_REDUCE_BY_RANGE and (self.check_pc_range(center) is False):
                continue

            if new_gt_points.__len__() < 5:  # too few points
                continue

            # put it on the road plane
            cur_height = (-d - a * center[0] - c * center[2]) / b
            move_height = new_gt_box3d[1] - cur_height
            new_gt_box3d[1] -= move_height
            new_gt_points[:, 1] -= move_height
            new_gt_obj.pos[1] -= move_height

            new_enlarged_box3d = new_gt_box3d.copy()
            new_enlarged_box3d[4] += 0.5
            new_enlarged_box3d[5] += 0.5  # enlarge new added box to avoid too nearby boxes

            cnt += 1
            new_corners = kitti_utils.boxes3d_to_corners3d(new_enlarged_box3d.reshape(1, 7))
            iou3d = kitti_utils.get_iou3d(new_corners, cur_gt_corners)
            valid_flag = iou3d.max() < 1e-8
            if not valid_flag:
                continue

            enlarged_box3d = new_gt_box3d.copy()
            enlarged_box3d[3] += 2  # remove the points above and below the object
            enlarged_box3d[4] += 0.2
            enlarged_box3d[5] += 0.2

            # --------wq:20191217
            gt_box3d_corners = kitti_utils.single_box3d_to_corner3d(enlarged_box3d)
            gt_box3d_corners[0:4, 1] = gt_box3d_corners[0:4, 1] + 2
            gt_box3d_corners[4:8, 1] = gt_box3d_corners[4:8, 1] - 2
            _, pts_in_3d_box_inds = kitti_utils.extract_pc_in_box3d(pts_rect, gt_box3d_corners)
            src_pts_flag[pts_in_3d_box_inds] = 0  # remove the original points which are inside the new box
            # --------wq:20191217

            new_pts_list.append(new_gt_points)
            new_pts_intensity_list.append(new_gt_intensity)
            cur_gt_boxes3d = np.concatenate((cur_gt_boxes3d, new_enlarged_box3d.reshape(1, 7)), axis=0)
            cur_gt_corners = np.concatenate((cur_gt_corners, new_corners), axis=0)
            extra_gt_boxes3d_list.append(new_gt_box3d.reshape(1, 7))
            extra_gt_obj_list.append(new_gt_obj)

        if new_pts_list.__len__() == 0:
            return False, pts_rect, pts_intensity, None, None

        extra_gt_boxes3d = np.concatenate(extra_gt_boxes3d_list, axis=0)
        # remove original points and add new points
        pts_rect = pts_rect[src_pts_flag == 1]
        pts_intensity = pts_intensity[src_pts_flag == 1]
        new_pts_rect = np.concatenate(new_pts_list, axis=0)
        new_pts_intensity = np.concatenate(new_pts_intensity_list, axis=0)
        pts_rect = np.concatenate((pts_rect, new_pts_rect), axis=0)
        pts_intensity = np.concatenate((pts_intensity, new_pts_intensity), axis=0)

        return True, pts_rect, pts_intensity, extra_gt_boxes3d, extra_gt_obj_list

    def data_augmentation(self, aug_pts_rect, aug_gt_boxes3d, gt_alpha, sample_id=None, mustaug=False, stage=1):
        """
        :param aug_pts_rect: (N, 3)
        :param aug_gt_boxes3d: (N, 7)
        :param gt_alpha: (N)
        :return:
        """
        aug_list = cfg.AUG_METHOD_LIST
        aug_enable = 1 - np.random.rand(3)
        if mustaug is True:
            aug_enable[0] = -1
            aug_enable[1] = -1
        aug_method = []
        if 'rotation' in aug_list and aug_enable[0] < cfg.AUG_METHOD_PROB[0]:
            angle = np.random.uniform(-np.pi / cfg.AUG_ROT_RANGE, np.pi / cfg.AUG_ROT_RANGE)
            aug_pts_rect = kitti_utils.rotate_pc_along_y(aug_pts_rect, rot_angle=angle)
            if stage == 1:
                # xyz change, hwl unchange
                aug_gt_boxes3d = kitti_utils.rotate_pc_along_y(aug_gt_boxes3d, rot_angle=angle)

                # calculate the ry after rotation
                x, z = aug_gt_boxes3d[:, 0], aug_gt_boxes3d[:, 2]
                beta = np.arctan2(z, x)
                new_ry = np.sign(beta) * np.pi / 2 + gt_alpha - beta
                aug_gt_boxes3d[:, 6] = new_ry  # TODO: not in [-np.pi / 2, np.pi / 2]
            elif stage == 2:
                # for debug stage-2, this implementation has little float precision difference with the above one
                assert aug_gt_boxes3d.shape[0] == 2
                aug_gt_boxes3d[0] = self.rotate_box3d_along_y(aug_gt_boxes3d[0], angle)
                aug_gt_boxes3d[1] = self.rotate_box3d_along_y(aug_gt_boxes3d[1], angle)
            else:
                raise NotImplementedError

            aug_method.append(['rotation', angle])

        if 'scaling' in aug_list and aug_enable[1] < cfg.AUG_METHOD_PROB[1]:
            scale = np.random.uniform(0.95, 1.05)
            aug_pts_rect = aug_pts_rect * scale
            aug_gt_boxes3d[:, 0:6] = aug_gt_boxes3d[:, 0:6] * scale
            aug_method.append(['scaling', scale])

        if 'flip' in aug_list and aug_enable[2] < cfg.AUG_METHOD_PROB[2]:
            # flip horizontal
            aug_pts_rect[:, 0] = -aug_pts_rect[:, 0]
            aug_gt_boxes3d[:, 0] = -aug_gt_boxes3d[:, 0]
            # flip orientation: ry > 0: pi - ry, ry < 0: -pi - ry
            if stage == 1:
                aug_gt_boxes3d[:, 6] = np.sign(aug_gt_boxes3d[:, 6]) * np.pi - aug_gt_boxes3d[:, 6]
            elif stage == 2:
                assert aug_gt_boxes3d.shape[0] == 2
                aug_gt_boxes3d[0, 6] = np.sign(aug_gt_boxes3d[0, 6]) * np.pi - aug_gt_boxes3d[0, 6]
                aug_gt_boxes3d[1, 6] = np.sign(aug_gt_boxes3d[1, 6]) * np.pi - aug_gt_boxes3d[1, 6]
            else:
                raise NotImplementedError

            aug_method.append('flip')

        return aug_pts_rect, aug_gt_boxes3d, aug_method


if __name__=='__main__':
    gt_database_dir = './gt_database/train_gt_database_3level_Car.pkl'
    kitti_dataset = KittiVoteDataset(root_dir='.', gt_database_dir=gt_database_dir, split='val', mode='EVAL')
    kitti_dataloader = DataLoader(kitti_dataset, batch_size=2, shuffle=True, num_workers=4)

    for i, batch in enumerate(kitti_dataloader):
        pts = batch['point_clouds']
        a = 1
