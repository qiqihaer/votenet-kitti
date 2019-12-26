# import _init_path
import os
import numpy as np
import pickle
import torch

# import lib.utils.roipool3d.roipool3d_utils as roipool3d_utils
from kitti.kitti_dataset import KittiDataset
import argparse
from kitti.kitti_utils import boxes3d_to_corners3d, extract_pc_in_box3d

parser = argparse.ArgumentParser()
parser.add_argument('--save_dir', type=str, default='./gt_database')
parser.add_argument('--class_name', type=str, default='Car')
parser.add_argument('--split', type=str, default='train')
args = parser.parse_args()


class GTDatabaseGenerator(KittiDataset):
    def __init__(self, root_dir, split='train', classes=args.class_name):
        super().__init__(root_dir, split=split)
        self.gt_database = None
        if classes == 'Car':
            self.classes = ('Background', 'Car', 'Van')
        elif classes == 'People':
            self.classes = ('Background', 'Pedestrian', 'Cyclist')
        elif classes == 'Pedestrian':
            self.classes = ('Background', 'Pedestrian')
        elif classes == 'Cyclist':
            self.classes = ('Background', 'Cyclist')
        else:
            assert False, "Invalid classes: %s" % classes

    def __len__(self):
        raise NotImplementedError

    def __getitem__(self, item):
        raise NotImplementedError

    def filtrate_objects(self, obj_list):
        valid_obj_list = []
        for obj in obj_list:
            if obj.cls_type not in self.classes:
                continue
            if obj.level_str not in ['Easy', 'Moderate', 'Hard']:
                continue
            valid_obj_list.append(obj)

        return valid_obj_list

    def generate_gt_database(self):
        car = []
        van = []
        for idx, sample_id in enumerate(self.image_idx_list):
            sample_id = int(sample_id)

            obj_list = self.filtrate_objects(self.get_label(sample_id))

            if obj_list.__len__() == 0:
                continue
            else:
                for obj in obj_list:
                    size = [obj.h, obj.w, obj.l, obj.ry]
                    type = obj.cls_type
                    if type == 'Car':
                        car.append(size)
                    elif type == 'Van':
                        van.append(size)
                    else:
                        print(sample_id)

        car = np.array(car)
        van = np.array(van)
        car_mean_size = car.mean(1)
        van_mean_size = van.mean(1)
        print(car_mean_size)
        print(van_mean_size)


if __name__ == '__main__':
    dataset = GTDatabaseGenerator(root_dir='.', split=args.split)
    os.makedirs(args.save_dir, exist_ok=True)

    dataset.generate_gt_database()

