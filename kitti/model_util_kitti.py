import numpy as np
import sys
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'utils'))


class KittiDatasetConfig(object):
    def __init__(self):
        self.num_class = 2
        self.num_heading_bin = 12
        self.num_size_cluster = 2

        self.type2class = {'Car': 0, 'Van': 1}
        self.class2type = {self.type2class[t]: t for t in self.type2class}
        self.type2onehotclass={'Car': 0}
        # self.type_mean_size = {'Car': np.array([1.52563191462, 1.62856739989, 3.88311640418]),
        #                        'Van': np.array([2.20310262529, 1.91235083532, 5.20625298329])}  # whl
        self.type_mean_size = {'Car': np.array([3.88311640418, 1.62856739989, 1.52563191462]),
                               'Van': np.array([5.20625298329, 1.91235083532, 2.20310262529])}  # lhw

        self.mean_size_arr = np.zeros((self.num_size_cluster, 3))
        for i in range(self.num_size_cluster):
            self.mean_size_arr[i,:] = self.type_mean_size[self.class2type[i]]

    def size2class(self, size, type_name):
        ''' Convert 3D box size (l,w,h) to size class and size residual '''
        size_class = self.type2class[type_name]
        size_residual = size - self.type_mean_size[type_name]
        return size_class, size_residual

    def class2size(self, pred_cls, residual):
        ''' Inverse function to size2class '''
        mean_size = self.type_mean_size[self.class2type[pred_cls]]
        return mean_size + residual

    def angle2class(self, angle):
        ''' Convert continuous angle to discrete class
            [optinal] also small regression number from
            class center angle to current angle.

            angle is from 0-2pi (or -pi~pi), class center at 0, 1*(2pi/N), 2*(2pi/N) ...  (N-1)*(2pi/N)
            return is class of int32 of 0,1,...,N-1 and a number such that
                class*(2pi/N) + number = angle
        '''
        num_class = self.num_heading_bin
        angle = angle % (2 * np.pi)
        assert (angle >= 0 and angle <= 2 * np.pi)
        angle_per_class = 2 * np.pi / float(num_class)
        shifted_angle = (angle + angle_per_class / 2) % (2 * np.pi)
        class_id = int(shifted_angle / angle_per_class)
        residual_angle = shifted_angle - (class_id * angle_per_class + angle_per_class / 2)
        return class_id, residual_angle

    def class2angle(self, pred_cls, residual, to_label_format=True):
        ''' Inverse function to angle2class '''
        num_class = self.num_heading_bin
        angle_per_class = 2 * np.pi / float(num_class)
        angle_center = pred_cls * angle_per_class
        angle = angle_center + residual
        if to_label_format and angle > np.pi:
            angle = angle - 2 * np.pi
        return angle

    # def param2obb(self, center, heading_class, heading_residual, size_class, size_residual):  # TODO:need to be changed
    #     print('------------------------------------------hhh------------------------------------------------------------------------------------')
    #     heading_angle = self.class2angle(heading_class, heading_residual)
    #     box_size = self.class2size(int(size_class), size_residual)
    #     obb = np.zeros((7,))
    #     obb[0:3] = center
    #     obb[3:6] = box_size
    #     obb[6] = heading_angle
    #     return obb
