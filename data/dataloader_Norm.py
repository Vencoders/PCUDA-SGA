import os
import glob
import h5py
import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d
import torch
from torch.utils.data import Dataset
from utils.pc_utils_Norm import (farthest_point_sample_np, scale_to_unit_cube, jitter_pointcloud,
                                rotate_shape, random_rotate_one_axis,farthest_point_sample,MixupS1,MixupA2)

from utils.data_utils import PointcloudScale, PointcloudToTensor, PointcloudTranslate, PointcloudRotate, PointcloudRotatePerturbation, \
                             PointcloudRandomCrop, PointcloudUpSampling,PointcloudRandomCropEdge,PointcloudNormalize,PointcloudJitter,PointcloudRandomInputDropout

import torchvision.transforms as transforms

from .transformations import *
import torch.utils.data as data


import copy

from . import data_utils as d_utils
from .plyfile import load_ply




eps = 10e-4
NUM_POINTS = 1024
idx_to_label = {0: "bathtub", 1: "bed", 2: "bookshelf", 3: "cabinet",
                4: "chair", 5: "lamp", 6: "monitor",
                7: "plant", 8: "sofa", 9: "table"}
label_to_idx = {"bathtub": 0, "bed": 1, "bookshelf": 2, "cabinet": 3,
                "chair": 4, "lamp": 5, "monitor": 6,
                "plant": 7, "sofa": 8, "table": 9}

trans = transforms.Compose(
    [
        PointcloudToTensor(),
        PointcloudScale(lo=0.7, hi=1.3, p=1),
        PointcloudRotatePerturbation(),
        PointcloudTranslate(0.2, p=1),
    ])



def load_data_h5py_scannet10(partition, dataroot):
    """
    Input:
        partition - train/test
    Return:
        data,label arrays
    """
    DATA_DIR = dataroot + '/PointDA_data/scannet_norm_curv'
    all_data = []
    all_label = []
    # print(os.path.join(DATA_DIR, '%s_*.h5' % partition))
    for h5_name in sorted(glob.glob(os.path.join(DATA_DIR, '%s_*.h5' % partition))):

        f = h5py.File(h5_name, 'r')
        data = f['data'][:]
        label = f['label'][:]
        f.close()
        all_data.append(data)
        all_label.append(label)
    all_data = np.concatenate(all_data, axis=0)
    all_label = np.concatenate(all_label, axis=0)
    return np.array(all_data).astype('float32'), np.array(all_label).astype('int64')





class ScanNet(Dataset):
    """
    scannet dataset for pytorch dataloader
    """
    def __init__(self, io, dataroot, partition='train', random_rotation=True):
        self.partition = partition
        self.random_rotation = random_rotation

        # read data
        self.data, self.label = load_data_h5py_scannet10(self.partition, dataroot)
        self.num_examples = self.data.shape[0]

        # split train to train part and validation part
        if partition == "train":
            self.train_ind = np.asarray([i for i in range(self.num_examples) if i % 10 < 8]).astype(np.int32)
            np.random.shuffle(self.train_ind)
            self.val_ind = np.asarray([i for i in range(self.num_examples) if i % 10 >= 8]).astype(np.int32)
            np.random.shuffle(self.val_ind)

        io.cprint("number of " + partition + " examples in scannet" + ": " + str(self.data.shape[0]))
        unique, counts = np.unique(self.label, return_counts=True)
        io.cprint("Occurrences count of classes in scannet " + partition + " set: " + str(dict(zip(unique, counts))))

    def __getitem__(self, item):
        # print("data.shape:",self.data.shape)
        pointcloud = np.copy(self.data[item])[:, :3]
        # print("pointcloud.shape:",pointcloud.shape)[3:] -> [3:7]
        norm_curv = np.copy(self.data[item])[:, 3:7].astype(np.float32)
        # print("norm_curv.shape:",norm_curv.shape)
        label = np.copy(self.label[item])
        pointcloud = scale_to_unit_cube(pointcloud)
        # Rotate ScanNet by -90 degrees
        pointcloud = self.rotate_pc(pointcloud)
        # sample according to farthest point sampling
        if pointcloud.shape[0] > NUM_POINTS:
            pointcloud = np.swapaxes(np.expand_dims(pointcloud, 0), 1, 2)
            norm_curv = np.swapaxes(np.expand_dims(norm_curv, 0), 1, 2)
            # print("norm_curv.shape:", norm_curv.shape)
            _, pointcloud, norm_curv = farthest_point_sample_np(pointcloud, norm_curv, NUM_POINTS)
            pointcloud = np.swapaxes(pointcloud.squeeze(), 1, 0).astype('float32')
            norm_curv = np.swapaxes(norm_curv.squeeze(), 1, 0).astype('float32')

        # apply data rotation and augmentation on train samples
        if self.random_rotation==True:
            pointcloud = random_rotate_one_axis(pointcloud, "z")
        if self.partition == 'train' and item not in self.val_ind:
            # pointcloud = jitter_pointcloud(pointcloud)


            pc_0 = pointcloud.copy()
            pc_1 = pointcloud.copy()


            # point_m1, point_m2 = MixupA2(pc_1)
            point_m1, point_m2 = MixupS1(pc_1)


            return (pc_0, label, norm_curv, point_m1, point_m2)


        #aug_test
        # pointcloud_1 = np.copy(self.data[item])[:, :3]
        # pointcloud_2 = np.copy(self.data[item])[:, :3]


        return (pointcloud, label, norm_curv)

        # return (pointcloud, label, norm_curv, point_m1, point_m3)

    def __len__(self):
        return self.data.shape[0]

    # scannet is rotated such that the up direction is the y axis
    def rotate_pc(self, pointcloud):
        pointcloud = rotate_shape(pointcloud, 'x', -np.pi / 2)
        return pointcloud


class ModelNet(Dataset):
    """
    modelnet dataset for pytorch dataloader
    """
    def __init__(self, io, dataroot, partition='train', random_rotation=True):
        self.partition = partition
        self.random_rotation = random_rotation
        self.pc_list = []
        self.lbl_list = []
        DATA_DIR = os.path.join(dataroot, "PointDA_data", "modelnet_norm_curv")

        npy_list = sorted(glob.glob(os.path.join(DATA_DIR, '*', partition, '*.npy')))

        for _dir in npy_list:
            self.pc_list.append(_dir)
            self.lbl_list.append(label_to_idx[_dir.split('/')[-3]])

        self.label = np.asarray(self.lbl_list)
        self.num_examples = len(self.pc_list)

        # split train to train part and validation part
        if partition == "train":
            self.train_ind = np.asarray([i for i in range(self.num_examples) if i % 10 < 8]).astype(np.int32)
            np.random.shuffle(self.train_ind)
            self.val_ind = np.asarray([i for i in range(self.num_examples) if i % 10 >= 8]).astype(np.int32)
            np.random.shuffle(self.val_ind)

        io.cprint("number of " + partition + " examples in modelnet : " + str(len(self.pc_list)))
        unique, counts = np.unique(self.label, return_counts=True)
        io.cprint("Occurrences count of classes in modelnet " + partition + " set: " + str(dict(zip(unique, counts))))

    def __getitem__(self, item):
        pointcloud = np.load(self.pc_list[item])[:, :3].astype(np.float32)
        # print("pointcloud.shape1:",pointcloud.shape)
        # print("type",type(pointcloud))
        # [3:] -> [3:7]
        norm_curv = np.load(self.pc_list[item])[:, 3:7].astype(np.float32)
        label = np.copy(self.label[item])
        pointcloud = scale_to_unit_cube(pointcloud)
        # sample according to farthest point sampling
        if pointcloud.shape[0] > NUM_POINTS:
            pointcloud = np.swapaxes(np.expand_dims(pointcloud, 0), 1, 2)
            # print("pointcloud.shape2:", pointcloud.shape)
            norm_curv = np.swapaxes(np.expand_dims(norm_curv, 0), 1, 2)
            _, pointcloud, norm_curv = farthest_point_sample_np(pointcloud, norm_curv, NUM_POINTS)
            # print("pointcloud.shape3:", pointcloud.shape)
            pointcloud = np.swapaxes(pointcloud.squeeze(), 1, 0).astype('float32')
            # print("pointcloud.shape4:", pointcloud.shape)
            norm_curv = np.swapaxes(norm_curv.squeeze(), 1, 0).astype('float32')



        # apply data rotation and augmentation on train samples
        if self.random_rotation==True:
            pointcloud = random_rotate_one_axis(pointcloud, "z")
        if self.partition == 'train' and item not in self.val_ind:
            # pointcloud = jitter_pointcloud(pointcloud)

            # print("=",pointcloud.shape)

            pc_0 = pointcloud.copy()
            pc_1 = pointcloud.copy()


            # point_m1, point_m2 = MixupA2(pc_1)
            point_m1, point_m2 = MixupS1(pc_1)

            return (pc_0, label, norm_curv, point_m1, point_m2)


        # aug_test
        # pointcloud_1 = np.copy(self.data[item])[:, :3].astype(np.float32)
        # pointcloud_2 = np.copy(self.data[item])[:, :3].astype(np.float32)


        return (pointcloud, label, norm_curv)
        # return (pointcloud, label)
        # return (pointcloud, label, norm_curv, point_m1, point_m3)

    def __len__(self):
        return len(self.pc_list)


class ShapeNet(Dataset):
    """
    Sahpenet dataset for pytorch dataloader
    """
    def __init__(self, io, dataroot, partition='train', random_rotation=True):
        self.partition = partition
        self.random_rotation = random_rotation
        self.pc_list = []
        self.lbl_list = []
        DATA_DIR = os.path.join(dataroot, "PointDA_data", "shapenet_norm_curv")

        npy_list = sorted(glob.glob(os.path.join(DATA_DIR, '*', partition, '*.npy')))

        for _dir in npy_list:
            self.pc_list.append(_dir)
            self.lbl_list.append(label_to_idx[_dir.split('/')[-3]])

        self.label = np.asarray(self.lbl_list)
        self.num_examples = len(self.pc_list)

        # split train to train part and validation part
        if partition == "train":
            self.train_ind = np.asarray([i for i in range(self.num_examples) if i % 10 < 8]).astype(np.int32)
            np.random.shuffle(self.train_ind)
            self.val_ind = np.asarray([i for i in range(self.num_examples) if i % 10 >= 8]).astype(np.int32)
            np.random.shuffle(self.val_ind)

        io.cprint("number of " + partition + " examples in shapenet: " + str(len(self.pc_list)))
        unique, counts = np.unique(self.label, return_counts=True)
        io.cprint("Occurrences count of classes in shapenet " + partition + " set: " + str(dict(zip(unique, counts))))

    def __getitem__(self, item):
        pointcloud = np.load(self.pc_list[item])[:, :3].astype(np.float32)
        # print("pointcloud.shape:",pointcloud.shape)[3:] -> [3:7]
        norm_curv = np.load(self.pc_list[item])[:, 3:7].astype(np.float32)
        label = np.copy(self.label[item])
        pointcloud = scale_to_unit_cube(pointcloud)
        # Rotate ShapeNet by -90 degrees
        pointcloud = self.rotate_pc(pointcloud, label)
        # sample according to farthest point sampling
        if pointcloud.shape[0] > NUM_POINTS:
            pointcloud = np.swapaxes(np.expand_dims(pointcloud, 0), 1, 2)
            norm_curv = np.swapaxes(np.expand_dims(norm_curv, 0), 1, 2)
            _, pointcloud, norm_curv = farthest_point_sample_np(pointcloud, norm_curv, NUM_POINTS)
            pointcloud = np.swapaxes(pointcloud.squeeze(), 1, 0).astype('float32')
            norm_curv = np.swapaxes(norm_curv.squeeze(), 1, 0).astype('float32')

        # apply data rotation and augmentation on train samples
        if self.random_rotation == True:
            pointcloud = random_rotate_one_axis(pointcloud, "z")
        if self.partition == 'train' and item not in self.val_ind:
            # pointcloud = jitter_pointcloud(pointcloud)

            pc_0 = pointcloud.copy()
            pc_1 = pointcloud.copy()


            # point_m1, point_m2 = MixupA2(pc_1)
            point_m1, point_m2 = MixupS1(pc_1)

            return (pc_0, label, norm_curv, point_m1, point_m2)

        # aug_test
        # pointcloud_1 = np.copy(self.data[item])[:, :3].astype(np.float32)
        # pointcloud_2 = np.copy(self.data[item])[:, :3].astype(np.float32)
        # print("==",pointcloud.shape)



        return (pointcloud, label, norm_curv)
        # return (pointcloud, label)
        # return (pointcloud, label, norm_curv, point_m1, point_m3)


    def __len__(self):
        return len(self.pc_list)

    # shpenet is rotated such that the up direction is the y axis in all shapes except plant
    def rotate_pc(self, pointcloud, label):
        if label.item(0) != label_to_idx["plant"]:
            pointcloud = rotate_shape(pointcloud, 'x', -np.pi / 2)
        return pointcloud





RESOLUTION = 128
TRANS = -1.6

def euler2mat(angle):
    """Convert euler angles to rotation matrix.
     :param angle: [3] or [b, 3]
     :return
        rotmat: [3] or [b, 3, 3]
    source
    https://github.com/ClementPinard/SfmLearner-Pytorch/blob/master/inverse_warp.py
    """

    if len(angle.size()) == 1:
        x, y, z = angle[0], angle[1], angle[2]
        _dim = 0
        _view = [3, 3]
    elif len(angle.size()) == 2:
        b, _ = angle.size()
        x, y, z = angle[:, 0], angle[:, 1], angle[:, 2]
        _dim = 1
        _view = [b, 3, 3]

    else:
        assert False

    cosz = torch.cos(z)
    sinz = torch.sin(z)

    # zero = torch.zeros([b], requires_grad=False, device=angle.device)[0]
    # one = torch.ones([b], requires_grad=False, device=angle.device)[0]
    zero = z.detach()*0
    one = zero.detach()+1
    zmat = torch.stack([cosz, -sinz, zero,
                        sinz, cosz, zero,
                        zero, zero, one], dim=_dim).reshape(_view)

    cosy = torch.cos(y)
    siny = torch.sin(y)

    ymat = torch.stack([cosy, zero, siny,
                        zero, one, zero,
                        -siny, zero, cosy], dim=_dim).reshape(_view)

    cosx = torch.cos(x)
    sinx = torch.sin(x)

    xmat = torch.stack([one, zero, zero,
                        zero, cosx, -sinx,
                        zero, sinx, cosx], dim=_dim).reshape(_view)

    rot_mat = xmat @ ymat @ zmat
    # print(rot_mat)
    return rot_mat



def distribute(depth, _x, _y, size_x, size_y, image_height, image_width):
    """
    Distributes the depth associated with each point to the discrete coordinates (image_height, image_width) in a region
    of size (size_x, size_y).
    :param depth:
    :param _x:
    :param _y:
    :param size_x:
    :param size_y:
    :param image_height:
    :param image_width:
    :return:
    """

    assert size_x % 2 == 0 or size_x == 1
    assert size_y % 2 == 0 or size_y == 1
    batch, _ = depth.size()
    epsilon = torch.tensor([1e-12], requires_grad=False, device=depth.device)
    _i = torch.linspace(-size_x / 2, (size_x / 2) - 1, size_x, requires_grad=False, device=depth.device)
    _j = torch.linspace(-size_y / 2, (size_y / 2) - 1, size_y, requires_grad=False, device=depth.device)

    extended_x = _x.unsqueeze(2).repeat([1, 1, size_x]) + _i  # [batch, num_points, size_x]
    extended_y = _y.unsqueeze(2).repeat([1, 1, size_y]) + _j  # [batch, num_points, size_y]

    extended_x = extended_x.unsqueeze(3).repeat([1, 1, 1, size_y])  # [batch, num_points, size_x, size_y]
    extended_y = extended_y.unsqueeze(2).repeat([1, 1, size_x, 1])  # [batch, num_points, size_x, size_y]

    extended_x.ceil_()
    extended_y.ceil_()

    value = depth.unsqueeze(2).unsqueeze(3).repeat([1, 1, size_x, size_y])  # [batch, num_points, size_x, size_y]

    # all points that will be finally used
    masked_points = ((extended_x >= 0)
                     * (extended_x <= image_height - 1)
                     * (extended_y >= 0)
                     * (extended_y <= image_width - 1)
                     * (value >= 0))

    true_extended_x = extended_x
    true_extended_y = extended_y

    # to prevent error
    extended_x = (extended_x % image_height)
    extended_y = (extended_y % image_width)

    # [batch, num_points, size_x, size_y]
    distance = torch.abs((extended_x - _x.unsqueeze(2).unsqueeze(3))
                         * (extended_y - _y.unsqueeze(2).unsqueeze(3)))
    weight = (masked_points.float()
          * (1 / (value + epsilon)))  # [batch, num_points, size_x, size_y]
    weighted_value = value * weight

    weight = weight.view([batch, -1])
    weighted_value = weighted_value.view([batch, -1])

    coordinates = (extended_x.view([batch, -1]) * image_width) + extended_y.view(
        [batch, -1])
    coord_max = image_height * image_width
    true_coordinates = (true_extended_x.view([batch, -1]) * image_width) + true_extended_y.view(
        [batch, -1])
    true_coordinates[~masked_points.view([batch, -1])] = coord_max
    weight_scattered = torch.zeros(
        [batch, image_width * image_height],
        device=depth.device).scatter_add(1, coordinates.long(), weight)

    masked_zero_weight_scattered = (weight_scattered == 0.0)
    weight_scattered += masked_zero_weight_scattered.float()

    weighed_value_scattered = torch.zeros(
        [batch, image_width * image_height],
        device=depth.device).scatter_add(1, coordinates.long(), weighted_value)

    return weighed_value_scattered,  weight_scattered


def points2depth(points, image_height, image_width, size_x=4, size_y=4):
    """
    :param points: [B, num_points, 3]
    :param image_width:
    :param image_height:
    :param size_x:
    :param size_y:
    :return:
        depth_recovered: [B, image_width, image_height]
    """

    epsilon = torch.tensor([1e-12], requires_grad=False, device=points.device)
    # epsilon not needed, kept here to ensure exact replication of old version
    coord_x = (points[:, :, 0] / (points[:, :, 2] + epsilon)) * (image_width / image_height)  # [batch, num_points]
    coord_y = (points[:, :, 1] / (points[:, :, 2] + epsilon))  # [batch, num_points]

    batch, total_points, _ = points.size()
    depth = points[:, :, 2]  # [batch, num_points]
    # pdb.set_trace()
    _x = ((coord_x + 1) * image_height) / 2
    _y = ((coord_y + 1) * image_width) / 2

    weighed_value_scattered, weight_scattered = distribute(
        depth=depth,
        _x=_x,
        _y=_y,
        size_x=size_x,
        size_y=size_y,
        image_height=image_height,
        image_width=image_width)

    depth_recovered = (weighed_value_scattered / weight_scattered).view([
        batch, image_height, image_width
    ])

    return depth_recovered


class PCViews:
    """For creating images from PC based on the view information. Faster as the
    repeated operations are done only once whie initialization.
    """

    def __init__(self):
        _views = np.asarray([
            [[0 * np.pi / 2, 0, np.pi / 2], [0, 0, TRANS]],
            [[1 * np.pi / 2, 0, np.pi / 2], [0, 0, TRANS]],
            [[2 * np.pi / 2, 0, np.pi / 2], [0, 0, TRANS]],
            [[3 * np.pi / 2, 0, np.pi / 2], [0, 0, TRANS]],
            [[0, -np.pi / 2, np.pi / 2], [0, 0, TRANS]],
            [[0, np.pi / 2, np.pi / 2], [0, 0, TRANS]]]
        )

        self.num_views = 6

        angle = torch.tensor(_views[:, 0, :]).float().cuda(1)
        self.rot_mat = euler2mat(angle).transpose(1, 2)
        self.translation = torch.tensor(_views[:, 1, :]).float().cuda(1)
        self.translation = self.translation.unsqueeze(1)

    def get_img(self, points):
        """Get image based on the prespecified specifications.

        Args:
            points (torch.tensor): of size [B, _, 3]
        Returns:
            img (torch.tensor): of size [B * self.num_views, RESOLUTION,
                RESOLUTION]
        """
        b, _, _ = points.shape
        v = self.translation.shape[0]

        _points = self.point_transform(
            points=torch.repeat_interleave(points, v, dim=0),
            rot_mat=self.rot_mat.repeat(b, 1, 1),
            translation=self.translation.repeat(b, 1, 1))

        img = points2depth(
            points=_points,
            image_height=RESOLUTION,
            image_width=RESOLUTION,
            size_x=1,
            size_y=1,
        )
        return img

    @staticmethod
    def point_transform(points, rot_mat, translation):
        """
        :param points: [batch, num_points, 3]
        :param rot_mat: [batch, 3]
        :param translation: [batch, 1, 3]
        :return:
        """
        rot_mat = rot_mat.to(points.device)
        translation = translation.to(points.device)
        points = torch.matmul(points, rot_mat)
        points = points - translation
        return points



