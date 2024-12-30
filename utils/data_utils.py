import numpy as np
import torch
import time
import random


def angle_axis(angle, axis):
    # type: (float, np.ndarray) -> float
    r"""Returns a 4x4 rotation matrix that performs a rotation around axis by angle
    Parameters
    ----------
    angle : float
        Angle to rotate by
    axis: np.ndarray
        Axis to rotate about
    Returns
    -------
    torch.Tensor
        3x3 rotation matrix
    """
    u = axis / np.linalg.norm(axis)
    cosval, sinval = np.cos(angle), np.sin(angle)

    # yapf: disable
    cross_prod_mat = np.array([[0.0, -u[2], u[1]],
                                [u[2], 0.0, -u[0]],
                                [-u[1], u[0], 0.0]])

    R = torch.from_numpy(
        cosval * np.eye(3)
        + sinval * cross_prod_mat
        + (1.0 - cosval) * np.outer(u, u)
    )
    # yapf: enable
    return R.float()

def fps(points, num):
    cids = []
    cid = np.random.choice(points.shape[0])
    cids.append(cid)
    id_flag = np.zeros(points.shape[0])
    id_flag[cid] = 1

    dist = torch.zeros(points.shape[0]) + 1e4
    dist = dist.type_as(points)
    while np.sum(id_flag) < num:
        dist_c = torch.norm(points - points[cids[-1]], p=2, dim=1)
        dist = torch.where(dist<dist_c, dist, dist_c)
        dist[id_flag == 1] = 1e4
        new_cid = torch.argmin(dist)
        id_flag[new_cid] = 1
        cids.append(new_cid)
    cids = torch.Tensor(cids)
    return cids

class PointcloudScale(object):
    def __init__(self, lo=0.7, hi=1.45, p=1):
        self.lo, self.hi = lo, hi
        self.p = p

    def __call__(self, points):
        if np.random.uniform(0, 1) > self.p:
            return points
        scaler = np.random.uniform(self.lo, self.hi)
        points[:, 0:3] *= scaler
        return points

class PointcloudFlip(object):
    def __init__(self,  p=1):
        self.p = p
    def __call__(self, points):
        if np.random.uniform(0, 1) > self.p:
            return points
        # 旋转翻转的轴，这里旋转z轴，即沿着xy平面进行翻转
        axis = 'z'
        axis_index = {'x': 0, 'y': 1, 'z': 2}[axis]
        # 将点云坐标中z轴的值取反，完成翻转
        coords_to_flip = points[:, axis_index]
        coords_flipped = -coords_to_flip
        # 将翻转过的值覆盖原有值
        points[:, axis_index] = coords_flipped
        return points

class PointcloudRotate(object):
    def __init__(self, axis=np.array([0.0, 0.0, 1.0]), p=1):
        self.axis = axis
        self.p = p

    def __call__(self, points):
        if np.random.uniform(0, 1) > self.p:
            return points
        if self.axis is None:
            angles = np.random.uniform(size=3) * 2 * np.pi
            Rx = angle_axis(angles[0], np.array([1.0, 0.0, 0.0]))
            Ry = angle_axis(angles[1], np.array([0.0, 1.0, 0.0]))
            Rz = angle_axis(angles[2], np.array([0.0, 0.0, 1.0]))

            rotation_matrix = torch.matmul(torch.matmul(Rz, Ry), Rx)
        else:
            rotation_angle = np.random.uniform() * 2 * np.pi
            rotation_matrix = angle_axis(rotation_angle, self.axis)

        normals = points.size(1) > 3
        if not normals:
            return torch.matmul(points, rotation_matrix.t())
        else:
            pc_xyz = points[:, 0:3]
            pc_normals = points[:, 3:]
            points[:, 0:3] = torch.matmul(pc_xyz, rotation_matrix.t())
            points[:, 3:] = torch.matmul(pc_normals, rotation_matrix.t())

            return points

class PointcloudRotatePerturbation(object):
    def __init__(self, angle_sigma=0.07, angle_clip=0.21, p=1):
        self.angle_sigma, self.angle_clip = angle_sigma, angle_clip
        self.p = p

    def _get_angles(self):
        angles = np.clip(
            self.angle_sigma * np.random.randn(3), -self.angle_clip, self.angle_clip
        )

        return angles

    def __call__(self, points):
        if np.random.uniform(0, 1) > self.p:
            return points
        angles = self._get_angles()
        Rx = angle_axis(angles[0], np.array([1.0, 0.0, 0.0]))
        Ry = angle_axis(angles[1], np.array([0.0, 1.0, 0.0]))
        Rz = angle_axis(angles[2], np.array([0.0, 0.0, 1.0]))

        rotation_matrix = torch.matmul(torch.matmul(Rz, Ry), Rx)

        normals = points.size(1) > 3
        if not normals:
            return torch.matmul(points, rotation_matrix.t())
        else:
            pc_xyz = points[:, 0:3]
            pc_normals = points[:, 3:]
            points[:, 0:3] = torch.matmul(pc_xyz, rotation_matrix.t())
            points[:, 3:] = torch.matmul(pc_normals, rotation_matrix.t())

            return points

class PointcloudJitter(object):
    def __init__(self, std=0.01, clip=0.05, p=1):
        self.std, self.clip = std, clip
        self.p = p

    def __call__(self, points):
        if np.random.uniform(0, 1) > self.p:
            return points
        jittered_data = (
            points.new(points.size(0), 3)
            .normal_(mean=0.0, std=self.std)
            .clamp_(-self.clip, self.clip)
        )
        points[:, 0:3] += jittered_data
        return points



class PointcloudSaltPepperNoise(object):
    def __init__(self, salt_pepper_ratio=0.2, salt_value=1, pepper_value=-1):
        self.salt_pepper_ratio = salt_pepper_ratio  # 噪声点占总点数的比例
        self.salt_value = salt_value  # '盐'噪声的值，可以设定一个极大值来模拟异常高的测量值
        self.pepper_value = pepper_value  # '胡椒'噪声的值，可以设定一个极小值来模拟异常低的测量值

    def __call__(self, points):
        num_points = points.shape[0]  # 点云中点的总数
        num_noise_points = int(num_points * self.salt_pepper_ratio)  # 计算要添加噪声的点数

        # 随机选择要添加噪声的点的索引
        noise_indices = np.random.choice(num_points, num_noise_points, replace=False)

        # 对选中的点添加 '盐' 噪声（较高的值）
        if self.salt_value is not None:
            points[noise_indices[:num_noise_points // 2], 0:3] = self.salt_value

        # 对选中的点添加 '胡椒' 噪声（较低的值）
        if self.pepper_value is not None:
            points[noise_indices[num_noise_points // 2:], 0:3] = self.pepper_value

        return points



class PointcloudDegradationNoise(object):
    def __init__(self, base_std=0.01, distance_factor=0.005, clip=0.05, p=1):
        self.base_std = base_std  # 基础标准差
        self.distance_factor = distance_factor  # 距离因子
        self.clip = clip  # 噪声裁剪值
        self.p = p  # 应用噪声的概率
        # 生成单位球上的随机点
        # self.reference_point = self.random_point_on_unit_sphere()

        # self.reference_point = np.array([0.866, 0, 0.5])

        # 定义可能的参考点列表
        # points = [
        #     np.array([0.866, 0, 0.5]),
        #     np.array([-0.866, 0, 0.5]),
        #     np.array([0.866, 0, -0.5]),
        #     np.array([-0.866, 0, -0.5])
        # ]

        points = [
            np.array([1 / np.sqrt(3), 1 / np.sqrt(3), 1 / np.sqrt(3)]),
            np.array([-1 / np.sqrt(3), 1 / np.sqrt(3), 1 / np.sqrt(3)]),
            np.array([1 / np.sqrt(3), -1 / np.sqrt(3), 1 / np.sqrt(3)]),
            np.array([1 / np.sqrt(3), 1 / np.sqrt(3), -1 / np.sqrt(3)]),
            np.array([-1 / np.sqrt(3), -1 / np.sqrt(3), 1 / np.sqrt(3)]),
            np.array([-1 / np.sqrt(3), 1 / np.sqrt(3), -1 / np.sqrt(3)]),
            np.array([1 / np.sqrt(3), -1 / np.sqrt(3), -1 / np.sqrt(3)]),
            np.array([-1 / np.sqrt(3), -1 / np.sqrt(3), -1 / np.sqrt(3)])
        ]

        # 随机选择一个参考点
        self.reference_point = random.choice(points)


    def random_point_on_unit_sphere(self):
        theta = np.random.uniform(0, 2*np.pi)
        #整个单位球
        # cos_phi = np.random.uniform(-1, 1)
        #约束
        cos_phi = np.random.uniform(-0.94, 0.94)
        phi = np.arccos(cos_phi)

        x = np.sin(phi) * np.cos(theta)
        y = np.sin(phi) * np.sin(theta)
        z = cos_phi
        return np.array([x, y, z])

    def __call__(self, points):
        if np.random.uniform(0, 1) > self.p:
            return points

        # 计算每个点到单位球上的参考点的欧氏距离
        distances = np.linalg.norm(points[:, 0:3] - self.reference_point, axis=1)


        # 计算每个点的噪声标准差，随距离增加而线性增加
        stds = self.base_std + self.distance_factor * distances

        # 为每个点生成噪声
        noise = np.random.normal(0, stds[:, np.newaxis])

        # 如果设置了噪声裁剪，则应用裁剪
        if self.clip is not None:
            noise = np.clip(noise, -self.clip, self.clip)

        # 将噪声添加到点云
        points[:, 0:3] += noise
        return points



class PointcloudTranslate(object):
    def __init__(self, translate_range=0.1, p=1):
        self.translate_range = translate_range
        self.p = p

    def __call__(self, points):
        if np.random.uniform(0, 1) > self.p:
            return points

        points = points.numpy()
        coord_min = np.min(points[:,:3], axis=0)
        coord_max = np.max(points[:,:3], axis=0)
        coord_diff = coord_max - coord_min

        translation = np.random.uniform(-self.translate_range, self.translate_range, size=(3)) * coord_diff
        points[:, 0:3] += translation

        # coord_min = np.min(points[:, 1], axis=0)
        # coord_max = np.max(points[:, 1], axis=0)
        # coord_diff = coord_max - coord_min
        #
        #
        # translation = 0.1 * coord_diff
        #
        # if translation < 0.075:
        #     translation = 0.05
        #
        # elif translation < 0.125:
        #     translation = 0.1
        #
        # elif translation < 0.175:
        #     translation = 0.15
        #
        # else:
        #     translation = 0.2
        #
        #
        # points[:, 1] += translation


        return torch.from_numpy(points).float()

class PointcloudToTensor(object):
    def __call__(self, points):
        return torch.from_numpy(points).float()

class PointcloudRandomInputDropout(object):
    def __init__(self, max_dropout_ratio=0.875, p=1):
        assert max_dropout_ratio >= 0 and max_dropout_ratio < 1
        self.max_dropout_ratio = max_dropout_ratio
        self.p = p

    def __call__(self, points):
        if np.random.uniform(0, 1) > self.p:
            return points
        pc = points.numpy()

        dropout_ratio = np.random.random() * self.max_dropout_ratio  # 0~0.875
        drop_idx = np.where(np.random.random((pc.shape[0])) <= dropout_ratio)[0]
        if len(drop_idx) > 0:
            pc[drop_idx] = pc[0]  # set to the first point

        return torch.from_numpy(pc).float()

class PointcloudSample(object):
    def __init__(self, num_pt=4096):
        self.num_points = num_pt

    def __call__(self, points):
        pc = points.numpy()
        # pt_idxs = np.arange(0, self.num_points)
        pt_idxs = np.arange(0, points.shape[0])
        np.random.shuffle(pt_idxs)
        pc = pc[pt_idxs[0:self.num_points], :]
        return torch.from_numpy(pc).float()

class PointcloudNormalize(object):
    def __init__(self, radius=1):
        self.radius = radius

    def pc_normalize(self, pc):
        l = pc.shape[0]
        centroid = np.mean(pc, axis=0)
        pc = pc - centroid
        m = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))
        pc = pc / m
        return pc

    def __call__(self, points):
        pc = points.numpy()
        pc[:, 0:3] = self.pc_normalize(pc[:, 0:3])
        return torch.from_numpy(pc).float()

class PointcloudRemoveInvalid(object):
    def __init__(self, invalid_value=0):
        self.invalid_value = invalid_value

    def __call__(self, points):
        pc = points.numpy()
        valid = np.sum(pc, axis=1) != self.invalid_value
        pc = pc[valid, :]
        return torch.from_numpy(pc).float()


# class PointcloudSelfOcclusion(object):
#     def __init__(self, normal_vector =np.array([1.0, 1.0, 1.0]), plane_point =np.array([1.0, 1.0, 1.0]),grid_size = 0.02,p=1):
#         self.normal_vector  = normal_vector
#         self.plane_point  = plane_point
#         self.p = p
#         self.grid_size = grid_size
#
#
#     def __call__(self, points):
#         if np.random.uniform(0, 1) > self.p:
#             return points
#
#         points = points.numpy()
#
#         normal_vector = np.array(self.normal_vector) / np.linalg.norm(self.normal_vector)  # 单位化法向量
#
#         point_to_plane_point = np.array(points) - np.array(self.plane_point)
#
#         distance = np.dot(point_to_plane_point, normal_vector)[:, np.newaxis]
#
#         projected_point = np.array(points) - distance * normal_vector
#
#         # projected_points = np.array([projected_point for p in points])
#
#         grid_coordinates = np.floor(projected_point[:, :2] / self.grid_size).astype(int)
#
#         grid_dict = {}
#         for point, projected_point, grid_coord in zip(points, projected_point, grid_coordinates):
#             key = (grid_coord[0], grid_coord[1])
#             # print(key)
#             projected_distance = np.dot(projected_point, normal_vector)
#             if key not in grid_dict or projected_distance < grid_dict[key][1]:
#                 grid_dict[key] = (point, projected_distance)
#         processed_points = [v[0] for v in grid_dict.values()]
#
#         while len(processed_points) < 1024:
#             processed_points.append(points[0])
#
#         processed_points = np.array(processed_points)
#
#
#         return torch.from_numpy(processed_points).float()




class PointcloudDensity(object):
    def __init__(self,v_point=np.array([1, 0, 0]),gate=1.2, p=1):
        self.p = p
        self.v_point = v_point
        self.gate = gate


    def __call__(self, points):
        if np.random.uniform(0, 1) > self.p:
            return points

        # points = points.numpy()

        dist = np.sqrt((self.v_point ** 2).sum())
        max_dist = dist + 1
        min_dist = dist - 1
        dist = np.linalg.norm(points - self.v_point.reshape(1, 3), axis=1)
        dist = (dist - min_dist) / (max_dist - min_dist)
        r_list = np.random.uniform(0, 1, points.shape[0])
        tmp_pc = points[dist * self.gate < (r_list)]

        num_points_to_add = 1024 - tmp_pc.shape[0]

        additional_points = tmp_pc[0].repeat(num_points_to_add, 1)
        processed_points = torch.cat((tmp_pc, additional_points), dim=0)

        return processed_points






class PointcloudDensityWeak(object):
    def __init__(self,v_point=np.array([1, 0, 0]),gate=0.6, p=1):
        self.p = p
        self.v_point = v_point
        self.gate = gate


    def __call__(self, points):
        if np.random.uniform(0, 1) > self.p:
            return points

        # points = points.numpy()

        dist = np.sqrt((self.v_point ** 2).sum())
        max_dist = dist + 1
        min_dist = dist - 1
        dist = np.linalg.norm(points - self.v_point.reshape(1, 3), axis=1)
        dist = (dist - min_dist) / (max_dist - min_dist)
        r_list = np.random.uniform(0, 1, points.shape[0])
        tmp_pc = points[dist * self.gate < (r_list)]

        num_points_to_add = 1024 - tmp_pc.shape[0]

        additional_points = tmp_pc[0].repeat(num_points_to_add, 1)

        processed_points = torch.cat((tmp_pc, additional_points), dim=0)


        return processed_points


class PointcloudSelfOcclusion(object):
    def __init__(self,pixel_size=0.1, p=1):
        self.p = p
        self.pixel_size = pixel_size


    def __call__(self, points):
        if np.random.uniform(0, 1) > self.p:
            return points

        # points = points.numpy()

        pixel = int(2 / self.pixel_size)
        rotated_pc = rotate_point_cloud_3d(points)
        pc_compress = (rotated_pc[:, 2] + 1) / 2 * pixel * pixel + (rotated_pc[:, 1] + 1) / 2 * pixel
        points_list = [None for i in range((pixel + 5) * (pixel + 5))]
        pc_compress = pc_compress.astype(int)
        for index, point in enumerate(rotated_pc):
            compress_index = pc_compress[index]
            if compress_index > len(points_list):
                print('out of index:', compress_index, len(points_list), point, points[index], (points[index] ** 2).sum(),
                      (point ** 2).sum())
            if points_list[compress_index] is None:
                points_list[compress_index] = index
            elif point[0] > rotated_pc[points_list[compress_index]][0]:
                points_list[compress_index] = index
        points_list = list(filter(lambda x: x is not None, points_list))
        points = points[points_list]

        num_points_to_add = 1024 - points.shape[0]

        additional_points = points[0].repeat(num_points_to_add, 1)

        processed_points = torch.cat((points, additional_points), dim=0)


        return processed_points 


class PointcloudDrop(object):
    def __init__(self, p=1):
        self.p = p

    def __call__(self, points):
        if np.random.uniform(0, 1) > self.p:
            return points

        num_point = 1024
        # p为舍弃的点的占比
        p = np.random.uniform(low=0, high=0.45)
        # 从点云中随机选一个点，并计算其余点与其的距离，并按距离将索引进行升序
        random_point = np.random.randint(0, points.shape[0])
        index = np.linalg.norm(points - points[random_point].reshape(1, 3), axis=1).argsort()
        # 将距离锚点最近的点舍弃
        tmp_pc = points[index[int(points.shape[0] * p):]]
        # 补齐丢弃点的数量，便于后续处理
        num_pad = np.ceil(num_point / tmp_pc.shape[0]).astype(np.int32)
        processed_points = np.tile(tmp_pc, (num_pad, 1))[:num_point]

        return processed_points




class PointcloudDropWeak(object):
    def __init__(self, p=1):
        self.p = p

    def __call__(self, points):
        if np.random.uniform(0, 1) > self.p:
            return points

        num_point = 1024
        p = np.random.uniform(low=0, high=0.25)
        random_point = np.random.randint(0, points.shape[0])
        index = np.linalg.norm(points - points[random_point].reshape(1, 3), axis=1).argsort()

        tmp_pc = points[index[int(points.shape[0] * p):]]
        num_pad = np.ceil(num_point / tmp_pc.shape[0]).astype(np.int32)
        processed_points = np.tile(tmp_pc, (num_pad, 1))[:num_point]

        return processed_points



# def drop_hole(pc, p):
#     random_point = np.random.randint(0, pc.shape[0])
#     index = np.linalg.norm(pc - pc[random_point].reshape(1,3), axis=1).argsort()
#     return pc[index[int(pc.shape[0] * p):]]


def rotate_point_cloud_3d(pc):
    rotation_angle = np.random.rand(3) * 2 * np.pi
    cosval = np.cos(rotation_angle)
    sinval = np.sin(rotation_angle)
    rotation_matrix_1 = np.array([[cosval[0], 0, sinval[0]],
                                [0, 1, 0],
                                [-sinval[0], 0, cosval[0]]])
    rotation_matrix_2 = np.array([[1, 0, 0],
                                [0, cosval[1], -sinval[1]],
                                [0, sinval[1], cosval[1]]])
    rotation_matrix_3 = np.array([[cosval[2], -sinval[2], 0],
                                 [sinval[2], cosval[2], 0],
                                 [0, 0, 1]])
    rotation_matrix = np.matmul(np.matmul(rotation_matrix_1, rotation_matrix_2), rotation_matrix_3)
    rotated_data = np.dot(pc.reshape((-1, 3)), rotation_matrix)

    return rotated_data


class PointcloudRandomCrop(object):
    def __init__(self, x_min=0.6, x_max=1.1, ar_min=0.75, ar_max=1.33, p=1, min_num_points=512, max_try_num=10):
        self.x_min = x_min
        self.x_max = x_max

        self.ar_min = ar_min
        self.ar_max = ar_max

        self.p = p

        self.max_try_num = max_try_num
        self.min_num_points = min_num_points

    def __call__(self, points):
        if np.random.uniform(0, 1) > self.p:
            return points
        points = points.numpy()

        isvalid = False
        try_num = 0
        while not isvalid:
            coord_min = np.min(points[:,:3], axis=0)
            coord_max = np.max(points[:,:3], axis=0)
            coord_diff = coord_max - coord_min
            # resampling later, so only consider crop here
            new_coord_range = np.zeros(3)
            new_coord_range[0] = np.random.uniform(self.x_min, self.x_max)
            ar = np.random.uniform(self.ar_min, self.ar_max)
            # new_coord_range[1] = np.random.uniform(self.ar_min, self.ar_max) * new_coord_range[0]
            # new_coord_range[2] = np.random.uniform(self.ar_min, self.ar_max) * new_coord_range[0]
            new_coord_range[1] = new_coord_range[0] * ar
            new_coord_range[2] = new_coord_range[0] / ar
            # new_coord_range = np.where(new_coord_range>1, 1, new_coord_range)

            new_coord_min = np.random.uniform(0, 1-new_coord_range)
            new_coord_max = new_coord_min + new_coord_range

            new_coord_min = coord_min + coord_diff * new_coord_min
            new_coord_max = coord_min + coord_diff * new_coord_max

            new_indices = (points[:,:3] > new_coord_min) & (points[:, :3] < new_coord_max)
            new_indices_crop = np.sum(new_indices, axis=1) == 3
            drop_indices_crop = np.sum(new_indices, axis=1) != 3

            new_points_crop = points[new_indices_crop]
            # re_points_crop = points[drop_indices_crop]

            # print(new_points_crop.shape[0],re_points_crop.shape[0])


            # new_indices_re = np.sum(new_indices, axis=1) != 3
            # new_points_re = points[new_indices_re]

            # other_num = points.shape[0] - new_points.shape[0]
            # if new_points.shape[0] > 0:
            #     isvalid = True
            if new_points_crop.shape[0] >= self.min_num_points and new_points_crop.shape[0] < 828:

                points[drop_indices_crop] = points[0]  # set to the first point
                # return torch.from_numpy(points).float()

                isvalid = True

            try_num += 1
            if try_num > self.max_try_num:
                # print(new_points.shape[0])
                return torch.from_numpy(points).float()


        # other_indices = np.random.choice(np.arange(new_points.shape[0]), other_num)
        # other_points = new_points[other_indices]
        # new_points = np.concatenate([new_points, other_points], axis=0)

        # new_points[:,:3] = (new_points[:,:3] - new_coord_min) / (new_coord_max - new_coord_min) * coord_diff + coord_min
        return torch.from_numpy(points).float()

class PointcloudRandomCropWeak(object):
    def __init__(self, x_min=0.6, x_max=1.1, ar_min=0.75, ar_max=1.33, p=1, min_num_points=512, max_try_num=10):
        self.x_min = x_min
        self.x_max = x_max

        self.ar_min = ar_min
        self.ar_max = ar_max

        self.p = p

        self.max_try_num = max_try_num
        self.min_num_points = min_num_points

    def __call__(self, points):
        if np.random.uniform(0, 1) > self.p:
            return points
        points = points.numpy()

        isvalid = False
        try_num = 0
        while not isvalid:
            coord_min = np.min(points[:,:3], axis=0)
            coord_max = np.max(points[:,:3], axis=0)
            coord_diff = coord_max - coord_min
            # resampling later, so only consider crop here
            new_coord_range = np.zeros(3)
            new_coord_range[0] = np.random.uniform(self.x_min, self.x_max)
            ar = np.random.uniform(self.ar_min, self.ar_max)
            # new_coord_range[1] = np.random.uniform(self.ar_min, self.ar_max) * new_coord_range[0]
            # new_coord_range[2] = np.random.uniform(self.ar_min, self.ar_max) * new_coord_range[0]
            new_coord_range[1] = new_coord_range[0] * ar
            new_coord_range[2] = new_coord_range[0] / ar
            # new_coord_range = np.where(new_coord_range>1, 1, new_coord_range)

            new_coord_min = np.random.uniform(0, 1-new_coord_range)
            new_coord_max = new_coord_min + new_coord_range

            new_coord_min = coord_min + coord_diff * new_coord_min
            new_coord_max = coord_min + coord_diff * new_coord_max

            new_indices = (points[:,:3] > new_coord_min) & (points[:, :3] < new_coord_max)
            new_indices_crop = np.sum(new_indices, axis=1) == 3
            drop_indices_crop = np.sum(new_indices, axis=1) != 3

            new_points_crop = points[new_indices_crop]
            # re_points_crop = points[drop_indices_crop]

            # print(new_points_crop.shape[0],re_points_crop.shape[0])


            # new_indices_re = np.sum(new_indices, axis=1) != 3
            # new_points_re = points[new_indices_re]

            # other_num = points.shape[0] - new_points.shape[0]
            # if new_points.shape[0] > 0:
            #     isvalid = True
            if new_points_crop.shape[0] >= 716 and new_points_crop.shape[0] < 928:

                points[drop_indices_crop] = points[0]  # set to the first point
                # return torch.from_numpy(points).float()

                isvalid = True

            try_num += 1
            if try_num > self.max_try_num:
                # print(new_points.shape[0])
                return torch.from_numpy(points).float()


        # other_indices = np.random.choice(np.arange(new_points.shape[0]), other_num)
        # other_points = new_points[other_indices]
        # new_points = np.concatenate([new_points, other_points], axis=0)

        # new_points[:,:3] = (new_points[:,:3] - new_coord_min) / (new_coord_max - new_coord_min) * coord_diff + coord_min
        return torch.from_numpy(points).float()

class PointcloudRandomCropEdge(object):
    def __init__(self, x_min=0.6, x_max=1.1, ar_min=0.75, ar_max=1.33, p=1, min_num_points=256, max_try_num=10):
        self.x_min = x_min
        self.x_max = x_max

        self.ar_min = ar_min
        self.ar_max = ar_max

        self.p = p

        self.max_try_num = max_try_num
        self.min_num_points = min_num_points

    def __call__(self, points):
        if np.random.uniform(0, 1) > self.p:
            return points
        points = points.numpy()

        isvalid = False
        try_num = 0
        while not isvalid:
            coord_min = np.min(points[:,:3], axis=0)
            coord_max = np.max(points[:,:3], axis=0)
            coord_diff = coord_max - coord_min
            # resampling later, so only consider crop here
            new_coord_range = np.zeros(3)
            new_coord_range[0] = np.random.uniform(self.x_min, self.x_max)
            ar = np.random.uniform(self.ar_min, self.ar_max)
            # new_coord_range[1] = np.random.uniform(self.ar_min, self.ar_max) * new_coord_range[0]
            # new_coord_range[2] = np.random.uniform(self.ar_min, self.ar_max) * new_coord_range[0]
            new_coord_range[1] = new_coord_range[0] * ar
            new_coord_range[2] = new_coord_range[0] / ar
            # new_coord_range = np.where(new_coord_range>1, 1, new_coord_range)

            new_coord_min = np.random.uniform(0, 1-new_coord_range)
            new_coord_max = new_coord_min + new_coord_range

            new_coord_min = coord_min + coord_diff * new_coord_min
            new_coord_max = coord_min + coord_diff * new_coord_max

            new_indices = (points[:,:3] > new_coord_min) & (points[:, :3] < new_coord_max)
            new_indices = np.sum(new_indices, axis=1) != 3
            new_points = points[new_indices]

            # other_num = points.shape[0] - new_points.shape[0]
            # if new_points.shape[0] > 0:
            #     isvalid = True
            if new_points.shape[0] >= self.min_num_points and new_points.shape[0] < points.shape[0]:
                isvalid = True

            try_num += 1
            if try_num > self.max_try_num:
                # print(new_points.shape[0])
                return torch.from_numpy(points).float()


        # other_indices = np.random.choice(np.arange(new_points.shape[0]), other_num)
        # other_points = new_points[other_indices]
        # new_points = np.concatenate([new_points, other_points], axis=0)

        # new_points[:,:3] = (new_points[:,:3] - new_coord_min) / (new_coord_max - new_coord_min) * coord_diff + coord_min
        return torch.from_numpy(new_points).float()

class PointcloudRandomCutout(object):
    def __init__(self, ratio_min=0.3, ratio_max=0.6, p=1, min_num_points=4096, max_try_num=10):
        self.ratio_min = ratio_min
        self.ratio_max = ratio_max
        self.p = p
        self.min_num_points = min_num_points
        self.max_try_num = max_try_num

    def __call__(self, points):
        if np.random.uniform(0, 1) > self.p:
            return points
        points = points.numpy()
        try_num = 0
        valid = False
        while not valid:
            coord_min = np.min(points[:,:3], axis=0)
            coord_max = np.max(points[:,:3], axis=0)
            coord_diff = coord_max - coord_min

            cut_ratio = np.random.uniform(self.ratio_min, self.ratio_max, 3)
            new_coord_min = np.random.uniform(0, 1-cut_ratio)
            new_coord_max= new_coord_min + cut_ratio

            new_coord_min = coord_min + new_coord_min * coord_diff
            new_coord_max = coord_min + new_coord_max * coord_diff

            cut_indices = (points[:, :3] > new_coord_min) & (points[:, :3] < new_coord_max)
            cut_indices = np.sum(cut_indices, axis=1) == 3

            # print(np.sum(cut_indices))
            # other_indices = (points[:, :3] < new_coord_min) | (points[:, :3] > new_coord_max)
            # other_indices = np.sum(other_indices, axis=1) == 3
            try_num += 1

            if try_num > self.max_try_num:
                return torch.from_numpy(points).float()

            # cut the points, sampling later

            if points.shape[0] - np.sum(cut_indices) >= self.min_num_points and np.sum(cut_indices) > 0:
                # print (np.sum(cut_indices))
                points = points[cut_indices==False]
                valid = True

        # if np.sum(other_indices) > 0:
        #     comp_indices = np.random.choice(np.arange(np.sum(other_indices)), np.sum(cut_indices))
        #     points[cut_indices] = points[comp_indices]
        return torch.from_numpy(points).float()

class PointcloudUpSampling(object):
    def __init__(self, max_num_points = 1024, radius=0.1, nsample=5, centroid="fps"):
        self.max_num_points = max_num_points
        # self.radius = radius
        self.centroid = centroid
        self.nsample = nsample

    def __call__(self, points):
        t0 = time.time()

        p_num = points.shape[0]
        if p_num > self.max_num_points:
            return points

        c_num = self.max_num_points - p_num

        if self.centroid == "random":
            cids = np.random.choice(np.arange(p_num), c_num)
        else:
            assert self.centroid == "fps"
            fps_num = c_num / self.nsample
            fps_ids = fps(points, fps_num)
            cids = np.random.choice(fps_ids, c_num)

        xyzs = points[:, :3]
        loc_matmul = torch.matmul(xyzs, xyzs.t())
        loc_norm = xyzs * xyzs
        r = torch.sum(loc_norm, -1, keepdim=True)

        r_t = r.t()  # 转置
        dist = r - 2 * loc_matmul + r_t
        # adj_matrix = torch.sqrt(dist + 1e-6)

        dist = dist[cids]
        # adj_sort = torch.argsort(adj_matrix, 1)
        adj_topk = torch.topk(dist, k=self.nsample, dim=1, largest=False)[1]

        # uniform = np.random.uniform(0, 1, (cids.shape[0], self.nsample*2))
        #
        # median = np.median(uniform, axis=1, keepdims=True)


        # choice = adj_topk[uniform > median]  # (c_num, n_samples)
        # choice = adj_sort[:, 0:self.nsample*2][uniform > median]  # (c_num, n_samples)

        # print("*",choice.shape)


        # choice = choice.reshape(-1, self.nsample)

        sample_points = points[adj_topk]  # (c_num, n_samples, 3)

        new_points = torch.mean(sample_points, dim=1)
        new_points = torch.cat([points, new_points], 0)

        return new_points

def points_sampler(points, num):
    pt_idxs = np.arange(0, points.shape[0])
    np.random.shuffle(pt_idxs)
    points = points[pt_idxs[0:num], :]
    return points

class PointcloudScaleAndTranslate(object):
    def __init__(self, scale_low=2. / 3., scale_high=3. / 2., translate_range=0.2):
        self.scale_low = scale_low
        self.scale_high = scale_high
        self.translate_range = translate_range

    def __call__(self, pc, device):
        bsize = pc.size()[0]
        dim = pc.size()[-1]

        for i in range(bsize):
            xyz1 = np.random.uniform(low=self.scale_low, high=self.scale_high, size=[dim])
            xyz2 = np.random.uniform(low=-self.translate_range, high=self.translate_range, size=[dim])
            
            pc[i, :, 0:3] = torch.mul(pc[i, :, 0:3], torch.from_numpy(xyz1).float().to(device)) + torch.from_numpy(xyz2).float().to(device)
            
        return pc