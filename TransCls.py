import numpy as np
import torch
import utils.pc_utils as pc_utils


def transcls_input(X, device='cuda:0'):
    """
    Translate point cloud.
    Input:
        args - commmand line arguments
        X - Point cloud [B, N, C]
        device - cuda/cpu
    Return:
        Trans_X - Translate point cloud
        pos_label - {0,1,2,3} indicating the rotate-angle 0.05, 0.10, 0.15, 0.20 respectively
    """
    # print("X_shape:",X.size())
    # print("X_type",type(X))

    batch_size, _, num_points = X.size()
    Trans_X = X.clone().cpu().numpy()  # [B, C, N]
    trans_label_x = torch.zeros(batch_size).to(device)
    trans_label_y = torch.zeros(batch_size).to(device)
    trans_label_z = torch.zeros(batch_size).to(device)
    Trans_X = np.transpose(Trans_X,(0,2,1))   # [B, N, C]

    count = np.zeros(4)

    for b in range(batch_size):

        coord_min = np.min(Trans_X[b ,:, :3], axis=0)
        coord_max = np.max(Trans_X[b, :, :3], axis=0)

        coord_diff = coord_max - coord_min
        # print("1",coord_diff)

        coord_diff_x = coord_diff[0]
        coord_diff_y = coord_diff[1]
        coord_diff_z = coord_diff[2]

        # print("x",coord_diff_x)
        # print("y",coord_diff_y)
        # print("z",coord_diff_z)


        translation_x = 0.1 * coord_diff_x

        if translation_x < 0.1:
            translation_x = 0.05
            # count[0] += 1
        elif translation_x < 0.13:
            translation_x = 0.1
            # count[1] += 1
        elif translation_x < 0.16:
            translation_x = 0.15
            # count[2] += 1
        else:
            translation_x = 0.2
            # count[3] += 1
        Trans_X[b ,:, 0] += translation_x
        trans_label_x[b] = translation_x * 20 -1


        translation_y = 0.1 * coord_diff_y

        if translation_y < 0.1:
            translation_y = 0.05
            # count[0] += 1
        elif translation_y < 0.13:
            translation_y = 0.1
            # count[1] += 1
        elif translation_y < 0.16:
            translation_y = 0.15
            # count[2] += 1
        else:
            translation_y = 0.2
            # count[3] += 1
        Trans_X[b ,:, 1] += translation_y

        trans_label_y[b] = translation_y * 20 -1


        # translation_z = 0.1 * coord_diff_z
        #
        # if translation_z < 0.07:
        #     translation_z = 0.05
        #     # count[0] += 1
        # elif translation_z < 0.11:
        #     translation_z = 0.1
        #     # count[1] += 1
        # elif translation_z < 0.14:
        #     translation_z = 0.15
        #     # count[2] += 1
        # else:
        #     translation_z = 0.2
        #     # count[3] += 1
        # Trans_X[b ,:, 2] += translation_z
        #
        # trans_label_z[b] = translation_z * 20 -1

    # print(count[0],count[1],count[2],count[3])

    trans_label_x = trans_label_x.long().to(device)
    trans_label_y = trans_label_y.long().to(device)
    # trans_label_z = trans_label_z.long().to(device)



    Trans_X = torch.from_numpy(Trans_X).to(device)
    Trans_X = Trans_X.transpose(1, 2)  # [B, C, N]
    points_perm = torch.randperm(num_points).to(device)  # draw random permutation of points in the shape
    Trans_X = Trans_X[:, :, points_perm]

    return Trans_X, (trans_label_x , trans_label_y)
    # return Trans_X, (trans_label_x, trans_label_y , trans_label_z)
    # return Trans_X, trans_label_y



def calc_loss(args, logits, pos_vals, criterion):
    """
    Calc. TransCls loss.
    Return: loss
    """

    trans_label_x, trans_label_y = pos_vals
    # trans_label_y, trans_label_z = pos_vals

    # trans_label_y = pos_vals


    loss = criterion(logits['trans_cls1'], trans_label_x) + criterion(logits['trans_cls2'], trans_label_y)

    # loss = criterion(logits['trans_cls1'], trans_label_x) + criterion(logits['trans_cls2'], trans_label_y) + criterion(logits['trans_cls3'], trans_label_z)

    # loss = criterion(logits['trans_cls'], trans_label_y)

    loss *= args.TransCls_weight
    return loss