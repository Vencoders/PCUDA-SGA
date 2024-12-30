import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
# from knn_cuda import KNN

import utils.pc_utils as pc_utils
import random
# from timm.models.layers import DropPath, trunc_normal_
# from extensions.chamfer_dist import ChamferDistanceL1, ChamferDistanceL2, ChamferDistanceL2_withnormal

from utils.trans_norm import TransNorm2d
from torch.autograd import Function

K = 20


def knn(x, k):
    inner = -2 * torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x ** 2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)

    # (batch_size, num_points, k)
    idx = pairwise_distance.topk(k=k, dim=-1)[1]
    return idx


def get_graph_feature(x, args, k=20, idx=None):
    batch_size = x.size(0)
    num_points = x.size(2)

    # print("xx:",x.shape)

    x = x.view(batch_size, -1, num_points)

    # print("xxx:",x.shape)


    if idx is None:
        idx = knn(x, k=k)  # (batch_size, num_points, k)
    # Run on cpu or gpu
    device = torch.device("cuda:" + str(x.get_device()) if args.cuda else "cpu")

    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points

    idx = idx + idx_base

    idx = idx.view(-1)

    _, num_dims, _ = x.size()

    x = x.transpose(2, 1).contiguous()  # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims)
    feature = x.view(batch_size * num_points, -1)[idx, :]  # matrix [k*num_points*batch_size,3]
    feature = feature.view(batch_size, num_points, k, num_dims)
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)

    feature = torch.cat((feature - x, x), dim=3).permute(0, 3, 1, 2)

    return feature


def l2_norm(input, axit=1):
    norm = torch.norm(input, 2, axit, True)
    output = torch.div(input, norm)
    return output


class conv_2d(nn.Module):
    def __init__(self, in_ch, out_ch, kernel, activation='relu', bias=True):
        super(conv_2d, self).__init__()
        if activation == 'relu':
            self.conv = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=kernel, bias=bias),
                # nn.BatchNorm2d(out_ch),
                nn.InstanceNorm2d(out_ch), #之前一直用的这个
                # TransNorm2d(out_ch),
                # nn.LayerNorm([out_ch, 1024, 20]),
                nn.ReLU(inplace=True)
            )
        elif activation == 'leakyrelu':
            self.conv = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=kernel, bias=bias),
                # nn.BatchNorm2d(out_ch),
                nn.InstanceNorm2d(out_ch),#之前一直用的这个
                # TransNorm2d(out_ch),
                # nn.LayerNorm([out_ch, 1024, 20]),
                nn.LeakyReLU(negative_slope=0.2, inplace=True)
            )

    def forward(self, x):
        x = self.conv(x)
        return x



class fc_layer(nn.Module):
    def __init__(self, in_ch, out_ch, bn=False, activation='relu', bias=True):
        super(fc_layer, self).__init__()
        if activation == 'relu':
            self.ac = nn.ReLU(inplace=True)
        elif activation == 'leakyrelu':
            self.ac = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        if bn:
            self.fc = nn.Sequential(
                nn.Linear(in_ch, out_ch, bias=bias),
                # nn.BatchNorm1d(out_ch),
                nn.LayerNorm(out_ch),
                self.ac
            )
        else:
            self.fc = nn.Sequential(
                nn.Linear(in_ch, out_ch, bias=bias),
                self.ac
            )

    def forward(self, x):
        x = l2_norm(x, 1)
        x = self.fc(x)
        return x


class transform_net(nn.Module):
    """ Input (XYZ) Transform Net, input is BxNx3 gray image
        Return: Transformation matrix of size 3xK """

    def __init__(self, args, in_ch, out=3):
        super(transform_net, self).__init__()
        self.K = out
        self.args = args

        activation = 'leakyrelu' if args.model == 'dgcnn' else 'relu'
        bias = False if args.model == 'dgcnn' else True

        self.conv2d1 = conv_2d(in_ch, 64, kernel=1, activation=activation, bias=bias)
        self.conv2d2 = conv_2d(64, 128, kernel=1, activation=activation, bias=bias)
        self.conv2d3 = conv_2d(128, 1024, kernel=1, activation=activation, bias=bias)
        self.fc1 = fc_layer(1024, 512, activation=activation, bias=bias, bn=True)
        self.fc2 = fc_layer(512, 256, activation=activation, bn=True)
        self.fc3 = nn.Linear(256, out * out)

    def forward(self, x):
        device = torch.device("cuda:" + str(x.get_device()) if self.args.cuda else "cpu")

        x = self.conv2d1(x)
        x = self.conv2d2(x)
        if self.args.model == "dgcnn":
            x = x.max(dim=-1, keepdim=False)[0]
            x = torch.unsqueeze(x, dim=3)
        x = self.conv2d3(x)
        x, _ = torch.max(x, dim=2, keepdim=False)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)

        iden = torch.eye(self.K).view(1, self.K * self.K).repeat(x.size(0), 1)
        iden = iden.to(device)
        x = x + iden
        x = x.view(x.size(0), self.K, self.K)
        return x


class PointNet(nn.Module):
    def __init__(self, args, num_class=10):
        super(PointNet, self).__init__()
        self.args = args

        self.trans_net1 = transform_net(args, 3, 3)
        self.trans_net2 = transform_net(args, 64, 64)
        self.conv1 = conv_2d(3, 64, 1)
        self.conv2 = conv_2d(64, 64, 1)
        self.conv3 = conv_2d(64, 64, 1)
        self.conv4 = conv_2d(64, 128, 1)
        self.conv5 = conv_2d(128, 1024, 1)

        num_f_prev = 64 + 64 + 64 + 128

        self.cls_C = class_classifier(args, 1024, 10)
        self.domain_C = domain_classifier(args, 1024, 2)
        self.rotcls_C1 = linear_classifier(1024, 4)
        self.rotcls_C2 = linear_classifier(1024, 4)
        self.defcls_C = ssl_classifier(args, 1024, 27)
        self.DecoderFC = DecoderFC(args, 1024)
        self.DefRec = RegionReconstruction(args, num_f_prev + 1024)
        self.normreg_C = nn.Conv1d(1024, 4, kernel_size=1, bias=False)

    def forward(self, x, alpha=0, activate_DefRec=False):
        num_points = x.size(2)
        x = torch.unsqueeze(x, dim=3)

        cls_logits = {}

        transform = self.trans_net1(x)
        x = x.transpose(2, 1)
        x = x.squeeze(dim=3)
        x = torch.bmm(x, transform)
        x = x.unsqueeze(3)
        x = x.transpose(2, 1)
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        transform = self.trans_net2(x2)
        x = x2.transpose(2, 1)
        x = x.squeeze(dim=3)
        x = torch.bmm(x, transform)
        x = x.unsqueeze(3)
        x = x.transpose(2, 1)
        x3 = self.conv3(x)
        x4 = self.conv4(x3)
        x_cat = torch.cat((x1, x2, x3, x4), dim=1)

        x5 = self.conv5(x4)
        x5_pool, _ = torch.max(x5, dim=2, keepdim=False)
        x = x5_pool.squeeze(dim=2)  # batchsize*1024

        cls_logits["cls"] = self.cls_C(x)
        if alpha is not 0:
            reverse_x = ReverseLayerF.apply(x, alpha)
            cls_logits["domain_cls"] = self.domain_C(reverse_x)
        cls_logits["rot_cls1"] = self.rotcls_C1(x)
        cls_logits["rot_cls2"] = self.rotcls_C2(x)
        cls_logits["def_cls"] = self.defcls_C(x)
        # cls_logits["curv_conf"] = self.curvconfreg_C(x)
        # cls_logits["norm_reg"] = self.normreg_C(x5).permute(0, 2, 1)
        cls_logits["decoder"] = self.DecoderFC(x)

        if activate_DefRec:
            DefRec_input = torch.cat((x_cat.squeeze(dim=3), x5_pool.repeat(1, 1, num_points)), dim=1)
            cls_logits["DefRec"] = self.DefRec(DefRec_input)

        return cls_logits


class DGCNN(nn.Module):
    def __init__(self, args):
        super(DGCNN, self).__init__()
        self.args = args
        self.k = K

        self.input_transform_net = transform_net(args, 6, 3)

        self.conv1 = conv_2d(6, 64, kernel=1, bias=False, activation='leakyrelu')
        self.conv2 = conv_2d(64 * 2, 64, kernel=1, bias=False, activation='leakyrelu')
        self.conv3 = conv_2d(64 * 2, 128, kernel=1, bias=False, activation='leakyrelu')
        self.conv4 = conv_2d(128 * 2, 256, kernel=1, bias=False, activation='leakyrelu')



        num_f_prev_5 = 64 + 64 + 128 + 256
        self.bn5 = nn.BatchNorm1d(512)
        self.conv5 = nn.Conv1d(num_f_prev_5, 512, kernel_size=1, bias=False)

        self.cls_C = class_classifier(args, 1024, 10)
        self.domain_C = domain_classifier(args, 1024, 2)
        self.rotcls_C1 = linear_classifier(1024, 4)
        self.rotcls_C2 = linear_classifier(1024, 4)
        # self.transcls_C = linear_classifier(1024, 4)
        self.transcls_C1 = linear_classifier(1024, 4)
        self.transcls_C2 = linear_classifier(1024, 4)
        # self.transcls_C3 = linear_classifier(1024, 4)
        self.defcls_C = linear_classifier(1024, 27)
        # self.defcls_C2 = linear_classifier(1024, 27)
        # self.normreg_C = nn.Conv1d(1024, 4, kernel_size=1, bias=False)
        # self.curvconfreg_C = linear_classifier(1)
        self.DecoderFC = DecoderFC(args, 1024)

        self.DefRec = RegionReconstruction(args, num_f_prev_5 + 1024)

        # self.MasSurf = MaskSurf(trans_dim = 384,drop_path_rate = 0.1,decoder_depth = 4,decoder_num_heads = 6,group_size = 32,num_group=64)

        self.inv_head = nn.Sequential(
                            nn.Linear(512 * 2, 512),
                            nn.ReLU(inplace=True),
                            nn.Linear(512, 128),
                            )


    def forward(self, x, alpha=0, activate_DefRec=False, activate_MasSurf=False):

        # MaskSurf_input = x.clone()

        if activate_MasSurf:
            x = x[:,:,:3]
            x = x.permute(0,2,1)

        batch_size = x.size(0)
        num_points = x.size(2)
        cls_logits = {}


        # returns a tensor of (batch_size, 6, #points, #neighboors)
        # interpretation: each point is represented by 20 NN, each of size 6
        # x0 = get_graph_feature(x, self.args, k=self.k)  # x0: [b, 6, 1024, 20]
        # align to a canonical space (e.g., apply rotation such that all inputs will have the same rotation)
        # transformd_x0 = self.input_transform_net(x0)  # transformd_x0: [3, 3]
        # x = torch.matmul(transformd_x0, x)

        # returns a tensor of (batch_size, 6, #points, #neighboors)
        # interpretation: each point is represented by 20 NN, each of size 6
        x = get_graph_feature(x, self.args, k=self.k)  # x: [b, 6, 1024, 20]
        # process point and inflate it from 6 to e.g., 64
        x = self.conv1(x)  # x: [b, 64, 1024, 20]
        # per each feature (from e.g., 64) take the max value from the representative vectors
        # Conceptually this means taking the neighbor that gives the highest feature value.
        # returns a tensor of size e.g., (batch_size, 64, #points)
        x1 = x.max(dim=-1, keepdim=False)[0]




        x = get_graph_feature(x1, self.args, k=self.k)
        x = self.conv2(x)
        x2 = x.max(dim=-1, keepdim=False)[0]



        x = get_graph_feature(x2, self.args, k=self.k)
        x = self.conv3(x)
        x3 = x.max(dim=-1, keepdim=False)[0]

        x11 = F.adaptive_max_pool1d(x3, 1).view(batch_size, -1)
        x22 = F.adaptive_avg_pool1d(x3, 1).view(batch_size, -1)
        feat_3 = torch.cat((x11, x22), 1)



        x = get_graph_feature(x3, self.args, k=self.k)
        x = self.conv4(x)
        x4 = x.max(dim=-1, keepdim=False)[0]



        x11 = F.adaptive_max_pool1d(x4, 1).view(batch_size, -1)
        x22 = F.adaptive_avg_pool1d(x4, 1).view(batch_size, -1)
        feat_4 = torch.cat((x11, x22), 1)

        x_cat = torch.cat((x1, x2, x3, x4), dim=1)
        x5 = self.conv5(x_cat)  # [b, 1024, 1024]
        x5 = F.leaky_relu(self.bn5(x5), negative_slope=0.2)
        # print("---5--", x5.shape)
        x1 = F.adaptive_max_pool1d(x5, 1).view(batch_size, -1)
        x2 = F.adaptive_avg_pool1d(x5, 1).view(batch_size, -1)
        x = torch.cat((x1, x2), 1)

        # # add noise
        # esp_a = x.data.new(x.size()).normal_(1., self.args.zeta)
        # esp_b = x.data.new(x.size()).normal_(0., self.args.zeta)
        # x = x * esp_a + esp_b

        feat_5 = x


        inv_feat_5 = self.inv_head(feat_5)



        # x5 = F.leaky_relu(self.bn5(x), negative_slope=0.2)

        # Per feature take the point that have the highest (absolute) value.
        # Generate a feature vector for the whole shape
        # x5_pool = F.adaptive_max_pool1d(x5, 1).view(batch_size, -1)
        # x = x5_pool

        cls_logits["cls"] = self.cls_C(x)  #attention
        # cls_logits["cls_pro"] = self.cls_C(x_pro)  #attention

        if alpha is not 0:
            reverse_x = ReverseLayerF.apply(x, alpha)
            cls_logits["domain_cls"] = self.domain_C(reverse_x)
        cls_logits["rot_cls1"] = self.rotcls_C1(x)
        cls_logits["rot_cls2"] = self.rotcls_C2(x)
        # cls_logits["trans_cls"] = self.transcls_C(x)
        cls_logits["trans_cls1"] = self.transcls_C1(x)
        cls_logits["trans_cls2"] = self.transcls_C2(x)
        # cls_logits["trans_cls3"] = self.transcls_C3(x)
        cls_logits["def_cls"] = self.defcls_C(x)
        # cls_logits["def_cls_n"] = self.defcls_C2(x)

        # cls_logits["curv_conf"] = self.curvconfreg_C(x)
        # cls_logits["norm_reg"] = self.normreg_C(x5).permute(0, 2, 1)
        cls_logits["decoder"] = self.DecoderFC(x)

        if activate_DefRec:
            DefRec_input = torch.cat((x_cat, x.unsqueeze(2).repeat(1, 1, num_points)), dim=1)
            cls_logits["DefRec"] = self.DefRec(DefRec_input)

        # if activate_MasSurf:
        #     cls_logits["MasSurf"] =self.MasSurf(MaskSurf_input)
        #     # print("inloss:",cls_logits["MasSurf"])



        if self.args.tool == 'LBE':

            esp3 = feat_3.data.new(feat_3.size()).normal_(0., self.args.zeta)
            h3 = feat_3 + esp3
            esp4 = feat_4.data.new(feat_4.size()).normal_(0., self.args.zeta)
            h4 = feat_4 + esp4
            esp5 = feat_5.data.new(feat_5.size()).normal_(0., self.args.zeta)

            h5 = feat_5 + esp5
            z = self.inv_head(h5)
            if self.args.normalize:
                z = nn.functional.normalize(z, dim=1)
            out = (cls_logits ,feat_3, feat_4, feat_5, h3, h4, h5, z)

        elif self.args.tool == 'RC':

            if self.args.normalize:
                inv_feat_5 = nn.functional.normalize(inv_feat_5, dim=1)
            out = (cls_logits, inv_feat_5, x)

        elif self.args.tool == 'BYOL':

            out = (cls_logits, x)

        elif self.args.tool == 'orig':

            out = (cls_logits,inv_feat_5)

        return out


class ResNet(nn.Module):
    def __init__(self, model, feat_dim=2048):
        super(ResNet, self).__init__()
        self.resnet = model
        self.resnet.fc = nn.Identity()

        self.inv_head = nn.Sequential(
            nn.Linear(feat_dim, 512, bias=False),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 256, bias=False)
        )

    def forward(self, x):
        x = self.resnet(x)
        x = self.inv_head(x)

        return x






# class TransformerDecoder(nn.Module):
#     def __init__(self, embed_dim=384, depth=4, num_heads=6, mlp_ratio=4., qkv_bias=False, qk_scale=None,
#                  drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1, norm_layer=nn.LayerNorm):
#         super().__init__()
#         self.blocks = nn.ModuleList([
#             Block(
#                 dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
#                 drop=drop_rate, attn_drop=attn_drop_rate,
#                 drop_path=drop_path_rate[i] if isinstance(drop_path_rate, list) else drop_path_rate
#             )
#             for i in range(depth)])
#         self.norm = norm_layer(embed_dim)
#         self.head = nn.Identity()
#
#         self.apply(self._init_weights)
#
#     def _init_weights(self, m):
#         if isinstance(m, nn.Linear):
#             nn.init.xavier_uniform_(m.weight)
#             if isinstance(m, nn.Linear) and m.bias is not None:
#                 nn.init.constant_(m.bias, 0)
#         elif isinstance(m, nn.LayerNorm):
#             nn.init.constant_(m.bias, 0)
#             nn.init.constant_(m.weight, 1.0)
#
#     def forward(self, x, pos, return_token_num):
#         for _, block in enumerate(self.blocks):
#             x = block(x + pos)
#
#         x = self.head(self.norm(x[:, -return_token_num:]))  # only return the mask tokens predict pixel
#         return x

class Group(nn.Module):  # FPS + KNN
    def __init__(self, num_group, group_size):
        super().__init__()
        self.num_group = num_group
        self.group_size = group_size
        self.knn = KNN(k=self.group_size, transpose_mode=True)

    def forward(self, xyz):
        '''
            input: B N 3
            ---------------------------
            output: B G M 3
            center : B G 3
        '''

        xyz = xyz.transpose(2, 1)

        batch_size, num_points, _ = xyz.shape
        # fps the centers out
        center = pc_utils.fps(xyz, self.num_group) # B G 3
        # knn to get the neighborhood
        _, idx = self.knn(xyz, center) # B G M
        assert idx.size(1) == self.num_group
        assert idx.size(2) == self.group_size
        idx_base = torch.arange(0, batch_size, device=xyz.device).view(-1, 1, 1) * num_points
        idx = idx + idx_base
        idx = idx.view(-1)
        neighborhood = xyz.view(batch_size * num_points, -1)[idx, :]
        neighborhood = neighborhood.view(batch_size, self.num_group, self.group_size, 3).contiguous()
        # normalize
        neighborhood = neighborhood - center.unsqueeze(2)
        return neighborhood, center




## Transformers
# class Mlp(nn.Module):
#     def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
#         super().__init__()
#         out_features = out_features or in_features
#         hidden_features = hidden_features or in_features
#         self.fc1 = nn.Linear(in_features, hidden_features)
#         self.act = act_layer()
#         self.fc2 = nn.Linear(hidden_features, out_features)
#         self.drop = nn.Dropout(drop)
#
#     def forward(self, x):
#         x = self.fc1(x)
#         x = self.act(x)
#         x = self.drop(x)
#         x = self.fc2(x)
#         x = self.drop(x)
#         return x

# class Attention(nn.Module):
#     def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
#         super().__init__()
#         self.num_heads = num_heads
#         head_dim = dim // num_heads
#         # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
#         self.scale = qk_scale or head_dim ** -0.5
#         self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
#         self.attn_drop = nn.Dropout(attn_drop)
#         self.proj = nn.Linear(dim, dim)
#         self.proj_drop = nn.Dropout(proj_drop)
#
#     def forward(self, x):
#         B, N, C = x.shape
#         qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
#         q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)
#
#         attn = (q @ k.transpose(-2, -1)) * self.scale
#         attn = attn.softmax(dim=-1)
#         attn = self.attn_drop(attn)
#
#         x = (attn @ v).transpose(1, 2).reshape(B, N, C)
#         x = self.proj(x)
#         x = self.proj_drop(x)
#         return x

# class Block(nn.Module):
#     def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
#                  drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
#         super().__init__()
#         self.norm1 = norm_layer(dim)
#
#         # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
#         self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
#         self.norm2 = norm_layer(dim)
#         mlp_hidden_dim = int(dim * mlp_ratio)
#         self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
#
#         self.attn = Attention(
#             dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
#
#     def forward(self, x):
#         x = x + self.drop_path(self.attn(self.norm1(x)))
#         x = x + self.drop_path(self.mlp(self.norm2(x)))
#         return x

# class TransformerEncoder(nn.Module):
#     def __init__(self, embed_dim=768, depth=4, num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None,
#                  drop_rate=0., attn_drop_rate=0., drop_path_rate=0.):
#         super().__init__()
#
#         self.blocks = nn.ModuleList([
#             Block(
#                 dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
#                 drop=drop_rate, attn_drop=attn_drop_rate,
#                 drop_path=drop_path_rate[i] if isinstance(drop_path_rate, list) else drop_path_rate
#             )
#             for i in range(depth)])
#
#     def forward(self, x, pos):
#         for _, block in enumerate(self.blocks):
#             x = block(x + pos)
#         return x

# class Encoder(nn.Module):   ## Embedding module
#     def __init__(self, encoder_channel):
#         super().__init__()
#         self.encoder_channel = encoder_channel
#         self.first_conv = nn.Sequential(
#             nn.Conv1d(3, 128, 1),
#             nn.BatchNorm1d(128),
#             nn.ReLU(inplace=True),
#             nn.Conv1d(128, 256, 1)
#         )
#         self.second_conv = nn.Sequential(
#             nn.Conv1d(512, 512, 1),
#             nn.BatchNorm1d(512),
#             nn.ReLU(inplace=True),
#             nn.Conv1d(512, self.encoder_channel, 1)
#         )
#
#     def forward(self, point_groups):
#         '''
#             point_groups : B G N 3
#             -----------------
#             feature_global : B G C
#         '''
#         bs, g, n , _ = point_groups.shape
#         point_groups = point_groups.reshape(bs * g, n, 3)
#         # encoder
#         feature = self.first_conv(point_groups.transpose(2,1))  # BG 256 n
#         feature_global = torch.max(feature,dim=2,keepdim=True)[0]  # BG 256 1
#         feature = torch.cat([feature_global.expand(-1,-1,n), feature], dim=1)# BG 512 n
#         feature = self.second_conv(feature) # BG 1024 n
#         feature_global = torch.max(feature, dim=2, keepdim=False)[0] # BG 1024
#         return feature_global.reshape(bs, g, self.encoder_channel)

# Pretrain model
# class MaskTransformer(nn.Module):
#     def __init__(self, mask_ratio, trans_dim, depth, drop_path_rate, num_heads, encoder_dims, mask_type):
#         super().__init__()
#         # self.config = config
#         # define the transformer argparse
#         self.mask_ratio = mask_ratio
#         self.trans_dim = trans_dim
#         self.depth = depth
#         self.drop_path_rate = drop_path_rate
#         self.num_heads = num_heads
#         # print_log(f'[args] {config.transformer_config}', logger = 'Transformer')
#         # embedding
#         self.encoder_dims =  encoder_dims
#         self.encoder = Encoder(encoder_channel = self.encoder_dims)
#
#         self.mask_type = mask_type
#
#         self.pos_embed = nn.Sequential(
#             nn.Linear(3, 128),
#             nn.GELU(),
#             nn.Linear(128, self.trans_dim),
#         )
#
#         dpr = [x.item() for x in torch.linspace(0, self.drop_path_rate, self.depth)]
#         self.blocks = TransformerEncoder(
#             embed_dim = self.trans_dim,
#             depth = self.depth,
#             drop_path_rate = dpr,
#             num_heads = self.num_heads,
#         )
#
#         self.norm = nn.LayerNorm(self.trans_dim)
#         self.apply(self._init_weights)
#
#     def _init_weights(self, m):
#         if isinstance(m, nn.Linear):
#             trunc_normal_(m.weight, std=.02)
#             if isinstance(m, nn.Linear) and m.bias is not None:
#                 nn.init.constant_(m.bias, 0)
#         elif isinstance(m, nn.LayerNorm):
#             nn.init.constant_(m.bias, 0)
#             nn.init.constant_(m.weight, 1.0)
#         elif isinstance(m, nn.Conv1d):
#             trunc_normal_(m.weight, std=.02)
#             if m.bias is not None:
#                 nn.init.constant_(m.bias, 0)
#
#     def _mask_center_block(self, center, noaug=False):
#         '''
#             center : B G 3
#             --------------
#             mask : B G (bool)
#         '''
#         # skip the mask
#         if noaug or self.mask_ratio == 0:
#             return torch.zeros(center.shape[:2]).bool()
#         # mask a continuous part
#         mask_idx = []
#         for points in center:
#             # G 3
#             points = points.unsqueeze(0)  # 1 G 3
#             index = random.randint(0, points.size(1) - 1)
#             distance_matrix = torch.norm(points[:, index].reshape(1, 1, 3) - points, p=2,
#                                          dim=-1)  # 1 1 3 - 1 G 3 -> 1 G
#
#             idx = torch.argsort(distance_matrix, dim=-1, descending=False)[0]  # G
#             ratio = self.mask_ratio
#             mask_num = int(ratio * len(idx))
#             mask = torch.zeros(len(idx))
#             mask[idx[:mask_num]] = 1
#             mask_idx.append(mask.bool())
#
#         bool_masked_pos = torch.stack(mask_idx).to(center.device)  # B G
#
#         return bool_masked_pos
#
#     def _mask_center_rand(self, center, noaug = False):
#         '''
#             center : B G 3
#             --------------
#             mask : B G (bool)
#         '''
#         B, G, _ = center.shape
#         # skip the mask
#         if noaug or self.mask_ratio == 0:
#             return torch.zeros(center.shape[:2]).bool()
#
#         self.num_mask = int(self.mask_ratio * G)
#
#         overall_mask = np.zeros([B, G])
#         for i in range(B):
#             mask = np.hstack([
#                 np.zeros(G-self.num_mask),
#                 np.ones(self.num_mask),
#             ])
#             np.random.shuffle(mask)
#             overall_mask[i, :] = mask
#         overall_mask = torch.from_numpy(overall_mask).to(torch.bool)
#
#         return overall_mask.to(center.device) # B G
#
#     def forward(self, neighborhood, center, noaug = False):
#         # generate mask
#         if self.mask_type == 'rand':
#             bool_masked_pos = self._mask_center_rand(center, noaug = noaug) # B G
#         else:
#             bool_masked_pos = self._mask_center_block(center, noaug = noaug)
#
#
#         group_input_tokens = self.encoder(neighborhood)  #  B G C
#
#         # print("mp:",bool_masked_pos.shape)
#         #
#         # print("ne:",neighborhood.shape)
#         #
#         # print("token:",group_input_tokens.shape)
#
#         batch_size, seq_len, C = group_input_tokens.size()
#
#         x_vis = group_input_tokens[~bool_masked_pos].reshape(batch_size, -1, C)
#
#         # print("x_v:",x_vis.shape)
#         # add pos embedding
#         # mask pos center
#         masked_center = center[~bool_masked_pos].reshape(batch_size, -1, 3)
#         pos = self.pos_embed(masked_center)
#
#         # print("x_:",x_vis.shape)
#         # print("pos:",pos.shape)
#
#         # transformer
#         x_vis = self.blocks(x_vis, pos)
#         x_vis = self.norm(x_vis)
#
#         return x_vis, bool_masked_pos

class class_classifier(nn.Module):
    def __init__(self, args, input_dim, num_class=10):
        super(class_classifier, self).__init__()

        activate = 'leakyrelu' if args.model == 'dgcnn' else 'relu'
        bias = True if args.model == 'dgcnn' else False

        self.mlp1 = fc_layer(input_dim, 512, bias=bias, activation=activate, bn=True)
        self.dp1 = nn.Dropout(p=args.dropout)
        self.mlp2 = fc_layer(512, 256, bias=True, activation=activate, bn=True)
        self.dp2 = nn.Dropout(p=args.dropout)
        self.mlp3 = nn.Linear(256, num_class)

    def forward(self, x):
        x = self.dp1(self.mlp1(x))
        x2 = self.dp2(self.mlp2(x))
        logits = self.mlp3(x2)
        return logits


class ssl_classifier(nn.Module):
    def __init__(self, args, input_dim, num_class):
        super(ssl_classifier, self).__init__()
        self.mlp1 = fc_layer(input_dim, 256)
        self.dp1 = nn.Dropout(p=args.dropout)
        self.mlp2 = nn.Linear(256, num_class)

    def forward(self, x):
        x = self.dp1(self.mlp1(x))
        logits = self.mlp2(x)
        return logits


class linear_classifier(nn.Module):
    def __init__(self, input_dim, num_class):
        super(linear_classifier, self).__init__()
        self.mlp1 = nn.Linear(input_dim, num_class)

    def forward(self, x):
        logits = self.mlp1(x)
        return logits


class domain_classifier(nn.Module):
    def __init__(self, args, input_dim, num_class=2):
        super(domain_classifier, self).__init__()

        activate = 'leakyrelu' if args.model == 'dgcnn' else 'relu'
        bias = True if args.model == 'dgcnn' else False

        self.mlp1 = fc_layer(input_dim, 512, bias=bias, activation=activate, bn=True)
        self.mlp2 = fc_layer(512, 256, bias=True, activation=activate, bn=True)
        self.mlp3 = nn.Linear(256, num_class)

    def forward(self, x):
        x = self.mlp1(x)
        x2 = self.mlp2(x)
        logits = self.mlp3(x2)
        return logits


class DecoderFC(nn.Module):
    def __init__(self, args, input_dim):
        super(DecoderFC, self).__init__()
        activate = 'leakyrelu' if args.model == 'dgcnn' else 'relu'
        bias = True if args.model == 'dgcnn' else False

        self.mlp1 = fc_layer(input_dim, 512, bias=bias, activation=activate, bn=True)
        self.mlp2 = fc_layer(512, 512, bias=True, activation=activate, bn=True)
        self.mlp3 = nn.Linear(512, args.output_pts * 3)

    def forward(self, x):
        x = self.mlp1(x)
        x = self.mlp2(x)
        x = self.mlp3(x)
        return x


# class MaskSurf(nn.Module):
#     def __init__(self, trans_dim,drop_path_rate,decoder_depth,decoder_num_heads,group_size,num_group):
#         super().__init__()
#         # print_log(f'[Point_MAE] ', logger ='Point_MAE')
#         self.trans_dim = trans_dim
#         self.MAE_encoder = MaskTransformer(mask_ratio=0.6, trans_dim=384, depth=12, drop_path_rate=0.1, num_heads=6,
#                                            encoder_dims=384, mask_type='rand')
#         self.group_size = group_size
#         self.num_group = num_group
#         self.drop_path_rate = drop_path_rate
#         self.mask_token = nn.Parameter(torch.zeros(1, 1, self.trans_dim))
#         self.decoder_pos_embed = nn.Sequential(
#             nn.Linear(3, 128),
#             nn.GELU(),
#             nn.Linear(128, self.trans_dim)
#         )
#
#         self.decoder_depth = decoder_depth
#         self.decoder_num_heads = decoder_num_heads
#         dpr = [x.item() for x in torch.linspace(0, self.drop_path_rate, self.decoder_depth)]
#         self.MAE_decoder = TransformerDecoder(
#             embed_dim=self.trans_dim,
#             depth=self.decoder_depth,
#             drop_path_rate=dpr,
#             num_heads=self.decoder_num_heads,
#         )
#
#         # print_log(f'[Point_MAE] divide point cloud into G{self.num_group} x S{self.group_size} points ...', logger ='Point_MAE')
#         # self.group_divider = Group(num_group = self.num_group, group_size = self.group_size)
#
#         # prediction head
#         self.increase_dim = nn.Sequential(
#             # nn.Conv1d(self.trans_dim, 1024, 1),
#             # nn.BatchNorm1d(1024),
#             # nn.LeakyReLU(negative_slope=0.2),
#             nn.Conv1d(self.trans_dim, 3*self.group_size, 1)
#         )
#
#         self.increase_dim2 = nn.Sequential(
#             # nn.Conv1d(self.trans_dim, 1024, 1),
#             # nn.BatchNorm1d(1024),
#             # nn.LeakyReLU(negative_slope=0.2),
#             nn.Conv1d(self.trans_dim, 3*self.group_size, 1)
#         )
#
#         trunc_normal_(self.mask_token, std=.02)
#
#         self.loss_func = ChamferDistanceL2_withnormal().cuda()
#
#     #     self.loss = config.loss
#     #     # loss
#     #     self.build_loss_func(self.loss)
#     #
#     # def build_loss_func(self, loss_type):
#     #     if loss_type == "cdl1":
#     #         self.loss_func = ChamferDistanceL1().cuda()
#     #     elif loss_type =='cdl2':
#     #         self.loss_func = ChamferDistanceL2().cuda()
#     #     else:
#     #         raise NotImplementedError
#     #         # self.loss_func = emd().cuda()
#
#     def forward(self, pts, vis = False, **kwargs):
#
#         neighborhood, neighborhood_normal, center = pc_utils.groupx(pts)
#         # print("x_0:",x.shape)
#         # neighborhood, neighborhood_normal, center = self.group_divider(pts)
#
#         x_vis, mask = self.MAE_encoder(neighborhood, center)
#         # x = x.reshape(batch_size, 3, -1)
#         # print("x_1:",x_vis.shape)
#
#         B,_,C = x_vis.shape # B VIS C
#
#         pos_emd_vis = self.decoder_pos_embed(center[~mask]).reshape(B, -1, C)
#
#         pos_emd_mask = self.decoder_pos_embed(center[mask]).reshape(B, -1, C)
#
#         _,N,_ = pos_emd_mask.shape
#         mask_token = self.mask_token.expand(B, N, -1)
#         x_full = torch.cat([x_vis, mask_token], dim=1)
#         pos_full = torch.cat([pos_emd_vis, pos_emd_mask], dim=1)
#
#         x_rec = self.MAE_decoder(x_full, pos_full, N)
#
#         B, M, C = x_rec.shape
#
#         # print("x_r:",x_rec.shape)
#
#         rebuild_points = self.increase_dim(x_rec.transpose(1, 2)).transpose(1, 2).reshape(B * M, -1, 3)  # B M 1024
#         rebuild_normal = self.increase_dim2(x_rec.transpose(1, 2)).transpose(1, 2).reshape(B * M, -1, 3)  # BM Gs 6
#
#         gt_points = neighborhood[mask].reshape(B*M,-1,3)
#         gt_normals = neighborhood_normal[mask].reshape(B * M, -1, 3)  ## BM, Gs, 6
#         loss_xyz, loss_normal = self.loss_func(rebuild_points, gt_points, rebuild_normal, gt_normals)
#
#
#         # print("new_x:",rebuild_points.shape)
#
#         if vis: #visualization
#             vis_points = neighborhood[~mask].reshape(B * (self.num_group - M), -1, 3)
#             full_vis = vis_points + center[~mask].unsqueeze(1)
#             full_rebuild = rebuild_points + center[mask].unsqueeze(1)
#             full = torch.cat([full_vis, full_rebuild], dim=0)
#             # full_points = torch.cat([rebuild_points,vis_points], dim=0)
#             full_center = torch.cat([center[mask], center[~mask]], dim=0)
#             # full = full_points + full_center.unsqueeze(1)
#             ret2 = full_vis.reshape(-1, 3).unsqueeze(0)
#             ret1 = full.reshape(-1, 3).unsqueeze(0)
#             # return ret1, ret2
#             return ret1, ret2, full_center
#         else:
#             return loss_xyz, loss_normal



class RegionReconstruction(nn.Module):
    """
    Region Reconstruction Network - Reconstruction of a deformed region.
    For more details see https://arxiv.org/pdf/2003.12641.pdf
    """

    def __init__(self, args, input_size):
        super(RegionReconstruction, self).__init__()
        self.args = args
        self.of1 = 256
        self.of2 = 256
        self.of3 = 128

        self.bn1 = nn.BatchNorm1d(self.of1)
        self.bn2 = nn.BatchNorm1d(self.of2)
        self.bn3 = nn.BatchNorm1d(self.of3)

        self.conv1 = nn.Conv1d(input_size, self.of1, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(self.of1, self.of2, kernel_size=1, bias=False)
        self.conv3 = nn.Conv1d(self.of2, self.of3, kernel_size=1, bias=False)
        self.conv4 = nn.Conv1d(self.of3, 3, kernel_size=1, bias=False)

    def forward(self, x):
        x = F.leaky_relu(self.bn1(self.conv1(x)), negative_slope=0.2)
        x = F.leaky_relu(self.bn2(self.conv2(x)), negative_slope=0.2)
        x = F.leaky_relu(self.bn3(self.conv3(x)), negative_slope=0.2)
        x = self.conv4(x)
        return x.permute(0, 2, 1)


class ReverseLayerF(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None


