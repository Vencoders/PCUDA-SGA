import numpy as np
import random
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import sklearn.metrics as metrics
import argparse
import copy
import utils.log
from data.dataloader_Norm import ScanNet, ModelNet, ShapeNet, label_to_idx
from data.dataloader_GraspNetPC import GraspNetRealPointClouds, GraspNetSynthetictPointClouds
from Models_Norm import PointNet, DGCNN
import moco.builder

import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import matplotlib.colors as mcolors






NWORKERS=4
MAX_LOSS = 9 * (10**9)

def str2bool(v):
    """
    Input:
        v - string
    output:
        True/False
    """
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


# ==================
# Argparse
# ==================
parser = argparse.ArgumentParser(description='DA on Point Clouds')
parser.add_argument('--exp_name', type=str, default='GAST_test',  help='Name of the experiment')
parser.add_argument('--out_path', type=str, default='./experiments', help='log folder path')
parser.add_argument('--dataroot', type=str, default=r'/data/zhc/gast/data', metavar='N', help='data path')
parser.add_argument('--model_file', type=str, default='model.ptdgcnn120', help='pretrained model file')
parser.add_argument('--trgt_dataset', type=str, default='scannet', choices=['Kin', 'RS','modelnet', 'shapenet', 'scannet'])
parser.add_argument('--model', type=str, default='dgcnn', choices=['pointnet', 'dgcnn'], help='Model to use')
parser.add_argument('--seed', type=int, default=1, help='random seed (default: 1)')
parser.add_argument('--gpus', type=lambda s: [int(item.strip()) for item in s.split(',')], default='1',
                    help='comma delimited of gpu ids to use. Use "-1" for cpu usage')
parser.add_argument('--test_batch_size', type=int, default=32, metavar='batch_size', help='Size of test batch per domain')
parser.add_argument('--output_pts', type=int, default=512, help='number of decoder points')
parser.add_argument('--dropout', type=float, default=0.5, help='dropout rate')
parser.add_argument('--tool', default="orig", type=str, help="orig/RC/LBE/IP/MIB")
parser.add_argument('--normalize', default=True, action='store_true', help='normalize')
parser.add_argument('--zeta', default=0.1, type=float, help='variance')
parser.add_argument('--lamb', default=1., type=float, help='weight of regularization term')

parser.add_argument(
    "--moco-k",
    default=65536,
    type=int,
    help="queue size; number of negative keys (default: 65536)",
)
parser.add_argument(
    "--moco-m",
    default=0.999,
    type=float,
    help="moco momentum of updating key encoder (default: 0.999)",
)
parser.add_argument(
    "--moco-t", default=0.07, type=float, help="softmax temperature (default: 0.07)"
)

args = parser.parse_args()

# ==================
# init
# ==================
io = utils.log.IOStream(args)
io.cprint(str(args))

# random.seed(1)
# np.random.seed(1)  # to get the same point choice in ModelNet and ScanNet leave it fixed
torch.manual_seed(args.seed)
args.cuda = (args.gpus[0] >= 0) and torch.cuda.is_available()
device = torch.device("cuda:" + str(args.gpus[0]) if args.cuda else "cpu")
if args.cuda:
    io.cprint('Using GPUs ' + str(args.gpus) + ',' + ' from ' +
              str(torch.cuda.device_count()) + ' devices available')
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
else:
    io.cprint('Using CPU')

# ==================
# loss function
# ==================
criterion = nn.CrossEntropyLoss()  # return the mean of CE over the batch


# ==================
# Read Test Data
# ==================
trgt_dataset = args.trgt_dataset
# data_func = {'modelnet': ModelNet, 'scannet': ScanNet, 'shapenet': ShapeNet}
# trgt_testset = data_func[trgt_dataset](io, args.dataroot, 'test')

if trgt_dataset == 'modelnet':
    trgt_testset = ModelNet(io, args.dataroot, 'test')

elif trgt_dataset == 'shapenet':
    trgt_testset = ShapeNet(io, args.dataroot, 'test')

elif trgt_dataset == 'scannet':
    trgt_testset = ScanNet(io, args.dataroot, 'test')

elif trgt_dataset == 'Kin':
    trgt_testset = GraspNetRealPointClouds(args.dataroot, mode='kinect', partition='test')

elif trgt_dataset == 'RS':
    trgt_testset = GraspNetRealPointClouds(args.dataroot, mode='realsense', partition='test')
# dataloaders for test
trgt_test_loader = DataLoader(trgt_testset, num_workers=NWORKERS, batch_size=args.test_batch_size)


# ==================
# Init Model
# ==================
# if args.model == 'pointnet':
#     model = PointNet(args)
#     model.load_state_dict(torch.load('./experiments/GAST/model.ptpointnet'))
# elif args.model == 'dgcnn':
#     model = DGCNN(args)
#     model.load_state_dict(torch.load('./experiments/GAST/' + args.model_file))
# else:
#     raise Exception("Not implemented")

model = moco.builder.MoCo_Model(args, queue_size=args.moco_k,
                      momentum=args.moco_m, temperature=args.moco_t)

model.load_state_dict(torch.load('./experiments/GAST/' + args.model_file))

model = model.to(device)

# Handle multi-gpu
if (device.type == 'cuda') and len(args.gpus) > 1:
    model = nn.DataParallel(model, args.gpus)
best_model = copy.deepcopy(model)


# ==================
# Test
# ==================
def test(test_loader, model=None, set_type="Target", partition="Val", epoch=0):

    # Run on cpu or gpu
    count = 0.0
    print_losses = {'cls': 0.0}
    batch_idx = 0
    with torch.no_grad():

        model.eval()
        test_pred = []
        test_true = []


        for data, labels, _  in test_loader:
            data, labels = data.to(device), labels.to(device).squeeze()
            data = data.permute(0, 2, 1)
            batch_size = data.size()[0]

            logits, _ = model(data, data, data,device, activate_N=True) #前面两个凑数

            loss = criterion(logits["cls"], labels)
            print_losses['cls'] += loss.item() * batch_size

            # evaluation metrics
            preds = logits["cls"].max(dim=1)[1]
            test_true.append(labels.cpu().numpy())
            test_pred.append(preds.detach().cpu().numpy())
            count += batch_size
            batch_idx += 1

    test_true = np.concatenate(test_true)
    test_pred = np.concatenate(test_pred)
    print_losses = {k: v * 1.0 / count for (k, v) in print_losses.items()}
    test_acc = io.print_progress(set_type, partition, epoch, print_losses, test_true, test_pred)
    conf_mat = metrics.confusion_matrix(test_true, test_pred, labels=list(label_to_idx.values())).astype(int)

    return test_acc, print_losses['cls'], conf_mat



# ==================
# t-SNE
# ==================
# features_list = []
# labels_list = []
# colors = ['#e6194b', '#3cb44b', '#ffe119', '#4363d8', '#f58231',
#           '#911eb4', '#46f0f0', '#f032e6', '#bcf60c', '#fabebe']
#
# cmap = mcolors.ListedColormap(colors)
#
# model.eval()  # 将模型设置为评估模式
# with torch.no_grad():
#     for data, labels, _  in trgt_test_loader:
#         data, labels = data.to(device), labels.to(device).squeeze()
#         data = data.permute(0, 2, 1)
#         # _, outputs = model(data, data, data, activate_N=True)
#         outputs, _ = model(data, data, data,device, activate_N=True)
#         features_list.append(outputs["cls"].cpu().numpy())
#         labels_list.append(labels.cpu().numpy())
#
# features = np.vstack(features_list)
# labels = np.hstack(labels_list)
#
# # 2. 使用t-SNE降维
# tsne = TSNE(n_components=2, random_state=42)
# features_tsne = tsne.fit_transform(features)
#
# # 3. 可视化结果
# plt.figure(figsize=(10, 10))
# plt.xticks([])
# plt.yticks([])
# scatter = plt.scatter(features_tsne[:, 0], features_tsne[:, 1], c=labels, cmap=cmap, s=25)
# # plt.colorbar(scatter, ticks=range(10))
# # plt.title('t-SNE visualization of extracted features')
# plt.savefig('tsne_visualization.png', dpi=300, bbox_inches='tight')  # 可以根据需要选择图像格式和参数
# # plt.show()






trgt_test_acc, trgt_test_loss, trgt_conf_mat = test(trgt_test_loader, best_model, "Target", "Test", 0)



io.cprint("target test accuracy: %.4f, target test loss: %.4f" % (trgt_test_acc, trgt_test_loss))
io.cprint("Test confusion matrix:")
io.cprint('\n' + str(trgt_conf_mat))

