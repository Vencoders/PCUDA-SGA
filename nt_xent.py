import torch
import torch.nn as nn

def gen_mask(k, feat_dim):
    mask = None
    for i in range(k):
        tmp_mask = torch.triu(torch.randint(0, 2, (feat_dim, feat_dim)), 1)
        tmp_mask = tmp_mask + torch.triu(1-tmp_mask,1).t()
        tmp_mask = tmp_mask.view(tmp_mask.shape[0], tmp_mask.shape[1],1)
        mask = tmp_mask if mask is None else torch.cat([mask,tmp_mask],2)
    return mask

def entropy(prob):
    # assume m x m x k input
    return -torch.sum(prob*torch.log(prob),1)

class NT_Xent(nn.Module):
    def __init__(self, batch_size, temperature, mask, probs, target):
        super(NT_Xent, self).__init__()
        self.batch_size = batch_size
        self.temperature = temperature
        self.mask = mask
        self.probs = probs
        self.target = target

        self.criterion = nn.CrossEntropyLoss(reduction="sum")
        self.similarity_f = nn.CosineSimilarity(dim=2)
        self.label = torch.LongTensor([i for i in range(10)])

    def forward(self, z_i, z_j):
        """
        We do not sample negative examples explicitly.
        Instead, given a positive pair, similar to (Chen et al., 2017), we treat the other 2(N − 1) augmented examples within a minibatch as negative examples.
        """
        p1 = torch.cat((z_i, z_j), dim=0)

        sim = self.similarity_f(p1.unsqueeze(1), p1.unsqueeze(0)) / self.temperature

        sim_i_j = torch.diag(sim, self.batch_size)
        sim_j_i = torch.diag(sim, -self.batch_size)

        positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(self.batch_size * 2, 1)

        probs = self.probs
        target = self.target

        target = torch.cat((target, target))
        # print(".",target)
        label = (self.label.unsqueeze(1).to(target.device) != target.unsqueeze(0))  # (C, N)
        # print("--",label)
        # print("*",label.shape)
        probs = torch.masked_select(probs, label.transpose(1, 0).to(probs.device)).reshape(self.batch_size * 2, -1)
        # print("**",probs.shape)
        # print("-",probs)
        # probs = probs.unsqueeze(1)  # (N, 1)

        mask_m, mask = self.mask

        negative_samples1 = sim[mask_m].reshape(self.batch_size * 2, -1)

        negative_samples2 = torch.where(mask.cuda(), negative_samples1, torch.tensor([float("-inf")]).cuda())

        labels = torch.zeros(self.batch_size * 2).long().cuda()

        logits1 = torch.cat((positive_samples, negative_samples1), dim=1) # common

        loss1 = self.criterion(logits1, labels)
        loss1 /= 2 * self.batch_size

        logits2 = torch.cat((positive_samples, probs), dim=1) # others

        loss2 = self.criterion(logits2, labels)
        loss2 /= 2 * self.batch_size

        
        return loss1, loss2


# class Contrast(nn.Module):
#     def __init__(self, batch_size, temperature, mask):
#         super(Contrast, self).__init__()
#         self.batch_size = batch_size
#         self.temperature = temperature
#         self.mask = mask
#         self.criterion = nn.CrossEntropyLoss(reduction="sum")
#         self.similarity_f = nn.CosineSimilarity(dim=2)
#
#     def forward(self, z_i, z_j):
#         '''
#         batch的大小为N，z_i为N个样本的增强特征1，z_j为N个样本的增强特征2
#         当z_i和z_j为两个点云增强特征时为模态内对比损失,当z_i和z_j分别为点云增强特征均值和对应图像增强特征时为跨模态对比损失
#         '''
#         # 将N个样本的两个增强版本拼接起来
#         p1 = torch.cat((z_i, z_j), dim=0)
#         # 计算相似度矩阵
#         sim = self.similarity_f(p1.unsqueeze(1), p1.unsqueeze(0)) / self.temperature
#         # 获取其中正样本对的相似度矩阵
#         sim_i_j = torch.diag(sim, self.batch_size)
#         sim_j_i = torch.diag(sim, -self.batch_size)
#         positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(self.batch_size * 2, 1)
#         # 获取其中负样本对的相似度矩阵
#         mask_m, mask = self.mask
#         negative_samples = sim[mask_m].reshape(self.batch_size * 2, -1)
#         # 将正样本对和负样本对的相似度拼接起来，其中第一个维度为正样本对的相似度，剩余维度为负样本对的维度，将labels设置为全0向量对应正样本对相似度的值
#         logits = torch.cat((positive_samples, negative_samples), dim=1)
#         labels = torch.zeros(self.batch_size * 2).long().cuda()
#         # 通过交叉熵作为损失函数的表达形式，只有放第一个维度的正样本对相似度越高，负样本对的相似度越低，该损失函数才能降低。
#         loss = self.criterion(logits, labels)
#         return loss
