# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F

from Models_Norm import PointNet, DGCNN



# import model.network as models


class MoCo_Model(nn.Module):
    def __init__(self, args, queue_size=65536, momentum=0.999, temperature=0.07):
        '''
        MoCoV2 model, taken from: https://github.com/facebookresearch/moco.

        Adapted for use in personal Boilerplate for unsupervised/self-supervised contrastive learning.

        Additionally, too inspiration from: https://github.com/HobbitLong/CMC.

        Args:
            init:
                args (dict): Program arguments/commandline arguments.

                queue_size (int): Length of the queue/memory, number of samples to store in memory. (default: 65536)

                momentum (float): Momentum value for updating the key_encoder. (default: 0.999)

                temperature (float): Temperature used in the InfoNCE / NT_Xent contrastive losses. (default: 0.07)

            forward:
                x_q (Tensor): Reprentation of view intended for the query_encoder.

                x_k (Tensor): Reprentation of view intended for the key_encoder.

        returns:

            logit (Tensor): Positve and negative logits computed as by InfoNCE loss. (bsz, queue_size + 1)

            label (Tensor): Labels of the positve and negative logits to be used in softmax cross entropy. (bsz, 1)

        '''
        super(MoCo_Model, self).__init__()

        self.queue_size = queue_size
        self.momentum = momentum
        self.temperature = temperature

        # assert self.queue_size % args.batch_size == 0  # for simplicity

        # Load model

        if args.model == 'pointnet':
            self.encoder_q = PointNet(args)
            self.encoder_k = PointNet(args)

        elif args.model == 'dgcnn':
            self.encoder_q = DGCNN(args)
            self.encoder_k = DGCNN(args)

        else:
            raise Exception("Not implemented")



        # self.encoder_q = getattr(models, args.model)(
        #     args, num_classes=128)  # Query Encoder
        #
        # self.encoder_k = getattr(models, args.model)(
        #     args, num_classes=128)  # Key Encoder


        # # Add the mlp head
        # self.encoder_q.fc = models.projection_MLP(args)
        # self.encoder_k.fc = models.projection_MLP(args)

        # Initialize the key encoder to have the same values as query encoder
        # Do not update the key encoder via gradient
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False

        # Create the queue to store negative samples
        self.register_buffer("queue", torch.randn(self.queue_size, 128))

        # Create pointer to store current position in the queue when enqueue and dequeue
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def momentum_update(self):
        '''
        Update the key_encoder parameters through the momentum update:


        key_params = momentum * key_params + (1 - momentum) * query_params

        '''

        # For each of the parameters in each encoder
        for p_q, p_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            p_k.data = p_k.data * self.momentum + p_q.detach().data * (1. - self.momentum)

    @torch.no_grad()
    def shuffled_idx(self, batch_size,device):
        '''
        Generation of the shuffled indexes for the implementation of ShuffleBN.

        https://github.com/HobbitLong/CMC.

        args:
            batch_size (Tensor.int()):  Number of samples in a batch

        returns:
            shuffled_idxs (Tensor.long()): A random permutation index order for the shuffling of the current minibatch

            reverse_idxs (Tensor.long()): A reverse of the random permutation index order for the shuffling of the
                                            current minibatch to get back original sample order

        '''

        # Generate shuffled indexes


        shuffled_idxs = torch.randperm(batch_size).long().to(device)

        reverse_idxs = torch.zeros(batch_size).long().to(device)

        value = torch.arange(batch_size).long().to(device)

        reverse_idxs.index_copy_(0, shuffled_idxs, value)

        return shuffled_idxs, reverse_idxs

    @torch.no_grad()
    def update_queue(self, feat_k):
        '''
        Update the memory / queue.

        Add batch to end of most recent sample index and remove the oldest samples in the queue.

        Store location of most recent sample index (ptr).

        Taken from: https://github.com/facebookresearch/moco

        args:
            feat_k (Tensor): Feature reprentations of the view x_k computed by the key_encoder.
        '''

        batch_size = feat_k.size(0)

        ptr = int(self.queue_ptr)

        # replace the keys at ptr (dequeue and enqueue)
        self.queue[ptr:ptr + batch_size, :] = feat_k

        # move pointer along to end of current batch
        ptr = (ptr + batch_size) % self.queue_size

        # Store queue pointer as register_buffer
        self.queue_ptr[0] = ptr

    def InfoNCE_logits(self, f_q, f_k):
        '''
        Compute the similarity logits between positive
         samples and positve to all negatives in the memory.

        args:
            f_q (Tensor): Feature reprentations of the view x_q computed by the query_encoder.

            f_k (Tensor): Feature reprentations of the view x_k computed by the key_encoder.

        returns:
            logit (Tensor): Positve and negative logits computed as by InfoNCE loss. (bsz, queue_size + 1)

            label (Tensor): Labels of the positve and negative logits to be used in softmax cross entropy. (bsz, 1)
        '''


        f_k = f_k.detach()

        # Get queue from register_buffer
        f_mem = self.queue.clone().detach()

        # Normalize the feature representations
        f_q = nn.functional.normalize(f_q, dim=1)
        f_k = nn.functional.normalize(f_k, dim=1)
        f_mem = nn.functional.normalize(f_mem, dim=1)

        # Compute sim between positive views
        pos = torch.bmm(f_q.view(f_q.size(0), 1, -1),
                        f_k.view(f_k.size(0), -1, 1)).squeeze(-1)


        # Compute sim between postive and all negatives in the memory
        neg = torch.mm(f_q, f_mem.transpose(1, 0))



        logits = torch.cat((pos, neg), dim=1)

        logits /= self.temperature

        # Create labels, first logit is postive, all others are negative
        labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()

        return logits, labels

    def forward(self, x_q, x_k, x,device, activate_C=False, activate_N=False, activate_S=False, activate_DefRec=False):

        if activate_S:
            logits , _ = self.encoder_q(x)
            logits_q , _ = self.encoder_q(x_q)
            logits_k , _ = self.encoder_q(x_k)

            return logits["cls"], logits_q["cls"], logits_k["cls"]

        if activate_N:

            # if activate_DefRec:
            #     logits, feat = self.encoder_q(x, activate_DefRec = True)
            #     feat = nn.functional.normalize(feat, dim=1)
            #     return logits, feat

            logits , feat = self.encoder_q(x)
            feat = nn.functional.normalize(feat, dim=1)

            return logits, feat

        if activate_C:

            batch_size = x_q.size(0)

            logits_o , feat_oo = self.encoder_q(x)

            # probs_o = F.softmax(logits_o["cls"], dim=1)

            # Feature representations of the query view from the query encoder
            logits_s, feat_qq = self.encoder_q(x_q)

            # # add noise
            # esp_a = feat_qq.data.new(feat_qq.size()).normal_(1., 0.1)
            # esp_b = feat_qq.data.new(feat_qq.size()).normal_(0., 0.1)
            #
            # feat_qq = feat_qq * esp_a + esp_b

            probs_s = F.softmax(logits_s["cls"], dim=1)

            # TODO: shuffle ids with distributed data parallel
            # Get shuffled and reversed indexes for the current minibatch
            shuffled_idxs, reverse_idxs = self.shuffled_idx(batch_size,device)

            # shuffled_idxs = shuffled_idxs.to('cuda:2')
            # reverse_idxs = reverse_idxs.to('cuda:2')

            with torch.no_grad():
                # Update the key encoder
                self.momentum_update()

                # Shuffle minibatch
                x_k = x_k[shuffled_idxs]

                # Feature representations of the shuffled key view from the key encoder
                logits_w, feat_kk = self.encoder_k(x_k)

                # probs_w = logits_w["cls"]
                probs_w = F.softmax(logits_w["cls"], dim=1)
                # reverse the shuffled samples to original position
                feat_kk = feat_kk[reverse_idxs]

            # Compute the logits for the InfoNCE contrastive loss.
            # logit, label = self.InfoNCE_logits(feat_q, feat_k)


            mem_buf = self.queue.clone().detach()

            feat_o = nn.functional.normalize(feat_oo, dim=1)

            feat_q = nn.functional.normalize(feat_qq, dim=1)
            feat_k = nn.functional.normalize(feat_kk, dim=1)
            mem_buf = nn.functional.normalize(mem_buf, dim=1)

            # #SCE
            # labels = torch.zeros(batch_size, dtype=torch.long).cuda()
            # sim_q_ktarget = torch.einsum('nc,nc->n', [feat_q, feat_k]).unsqueeze(-1)
            # sim_k_ktarget = torch.zeros(batch_size).unsqueeze(-1).cuda()
            #
            #
            # sim_q_queue = torch.einsum('nc,ck->nk', [feat_q, mem_buf.transpose(1, 0)])
            # sim_k_queue = torch.einsum('nc,ck->nk', [feat_k, mem_buf.transpose(1, 0)])
            #
            #
            # sim_q = torch.cat([sim_q_ktarget, sim_q_queue], dim=1)
            # sim_k = torch.cat([sim_k_ktarget, sim_k_queue], dim=1)
            #
            #
            # mask = nn.functional.one_hot(labels, 1 + mem_buf.shape[0])
            #
            #
            # logits_q = sim_q / 0.1
            # logits_k = sim_k / 0.05
            #
            # prob_k = nn.functional.softmax(logits_k, dim=1)
            # prob_q = nn.functional.normalize(
            #     0.5 * mask + (1 - 0.5) * prob_k, p=1, dim=1)

            # ReSSL
            logitso = torch.einsum('nc,ck->nk', [feat_o, mem_buf.transpose(1, 0)])
            logitsq = torch.einsum('nc,ck->nk', [feat_q, mem_buf.transpose(1, 0)])
            logitsk = torch.einsum('nc,ck->nk', [feat_k, mem_buf.transpose(1, 0)])

            # Update the queue/memory with the current key_encoder minibatch.
            self.update_queue(feat_k)

            # return prob_q, logits_q

            return logitsq, logitsk, logitso, feat_qq, feat_kk, probs_w, probs_s

            # return logit, label










# # Copyright (c) Meta Platforms, Inc. and affiliates.
#
# # This source code is licensed under the MIT license found in the
# # LICENSE file in the root directory of this source tree.
#
# import torch
# import torch.nn as nn
#
#
# class MoCo(nn.Module):
#     """
#     Build a MoCo model with: a query encoder, a key encoder, and a queue
#     https://arxiv.org/abs/1911.05722
#     """
#
#     def __init__(self, base_encoder, dim=128, K=65536, m=0.999, T=0.07, mlp=False):
#         """
#         dim: feature dimension (default: 128)
#         K: queue size; number of negative keys (default: 65536)
#         m: moco momentum of updating key encoder (default: 0.999)
#         T: softmax temperature (default: 0.07)
#         """
#         super(MoCo, self).__init__()
#
#         self.K = K
#         self.m = m
#         self.T = T
#
#         # create the encoders
#         # num_classes is the output fc dimension
#         self.encoder_q = base_encoder
#         self.encoder_k = base_encoder
#
#         if mlp:  # hack: brute-force replacement
#             dim_mlp = self.encoder_q.fc.weight.shape[1]
#             self.encoder_q.fc = nn.Sequential(
#                 nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.encoder_q.fc
#             )
#             self.encoder_k.fc = nn.Sequential(
#                 nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.encoder_k.fc
#             )
#
#         for param_q, param_k in zip(
#             self.encoder_q.parameters(), self.encoder_k.parameters()
#         ):
#             param_k.data.copy_(param_q.data)  # initialize
#             param_k.requires_grad = False  # not update by gradient
#
#         # create the queue
#         self.register_buffer("queue", torch.randn(dim, K))
#         self.queue = nn.functional.normalize(self.queue, dim=0)
#
#         self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))
#
#     @torch.no_grad()
#     def _momentum_update_key_encoder(self):
#         """
#         Momentum update of the key encoder
#         """
#         for param_q, param_k in zip(
#             self.encoder_q.parameters(), self.encoder_k.parameters()
#         ):
#
#             param_k.data = param_k.data * self.m + param_q.data * (1.0 - self.m)
#
#     @torch.no_grad()
#     def _dequeue_and_enqueue(self, keys):
#         # gather keys before updating queue
#         #
#
#         batch_size = keys.shape[0]
#
#         ptr = int(self.queue_ptr)
#         assert self.K % batch_size == 0  # for simplicity
#
#         # replace the keys at ptr (dequeue and enqueue)
#         self.queue[:, ptr : ptr + batch_size] = keys.T
#         ptr = (ptr + batch_size) % self.K  # move pointer
#
#         self.queue_ptr[0] = ptr
#
#     @torch.no_grad()
#     def _batch_shuffle_ddp(self, x):
#         """
#         Batch shuffle, for making use of BatchNorm.
#         *** Only support DistributedDataParallel (DDP) model. ***
#         """
#         # gather from all gpus
#         batch_size_this = x.shape[0]
#         x_gather = concat_all_gather(x)
#         batch_size_all = x_gather.shape[0]
#
#         num_gpus = batch_size_all // batch_size_this
#
#         # random shuffle index
#         idx_shuffle = torch.randperm(batch_size_all).cuda()
#
#         # broadcast to all gpus
#         torch.distributed.broadcast(idx_shuffle, src=0)
#
#         # index for restoring
#         idx_unshuffle = torch.argsort(idx_shuffle)
#
#         # shuffled index for this gpu
#         gpu_idx = torch.distributed.get_rank()
#         idx_this = idx_shuffle.view(num_gpus, -1)[gpu_idx]
#
#         return x_gather[idx_this], idx_unshuffle
#
#     @torch.no_grad()
#     def _batch_unshuffle_ddp(self, x, idx_unshuffle):
#         """
#         Undo batch shuffle.
#         *** Only support DistributedDataParallel (DDP) model. ***
#         """
#         # gather from all gpus
#         batch_size_this = x.shape[0]
#         x_gather = concat_all_gather(x)
#         batch_size_all = x_gather.shape[0]
#
#         num_gpus = batch_size_all // batch_size_this
#
#         # restored index for this gpu
#         gpu_idx = torch.distributed.get_rank()
#         idx_this = idx_unshuffle.view(num_gpus, -1)[gpu_idx]
#
#         return x_gather[idx_this]
#
#     def forward(self, im_q, im_k):
#         """
#         Input:
#             im_q: a batch of query images
#             im_k: a batch of key images
#         Output:
#             logits, targets
#         """
#
#         # compute query features
#         _, q = self.encoder_q(im_q)  # queries: NxC
#         q = nn.functional.normalize(q, dim=1)
#
#         # compute key features
#         with torch.no_grad():  # no gradient to keys
#             self._momentum_update_key_encoder()  # update the key encoder
#
#             # shuffle for making use of BN
#             # im_k, idx_unshuffle = self._batch_shuffle_ddp(im_k)
#
#             _, k = self.encoder_k(im_k)  # keys: NxC
#             k = nn.functional.normalize(k, dim=1)
#
#             # undo shuffle
#             # k = self._batch_unshuffle_ddp(k, idx_unshuffle)
#
#         # compute logits
#         # Einstein sum is more intuitive
#         # positive logits: Nx1
#         l_pos = torch.einsum("nc,nc->n", [q, k]).unsqueeze(-1)
#         # negative logits: NxK
#         l_neg = torch.einsum("nc,ck->nk", [q, self.queue.clone().detach()])
#
#         # logits: Nx(1+K)
#         logits = torch.cat([l_pos, l_neg], dim=1)
#
#         # apply temperature
#         logits /= self.T
#
#         # labels: positive key indicators
#         labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()
#
#         # dequeue and enqueue
#         self._dequeue_and_enqueue(k)
#
#         return logits, labels
#
#
# # utils
# @torch.no_grad()
# def concat_all_gather(tensor):
#     """
#     Performs all_gather operation on the provided tensors.
#     *** Warning ***: torch.distributed.all_gather has no gradient.
#     """
#     tensors_gather = [
#         torch.ones_like(tensor) for _ in range(torch.distributed.get_world_size())
#     ]
#     torch.distributed.all_gather(tensors_gather, tensor, async_op=False)
#
#     output = torch.cat(tensors_gather, dim=0)
#     return output
