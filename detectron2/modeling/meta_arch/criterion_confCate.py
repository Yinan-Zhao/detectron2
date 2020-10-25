# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Ke Sun (sunk@mail.ustc.edu.cn)
# ------------------------------------------------------------------------------

import torch
import torch.nn as nn
from torch.nn import functional as F
from scipy.optimize import linear_sum_assignment
from detectron2.utils.comm import get_world_size
import numpy as np

import pdb

def dice_loss(input, target, num_inst, background_channels=11, valid_mask=None, sigmoid_clip=False):
    if valid_mask is None:
        input = input.contiguous().view(input.size()[0], -1)
        target = target.contiguous().view(target.size()[0], -1).float()
    else:
        input = input.contiguous()[:,valid_mask]
        target = target.contiguous()[:,valid_mask].float()

    a_tmp = input * target
    b_tmp = input * input
    if sigmoid_clip:
        low_thresh = 0.3
        high_thresh = 0.7
        a_tmp[(target>high_thresh) & (input>high_thresh)] = 1
        a_tmp[(target<low_thresh) & (input<low_thresh)] = 0
        b_tmp[(target>high_thresh) & (input>high_thresh)] = 1
        b_tmp[(target<low_thresh) & (input<low_thresh)] = 0
    a = torch.sum(a_tmp, 1)
    b = torch.sum(b_tmp, 1)
    c = torch.sum(target * target, 1)

    d = (2*a+1) / (b+c+1)

    thing_loss = (1-d[background_channels:][c[background_channels:]!=0]).sum()
    thing_loss /= num_inst

    stuff_loss = (1-d[:background_channels]).mean()

    return stuff_loss, thing_loss


def dice_match(input, target, sigmoid_clip=False):
    smooth = 0.001
    if input.shape[0] == 0:
        return torch.empty(0)
    input = input.contiguous().view(input.size()[0], -1)
    target = target.contiguous().view(target.size()[0], -1).float()

    if sigmoid_clip:
        low_thresh = 0.3
        high_thresh = 0.7
        #input[(target>high_thresh) & (input>high_thresh)] = 1
        #input[(target<low_thresh) & (input<low_thresh)] = 0
        input[input>high_thresh] = 1
        input[input<low_thresh] = 0

    a = torch.sum(input * target, 1)
    b = torch.sum(input, 1)
    c = torch.sum(target, 1)

    d = (2*a+smooth) / (b+c+smooth)

    return d


def focal_loss(input, target, valid_mask=None, sigmoid_clip=False, alpha: float = 0.25, gamma: float = 2):
    """
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        alpha: (optional) Weighting factor in range (0,1) to balance
                positive vs negative examples. Default = -1 (no weighting).
        gamma: Exponent of the modulating factor (1 - p_t) to
               balance easy vs hard examples.
    Returns:
        Loss tensor
    """
    if valid_mask is None:
        input = input.contiguous().view(input.size()[0], -1)
        target = target.contiguous().view(target.size()[0], -1).float()
    else:
        input = input.contiguous()[:,valid_mask]
        target = target.contiguous()[:,valid_mask].float()

    prob = input
    ce_loss = F.binary_cross_entropy(prob, target, reduction="none")
    if sigmoid_clip:
        low_thresh = 0.3
        high_thresh = 0.7
        ce_loss[(target>high_thresh) & (input>high_thresh)] = 0
        ce_loss[(target<low_thresh) & (input<low_thresh)] = 0

    p_t = prob * target + (1 - prob) * (1 - target)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * target + (1 - alpha) * (1 - target)
        loss = alpha_t * loss

    loss_list = loss.mean(1) 
    loss_stuff = loss_list.mean()
    return loss_stuff

def conf_loss(input, target, neg_factor=10):
    ce_loss = F.cross_entropy(input, target, reduction="none")
    neg_idx = 8
    loss = ce_loss[target!=neg_idx].sum() + ce_loss[target==neg_idx].sum()/neg_factor
    loss = loss/ce_loss.shape[0]
    return loss

class CrossEntropy(nn.Module):
    def __init__(self, ignore_label=-1, weight=None):
        super(CrossEntropy, self).__init__()
        self.ignore_label = ignore_label
        self.criterion = nn.CrossEntropyLoss(weight=weight, 
                                             ignore_index=ignore_label)

    def forward(self, score, target):
        ph, pw = score.size(2), score.size(3)
        h, w = target.size(1), target.size(2)
        if ph != h or pw != w:
            score = F.upsample(
                    input=score, size=(h, w), mode='bilinear')

        loss = self.criterion(score, target)

        return loss


class MatchDiceConfCate(nn.Module):
    def __init__(self, ignore_label=-1, weight=None, background_channel=11, channel_shuffle=False, iou_use_smooth=True, sigmoid_clip=False, match_with_dice=False, no_focal_loss=False, focal_weight=1., factor_empty=5, conf_weight=5.):
        super(MatchDiceConfCate, self).__init__()
        self.ignore_label = ignore_label
        self.sigmoid_layer = nn.Sigmoid()
        self.softmax_layer = nn.Softmax(dim=2)
        self.background_channel = background_channel
        self.criterion_sem = CrossEntropy(ignore_label=ignore_label, weight=weight)
        self.channel_shuffle = channel_shuffle
        self.iou_use_smooth = iou_use_smooth
        self.sigmoid_clip = sigmoid_clip
        self.match_with_dice = match_with_dice
        self.no_focal_loss = no_focal_loss
        self.focal_weight = focal_weight
        self.factor_empty = factor_empty
        self.conf_weight = conf_weight

    def forward(self, score, target):
        target_inst, target_inst_sem_idx, target_sem, target_label, label_inst_num, label_background_num = target
        score_inst, score_conf = score
        ph, pw = score_inst.size(2), score_inst.size(3)
        h, w = target_inst.size(2), target_inst.size(3)

        if ph != h or pw != w:
            score_inst = F.upsample(
                    input=score_inst, size=(h, w), mode='bilinear')

        score_inst_sig = self.sigmoid_layer(score_inst)
        score_conf_sig = self.softmax_layer(score_conf)

        num_inst = sum(label_inst_num)
        num_inst = torch.as_tensor([num_inst], dtype=torch.float, device=score_inst.device)
        torch.distributed.all_reduce(num_inst)
        num_inst = torch.clamp(num_inst / get_world_size(), min=1).item()  

        with torch.no_grad():     
            score_inst_downsample = F.interpolate(input=score_inst_sig, size=(h//4, w//4), mode='bilinear') 
            target_inst_downsample = F.interpolate(input=target_inst.float(), size=(h//4, w//4), mode='nearest') 
            target_sem_downsample = F.interpolate(input=target_sem.float(), size=(h//4, w//4), mode='nearest') 

        #### loss ####
        loss_stuff_dice = 0.
        loss_thing_dice = 0.
        loss_stuff_focal = 0.
        loss_conf = 0.
        for b in range(score_inst_sig.shape[0]):
            score_inst_b = score_inst_sig[b:b+1,self.background_channel:] # 1 x C1 x H x W
            target_inst_b = target_inst[b:b+1,self.background_channel:self.background_channel+label_inst_num[b]] # 1 x C2 x H x W

            score_inst_downsample_b = score_inst_downsample[b:b+1,self.background_channel:]
            target_inst_downsample_b = target_inst_downsample[b:b+1,self.background_channel:self.background_channel+label_inst_num[b]]

            if self.channel_shuffle:
                rand_perm = np.random.permutation(score_inst_b.shape[1])
                score_inst_b = score_inst_b[:,rand_perm]
                score_inst_downsample_b = score_inst_downsample_b[:,rand_perm]

            dim_flatten = target_inst_downsample_b.shape[1]*score_inst_downsample_b.shape[1]
            with torch.no_grad():
                output_x = torch.reshape(score_inst_downsample_b.expand(target_inst_downsample_b.shape[1],-1,-1,-1), 
                    (dim_flatten,score_inst_downsample_b.shape[2], score_inst_downsample_b.shape[3]))
                label_x = torch.reshape(target_inst_downsample_b.expand(score_inst_downsample_b.shape[1],-1,-1,-1).permute(1,0,2,3), (dim_flatten,target_inst_downsample_b.shape[2], target_inst_downsample_b.shape[3]))
                if self.match_with_dice:
                    iou_flatten = dice_match(output_x.detach(), label_x.detach(), sigmoid_clip=self.sigmoid_clip)
                else:
                    iou_flatten = iou_pytorch(output_x.detach() > 0.5, label_x.detach() > 0.5, 
                        use_smooth=self.iou_use_smooth)
                iou_matrix = iou_flatten.view(target_inst_downsample_b.shape[1], score_inst_downsample_b.shape[1])
                if iou_matrix.shape[0]:
                    iou_matrix += score_conf_sig[b][:,target_inst_sem_idx[b,:label_inst_num[b]].long()].permute(1,0).detach()
                row_ind, col_ind = linear_sum_assignment(-iou_matrix.cpu().numpy())

            col_ind_empty = np.setdiff1d(np.arange(score_inst_b.shape[1]), col_ind)

            score_inst_sig_perm = torch.cat((score_inst_sig[b,:self.background_channel],
                                            score_inst_b[0,col_ind,:,:]),0)

            target_inst_perm = torch.cat((target_inst[b,:self.background_channel],
                                            target_inst_b[0,row_ind,:,:]),0).float()

            loss_stuff_dice_tmp, loss_thing_dice_tmp = dice_loss(score_inst_sig_perm,
                                                                target_inst_perm,
                                                                num_inst, 
                                                                background_channels=self.background_channel, 
                                                                valid_mask=None, 
                                                                sigmoid_clip=self.sigmoid_clip)

            target_conf = target_inst_sem_idx[b]
            loss_conf_tmp = conf_loss(torch.cat((score_conf[b,col_ind], score_conf[b,col_ind_empty]), 0), 
                                        target_conf.long(), 
                                        neg_factor=self.factor_empty)

            if self.no_focal_loss:
                loss_stuff_focal_tmp = 0
            else:
                loss_stuff_focal_tmp = focal_loss(score_inst_sig[b,:self.background_channel],
                                                target_inst[b,:self.background_channel].float(),
                                                valid_mask=None, 
                                                sigmoid_clip=self.sigmoid_clip)

            loss_stuff_dice += loss_stuff_dice_tmp
            loss_thing_dice += loss_thing_dice_tmp
            loss_stuff_focal += loss_stuff_focal_tmp
            loss_conf += loss_conf_tmp

        loss_stuff_focal = loss_stuff_focal / score_inst_sig.shape[0]
        loss_stuff_dice = loss_stuff_dice / score_inst_sig.shape[0]
        loss_conf = loss_conf / score_inst_sig.shape[0]

        loss = loss_stuff_focal*self.focal_weight + loss_stuff_dice + loss_thing_dice + loss_conf*self.conf_weight

        return loss/4.

