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
    if torch.isnan(thing_loss):
        pdb.set_trace()

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

def conf_loss(input, target, neg_factor=10, neg_idx=80):
    ce_loss = F.cross_entropy(input, target, reduction="none")
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

def MatchDice(score_inst_sig, target_inst, score_conf_softmax, gt_classes):
    assert score_inst_sig.shape[-2] == target_inst.shape[-2] and score_inst_sig.shape[-1] == target_inst.shape[-1]
    h, w = target_inst.size(2), target_inst.size(3)
    with torch.no_grad():     
        score_inst_downsample = F.interpolate(input=score_inst_sig, size=(h//4, w//4), mode='bilinear') 
        if target_inst.shape[1]:
            target_inst_downsample = F.interpolate(input=target_inst.float(), size=(h//4, w//4), mode='nearest') 
        else:
            target_inst_downsample = target_inst.new_full((target_inst.shape[0],target_inst.shape[1],target_inst.shape[2]//4,target_inst.shape[3]//4), False)

        dim_flatten = target_inst_downsample.shape[1]*score_inst_downsample.shape[1]
        
        output_x = torch.reshape(score_inst_downsample.expand(target_inst_downsample.shape[1],-1,-1,-1), 
            (dim_flatten,score_inst_downsample.shape[2], score_inst_downsample.shape[3]))
        label_x = torch.reshape(target_inst_downsample.expand(score_inst_downsample.shape[1],-1,-1,-1).permute(1,0,2,3), (dim_flatten,target_inst_downsample.shape[2], target_inst_downsample.shape[3]))

        iou_flatten = dice_match(output_x.detach(), label_x.detach(), sigmoid_clip=True)
            
        iou_matrix = iou_flatten.view(target_inst_downsample.shape[1], score_inst_downsample.shape[1])
        if iou_matrix.shape[0]:
            iou_matrix += score_conf_softmax[0][:,gt_classes].permute(1,0).detach()
        row_ind, col_ind = linear_sum_assignment(-iou_matrix.cpu().numpy())
        return row_ind, col_ind

