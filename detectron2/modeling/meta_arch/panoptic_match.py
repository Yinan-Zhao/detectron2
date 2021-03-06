# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
from torch import nn
import torch._utils
import torch.nn.functional as F

import os
import logging
import functools
import numpy as np

from detectron2.structures import ImageList, Boxes, Instances
from detectron2.structures.masks import polygons_to_bitmask
from detectron2.utils.comm import get_world_size

from ..postprocessing import detector_postprocess, sem_seg_postprocess
from .build import META_ARCH_REGISTRY
from .criterion_confCate import CrossEntropy, MatchDice, dice_loss, conf_loss, focal_loss

from PIL import Image

import pdb

BatchNorm2d = nn.BatchNorm2d
BN_MOMENTUM = 0.01
logger = logging.getLogger(__name__)

IGNORE_LABEL_SEM = 255
SIZE_DIVISIBILITY = 32
BACKGROUND_NUM = 53
FOREGROUND_NUM = 80

__all__ = ["PanopticMatch",
            "conv3x3",
            "BasicBlock",
            "Bottleneck",
            "HighResolutionModule",
            "HighResolutionNet",
            "blocks_dict"]

def is_dist_avail_and_initialized():
    if not torch.distributed.is_available():
        return False
    if not torch.distributed.is_initialized():
        return False
    return True


@META_ARCH_REGISTRY.register()
class PanopticMatch(nn.Module):
    """
    Implement the paper :paper:`PanopticFPN`.
    """

    def __init__(self, cfg):
        super().__init__()

        self.seg_model = get_seg_model(cfg)
        self.criterion_sem = CrossEntropy(ignore_label=IGNORE_LABEL_SEM-1)
        self.sigmoid_layer = nn.Sigmoid()
        self.softmax_layer = nn.Softmax(dim=2)

        self.register_buffer("pixel_mean", torch.Tensor(cfg.MODEL.PIXEL_MEAN).view(-1, 1, 1))
        self.register_buffer("pixel_std", torch.Tensor(cfg.MODEL.PIXEL_STD).view(-1, 1, 1))

    @property
    def device(self):
        return self.pixel_mean.device

    def forward(self, batched_inputs):
        """
        Args:
            batched_inputs: a list, batched outputs of :class:`DatasetMapper`.
                Each item in the list contains the inputs for one image.

                For now, each item in the list is a dict that contains:

                * "image": Tensor, image in (C, H, W) format.
                * "instances": Instances
                * "sem_seg": semantic segmentation ground truth.
                * Other information that's included in the original dicts, such as:
                  "height", "width" (int): the output resolution of the model, used in inference.
                  See :meth:`postprocess` for details.

        Returns:
            list[dict]:
                each dict is the results for one image. The dict contains the following keys:

                * "instances": see :meth:`GeneralizedRCNN.forward` for its format.
                * "sem_seg": see :meth:`SemanticSegmentor.forward` for its format.
                * "panoptic_seg": available when `PANOPTIC_FPN.COMBINE.ENABLED`.
                  See the return value of
                  :func:`combine_semantic_and_instance_outputs` for its format.
        """
        images = [x["image"].to(self.device) for x in batched_inputs]
        images = [(x - self.pixel_mean) / self.pixel_std for x in images]
        images = ImageList.from_tensors(images, SIZE_DIVISIBILITY)
        score_sem, score_inst, score_conf = self.seg_model(images.tensor)

        h, w = images.tensor.size(2), images.tensor.size(3)
        score_inst = F.upsample(
                input=score_inst, size=(h, w), mode='bilinear')
        score_sem = F.upsample(
                input=score_sem, size=(h, w), mode='bilinear')

        score_conf_softmax = self.softmax_layer(score_conf)

        score_inst_sig = self.sigmoid_layer(score_inst)
        score_inst_sig_stuff = score_inst_sig[:,:BACKGROUND_NUM]
        score_inst_sig_thing = score_inst_sig[:,BACKGROUND_NUM:]

        if "sem_seg" in batched_inputs[0]:
            gt_sem_seg = [x["sem_seg"].to(self.device) for x in batched_inputs]
            gt_sem_seg = ImageList.from_tensors(
                gt_sem_seg, SIZE_DIVISIBILITY, IGNORE_LABEL_SEM
            ).tensor
        else:
            gt_sem_seg = None
        
        if "instances" in batched_inputs[0]:
            gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
        else:
            gt_instances = None
        #pdb.set_trace()
        if self.training:
            assert (gt_sem_seg-1<0).sum() == 0
            sem_seg_losses = self.criterion_sem(score_sem, gt_sem_seg-1)

            gt_sem_seg[gt_sem_seg>BACKGROUND_NUM] = 0
            gt_stuff = F.one_hot(gt_sem_seg, num_classes=BACKGROUND_NUM+1).permute(0,3,1,2)
            gt_stuff = gt_stuff[:,1:]

            num_inst = sum([len(gt_instances[i]) for i in range(len(gt_instances))])
            num_inst = torch.as_tensor([num_inst], dtype=torch.float, device=self.device)
            if is_dist_avail_and_initialized():
                torch.distributed.all_reduce(num_inst)
            num_inst = torch.clamp(num_inst / get_world_size(), min=1).item()  

            loss_stuff_dice = 0.
            loss_thing_dice = 0.
            loss_stuff_focal = 0.
            loss_conf = 0.

            for i in range(len(batched_inputs)):
                gt_inst = gt_instances[i]
                gt_classes = gt_inst.gt_classes
                
                if gt_inst.has('gt_masks'):
                    gt_masks = gt_inst.gt_masks
                    masks = torch.stack([torch.from_numpy(polygons_to_bitmask(poly, gt_inst.image_size[0], gt_inst.image_size[1])).to(self.device) for poly in gt_masks.polygons], 0)
                    masks_pad = masks.new_full((masks.shape[0], images.tensor.shape[-2], images.tensor.shape[-1]), False)
                    masks_pad[:,:masks.shape[-2], :masks.shape[-1]].copy_(masks)
                else:
                    masks_pad = torch.zeros([0, images.tensor.shape[-2], images.tensor.shape[-1]], dtype=torch.bool, device=self.device)
                
                row_ind, col_ind = MatchDice(score_inst_sig_thing[i:i+1], torch.unsqueeze(masks_pad,0), score_conf_softmax[i:i+1], gt_classes)
                col_ind_empty = np.setdiff1d(np.arange(score_inst_sig_thing[i:i+1].shape[1]), col_ind)

                score_inst_sig_perm = torch.cat((score_inst_sig_stuff[i],
                                                score_inst_sig_thing[i,col_ind,:,:]),0)

                target_inst_perm = torch.cat((gt_stuff[i].float(),
                                            masks_pad[row_ind].float()),0)

                loss_stuff_dice_tmp, loss_thing_dice_tmp = dice_loss(score_inst_sig_perm,
                                                                    target_inst_perm,
                                                                    num_inst, 
                                                                    background_channels=BACKGROUND_NUM, 
                                                                    valid_mask=None, 
                                                                    sigmoid_clip=True)
                loss_stuff_dice += loss_stuff_dice_tmp
                loss_thing_dice += loss_thing_dice_tmp

                target_conf = gt_classes.new_full((score_conf.shape[1],), FOREGROUND_NUM)
                target_conf[:len(gt_classes[row_ind])] = gt_classes[row_ind]
                loss_conf_tmp = conf_loss(torch.cat((score_conf[i,col_ind], score_conf[i,col_ind_empty]), 0), 
                                            target_conf.long(), 
                                            neg_factor=10,
                                            neg_idx=FOREGROUND_NUM)
                loss_conf += loss_conf_tmp

                loss_stuff_focal_tmp = focal_loss(score_inst_sig_stuff[i],
                                                    gt_stuff[i].float(),
                                                    valid_mask=None, 
                                                    sigmoid_clip=True)
                loss_stuff_focal += loss_stuff_focal_tmp

            loss_stuff_focal = loss_stuff_focal / len(batched_inputs)
            loss_stuff_dice = loss_stuff_dice / len(batched_inputs)
            loss_conf = loss_conf / len(batched_inputs)

            loss_stuff_focal = loss_stuff_focal*100.
            loss_conf = loss_conf*5

        
            losses = {}
            losses.update({"loss_sem_seg": sem_seg_losses})
            losses.update({"loss_stuff_focal": loss_stuff_focal})
            losses.update({"loss_stuff_dice": loss_stuff_dice})
            losses.update({"loss_thing_dice": loss_thing_dice})
            losses.update({"loss_conf": loss_conf})
            return losses

        score_sem_null = score_sem.new_full((score_sem.shape[0], 1, score_sem.shape[-2], score_sem.shape[-1]), -1000.)
        processed_results = []
        for i in range(len(batched_inputs)):
            height = batched_inputs[i].get("height", images.image_sizes[i][0])
            width = batched_inputs[i].get("width", images.image_sizes[i][1])

            score_inst_sig_stuff_b = F.interpolate(
                score_inst_sig_stuff[i:i+1,:,:images.image_sizes[i][0],:images.image_sizes[i][1]], size=(height, width), mode="bilinear", align_corners=False
            )
            score_inst_sig_thing_b = F.interpolate(
                score_inst_sig_thing[i:i+1,:,:images.image_sizes[i][0],:images.image_sizes[i][1]], size=(height, width), mode="bilinear", align_corners=False
            )

            img_name = os.path.basename(batched_inputs[i]['file_name'])
            img_name_split = img_name.split('.')
            save_dir = '/home/yz9244/detectron2/output/vis_inst_sig'
            for j in range(80):
                pred_inst_tmp = np.asarray(255*(score_inst_sig_thing_b[0,j].cpu().numpy()), dtype=np.uint8)
                img = Image.fromarray(pred_inst_tmp)
                save_img = Image.new('RGB', (img.width, 2*img.height))
                                
                img = Image.fromarray(pred_inst_tmp)
                save_img.paste(img, (0, 0))

                pred_inst_tmp = np.asarray(255*(pred_inst_tmp>127), dtype=np.uint8)
                img = Image.fromarray(pred_inst_tmp)
                save_img.paste(img, (0, img.height))
                save_img.save(os.path.join(save_dir, img_name_split[0]+'_%02d.png'%(j)))

            res = {}

            score_sem_foreground = torch.log(torch.exp(score_sem[i:i+1,BACKGROUND_NUM:]).sum(dim=1,keepdim=True))
            sem_seg_result = torch.cat((score_sem_foreground,score_sem[i:i+1,:BACKGROUND_NUM]), 1)
            sem_seg_r = sem_seg_postprocess(sem_seg_result[0], images.image_sizes[i], height, width)

            res.update({"sem_seg": sem_seg_r})

            result = Instances((height, width))
            inst_sem_id = torch.argmax(score_conf_softmax[i], dim=1)
            scores = score_conf_softmax[i,range(score_conf.shape[1]),inst_sem_id]
            scores = scores[inst_sem_id!=FOREGROUND_NUM]
            pred_classes = inst_sem_id[inst_sem_id!=FOREGROUND_NUM]
            pred_masks = score_inst_sig_thing_b[0,inst_sem_id!=FOREGROUND_NUM]

            pred_mask_sum = torch.sum(pred_masks>0.5, (1,2))
            result.pred_masks = pred_masks[pred_mask_sum>0] > 0.5
            result.pred_classes = pred_classes[pred_mask_sum>0]
            result.scores = scores[pred_mask_sum>0]

            box_tmp = torch.zeros(result.pred_masks.shape[0],4)
            for j in range(result.pred_masks.shape[0]):
                nonzero_idx = torch.nonzero(result.pred_masks[j])
                box_tmp[j,0] = nonzero_idx[:,1].min().item()
                box_tmp[j,2] = nonzero_idx[:,1].max().item()
                box_tmp[j,1] = nonzero_idx[:,0].min().item()
                box_tmp[j,3] = nonzero_idx[:,0].max().item()
            result.pred_boxes = Boxes(box_tmp)

            #detector_r = detector_postprocess(result, height, width)
            detector_r = result
            res.update({"instances": detector_r})

            panoptic_r = combine_semantic_and_instance_outputs(
                    result.scores,
                    result.pred_classes,
                    pred_masks[pred_mask_sum>0],
                    score_inst_sig_stuff_b[0]
                )
            res.update({"panoptic_seg": panoptic_r})

            processed_results.append(res)

        return processed_results


def combine_semantic_and_instance_outputs(
    scores,
    pred_classes,
    score_inst_sig_thing,
    score_inst_sig_stuff,
):
    """
    Implement a simple combining logic following
    "combine_semantic_and_instance_predictions.py" in panopticapi
    to produce panoptic segmentation outputs.

    Args:
        instance_results: output of :func:`detector_postprocess`.
        semantic_results: an (H, W) tensor, each is the contiguous semantic
            category id

    Returns:
        panoptic_seg (Tensor): of shape (height, width) where the values are ids for each segment.
        segments_info (list[dict]): Describe each segment in `panoptic_seg`.
            Each dict contains keys "id", "category_id", "isthing".
    """
    score_inst_sig_thing = score_inst_sig_thing*scores[:,None,None]
    score_inst_sig = torch.cat((score_inst_sig_stuff, score_inst_sig_thing), 0)

    score_inst_idx = torch.argmax(score_inst_sig, dim=0)
    segmentIds = torch.unique(score_inst_idx)

    panoptic_seg = torch.zeros((score_inst_sig.shape[-2], score_inst_sig.shape[-1]), dtype=torch.int32)
    segments_info = []
    current_segment_id = 0

    for segmentId in segmentIds:
        current_segment_id += 1
        segmentId = segmentId.item()
        if segmentId < BACKGROUND_NUM:
            panoptic_seg[score_inst_idx == segmentId] = current_segment_id
            segments_info.append(
            {
                "id": current_segment_id,
                "isthing": False,
                "category_id": segmentId+1,})
        else:
            panoptic_seg[score_inst_idx == segmentId] = current_segment_id
            category_id = pred_classes[segmentId-BACKGROUND_NUM]
            segments_info.append(
            {
                "id": current_segment_id,
                "isthing": True,
                "category_id": category_id.item(),})

    return panoptic_seg, segments_info


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1,
                               bias=False)
        self.bn3 = BatchNorm2d(planes * self.expansion,
                               momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class HighResolutionModule(nn.Module):
    def __init__(self, num_branches, blocks, num_blocks, num_inchannels,
                 num_channels, fuse_method, multi_scale_output=True):
        super(HighResolutionModule, self).__init__()
        self._check_branches(
            num_branches, blocks, num_blocks, num_inchannels, num_channels)

        self.num_inchannels = num_inchannels
        self.fuse_method = fuse_method
        self.num_branches = num_branches

        self.multi_scale_output = multi_scale_output

        self.branches = self._make_branches(
            num_branches, blocks, num_blocks, num_channels)
        self.fuse_layers = self._make_fuse_layers()
        self.relu = nn.ReLU(inplace=True)

    def _check_branches(self, num_branches, blocks, num_blocks,
                        num_inchannels, num_channels):
        if num_branches != len(num_blocks):
            error_msg = 'NUM_BRANCHES({}) <> NUM_BLOCKS({})'.format(
                num_branches, len(num_blocks))
            logger.error(error_msg)
            raise ValueError(error_msg)

        if num_branches != len(num_channels):
            error_msg = 'NUM_BRANCHES({}) <> NUM_CHANNELS({})'.format(
                num_branches, len(num_channels))
            logger.error(error_msg)
            raise ValueError(error_msg)

        if num_branches != len(num_inchannels):
            error_msg = 'NUM_BRANCHES({}) <> NUM_INCHANNELS({})'.format(
                num_branches, len(num_inchannels))
            logger.error(error_msg)
            raise ValueError(error_msg)

    def _make_one_branch(self, branch_index, block, num_blocks, num_channels,
                         stride=1):
        downsample = None
        if stride != 1 or \
           self.num_inchannels[branch_index] != num_channels[branch_index] * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.num_inchannels[branch_index],
                          num_channels[branch_index] * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                BatchNorm2d(num_channels[branch_index] * block.expansion,
                            momentum=BN_MOMENTUM),
            )

        layers = []
        layers.append(block(self.num_inchannels[branch_index],
                            num_channels[branch_index], stride, downsample))
        self.num_inchannels[branch_index] = \
            num_channels[branch_index] * block.expansion
        for i in range(1, num_blocks[branch_index]):
            layers.append(block(self.num_inchannels[branch_index],
                                num_channels[branch_index]))

        return nn.Sequential(*layers)

    def _make_branches(self, num_branches, block, num_blocks, num_channels):
        branches = []

        for i in range(num_branches):
            branches.append(
                self._make_one_branch(i, block, num_blocks, num_channels))

        return nn.ModuleList(branches)

    def _make_fuse_layers(self):
        if self.num_branches == 1:
            return None

        num_branches = self.num_branches
        num_inchannels = self.num_inchannels
        fuse_layers = []
        for i in range(num_branches if self.multi_scale_output else 1):
            fuse_layer = []
            for j in range(num_branches):
                if j > i:
                    fuse_layer.append(nn.Sequential(
                        nn.Conv2d(num_inchannels[j],
                                  num_inchannels[i],
                                  1,
                                  1,
                                  0,
                                  bias=False),
                        BatchNorm2d(num_inchannels[i], momentum=BN_MOMENTUM)))
                elif j == i:
                    fuse_layer.append(None)
                else:
                    conv3x3s = []
                    for k in range(i-j):
                        if k == i - j - 1:
                            num_outchannels_conv3x3 = num_inchannels[i]
                            conv3x3s.append(nn.Sequential(
                                nn.Conv2d(num_inchannels[j],
                                          num_outchannels_conv3x3,
                                          3, 2, 1, bias=False),
                                BatchNorm2d(num_outchannels_conv3x3, 
                                            momentum=BN_MOMENTUM)))
                        else:
                            num_outchannels_conv3x3 = num_inchannels[j]
                            conv3x3s.append(nn.Sequential(
                                nn.Conv2d(num_inchannels[j],
                                          num_outchannels_conv3x3,
                                          3, 2, 1, bias=False),
                                BatchNorm2d(num_outchannels_conv3x3,
                                            momentum=BN_MOMENTUM),
                                nn.ReLU(inplace=True)))
                    fuse_layer.append(nn.Sequential(*conv3x3s))
            fuse_layers.append(nn.ModuleList(fuse_layer))

        return nn.ModuleList(fuse_layers)

    def get_num_inchannels(self):
        return self.num_inchannels

    def forward(self, x):
        if self.num_branches == 1:
            return [self.branches[0](x[0])]

        for i in range(self.num_branches):
            x[i] = self.branches[i](x[i])

        x_fuse = []
        for i in range(len(self.fuse_layers)):
            y = x[0] if i == 0 else self.fuse_layers[i][0](x[0])
            for j in range(1, self.num_branches):
                if i == j:
                    y = y + x[j]
                elif j > i:
                    width_output = x[i].shape[-1]
                    height_output = x[i].shape[-2]
                    y = y + F.interpolate(
                        self.fuse_layers[i][j](x[j]),
                        size=[height_output, width_output],
                        mode='bilinear')
                else:
                    y = y + self.fuse_layers[i][j](x[j])
            x_fuse.append(self.relu(y))

        return x_fuse


blocks_dict = {
    'BASIC': BasicBlock,
    'BOTTLENECK': Bottleneck
}


class HighResolutionNet(nn.Module):

    def __init__(self, config, **kwargs):
        extra = config.MODEL.HRNET
        super(HighResolutionNet, self).__init__()

        '''x_linspace = np.linspace(-1, 1, 256)
        y_linspace = np.linspace(-1, 1, 128)
        xv, yv = np.meshgrid(x_linspace, y_linspace)
        self.rela_coor = np.array([xv, yv], dtype=np.float32)'''

        # stem net
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1,
                               bias=False)
        self.bn1 = BatchNorm2d(64, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1,
                               bias=False)
        self.bn2 = BatchNorm2d(64, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        
        self.stage1_cfg = extra['STAGE1']
        num_channels = self.stage1_cfg['NUM_CHANNELS'][0]
        block = blocks_dict[self.stage1_cfg['BLOCK']]
        num_blocks = self.stage1_cfg['NUM_BLOCKS'][0]
        self.layer1 = self._make_layer(block, 64, num_channels, num_blocks)
        stage1_out_channel = block.expansion*num_channels

        self.stage2_cfg = extra['STAGE2']
        num_channels = self.stage2_cfg['NUM_CHANNELS']
        block = blocks_dict[self.stage2_cfg['BLOCK']]
        num_channels = [
            num_channels[i] * block.expansion for i in range(len(num_channels))]
        self.transition1 = self._make_transition_layer(
            [stage1_out_channel], num_channels)
        self.stage2, pre_stage_channels = self._make_stage(
            self.stage2_cfg, num_channels)

        self.stage3_cfg = extra['STAGE3']
        num_channels = self.stage3_cfg['NUM_CHANNELS']
        block = blocks_dict[self.stage3_cfg['BLOCK']]
        num_channels = [
            num_channels[i] * block.expansion for i in range(len(num_channels))]
        self.transition2 = self._make_transition_layer(
            pre_stage_channels, num_channels)
        self.stage3, pre_stage_channels = self._make_stage(
            self.stage3_cfg, num_channels)

        self.stage4_cfg = extra['STAGE4']
        num_channels = self.stage4_cfg['NUM_CHANNELS']
        block = blocks_dict[self.stage4_cfg['BLOCK']]
        num_channels = [
            num_channels[i] * block.expansion for i in range(len(num_channels))]
        self.transition3 = self._make_transition_layer(
            pre_stage_channels, num_channels)
        self.stage4, pre_stage_channels = self._make_stage(
            self.stage4_cfg, num_channels, multi_scale_output=True)
        
        last_inp_channels = np.int(np.sum(pre_stage_channels))

        self.last_layer_inst = nn.Sequential(
            nn.Conv2d(
                in_channels=last_inp_channels,
                out_channels=last_inp_channels,
                kernel_size=3,
                stride=1,
                padding=1),
            BatchNorm2d(last_inp_channels, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels=last_inp_channels,
                out_channels=last_inp_channels,
                kernel_size=3,
                stride=1,
                padding=1),
            BatchNorm2d(last_inp_channels, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels=last_inp_channels,
                out_channels=last_inp_channels,
                kernel_size=3,
                stride=1,
                padding=1),
            BatchNorm2d(last_inp_channels, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels=last_inp_channels,
                out_channels=extra.NUM_INSTANCES,
                kernel_size=extra.FINAL_CONV_KERNEL,
                stride=1,
                padding=1 if extra.FINAL_CONV_KERNEL == 3 else 0)
        )

        self.last_layer_background = nn.Sequential(
            nn.Conv2d(
                in_channels=last_inp_channels,
                out_channels=last_inp_channels,
                kernel_size=1,
                stride=1,
                padding=0),
            BatchNorm2d(last_inp_channels, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels=last_inp_channels,
                out_channels=BACKGROUND_NUM,
                kernel_size=extra.FINAL_CONV_KERNEL,
                stride=1,
                padding=1 if extra.FINAL_CONV_KERNEL == 3 else 0)
        )

        self.last_layer_sem = nn.Sequential(
            nn.Conv2d(
                in_channels=last_inp_channels,
                out_channels=last_inp_channels,
                kernel_size=1,
                stride=1,
                padding=0),
            BatchNorm2d(last_inp_channels, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels=last_inp_channels,
                out_channels=BACKGROUND_NUM+FOREGROUND_NUM,
                kernel_size=extra.FINAL_CONV_KERNEL,
                stride=1,
                padding=1 if extra.FINAL_CONV_KERNEL == 3 else 0)
        )

        self.last_layer_conf_conv = nn.Sequential(
            nn.Conv2d(
                in_channels=last_inp_channels,
                out_channels=2*last_inp_channels,
                kernel_size=3,
                stride=2,
                padding=1),
            BatchNorm2d(2*last_inp_channels, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels=2*last_inp_channels,
                out_channels=4*last_inp_channels,
                kernel_size=3,
                stride=2,
                padding=1),
            BatchNorm2d(4*last_inp_channels, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True),
        )

        self.num_instance = extra.NUM_INSTANCES
        self.last_layer_conf_fc = nn.Linear(4*last_inp_channels, extra.NUM_INSTANCES*(FOREGROUND_NUM+1))

    def _make_transition_layer(
            self, num_channels_pre_layer, num_channels_cur_layer):
        num_branches_cur = len(num_channels_cur_layer)
        num_branches_pre = len(num_channels_pre_layer)

        transition_layers = []
        for i in range(num_branches_cur):
            if i < num_branches_pre:
                if num_channels_cur_layer[i] != num_channels_pre_layer[i]:
                    transition_layers.append(nn.Sequential(
                        nn.Conv2d(num_channels_pre_layer[i],
                                  num_channels_cur_layer[i],
                                  3,
                                  1,
                                  1,
                                  bias=False),
                        BatchNorm2d(
                            num_channels_cur_layer[i], momentum=BN_MOMENTUM),
                        nn.ReLU(inplace=True)))
                else:
                    transition_layers.append(None)
            else:
                conv3x3s = []
                for j in range(i+1-num_branches_pre):
                    inchannels = num_channels_pre_layer[-1]
                    outchannels = num_channels_cur_layer[i] \
                        if j == i-num_branches_pre else inchannels
                    conv3x3s.append(nn.Sequential(
                        nn.Conv2d(
                            inchannels, outchannels, 3, 2, 1, bias=False),
                        BatchNorm2d(outchannels, momentum=BN_MOMENTUM),
                        nn.ReLU(inplace=True)))
                transition_layers.append(nn.Sequential(*conv3x3s))

        return nn.ModuleList(transition_layers)

    def _make_layer(self, block, inplanes, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                BatchNorm2d(planes * block.expansion, momentum=BN_MOMENTUM),
            )

        layers = []
        layers.append(block(inplanes, planes, stride, downsample))
        inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(inplanes, planes))

        return nn.Sequential(*layers)

    def _make_stage(self, layer_config, num_inchannels,
                    multi_scale_output=True):
        num_modules = layer_config['NUM_MODULES']
        num_branches = layer_config['NUM_BRANCHES']
        num_blocks = layer_config['NUM_BLOCKS']
        num_channels = layer_config['NUM_CHANNELS']
        block = blocks_dict[layer_config['BLOCK']]
        fuse_method = layer_config['FUSE_METHOD']

        modules = []
        for i in range(num_modules):
            # multi_scale_output is only used last module
            if not multi_scale_output and i == num_modules - 1:
                reset_multi_scale_output = False
            else:
                reset_multi_scale_output = True
            modules.append(
                HighResolutionModule(num_branches,
                                      block,
                                      num_blocks,
                                      num_inchannels,
                                      num_channels,
                                      fuse_method,
                                      reset_multi_scale_output)
            )
            num_inchannels = modules[-1].get_num_inchannels()

        return nn.Sequential(*modules), num_inchannels

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.layer1(x)

        x_list = []
        for i in range(self.stage2_cfg['NUM_BRANCHES']):
            if self.transition1[i] is not None:
                x_list.append(self.transition1[i](x))
            else:
                x_list.append(x)
        y_list = self.stage2(x_list)

        x_list = []
        for i in range(self.stage3_cfg['NUM_BRANCHES']):
            if self.transition2[i] is not None:
                x_list.append(self.transition2[i](y_list[-1]))
            else:
                x_list.append(y_list[i])
        y_list = self.stage3(x_list)

        x_list = []
        for i in range(self.stage4_cfg['NUM_BRANCHES']):
            if self.transition3[i] is not None:
                x_list.append(self.transition3[i](y_list[-1]))
            else:
                x_list.append(y_list[i])

        x = self.stage4(x_list)

        # Upsampling
        x0_h, x0_w = x[0].size(2), x[0].size(3)
        x1 = F.upsample(x[1], size=(x0_h, x0_w), mode='bilinear')
        x2 = F.upsample(x[2], size=(x0_h, x0_w), mode='bilinear')
        x3 = F.upsample(x[3], size=(x0_h, x0_w), mode='bilinear')

        x = torch.cat([x[0], x1, x2, x3], 1)

        x_sem = self.last_layer_sem(x)
        x_background = self.last_layer_background(x)
        x_inst = self.last_layer_inst(x)
        x_inst -= 2.
        x_inst = torch.cat((x_background, x_inst), dim=1)

        x_conf = self.last_layer_conf_conv(x)
        x_conf = torch.mean(x_conf,(2,3))
        x_conf = self.last_layer_conf_fc(x_conf)
        x_conf = x_conf.view(x_conf.shape[0], self.num_instance, FOREGROUND_NUM+1)

        return x_sem, x_inst, x_conf

    def init_weights(self, pretrained='',):
        logger.info('=> init weights from normal distribution')
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.001)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        if os.path.isfile(pretrained):
            pretrained_dict = torch.load(pretrained)
            logger.info('=> loading pretrained model {}'.format(pretrained))
            model_dict = self.state_dict()
            if 'state_dict' in pretrained_dict.keys():
                pretrained_dict = pretrained_dict['state_dict']
                pretrained_dict = {k[6:]: v for k, v in pretrained_dict.items()
                            if k[6:] in model_dict.keys()}
            else:
                pretrained_dict = {k: v for k, v in pretrained_dict.items()
                                   if k in model_dict.keys()}
            #for k, _ in pretrained_dict.items():
            #    logger.info(
            #        '=> loading {} pretrained model {}'.format(k, pretrained))
            model_dict.update(pretrained_dict)
            self.load_state_dict(model_dict)

def get_seg_model(cfg, **kwargs):
    model = HighResolutionNet(cfg, **kwargs)
    model.init_weights(cfg.MODEL.HRNET.PRETRAINED)

    return model
