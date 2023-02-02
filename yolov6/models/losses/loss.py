#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from yolov6.assigners.anchor_generator import generate_anchors
from yolov6.utils.general import dist2bbox, bbox2dist, dist2cor, cor2dist, xywh2xyxy, box_iou
from yolov6.utils.figure_iou import IOUloss
from yolov6.assigners.atss_assigner import ATSSAssigner
from yolov6.assigners.tal_assigner import TaskAlignedAssigner

class ComputeLoss:
    '''Loss computation func.'''
    def __init__(self, 
                 fpn_strides=[8, 16, 32],
                 grid_cell_size=5.0,
                 grid_cell_offset=0.5,
                 npro=31,
                 nalp=24,
                 nads=37,
                 ori_img_size=640,
                 warmup_epoch=4,
                 use_dfl=True,
                 reg_max=16,
                 iou_type='giou',
                 loss_weight={
                     'class': 3.0,
                     'iou': 2.5,
                     'corner': 1.0,
                     'dfl': 0.5}
                 ):
        
        self.fpn_strides = fpn_strides
        self.grid_cell_size = grid_cell_size
        self.grid_cell_offset = grid_cell_offset
        self.npro = npro
        self.nalp = nalp
        self.nads = nads
        self.ori_img_size = ori_img_size
        
        self.warmup_epoch = warmup_epoch
        self.warmup_assigner = ATSSAssigner(9, npro=self.npro, nalp=self.nalp, nads=self.nads)
        self.formal_assigner = ATSSAssigner(9, npro=self.npro, nalp=self.nalp, nads=self.nads)
        #self.formal_assigner = TaskAlignedAssigner(topk=13, npro=self.npro, nalp=self.nalp, nads=self.nads, alpha=1.0, beta=6.0)

        self.use_dfl = use_dfl
        self.reg_max = reg_max
        self.proj = nn.Parameter(torch.linspace(0, self.reg_max, self.reg_max + 1), requires_grad=False)
        self.iou_type = iou_type
        self.varifocal_loss = VarifocalLoss().cuda()
        self.bbox_loss = BboxLoss(self.reg_max, self.use_dfl, self.iou_type).cuda()
        self.corner_loss = CornerLoss().cuda()
        self.loss_weight = loss_weight       
        
    def __call__(
        self,
        outputs,
        targets,
        epoch_num,
        step_num
    ):
        feats, pred_pro_scores, pred_alp_scores, \
        pred_ad0_scores, pred_ad1_scores, pred_ad2_scores, \
        pred_ad3_scores, pred_ad4_scores, pred_ad5_scores, pred_reg_distri, pred_cor_distri = outputs
        pred_ads_scores = [pred_ad0_scores, pred_ad1_scores, pred_ad2_scores, pred_ad3_scores, pred_ad4_scores, pred_ad5_scores]

        anchors, anchor_points, n_anchors_list, stride_tensor = \
               generate_anchors(feats, self.fpn_strides, self.grid_cell_size, self.grid_cell_offset, device=feats[0].device)
   
        assert pred_pro_scores.type() == pred_reg_distri.type()
        gt_point_scale = torch.full((1,12), self.ori_img_size).type_as(pred_pro_scores)
        batch_size = pred_pro_scores.shape[0]

        # targets
        targets =self.preprocess(targets, batch_size, gt_point_scale)
        gt_pro = targets[:, :, 0]
        gt_alp = targets[:, :, 1]
        gt_ads = targets[:, :, 2:8]
        gt_bboxes = targets[:, :, 8:12] #xyxy
        gt_corners = targets[:, :, 12:]
        mask_gt = (gt_bboxes.sum(-1, keepdim=True) > 0).float()

        
        # pboxes
        anchor_points_s = anchor_points / stride_tensor
        pred_bboxes = self.bbox_decode(anchor_points_s, pred_reg_distri) #xyxy
        pred_corners = self.corner_decode(anchor_points_s, pred_cor_distri)

        try:
            target_pro, target_alp, target_ads, target_bboxes, target_corners, target_pro_scores, target_alp_scores, target_ads_scores, fg_mask = \
                    self.formal_assigner(
                        anchors,
                        n_anchors_list,
                        gt_pro,
                        gt_alp,
                        gt_ads,
                        gt_bboxes,
                        gt_corners,
                        mask_gt,
                        pred_bboxes.detach() * stride_tensor)
            # if epoch_num < self.warmup_epoch:
            #     target_pro, target_alp, target_ads, target_bboxes, target_corners, target_pro_scores, target_alp_scores, target_ads_scores, fg_mask = \
            #         self.warmup_assigner(
            #             anchors,
            #             n_anchors_list,
            #             gt_pro,
            #             gt_alp,
            #             gt_ads,
            #             gt_bboxes,
            #             gt_corners,
            #             mask_gt,
            #             pred_bboxes.detach() * stride_tensor)
            # else:
            #     target_pro, target_alp, target_ads, target_bboxes, target_corners, target_pro_scores, target_alp_scores, target_ads_scores, fg_mask = \
            #         self.formal_assigner(
            #             pred_pro_scores.detach(),
            #             pred_alp_scores.detach(),
            #             pred_ads_scores,
            #             pred_bboxes.detach() * stride_tensor,
            #             pred_corners.detach() * stride_tensor,
            #             anchor_points,
            #             gt_pro,
            #             gt_alp,
            #             gt_ads,
            #             gt_bboxes,
            #             gt_corners,
            #             mask_gt)

        except RuntimeError:
            print(
                "OOM RuntimeError is raised due to the huge memory cost during label assignment. \
                    CPU mode is applied in this batch. If you want to avoid this issue, \
                    try to reduce the batch size or image size."
            )
            torch.cuda.empty_cache()
            print("------------CPU Mode for This Batch-------------")
            _anchors = anchors.cpu().float()
            _n_anchors_list = n_anchors_list
            _gt_pro = gt_pro.cpu().float()
            _gt_alp = gt_alp.cpu().float()
            _gt_ads = gt_ads.cpu().float()
            _gt_bboxes = gt_bboxes.cpu().float()
            _gt_corners = gt_corners.cpu().float()
            _mask_gt = mask_gt.cpu().float()
            _pred_bboxes = pred_bboxes.detach().cpu().float()
            _stride_tensor = stride_tensor.cpu().float()

            target_pro, target_alp, target_ads, target_bboxes, target_corners, target_pro_scores, target_alp_scores, target_ads_scores, fg_mask = \
                self.formal_assigner(
                    _anchors,
                    _n_anchors_list,
                    _gt_pro,
                    _gt_alp,
                    _gt_ads,
                    _gt_bboxes,
                    _gt_corners,
                    _mask_gt,
                    _pred_bboxes * _stride_tensor)
            # if epoch_num < self.warmup_epoch:
            #     _anchors = anchors.cpu().float()
            #     _n_anchors_list = n_anchors_list
            #     _gt_pro = gt_pro.cpu().float()
            #     _gt_alp = gt_alp.cpu().float()
            #     _gt_ads = gt_ads.cpu().float()
            #     _gt_bboxes = gt_bboxes.cpu().float()
            #     _gt_corners = gt_corners.cpu().float()
            #     _mask_gt = mask_gt.cpu().float()
            #     _pred_bboxes = pred_bboxes.detach().cpu().float()
            #     _stride_tensor = stride_tensor.cpu().float()

            #     target_pro, target_alp, target_ads, target_bboxes, target_corners, target_pro_scores, target_alp_scores, target_ads_scores, fg_mask = \
            #         self.warmup_assigner(
            #             _anchors,
            #             _n_anchors_list,
            #             _gt_pro,
            #             _gt_alp,
            #             _gt_ads,
            #             _gt_bboxes,
            #             _gt_corners,
            #             _mask_gt,
            #             _pred_bboxes * _stride_tensor)

            # else:
            #     _pred_pro_scores = pred_pro_scores.detach().cpu().float()
            #     _pred_alp_scores = pred_alp_scores.detach().cpu().float()
            #     _pred_ads_scores = [pred_ads_scores[i].cpu().float() for i in range(5)]
            #     _pred_bboxes = pred_bboxes.detach().cpu().float()
            #     _pred_corners = pred_corners.detach().cpu().float()
            #     _anchor_points = anchor_points.cpu().float()
            #     _gt_pro = gt_pro.cpu().float()
            #     _gt_alp = gt_alp.cpu().float()
            #     _gt_ads = gt_ads.cpu().float()
            #     _gt_bboxes = gt_bboxes.cpu().float()
            #     _gt_corners = gt_corners.cpu().float()
            #     _mask_gt = mask_gt.cpu().float()
            #     _stride_tensor = stride_tensor.cpu().float()

            #     target_pro, target_alp, target_ads, target_bboxes, target_corners, target_pro_scores, target_alp_scores, target_ads_scores, fg_mask = \
            #         self.formal_assigner(
            #             _pred_pro_scores,
            #             _pred_alp_scores,
            #             _pred_ads_scores,
            #             _pred_bboxes * _stride_tensor,
            #             _pred_corners * _stride_tensor,
            #             _anchor_points,
            #             _gt_pro,
            #             _gt_alp,
            #             _gt_ads,
            #             _gt_bboxes,
            #             _gt_corners,
            #             _mask_gt)

            target_pro = target_pro.cuda()
            target_alp = target_alp.cuda()
            for i in range(6):
                target_ads[i] = target_ads[i].cuda()
                target_ads_scores[i] = target_ads_scores[i].cuda()
            target_bboxes = target_bboxes.cuda()
            target_corners = target_corners.cuda()
            target_pro_scores = target_pro_scores.cuda()
            target_alp_scores = target_alp_scores.cuda()
            fg_mask = fg_mask.cuda()
        #Dynamic release GPU memory
        if step_num % 10 == 0:
            torch.cuda.empty_cache()

        # rescale bbox
        target_bboxes /= stride_tensor
        target_corners /= stride_tensor
        
        # pro loss
        target_pro = torch.where(fg_mask > 0, target_pro, torch.full_like(target_pro, self.npro))
        one_hot_pro = F.one_hot(target_pro.long(), self.npro + 1)[..., :-1]
        loss_pro = self.varifocal_loss(pred_pro_scores, target_pro_scores, one_hot_pro)

        # alp loss
        target_alp = torch.where(fg_mask > 0, target_alp, torch.full_like(target_alp, self.nalp))
        one_hot_alp = F.one_hot(target_alp.long(), self.nalp + 1)[..., :-1]
        loss_alp = self.varifocal_loss(pred_alp_scores, target_alp_scores, one_hot_alp)

        # ads loss
        loss_ads_list = []
        for i in range(6):
            target_1_ads = torch.where(fg_mask > 0, target_ads[i], torch.full_like(target_ads[i], self.nads))
            one_hot_1_ads = F.one_hot(target_1_ads.long(), self.nads + 1)[..., :-1]
            loss_ads_list.append(self.varifocal_loss(pred_ads_scores[i], target_ads_scores[i], one_hot_1_ads))
            
        target_pro_scores_sum = target_pro_scores.sum()
		# avoid devide zero error, devide by zero will cause loss to be inf or nan.
        # if target_scores_sum is 0, loss_cls equals to 0 alson 
        if target_pro_scores_sum > 0:
            loss_pro /= target_pro_scores_sum
            
        target_alp_scores_sum = target_alp_scores.sum()
        if target_alp_scores_sum > 0:
            loss_alp /= target_alp_scores_sum
        
        total_target_ads_scores_sum = 0
        for i in range(6):
            target_ads_scores_sum = target_ads_scores[i].sum()
            total_target_ads_scores_sum += target_ads_scores_sum
            if target_ads_scores_sum > 0:
                loss_ads_list[i] = loss_ads_list[i] / target_ads_scores_sum
        loss_ads = loss_ads_list[0]
        for i in range(1, 6):
            loss_ads += loss_ads_list[i]

        loss_cls = (loss_pro + loss_alp + loss_ads) / 8.0

        target_scores_sum = (target_pro_scores_sum + target_alp_scores_sum + total_target_ads_scores_sum) / 8.0
        
        # bbox loss
        loss_iou, loss_dfl = self.bbox_loss(pred_reg_distri, pred_bboxes, anchor_points_s, target_bboxes,
                                            target_pro_scores, target_alp_scores, target_ads_scores, target_scores_sum, fg_mask)

        # corner loss
        loss_cor = self.corner_loss(pred_corners, target_corners, target_scores_sum, fg_mask)
        
        loss = self.loss_weight['class'] * loss_cls + \
               self.loss_weight['iou'] * loss_iou + \
               self.loss_weight['corner'] * loss_cor + \
               self.loss_weight['dfl'] * loss_dfl
       
        return loss, \
            torch.cat(((self.loss_weight['iou'] * loss_iou).unsqueeze(0), 
                         (self.loss_weight['corner'] * loss_cor).unsqueeze(0),
                         (self.loss_weight['dfl'] * loss_dfl).unsqueeze(0),
                         (self.loss_weight['class'] * loss_cls).unsqueeze(0),
                         (loss_pro).unsqueeze(0),
                         (loss_alp).unsqueeze(0),
                         (loss_ads / 6).unsqueeze(0))).detach()
     
    def preprocess(self, targets, batch_size, scale_tensor):
        targets_list = np.zeros((batch_size, 1, 20)).tolist()
        for i, item in enumerate(targets.cpu().numpy().tolist()):
            targets_list[int(item[0])].append(item[1:])
        max_len = max((len(l) for l in targets_list))
        targets = torch.from_numpy(np.array(list(map(lambda l:l + [[-1,-1,-1,-1,-1,-1,-1,-1,0,0,0,0,0,0,0,0,0,0,0,0]]*(max_len - len(l)), targets_list)))[:,1:,:]).to(targets.device)
        batch_target = targets[:, :, 8:].mul_(scale_tensor)
        targets[..., 8:12] = xywh2xyxy(batch_target[..., :4])
        targets[..., 12:] = batch_target[..., 4:]
        return targets

    def bbox_decode(self, anchor_points, pred_dist):
        if self.use_dfl:
            batch_size, n_anchors, _ = pred_dist.shape
            pred_dist = F.softmax(pred_dist.view(batch_size, n_anchors, 4, self.reg_max + 1), dim=-1).matmul(self.proj.to(pred_dist.device))
        return dist2bbox(pred_dist, anchor_points)
    
    def corner_decode(self, anchor_points, pred_dist):
        return dist2cor(pred_dist, anchor_points)


class VarifocalLoss(nn.Module):
    def __init__(self):
        super(VarifocalLoss, self).__init__()

    def forward(self, pred_score, gt_score, label, alpha=0.75, gamma=2.0):

        weight = alpha * pred_score.pow(gamma) * (1 - label) + gt_score * label
        with torch.cuda.amp.autocast(enabled=False):
            loss = (F.binary_cross_entropy(pred_score.float(), gt_score.float(), reduction='none') * weight).sum()

        return loss


class BboxLoss(nn.Module):
    def __init__(self, reg_max, use_dfl=False, iou_type='giou'):
        super(BboxLoss, self).__init__()
        self.iou_loss = IOUloss(box_format='xyxy', iou_type=iou_type, eps=1e-10)
        self.reg_max = reg_max
        self.use_dfl = use_dfl

    def forward(self, pred_dist, pred_bboxes, anchor_points,
                target_bboxes, target_pro_scores, target_alp_scores, target_ads_scores, target_scores_sum, fg_mask):

        # select positive samples mask
        num_pos = fg_mask.sum()
        if num_pos > 0:
            # iou loss
            bbox_mask = fg_mask.unsqueeze(-1).repeat([1, 1, 4])
            pred_bboxes_pos = torch.masked_select(pred_bboxes,
                                                  bbox_mask).reshape([-1, 4])
            target_bboxes_pos = torch.masked_select(
                target_bboxes, bbox_mask).reshape([-1, 4])
            target_score = target_pro_scores.sum(-1) + target_alp_scores.sum(-1)
            for i in range(6):
                target_score += target_ads_scores[i].sum(-1)
            target_score /= 8.0
            bbox_weight = torch.masked_select(
                target_score, fg_mask).unsqueeze(-1)
            loss_iou = self.iou_loss(pred_bboxes_pos,
                                     target_bboxes_pos) * bbox_weight
            if target_scores_sum == 0:
                loss_iou = loss_iou.sum()
            else:
                loss_iou = loss_iou.sum() / target_scores_sum
               
            # dfl loss
            if self.use_dfl:
                dist_mask = fg_mask.unsqueeze(-1).repeat(
                    [1, 1, (self.reg_max + 1) * 4])
                pred_dist_pos = torch.masked_select(
                    pred_dist, dist_mask).reshape([-1, 4, self.reg_max + 1])
                target_ltrb = bbox2dist(anchor_points, target_bboxes, self.reg_max)
                target_ltrb_pos = torch.masked_select(
                    target_ltrb, bbox_mask).reshape([-1, 4])
                loss_dfl = self._df_loss(pred_dist_pos,
                                        target_ltrb_pos) * bbox_weight
                if target_scores_sum == 0:
                    loss_dfl = loss_dfl.sum()
                else:
                    loss_dfl = loss_dfl.sum() / target_scores_sum
            else:
                loss_dfl = pred_dist.sum() * 0.

        else:
            loss_iou = pred_dist.sum() * 0.
            loss_dfl = pred_dist.sum() * 0.

        return loss_iou, loss_dfl

    def _df_loss(self, pred_dist, target):
        target_left = target.to(torch.long)
        target_right = target_left + 1
        weight_left = target_right.to(torch.float) - target
        weight_right = 1 - weight_left
        loss_left = F.cross_entropy(
            pred_dist.view(-1, self.reg_max + 1), target_left.view(-1), reduction='none').view(
            target_left.shape) * weight_left
        loss_right = F.cross_entropy(
            pred_dist.view(-1, self.reg_max + 1), target_right.view(-1), reduction='none').view(
            target_left.shape) * weight_right
        return (loss_left + loss_right).mean(-1, keepdim=True)

class CornerLoss(nn.Module):
    def __init__(self):
        super(CornerLoss, self).__init__()
        self.w_loss = WingLoss()
    
    def forward(self, pred, target, score_sum, fg_mask):
        num_pos = fg_mask.sum()
        if num_pos > 0:
            corners_mask = fg_mask.unsqueeze(-1).repeat([1, 1, 8])
            pred_corners_pos = torch.masked_select(pred,
                corners_mask).reshape([-1, 8])
            target_corners_pos = torch.masked_select(
                target, corners_mask).reshape([-1, 8])
            w_loss = self.w_loss(pred_corners_pos, target_corners_pos)
            loss = w_loss
            if score_sum == 0:
                loss = loss.sum() / 8.0
            else:
                loss = loss.sum() / (8.0 * score_sum)
        else:
            loss = pred.sum() * 0.
        return loss

class WingLoss(nn.Module):
    def __init__(self, w=5, e=2):
        super(WingLoss, self).__init__()
        self.w = w
        self.e = e
        self.C = self.w - self.w * np.log(1 + self.w / self.e)

    def forward(self, x, t):
        weight = torch.ones_like(t)
        weight[torch.where(t==-1)] = 0
        diff = weight * (x - t)
        abs_diff = diff.abs()
        flag = (abs_diff.data < self.w).float()
        y = flag * self.w * torch.log(1 + abs_diff / self.e) + (1 - flag) * (abs_diff - self.C)
        return y
