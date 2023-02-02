import torch
import torch.nn as nn
import torch.nn.functional as F
from yolov6.assigners.assigner_utils import select_candidates_in_gts, select_highest_overlaps, iou_calculator, dist_calculator

class TaskAlignedAssigner(nn.Module):
    def __init__(self,
                 topk=13,
                 npro=31,
                 nalp=24,
                 nads=37,
                 alpha=1.0,
                 beta=6.0, 
                 eps=1e-9):
        super(TaskAlignedAssigner, self).__init__()
        self.topk = topk
        self.npro = npro
        self.nalp = nalp
        self.nads = nads
        self.pro_bg_idx = npro
        self.alp_bg_idx = nalp
        self.ads_bg_idx = nads
        self.alpha = alpha
        self.beta = beta
        self.eps = eps

    @torch.no_grad()
    def forward(self,
                pd_pro_scores,
                pd_alp_scores,
                pd_ads_scores,
                pd_bboxes,
                pd_corners,
                anc_points,
                gt_pro,
                gt_alp,
                gt_ads,
                gt_bboxes,
                gt_corners,
                mask_gt):
        """This code referenced to
           https://github.com/Nioolek/PPYOLOE_pytorch/blob/master/ppyoloe/assigner/tal_assigner.py

        Args:
            pd_pro_scores (Tensor): shape(bs, num_total_anchors, npro)
            pd_alp_scores (Tensor): shape(bs, num_total_anchors, nalp)
            pd_ads_scores (List include 5 Tensor): shape(bs, num_total_anchors, nads)
            pd_bboxes (Tensor): shape(bs, num_total_anchors, 4)
            anc_points (Tensor): shape(num_total_anchors, 2)
            gt_pro (Tensor): shape(bs, n_max_boxes, 1)
            gt_alp (Tensor): shape(bs, n_max_boxes, 1)
            gt_ads (Tensor): shape(bs, n_max_boxes, 5)
            gt_bboxes (Tensor): shape(bs, n_max_boxes, 4)
            gt_corners (Tensor): shape(bs, n_max_boxes, 8)
            mask_gt (Tensor): shape(bs, n_max_boxes, 1)
        Returns:
            target_pro (Tensor): shape(bs, num_total_anchors)
            target_alp (Tensor): shape(bs, num_total_anchors)
            target_ads (List include 5 Tensor): shape(bs, num_total_anchors)
            target_bboxes (Tensor): shape(bs, num_total_anchors, 4)
            target_pro_score (Tensor): shape(bs, num_total_anchors, npro)
            target_alp_score (Tensor): shape(bs, num_total_anchors, nalp)
            target_ad0-ad4_score (List inclue 5 Tensor): shape(bs, num_total_anchors, nads)
            fg_mask (Tensor): shape(bs, num_total_anchors)
        """
        self.n_anchors = anc_points.size(0)
        self.bs = gt_bboxes.size(0)
        self.n_max_boxes = gt_bboxes.size(1)

        if self.n_max_boxes == 0:
            device = gt_bboxes.device
            target_ads = []
            for i in range(5):
                target_ads.append(torch.full( [self.bs, self.n_anchors], self.ads_bg_idx).to(device))
            target_ad0_4 = []
            for i in range(5):
                target_ad0_4.append(torch.zeros([self.bs, self.n_anchors, self.nads]).to(device))
            return torch.full( [self.bs, self.n_anchors], self.pro_bg_idx).to(device), \
                   torch.full( [self.bs, self.n_anchors], self.alp_bg_idx).to(device), \
                   target_ads, \
                   torch.zeros([self.bs, self.n_anchors, 4]).to(device), \
                   torch.zeros([self.bs, self.n_anchors, 8]).to(device), \
                   torch.zeros([self.bs, self.n_anchors, self.npro]).to(device), \
                   torch.zeros([self.bs, self.n_anchors, self.nalp]).to(device), \
                   target_ad0_4, \
                   torch.zeros([self.bs, self.n_anchors]).to(device)

        mask_pos, align_metric, overlaps = self.get_pos_mask(
            pd_pro_scores, pd_bboxes, gt_pro, gt_bboxes, anc_points, mask_gt)

        target_gt_idx, fg_mask, mask_pos = select_highest_overlaps(
            mask_pos, overlaps, self.n_max_boxes)

        # assigned target
        target_pro, target_alp, target_ad0_4, target_bboxes, target_corners, target_pro_scores, target_alp_scores, target_ad0_4_scores = self.get_targets(
            gt_pro, gt_alp, gt_ads, gt_bboxes, gt_corners, target_gt_idx, fg_mask)

#####################################################################   TODO   ###########################################################################
        # normalize
        align_metric *= mask_pos
        pos_align_metrics = align_metric.max(axis=-1, keepdim=True)[0]
        pos_overlaps = (overlaps * mask_pos).max(axis=-1, keepdim=True)[0]
        norm_align_metric = (align_metric * pos_overlaps / (pos_align_metrics + self.eps)).max(-2)[0].unsqueeze(-1)
        target_pro_scores = target_pro_scores * norm_align_metric

        return target_pro, target_alp, target_ad0_4, target_bboxes, target_corners, target_pro_scores, target_alp_scores, target_ad0_4_scores, fg_mask.bool()

    def get_pos_mask(self,
                     pd_scores,
                     pd_bboxes,
                     gt_labels,
                     gt_bboxes,
                     anc_points,
                     mask_gt):

        # get anchor_align metric
        align_metric, overlaps = self.get_box_metrics(pd_scores, pd_bboxes, gt_labels, gt_bboxes)
        # get in_gts mask
        mask_in_gts = select_candidates_in_gts(anc_points, gt_bboxes)
        # get topk_metric mask
        mask_topk = self.select_topk_candidates(
            align_metric * mask_in_gts, topk_mask=mask_gt.repeat([1, 1, self.topk]).bool())
        # merge all mask to a final mask
        mask_pos = mask_topk * mask_in_gts * mask_gt

        return mask_pos, align_metric, overlaps

    def get_box_metrics(self,
                        pd_scores,
                        pd_bboxes,
                        gt_labels,
                        gt_bboxes):

        pd_scores = pd_scores.permute(0, 2, 1)
        gt_labels = gt_labels.to(torch.long)
        ind = torch.zeros([2, self.bs, self.n_max_boxes], dtype=torch.long)
        ind[0] = torch.arange(end=self.bs).view(-1, 1).repeat(1, self.n_max_boxes)
        ind[1] = gt_labels.squeeze(-1)
        bbox_scores = pd_scores[ind[0], ind[1]]

        overlaps = iou_calculator(gt_bboxes, pd_bboxes)
        align_metric = bbox_scores.pow(self.alpha) * overlaps.pow(self.beta)

        return align_metric, overlaps

    def select_topk_candidates(self,
                               metrics,
                               largest=True,
                               topk_mask=None):

        num_anchors = metrics.shape[-1]
        topk_metrics, topk_idxs = torch.topk(
            metrics, self.topk, axis=-1, largest=largest)
        if topk_mask is None:
            topk_mask = (topk_metrics.max(axis=-1, keepdim=True) > self.eps).tile(
                [1, 1, self.topk])
        topk_idxs = torch.where(topk_mask, topk_idxs, torch.zeros_like(topk_idxs))
        is_in_topk = F.one_hot(topk_idxs, num_anchors).sum(axis=-2)
        is_in_topk = torch.where(is_in_topk > 1,
            torch.zeros_like(is_in_topk), is_in_topk)
        return is_in_topk.to(metrics.dtype)

    def get_targets(self,
                    gt_pro,
                    gt_alp,
                    gt_ads,
                    gt_bboxes,
                    gt_corners,
                    target_gt_idx,
                    fg_mask):

        # assigned target labels
        batch_ind = torch.arange(end=self.bs, dtype=torch.int64, device=gt_pro.device)[...,None]
        target_gt_idx = target_gt_idx + batch_ind * self.n_max_boxes
        target_pro = gt_pro.long().flatten()[target_gt_idx]
        target_alp = gt_alp.long().flatten()[target_gt_idx]
        target_ad0_4 = []
        for i in range(5):
            target_ads = gt_ads[:, :, i].long().flatten()[target_gt_idx]
            target_ad0_4.append(target_ads)

        # assigned target boxes
        target_bboxes = gt_bboxes.reshape([-1, 4])[target_gt_idx]

        # assigned target corners
        target_corners = gt_corners.reshape([-1, 8])[target_gt_idx]

        # assigned target pro scores
        target_pro[target_pro<0] = 0
        target_pro_scores = F.one_hot(target_pro, self.npro)
        fg_pro_scores_mask = fg_mask[:, :, None].repeat(1, 1, self.npro)
        target_pro_scores = torch.where(fg_pro_scores_mask > 0, target_pro_scores,
                                            torch.full_like(target_pro_scores, 0))
        
        # assigned target alp scores
        target_alp[target_alp<0] = 0
        target_alp_scores = F.one_hot(target_alp, self.nalp)
        fg_alp_scores_mask = fg_mask[:, :, None].repeat(1, 1, self.nalp)
        target_alp_scores = torch.where(fg_alp_scores_mask > 0, target_alp_scores,
                                            torch.full_like(target_alp_scores, 0))
        # assigned target ads scores
        target_ad0_4_scores = []
        for i in range(5):
            target_ads = target_ad0_4[i]
            target_ads[target_ads<0] = 0
            target_ads_scores = F.one_hot(target_ads, self.nads)
            fg_ads_scores_mask = fg_mask[:, :, None].repeat(1, 1, self.nads)
            target_ads_scores = torch.where(fg_ads_scores_mask > 0, target_ads_scores,
                                                torch.full_like(target_ads_scores, 0))
            target_ad0_4_scores.append(target_ads_scores)

        return target_pro, target_alp, target_ad0_4, target_bboxes, target_corners, target_pro_scores, target_alp_scores, target_ad0_4_scores
