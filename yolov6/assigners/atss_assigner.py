import torch
import torch.nn as nn
import torch.nn.functional as F
from yolov6.assigners.iou2d_calculator import iou2d_calculator
from yolov6.assigners.assigner_utils import dist_calculator, select_candidates_in_gts, select_highest_overlaps, iou_calculator

class ATSSAssigner(nn.Module):
    '''Adaptive Training Sample Selection Assigner'''
    def __init__(self,
                 topk=9,
                 npro=31,
                 nalp=24,
                 nads=37):
        super(ATSSAssigner, self).__init__()
        self.topk = topk
        self.npro = npro
        self.nalp = nalp
        self.nads = nads
        self.pro_bg_idx = npro
        self.alp_bg_idx = nalp
        self.ads_bg_idx = nads

    @torch.no_grad()
    def forward(self,
                anc_bboxes,
                n_level_bboxes,
                gt_pro,
                gt_alp,
                gt_ads,
                gt_bboxes,
                gt_corners,
                mask_gt,
                pd_bboxes):
        r"""This code is based on
            https://github.com/fcjian/TOOD/blob/master/mmdet/core/bbox/assigners/atss_assigner.py

        Args:
            anc_bboxes (Tensor): shape(num_total_anchors, 4)
            n_level_bboxes (List):len(3)
            gt_pro (Tensor): shape(bs, n_max_boxes, 1)
            gt_alp (Tensor): shape(bs, n_max_boxes, 1)
            gt_ads (Tensor): shape(bs, n_max_boxes, 6)
            gt_bboxes (Tensor): shape(bs, n_max_boxes, 4)
            gt_corners (Tensor): shape(bs, n_max_boxes, 8)
            mask_gt (Tensor): shape(bs, n_max_boxes, 1)
            pd_bboxes (Tensor): shape(bs, n_max_boxes, 4)
        Returns:
            target_pro (Tensor): shape(bs, num_total_anchors)
            target_alp (Tensor): shape(bs, num_total_anchors)
            target_ads (List include 6 Tensor): shape(bs, num_total_anchors)
            target_bboxes (Tensor): shape(bs, num_total_anchors, 4)
            target_corners (Tensor): shape(bs, num_total_anchors, 8)
            target_pro_score (Tensor): shape(bs, num_total_anchors, npro)
            target_alp_score (Tensor): shape(bs, num_total_anchors, nalp)
            target_ad0-ad4_score (List inclue 6 Tensor): shape(bs, num_total_anchors, nads)
            fg_mask (Tensor): shape(bs, num_total_anchors)
        """
        self.n_anchors = anc_bboxes.size(0)
        self.bs = gt_bboxes.size(0)
        self.n_max_boxes = gt_bboxes.size(1)

        if self.n_max_boxes == 0:
            device = gt_bboxes.device
            target_ads = []
            for i in range(6):
                target_ads.append(torch.full( [self.bs, self.n_anchors], self.ads_bg_idx).to(device))
            target_ad0_5 = []
            for i in range(6):
                target_ad0_5.append(torch.zeros([self.bs, self.n_anchors, self.nads]).to(device))
            return torch.full( [self.bs, self.n_anchors], self.pro_bg_idx).to(device), \
                   torch.full( [self.bs, self.n_anchors], self.alp_bg_idx).to(device), \
                   target_ads, \
                   torch.zeros([self.bs, self.n_anchors, 4]).to(device), \
                   torch.zeros([self.bs, self.n_anchors, 8]).to(device), \
                   torch.zeros([self.bs, self.n_anchors, self.npro]).to(device), \
                   torch.zeros([self.bs, self.n_anchors, self.nalp]).to(device), \
                   target_ad0_5, \
                   torch.zeros([self.bs, self.n_anchors]).to(device)


        overlaps = iou2d_calculator(gt_bboxes.reshape([-1, 4]), anc_bboxes)
        overlaps = overlaps.reshape([self.bs, -1, self.n_anchors])

        distances, ac_points = dist_calculator(gt_bboxes.reshape([-1, 4]), anc_bboxes)
        distances = distances.reshape([self.bs, -1, self.n_anchors])

        is_in_candidate, candidate_idxs = self.select_topk_candidates(
            distances, n_level_bboxes, mask_gt)

        overlaps_thr_per_gt, iou_candidates = self.thres_calculator(
            is_in_candidate, candidate_idxs, overlaps)

        # select candidates iou >= threshold as positive
        is_pos = torch.where(
            iou_candidates > overlaps_thr_per_gt.repeat([1, 1, self.n_anchors]),
            is_in_candidate, torch.zeros_like(is_in_candidate))

        is_in_gts = select_candidates_in_gts(ac_points, gt_bboxes)
        mask_pos = is_pos * is_in_gts * mask_gt

        target_gt_idx, fg_mask, mask_pos = select_highest_overlaps(
            mask_pos, overlaps, self.n_max_boxes)

        # assigned target
        target_pro, target_alp, target_ad0_5, target_bboxes, target_corners, target_pro_scores, target_alp_scores, target_ad0_5_score = self.get_targets(
            gt_pro, gt_alp, gt_ads, gt_bboxes, gt_corners, target_gt_idx, fg_mask)

        # soft label with iou
        if pd_bboxes is not None:
            ious = iou_calculator(gt_bboxes, pd_bboxes) * mask_pos
            ious = ious.max(axis=-2)[0].unsqueeze(-1)
            target_pro_scores *= ious
            target_alp_scores *= ious
            for i in range(6):
                target_ad0_5_score[i] *= ious
        
        for i in range(6):
            target_ad0_5[i] = target_ad0_5[i].long()

        return target_pro.long(), target_alp.long(), target_ad0_5, target_bboxes, target_corners, target_pro_scores, target_alp_scores, target_ad0_5_score, fg_mask.bool()

    def select_topk_candidates(self,
                               distances,
                               n_level_bboxes,
                               mask_gt):

        mask_gt = mask_gt.repeat(1, 1, self.topk).bool()
        level_distances = torch.split(distances, n_level_bboxes, dim=-1)
        is_in_candidate_list = []
        candidate_idxs = []
        start_idx = 0
        for per_level_distances, per_level_boxes in zip(level_distances, n_level_bboxes):

            end_idx = start_idx + per_level_boxes
            selected_k = min(self.topk, per_level_boxes)
            _, per_level_topk_idxs = per_level_distances.topk(selected_k, dim=-1, largest=False)
            candidate_idxs.append(per_level_topk_idxs + start_idx)
            per_level_topk_idxs = torch.where(mask_gt,
                per_level_topk_idxs, torch.zeros_like(per_level_topk_idxs))
            is_in_candidate = F.one_hot(per_level_topk_idxs, per_level_boxes).sum(dim=-2)
            is_in_candidate = torch.where(is_in_candidate > 1,
                torch.zeros_like(is_in_candidate), is_in_candidate)
            is_in_candidate_list.append(is_in_candidate.to(distances.dtype))
            start_idx = end_idx

        is_in_candidate_list = torch.cat(is_in_candidate_list, dim=-1)
        candidate_idxs = torch.cat(candidate_idxs, dim=-1)

        return is_in_candidate_list, candidate_idxs

    def thres_calculator(self,
                         is_in_candidate,
                         candidate_idxs,
                         overlaps):

        n_bs_max_boxes = self.bs * self.n_max_boxes
        _candidate_overlaps = torch.where(is_in_candidate > 0,
            overlaps, torch.zeros_like(overlaps))
        candidate_idxs = candidate_idxs.reshape([n_bs_max_boxes, -1])
        assist_idxs = self.n_anchors * torch.arange(n_bs_max_boxes, device=candidate_idxs.device)
        assist_idxs = assist_idxs[:,None]
        faltten_idxs = candidate_idxs + assist_idxs
        candidate_overlaps = _candidate_overlaps.reshape(-1)[faltten_idxs]
        candidate_overlaps = candidate_overlaps.reshape([self.bs, self.n_max_boxes, -1])

        overlaps_mean_per_gt = candidate_overlaps.mean(axis=-1, keepdim=True)
        overlaps_std_per_gt = candidate_overlaps.std(axis=-1, keepdim=True)
        overlaps_thr_per_gt = overlaps_mean_per_gt + overlaps_std_per_gt

        return overlaps_thr_per_gt, _candidate_overlaps

    def get_targets(self,
                    gt_pro,
                    gt_alp,
                    gt_ads,
                    gt_bboxes,
                    gt_corners,
                    target_gt_idx,
                    fg_mask):
        # aget ads
        gt_ad0_5 = [gt_ads[:, :, i] for i in range(6)]

        # assigned target pro
        batch_idx = torch.arange(self.bs, dtype=gt_pro.dtype, device=gt_pro.device)
        batch_idx = batch_idx[...,None]
        target_gt_idx = (target_gt_idx + batch_idx * self.n_max_boxes).long()
        target_pro = gt_pro.flatten()[target_gt_idx.flatten()]
        target_pro = target_pro.reshape([self.bs, self.n_anchors])
        target_pro = torch.where(fg_mask > 0,
            target_pro, torch.full_like(target_pro, self.pro_bg_idx))

        # assigned target alp
        target_alp = gt_alp.flatten()[target_gt_idx.flatten()]
        target_alp = target_alp.reshape([self.bs, self.n_anchors])
        target_alp = torch.where(fg_mask > 0,
            target_alp, torch.full_like(target_alp, self.alp_bg_idx))

        # assigned target ads
        target_ad0_5 = []
        for i in range(6):
            target_ads = gt_ad0_5[i].flatten()[target_gt_idx.flatten()]
            target_ads = target_ads.reshape([self.bs, self.n_anchors])
            target_ads = torch.where(fg_mask > 0,
                target_ads, torch.full_like(target_ads, self.ads_bg_idx))
            target_ad0_5.append(target_ads)

        # assigned target boxes
        target_bboxes = gt_bboxes.reshape([-1, 4])[target_gt_idx.flatten()]
        target_bboxes = target_bboxes.reshape([self.bs, self.n_anchors, 4])

        # assigned target corners
        target_corners = gt_corners.reshape([-1, 8])[target_gt_idx.flatten()]
        target_corners = target_corners.reshape([self.bs, self.n_anchors, 8])

        # assigned target pro scores
        target_pro_scores = F.one_hot(target_pro.long(), self.npro + 1).float()
        target_pro_scores = target_pro_scores[:, :, :self.npro]

        # assigned target alp scores
        target_alp_scores = F.one_hot(target_alp.long(), self.nalp + 1).float()
        target_alp_scores = target_alp_scores[:, :, :self.nalp]

        # assigned target ads scores
        target_ad0_5_score = []
        for i in range(6):
            target_ads_score = F.one_hot(target_ad0_5[i].long(), self.nads + 1).float()
            target_ads_score = target_ads_score[:, :, :self.nads]
            target_ad0_5_score.append(target_ads_score)

        return target_pro, target_alp, target_ad0_5, target_bboxes, target_corners, target_pro_scores, target_alp_scores, target_ad0_5_score
