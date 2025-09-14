# Copyright (c) OpenMMLab. All rights reserved.
import torch


from mmdet.core.bbox.iou_calculators import build_iou_calculator
from mmdet.core.bbox.assigners.assign_result import AssignResult
from mmdet.core.bbox.assigners.base_assigner import BaseAssigner
from ..builder import ROTATED_BBOX_ASSIGNERS


@ROTATED_BBOX_ASSIGNERS.register_module()
class MixAssigner(BaseAssigner):
    """Assign a corresponding gt bbox or background to each bbox.

    Each proposals will be assigned with `-1`, or a semi-positive integer
    indicating the ground truth index.

    - -1: negative sample, no assigned gt
    - semi-positive integer: positive sample, index (0-based) of assigned gt

    Args:
        pos_iou_thr (float): IoU threshold for positive bboxes.
        neg_iou_thr (float or tuple): IoU threshold for negative bboxes.
        min_pos_iou (float): Minimum iou for a bbox to be considered as a
            positive bbox. Positive samples can have smaller IoU than
            pos_iou_thr due to the 4th step (assign max IoU sample to each gt).
            `min_pos_iou` is set to avoid assigning bboxes that have extremely
            small iou with GT as positive samples. It brings about 0.3 mAP
            improvements in 1x schedule but does not affect the performance of
            3x schedule. More comparisons can be found in
            `PR #7464 <https://github.com/open-mmlab/mmdetection/pull/7464>`_.
        gt_max_assign_all (bool): Whether to assign all bboxes with the same
            highest overlap with some gt to that gt.
        ignore_iof_thr (float): IoF threshold for ignoring bboxes (if
            `gt_bboxes_ignore` is specified). Negative values mean not
            ignoring any bboxes.
        ignore_wrt_candidates (bool): Whether to compute the iof between
            `bboxes` and `gt_bboxes_ignore`, or the contrary.
        match_low_quality (bool): Whether to allow low quality matches. This is
            usually allowed for RPN and single stage detectors, but not allowed
            in the second stage. Details are demonstrated in Step 4.
        gpu_assign_thr (int): The upper bound of the number of GT for GPU
            assign. When the number of gt is above this threshold, will assign
            on CPU device. Negative values mean not assign on CPU.
    """

    def __init__(self,
                 pos_topk,
                 neg_iou_thr,
                 ignore_iof_thr=-1,
                 ignore_wrt_candidates=True,
                 gpu_assign_thr=-1,
                 iou_calculator_pos=dict(type='BboxOverlaps2D'),
                 iou_calculator_neg=dict(type='BboxOverlaps2D'),
                 assign_metric='iou'):
        self.pos_topk = pos_topk
        self.neg_iou_thr = neg_iou_thr
        self.ignore_iof_thr = ignore_iof_thr
        self.ignore_wrt_candidates = ignore_wrt_candidates
        self.gpu_assign_thr = gpu_assign_thr
        self.iou_calculator_pos = build_iou_calculator(iou_calculator_pos)
        self.iou_calculator_neg = build_iou_calculator(iou_calculator_neg)
        self.assign_metric = assign_metric

    def assign(self, bboxes, gt_bboxes, gt_bboxes_redundancy, gt_bboxes_ignore=None, gt_labels=None, ast_bboxes=None):
        """Assign gt to bboxes.

        This method assign a gt bbox to every bbox (proposal/anchor), each bbox
        will be assigned with -1, or a semi-positive number. -1 means negative
        sample, semi-positive number is the index (0-based) of assigned gt.
        The assignment is done in following steps, the order matters.

        1. assign every bbox to the background
        2. assign proposals whose iou with all gts < neg_iou_thr to 0
        3. for each bbox, if the iou with its nearest gt >= pos_iou_thr,
           assign it to that bbox
        4. for each gt bbox, assign its nearest proposals (may be more than
           one) to itself

        Args:
            bboxes (Tensor): Bounding boxes to be assigned, shape(n, 4).
            gt_bboxes (Tensor): Groundtruth boxes, shape (k, 4).
            gt_bboxes_ignore (Tensor, optional): Ground truth bboxes that are
                labelled as `ignored`, e.g., crowd boxes in COCO.
            gt_labels (Tensor, optional): Label of gt_bboxes, shape (k, ).

        Returns:
            :obj:`AssignResult`: The assign result.

        Example:
            >>> self = MaxIoUAssigner(0.5, 0.5)
            >>> bboxes = torch.Tensor([[0, 0, 10, 10], [10, 10, 20, 20]])
            >>> gt_bboxes = torch.Tensor([[0, 0, 10, 9]])
            >>> assign_result = self.assign(bboxes, gt_bboxes)
            >>> expected_gt_inds = torch.LongTensor([1, 0])
            >>> assert torch.all(assign_result.gt_inds == expected_gt_inds)
        """
        assign_on_cpu = True if (self.gpu_assign_thr > 0) and (
            gt_bboxes.shape[0] > self.gpu_assign_thr) else False
        # compute overlap and assign gt on CPU when number of GT is large
        if assign_on_cpu:
            device = bboxes.device
            bboxes = bboxes.cpu()
            gt_bboxes = gt_bboxes.cpu()
            if gt_bboxes_redundancy is not None:
                gt_bboxes_redundancy = gt_bboxes_redundancy.cpu()
            if gt_bboxes_ignore is not None:
                gt_bboxes_ignore = gt_bboxes_ignore.cpu()
            if gt_labels is not None:
                gt_labels = gt_labels.cpu()

        if gt_bboxes_redundancy is not None:

            overlaps_neg = self.iou_calculator_neg(gt_bboxes_redundancy, bboxes)
            overlaps_pos = self.iou_calculator_pos(gt_bboxes,  bboxes, mode=self.assign_metric, ast_bboxes=ast_bboxes)

            if (self.ignore_iof_thr > 0 and gt_bboxes_ignore is not None
                    and gt_bboxes_ignore.numel() > 0 and bboxes.numel() > 0):
                if self.ignore_wrt_candidates:
                    ignore_overlaps = self.iou_calculator_neg(
                        bboxes, gt_bboxes_ignore, mode='iof')
                    ignore_max_overlaps, _ = ignore_overlaps.max(dim=1)
                else:
                    ignore_overlaps = self.iou_calculator_neg(
                        gt_bboxes_ignore, bboxes, mode='iof')
                    ignore_max_overlaps, _ = ignore_overlaps.max(dim=0)
                overlaps_neg[:, ignore_max_overlaps > self.ignore_iof_thr] = -1       
                overlaps_pos[:, ignore_max_overlaps > self.ignore_iof_thr] = -1

            assign_result = self.assign_mix_overlaps(overlaps_pos, overlaps_neg, gt_labels)
        else:
            overlaps_pos = self.iou_calculator_pos(gt_bboxes,  bboxes, mode=self.assign_metric, ast_bboxes=ast_bboxes)

            if (self.ignore_iof_thr > 0 and gt_bboxes_ignore is not None
                    and gt_bboxes_ignore.numel() > 0 and bboxes.numel() > 0):
                if self.ignore_wrt_candidates:
                    ignore_overlaps = self.iou_calculator_neg(
                        bboxes, gt_bboxes_ignore, mode='iof')
                    ignore_max_overlaps, _ = ignore_overlaps.max(dim=1)
                else:
                    ignore_overlaps = self.iou_calculator_neg(
                        gt_bboxes_ignore, bboxes, mode='iof')
                    ignore_max_overlaps, _ = ignore_overlaps.max(dim=0)     
                overlaps_pos[:, ignore_max_overlaps > self.ignore_iof_thr] = -1

            assign_result = self.assign_wrt_ranking(overlaps_pos, gt_labels)


        if assign_on_cpu:
            assign_result.gt_inds = assign_result.gt_inds.to(device)
            assign_result.max_overlaps = assign_result.max_overlaps.to(device)
            if assign_result.labels is not None:
                assign_result.labels = assign_result.labels.to(device)
        return assign_result

    def assign_mix_overlaps(self, overlaps_pos, overlaps_neg, gt_labels=None):
        """Assign w.r.t. the overlaps of bboxes with gts.

        Args:
            overlaps (Tensor): Overlaps between k gt_bboxes and n bboxes,
                shape(k, n).
            gt_labels (Tensor, optional): Labels of k gt_bboxes, shape (k, ).

        Returns:
            :obj:`AssignResult`: The assign result.
        """
        assert overlaps_pos.size(1) == overlaps_neg.size(1)
        num_gts, num_bboxes = overlaps_pos.size(0), overlaps_pos.size(1)

        # 1. assign -1 by default
        assigned_gt_inds = overlaps_pos.new_full((num_bboxes, ),
                                             -1,
                                             dtype=torch.long)

        if num_gts == 0 or num_bboxes == 0:
            # No ground truth or boxes, return empty assignment
            max_overlaps = overlaps_pos.new_zeros((num_bboxes, ))
            if num_gts == 0:
                # No truth, assign everything to background
                assigned_gt_inds[:] = 0
            if gt_labels is None:
                assigned_labels = None
            else:
                assigned_labels = overlaps_pos.new_full((num_bboxes, ),
                                                    -1,
                                                    dtype=torch.long)
            return AssignResult(
                num_gts,
                assigned_gt_inds,
                max_overlaps,
                labels=assigned_labels)

        # for each anchor, which gt best overlaps with it
        # for each anchor, the max iou of all gts
        max_overlaps_neg, argmax_overlaps_neg = overlaps_neg.max(dim=0)

        max_overlaps, argmax_overlaps = overlaps_pos.max(dim=0)
        # for each gt, topk anchors
        # for each gt, the topk of all proposals
        gt_max_overlaps, gt_argmax_overlaps = overlaps_pos.topk(self.pos_topk, dim=1, largest=True, sorted=True)


        # 2. assign negative: below
        # the negative inds are set to be 0
        if isinstance(self.neg_iou_thr, float):
            assigned_gt_inds[(max_overlaps_neg >= 0)
                             & (max_overlaps_neg < self.neg_iou_thr)] = 0
        elif isinstance(self.neg_iou_thr, tuple):
            assert len(self.neg_iou_thr) == 2
            assigned_gt_inds[(max_overlaps_neg >= self.neg_iou_thr[0])
                             & (max_overlaps_neg < self.neg_iou_thr[1])] = 0

        # 3. assign positive: above positive IoU threshold
        for i in range(num_gts):
            for j in range(self.pos_topk):
                max_overlap_inds = overlaps_pos[i,:] == gt_max_overlaps[i,j]
                assigned_gt_inds[max_overlap_inds] = i + 1


        if gt_labels is not None:
            assigned_labels = assigned_gt_inds.new_full((num_bboxes, ), -1)
            pos_inds = torch.nonzero(
                assigned_gt_inds > 0, as_tuple=False).squeeze()
            if pos_inds.numel() > 0:
                assigned_labels[pos_inds] = gt_labels[
                    assigned_gt_inds[pos_inds] - 1]
        else:
            assigned_labels = None

        return AssignResult(
            num_gts, assigned_gt_inds, max_overlaps, labels=assigned_labels)


    def assign_wrt_ranking(self,  overlaps, gt_labels=None):
        num_gts, num_bboxes = overlaps.size(0), overlaps.size(1)

        # 1. assign -1 by default
        assigned_gt_inds = overlaps.new_full((num_bboxes,),
                                             -1,
                                             dtype=torch.long)

        if num_gts == 0 or num_bboxes == 0:
            # No ground truth or boxes, return empty assignment
            max_overlaps = overlaps.new_zeros((num_bboxes,))
            if num_gts == 0:
                # No truth, assign everything to background
                assigned_gt_inds[:] = 0
            if gt_labels is None:
                assigned_labels = None
            else:
                assigned_labels = overlaps.new_full((num_bboxes,),
                                                    -1,
                                                    dtype=torch.long)
            return AssignResult(
                num_gts,
                assigned_gt_inds,
                max_overlaps,
                labels=assigned_labels)

        # for each anchor, which gt best overlaps with it
        # for each anchor, the max iou of all gts
        max_overlaps, argmax_overlaps = overlaps.max(dim=0)
        # for each gt, topk anchors
        # for each gt, the topk of all proposals
        gt_max_overlaps, gt_argmax_overlaps = overlaps.topk(self.pos_topk, dim=1, largest=True, sorted=True)  # gt_argmax_overlaps [num_gt, k]


        assigned_gt_inds[(max_overlaps >= 0)
                             & (max_overlaps < 0.8)] = 0

        for i in range(num_gts):
            for j in range(self.pos_topk):
                max_overlap_inds = overlaps[i,:] == gt_max_overlaps[i,j]
                assigned_gt_inds[max_overlap_inds] = i + 1

        if gt_labels is not None:
            assigned_labels = assigned_gt_inds.new_full((num_bboxes,), -1)
            pos_inds = torch.nonzero(
                assigned_gt_inds > 0, as_tuple=False).squeeze()
            if pos_inds.numel() > 0:
                assigned_labels[pos_inds] = gt_labels[
                    assigned_gt_inds[pos_inds] - 1]
        else:
            assigned_labels = None

        return AssignResult(
            num_gts, assigned_gt_inds, max_overlaps, labels=assigned_labels)
