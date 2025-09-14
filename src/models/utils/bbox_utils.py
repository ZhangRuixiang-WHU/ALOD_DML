import warnings
from collections.abc import Sequence
from typing import List, Optional, Tuple, Union
import types

import numpy as np
import math
import torch
from mmdet.core.mask.structures import BitmapMasks
from mmrotate.core.bbox.transforms import poly2obb_le90, obb2poly_le90
from torch.nn import functional as F

import sklearn.mixture as skm


def resize_image(inputs, resize_ratio=0.5):
    down_inputs = F.interpolate(inputs, 
                                scale_factor=resize_ratio, 
                                mode='nearest')
    
    return down_inputs


def pop_elements(_list, count):
    for idx in range(count):
        _list.pop(idx)
    return _list

def pointobb2thetaobb(pointobb):
    return poly2obb_le90(pointobb)

def thetaobb2pointobb(thetaobb):
    return obb2poly_le90(thetaobb).reshape(
        -1, 2
    ) 

def bbox2points(box):
    min_x, min_y, max_x, max_y = torch.split(box[:, :4], [1, 1, 1, 1], dim=1)

    return torch.cat(
        [min_x, min_y, max_x, min_y, max_x, max_y, min_x, max_y], dim=1
    ).reshape(
        -1, 2
    )  # n*4,2


def bbox2points(box):
    min_x, min_y, max_x, max_y = torch.split(box[:, :4], [1, 1, 1, 1], dim=1)

    return torch.cat(
        [min_x, min_y, max_x, min_y, max_x, max_y, min_x, max_y], dim=1
    ).reshape(
        -1, 2
    )  # n*4,2


def points2bbox(point, max_w, max_h):
    point = point.reshape(-1, 4, 2)
    if point.size()[0] > 0:
        min_xy = point.min(dim=1)[0]
        max_xy = point.max(dim=1)[0]
        xmin = min_xy[:, 0].clamp(min=0, max=max_w)
        ymin = min_xy[:, 1].clamp(min=0, max=max_h)
        xmax = max_xy[:, 0].clamp(min=0, max=max_w)
        ymax = max_xy[:, 1].clamp(min=0, max=max_h)
        min_xy = torch.stack([xmin, ymin], dim=1)
        max_xy = torch.stack([xmax, ymax], dim=1)
        return torch.cat([min_xy, max_xy], dim=1)  # n,4
    else:
        return point.new_zeros(0, 4)


def check_is_tensor(obj):
    """Checks whether the supplied object is a tensor."""
    if not isinstance(obj, torch.Tensor):
        raise TypeError("Input type is not a torch.Tensor. Got {}".format(type(obj)))


def normal_transform_pixel(
    height: int,
    width: int,
    eps: float = 1e-14,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    tr_mat = torch.tensor(
        [[1.0, 0.0, -1.0], [0.0, 1.0, -1.0], [0.0, 0.0, 1.0]],
        device=device,
        dtype=dtype,
    )  # 3x3

    # prevent divide by zero bugs
    width_denom: float = eps if width == 1 else width - 1.0
    height_denom: float = eps if height == 1 else height - 1.0

    tr_mat[0, 0] = tr_mat[0, 0] * 2.0 / width_denom
    tr_mat[1, 1] = tr_mat[1, 1] * 2.0 / height_denom

    return tr_mat.unsqueeze(0)  # 1x3x3


def normalize_homography(
    dst_pix_trans_src_pix: torch.Tensor,
    dsize_src: Tuple[int, int],
    dsize_dst: Tuple[int, int],
) -> torch.Tensor:
    check_is_tensor(dst_pix_trans_src_pix)

    if not (
        len(dst_pix_trans_src_pix.shape) == 3
        or dst_pix_trans_src_pix.shape[-2:] == (3, 3)
    ):
        raise ValueError(
            "Input dst_pix_trans_src_pix must be a Bx3x3 tensor. Got {}".format(
                dst_pix_trans_src_pix.shape
            )
        )

    # source and destination sizes
    src_h, src_w = dsize_src
    dst_h, dst_w = dsize_dst

    # compute the transformation pixel/norm for src/dst
    src_norm_trans_src_pix: torch.Tensor = normal_transform_pixel(src_h, src_w).to(
        dst_pix_trans_src_pix
    )
    src_pix_trans_src_norm = torch.inverse(src_norm_trans_src_pix.float()).to(
        src_norm_trans_src_pix.dtype
    )
    dst_norm_trans_dst_pix: torch.Tensor = normal_transform_pixel(dst_h, dst_w).to(
        dst_pix_trans_src_pix
    )

    # compute chain transformations
    dst_norm_trans_src_norm: torch.Tensor = dst_norm_trans_dst_pix @ (
        dst_pix_trans_src_pix @ src_pix_trans_src_norm
    )
    return dst_norm_trans_src_norm


def warp_affine(
    src: torch.Tensor,
    M: torch.Tensor,
    dsize: Tuple[int, int],
    mode: str = "bilinear",
    padding_mode: str = "zeros",
    align_corners: Optional[bool] = None,
) -> torch.Tensor:
    if not isinstance(src, torch.Tensor):
        raise TypeError(
            "Input src type is not a torch.Tensor. Got {}".format(type(src))
        )

    if not isinstance(M, torch.Tensor):
        raise TypeError("Input M type is not a torch.Tensor. Got {}".format(type(M)))

    if not len(src.shape) == 4:
        raise ValueError("Input src must be a BxCxHxW tensor. Got {}".format(src.shape))

    if not (len(M.shape) == 3 or M.shape[-2:] == (2, 3)):
        raise ValueError("Input M must be a Bx2x3 tensor. Got {}".format(M.shape))

    # TODO: remove the statement below in kornia v0.3
    if align_corners is None:
        message: str = (
            "The align_corners default value has been changed. By default now is set True "
            "in order to match cv2.warpAffine."
        )
        warnings.warn(message)
        # set default value for align corners
        align_corners = True

    B, C, H, W = src.size()

    # we generate a 3x3 transformation matrix from 2x3 affine

    dst_norm_trans_src_norm: torch.Tensor = normalize_homography(M, (H, W), dsize)

    src_norm_trans_dst_norm = torch.inverse(dst_norm_trans_src_norm.float())

    grid = F.affine_grid(
        src_norm_trans_dst_norm[:, :2, :],
        [B, C, dsize[0], dsize[1]],
        align_corners=align_corners,
    )

    return F.grid_sample(
        src.float(),
        grid,
        align_corners=align_corners,
        mode=mode,
        padding_mode=padding_mode,
    ).to(src.dtype)


class Transform2D:
    @staticmethod
    def transform_rbboxes(bbox, M, out_shape): ## change to rbbox
        if isinstance(bbox, Sequence):
            assert len(bbox) == len(M)
            return [
                Transform2D.transform_rbboxes(b, m, o)
                for b, m, o in zip(bbox, M, out_shape)
            ]
        else:
            if bbox.shape[0] == 0:
                return bbox
            score = None
            if bbox.shape[1] > 5:
                score = bbox[:, 5:]
            points = thetaobb2pointobb(bbox[:, :5])
            points = torch.cat(
                [points, points.new_ones(points.shape[0], 1)], dim=1
            )  # n,3
            points = torch.matmul(M, points.t()).t()
            points = points[:, :2] / points[:, 2:3]
            bbox = pointobb2thetaobb(points)#, out_shape[1], out_shape[0])
            if score is not None:
                return torch.cat([bbox, score], dim=1)
            return bbox
        
    @staticmethod
    def transform_bboxes(bbox, M, out_shape):
        if isinstance(bbox, Sequence):
            assert len(bbox) == len(M)
            return [
                Transform2D.transform_bboxes(b, m, o)
                for b, m, o in zip(bbox, M, out_shape)
            ]
        else:
            if bbox.shape[0] == 0:
                return bbox
            score = None
            if bbox.shape[1] > 4:
                score = bbox[:, 4:]
            points = bbox2points(bbox[:, :4])
            points = torch.cat(
                [points, points.new_ones(points.shape[0], 1)], dim=1
            )  # n,3
            points = torch.matmul(M, points.t()).t()
            points = points[:, :2] / points[:, 2:3]
            bbox = points2bbox(points, out_shape[1], out_shape[0])
            if score is not None:
                return torch.cat([bbox, score], dim=1)
            return bbox

    @staticmethod
    def transform_masks(
        mask: Union[BitmapMasks, List[BitmapMasks]],
        M: Union[torch.Tensor, List[torch.Tensor]],
        out_shape: Union[list, List[list]],
    ):
        if isinstance(mask, Sequence):
            assert len(mask) == len(M)
            return [
                Transform2D.transform_masks(b, m, o)
                for b, m, o in zip(mask, M, out_shape)
            ]
        else:
            if mask.masks.shape[0] == 0:
                return BitmapMasks(np.zeros((0, *out_shape)), *out_shape)
            mask_tensor = (
                torch.from_numpy(mask.masks[:, None, ...]).to(M.device).to(M.dtype)
            )
            return BitmapMasks(
                warp_affine(
                    mask_tensor,
                    M[None, ...].expand(mask.masks.shape[0], -1, -1),
                    out_shape,
                )
                .squeeze(1)
                .cpu()
                .numpy(),
                out_shape[0],
                out_shape[1],
            )

    @staticmethod
    def transform_image(img, M, out_shape):
        if isinstance(img, Sequence):
            assert len(img) == len(M)
            return [
                Transform2D.transform_image(b, m, shape)
                for b, m, shape in zip(img, M, out_shape)
            ]
        else:
            if img.dim() == 2:
                img = img[None, None, ...]
            elif img.dim() == 3:
                img = img[None, ...]

            return (
                warp_affine(img.float(), M[None, ...], out_shape, mode="nearest")
                .squeeze()
                .to(img.dtype)
            )

def gmm_policy(scores, given_gt_thr=0.5, policy='middle'):
    """The policy of choosing pseudo label.

    The previous GMM-B policy is used as default.
    1. Use the predicted bbox to fit a GMM with 2 center.
    2. Find the predicted bbox belonging to the positive
        cluster with highest GMM probability.
    3. Take the class score of the finded bbox as gt_thr.

    Args:
        scores (nd.array): The scores.

    Returns:
        float: Found gt_thr.

    """
    if len(scores) < 4:
        return given_gt_thr
    if isinstance(scores, torch.Tensor):
        scores = scores.cpu().numpy()
    if len(scores.shape) == 1:
        scores = scores[:, np.newaxis]
    means_init = [[np.min(scores)], [np.max(scores)]]
    weights_init = [1 / 2, 1 / 2]
    precisions_init = [[[1.0]], [[1.0]]]
    gmm = skm.GaussianMixture(
        2,
        weights_init=weights_init,
        means_init=means_init,
        precisions_init=precisions_init)
    gmm.fit(scores)
    gmm_assignment = gmm.predict(scores)
    gmm_scores = gmm.score_samples(scores)
    gmm_mean_1 = gmm.means_[1][0]
    assert policy in ['middle', 'high']
    if policy == 'high':
        if (gmm_assignment == 1).any():
            gmm_scores[gmm_assignment == 0] = -np.inf
            indx = np.argmax(gmm_scores, axis=0)
            pos_indx = (gmm_assignment == 1) & (
                scores >= scores[indx]).squeeze()
            pos_thr = float(scores[pos_indx].min())
            # pos_thr = max(given_gt_thr, pos_thr)
        else:
            pos_thr = given_gt_thr
    elif policy == 'middle':
        if (gmm_assignment == 1).any():
            pos_thr = float(scores[gmm_assignment == 1].min())
            # pos_thr = max(given_gt_thr, pos_thr)
        else:
            pos_thr = given_gt_thr

    pos_thr = max(gmm_mean_1,pos_thr)

    return pos_thr



# def thr_select_policy(scores, given_gt_thr=0.5, percent=35, vaild_len=100, policy='high'):
def thr_select_policy(scores, given_gt_thr=0.5, percent=35, vaild_len=100, policy='high'):
    """The policy of choosing pseudo label

    The previous GMM-B policy is used as default.
    1. Use the predicted bbox to fit a GMM with 2 center.
    2. Find the predicted bbox belonging to the positive
        cluster with highest GMM probability.
    3. Take the class score of the finded bbox as gt_thr.

    Args:
        scores (nd.array): The scores.

    Returns:
        float: Found gt_thr.

    """
    if len(scores) < vaild_len:
        return given_gt_thr
    
    if isinstance(scores, torch.Tensor):
        scores = scores.cpu().numpy()
    if isinstance(scores,list):
        scores = np.array(scores)
    if len(scores.shape) == 1:
        scores = scores[:, np.newaxis]
        
        
    if isinstance(percent,int):
        pos_thr = np.percentile(scores,percent)
        return pos_thr
    else:
        means_init = [[np.min(scores)], [np.max(scores)]]
        weights_init = [1 / 2, 1 / 2]
        precisions_init = [[[1.0]], [[1.0]]]
        gmm = skm.GaussianMixture(
            2,
            weights_init=weights_init,
            means_init=means_init,
            precisions_init=precisions_init)
        gmm.fit(scores)
        pos_thr = (gmm.means_[0][0]+gmm.means_[1][0])/2
        
        if percent == 'gmm' or  percent == 'gmm-mean':
            return pos_thr
        else:
            print("The percent of gmm must be in ['gmm', 'gmm-mean']")




def filter_invalid(bbox, label=None, score=None, mask=None, thr=0.0, min_size=0):
    if score is not None:
        valid = score > thr
        bbox = bbox[valid]
        if label is not None:
            label = label[valid]
        if mask is not None:
            mask = BitmapMasks(mask.masks[valid.cpu().numpy()], mask.height, mask.width)
    if min_size is not None:
        bw = bbox[:, 2]
        bh = bbox[:, 3]
        # bw = bbox[:, 2] - bbox[:, 0]
        # bh = bbox[:, 3] - bbox[:, 1]
        valid = (bw > min_size) & (bh > min_size)
        bbox = bbox[valid]
        if label is not None:
            label = label[valid]
        if mask is not None:
            mask = BitmapMasks(mask.masks[valid.cpu().numpy()], mask.height, mask.width)
    return bbox, label, mask


def get_adaptive_thr(class_metrix, cls_thr, bbox_list, label_list,vaild_len=10,percent=30,min=0.2,max=0.6, self_vaild_len = False):
    if self_vaild_len:
        thr_vaild_len = {0:500, 1:900, 2:150, 3:20, 4:500, 5:100, 6:20, 7:20, 8:20, 9:20, 10:100, 11:20, 12:20, 13:20, 14:20, 15:20, 16:20, 17:20, 18:20}
        for proposal, proposal_label in zip(bbox_list, label_list):
            for box, cls in zip(proposal,proposal_label):
                if len(class_metrix[cls.item()]) >= (thr_vaild_len[cls.item()]*5):
                    _ = class_metrix[cls.item()].pop(0)
                class_metrix[cls.item()].append(box[4].item())
        for key,value in class_metrix.items():
            if len(value) > thr_vaild_len[key]:
                cls_thr[key] = np.clip(np.percentile(value,percent),min,max)
            else:
                cls_thr[key] = np.clip(cls_thr[key],min,max)
        return class_metrix,cls_thr
    else:
        for proposal, proposal_label in zip(bbox_list, label_list):
            for box, cls in zip(proposal,proposal_label):
                if len(class_metrix[cls.item()]) >= 500:
                    _ = class_metrix[cls.item()].pop(0)
                class_metrix[cls.item()].append(box[4].item())
        for key,value in class_metrix.items():
            if len(value) > vaild_len:
                cls_thr[key] = np.clip(np.percentile(value,percent),min,max)
            else:
                cls_thr[key] = np.clip(cls_thr[key],min,max)
        return class_metrix,cls_thr

def get_adaptive_thr_gmm(class_metrix, cls_thr, bbox_list, label_list,vaild_len=10,percent=30,min=0.2,max=0.6, self_vaild_len = False):
    if self_vaild_len:
        thr_vaild_len = {0:500, 1:900, 2:150, 3:20, 4:500, 5:100, 6:20, 7:20, 8:20, 9:20, 10:100, 11:20, 12:20, 13:20, 14:20, 15:20, 16:20, 17:20, 18:20}
        for proposal, proposal_label in zip(bbox_list, label_list):
            for box, cls in zip(proposal,proposal_label):
                if len(class_metrix[cls.item()]) >= (thr_vaild_len[cls.item()]*5):
                    _ = class_metrix[cls.item()].pop(0)
                class_metrix[cls.item()].append(box[4].item())
        for key,value in class_metrix.items():
            if len(value) > thr_vaild_len[key]:
                gmm_thr = gmm_policy(np.array(value),given_gt_thr=cls_thr[key])
                cls_thr[key] = np.clip(gmm_thr,min,max)
            else:
                cls_thr[key] = np.clip(cls_thr[key],min,max)
        return class_metrix,cls_thr
    else:
        for proposal, proposal_label in zip(bbox_list, label_list):
            for box, cls in zip(proposal,proposal_label):
                if len(class_metrix[cls.item()]) >= 200:
                    _ = class_metrix[cls.item()].pop(0)
                class_metrix[cls.item()].append(box[4].item())
        for key,value in class_metrix.items():
            if len(value) > vaild_len:
                gmm_thr = gmm_policy(np.array(value),given_gt_thr=cls_thr[key])
                cls_thr[key] = np.clip(gmm_thr,min,max)
            else:
                cls_thr[key] = np.clip(cls_thr[key],min,max)
        return class_metrix,cls_thr


def get_adaptive_cls_base_thr(class_metrix, cls_thr, bbox_list, label_list,percent=35,min=0.5,max=0.9,vaild_len=100,keep_len = 200):
    preset_vaild_len = {0:500, 1:900, 2:150, 3:20, 4:500, 5:100, 6:20, 7:20, 8:20, 9:20, 10:100, 11:20, 12:20, 13:20, 14:20, 15:20, 16:20, 17:20, 18:20}
    for proposal, proposal_label in zip(bbox_list, label_list):
        for box, cls in zip(proposal,proposal_label):
            if keep_len == 'pre-set':
                if len(class_metrix[cls.item()]) >= (preset_vaild_len[cls.item()]*5):
                    _ = class_metrix[cls.item()].pop(0)
            else:
                if len(class_metrix[cls.item()]) >= keep_len:
                    _ = class_metrix[cls.item()].pop(0)
            class_metrix[cls.item()].append(box[5].item())

    for key,value in class_metrix.items():
        if vaild_len == 'pre-set':
            thr = thr_select_policy(np.array(value),given_gt_thr=cls_thr[key],percent=percent,vaild_len=preset_vaild_len[key])
        else:
            thr = thr_select_policy(np.array(value),given_gt_thr=cls_thr[key],percent=percent,vaild_len=vaild_len)
        cls_thr[key] = np.clip(thr,min,max)
    return class_metrix,cls_thr

def filter_invalid_with_adaptive_thr_cls(bbox, label, score, cls_thr, min_size=0):
    if bbox.size(0) == 0:
        return bbox,label
    else:
        valid = torch.tensor([sco > cls_thr[cls.item()] for sco, cls in zip(score,label)])
        bbox = bbox[valid]
        label = label[valid]
        if min_size is not None:
            bw = bbox[:, 2]
            bh = bbox[:, 3]
            valid = (bw > min_size) & (bh > min_size)
            bbox = bbox[valid]
            label = label[valid]
        return bbox, label



def get_adaptive_percent_thr(score_queue, percent_thr, bbox_list, label_list, percent=35, min=0.5, max=0.9, vaild_len=1000, keep_len = 2000):
    
    for proposal, proposal_label in zip(bbox_list, label_list):
        for box, cls in zip(proposal,proposal_label):
            if len(score_queue) >= keep_len:
                _ = score_queue.pop(0)
            score_queue.append(box[5].item())
    thr = thr_select_policy(np.array(score_queue),given_gt_thr=percent_thr,percent=percent,vaild_len=vaild_len)
    percent_thr = np.clip(thr,min,max)

    return score_queue, percent_thr


def filter_invalid_with_adaptive_percent(bbox, label, score, percent_thr, min_size=0):
    if bbox.size(0) == 0:
        return bbox,label
    else:
        valid = torch.tensor([sco > percent_thr for sco, cls in zip(score,label)])
        bbox = bbox[valid]
        label = label[valid]
        if min_size is not None:
            bw = bbox[:, 2]
            bh = bbox[:, 3]
            valid = (bw > min_size) & (bh > min_size)
            bbox = bbox[valid]
            label = label[valid]
        return bbox, label

def get_adaptive_reg_cls_thr(class_metrix_list, cls_thr_list, bbox_list, label_list,anchor_iou_list, vaild_len=10,percent=30,min=0.2,max=0.6,self_vaild_len = False):
    if self_vaild_len:
        thr_vaild_len = {0:100, 1:100, 2:100, 3:20, 4:100, 5:100, 6:20, 7:20, 8:20, 9:20, 10:100, 11:20, 12:20, 13:20, 14:20, 15:20, 16:20, 17:20, 18:20}
        reg_level = len(class_metrix_list)
        for proposal, proposal_label, proposal_anchor_iou in zip(bbox_list, label_list, anchor_iou_list):
            for box, cls, iou in zip(proposal,proposal_label,proposal_anchor_iou):
                reg_level_idx = math.floor(iou.item()*reg_level)
                if reg_level_idx == reg_level:
                    reg_level_idx = reg_level - 1
                if len(class_metrix_list[reg_level_idx][cls.item()]) >= (thr_vaild_len[cls.item()]*2):
                    _ = class_metrix_list[reg_level_idx][cls.item()].pop(0)
                class_metrix_list[reg_level_idx][cls.item()].append(box[4].item())
        for ind in range(reg_level):
            for key,value in class_metrix_list[ind].items():
                if len(value) > thr_vaild_len[key]:
                    cls_thr_list[ind][key] = np.clip(np.percentile(value,percent),min,max)
                else:
                    cls_thr_list[ind][key] = np.clip(cls_thr_list[ind][key],min,max)
        return class_metrix_list, cls_thr_list
    else:
        reg_level = len(class_metrix_list)
        for proposal, proposal_label, proposal_anchor_iou in zip(bbox_list, label_list, anchor_iou_list):
            for box, cls, iou in zip(proposal,proposal_label,proposal_anchor_iou):
                reg_level_idx = math.floor(iou.item()*reg_level)
                if reg_level_idx == reg_level:
                    reg_level_idx = reg_level - 1
                if len(class_metrix_list[reg_level_idx][cls.item()]) >= 200:
                    _ = class_metrix_list[reg_level_idx][cls.item()].pop(0)
                class_metrix_list[reg_level_idx][cls.item()].append(box[4].item())
        for ind in range(reg_level):
            for key,value in class_metrix_list[ind].items():
                if len(value) > vaild_len:
                    cls_thr_list[ind][key] = np.clip(np.percentile(value,percent),min,max)
                else:
                    cls_thr_list[ind][key] = np.clip(cls_thr_list[ind][key],min,max)
        return class_metrix_list, cls_thr_list


def get_adaptive_reg_cls_thr_gmm(class_metrix_list, cls_thr_list, bbox_list, label_list,anchor_iou_list, vaild_len=10,percent=30,min=0.2,max=0.6,self_vaild_len = False):
    if self_vaild_len:
        thr_vaild_len = {0:100, 1:100, 2:100, 3:20, 4:100, 5:100, 6:20, 7:20, 8:20, 9:20, 10:100, 11:20, 12:20, 13:20, 14:20, 15:20, 16:20, 17:20, 18:20}
        reg_level = len(class_metrix_list)
        for proposal, proposal_label, proposal_anchor_iou in zip(bbox_list, label_list, anchor_iou_list):
            for box, cls, iou in zip(proposal,proposal_label,proposal_anchor_iou):
                reg_level_idx = math.floor(iou.item()*reg_level)
                if reg_level_idx == reg_level:
                    reg_level_idx = reg_level - 1
                if len(class_metrix_list[reg_level_idx][cls.item()]) >= (thr_vaild_len[cls.item()]*2):
                    _ = class_metrix_list[reg_level_idx][cls.item()].pop(0)
                class_metrix_list[reg_level_idx][cls.item()].append(box[5].item())
        for ind in range(reg_level):
            for key,value in class_metrix_list[ind].items():
                if len(value) > thr_vaild_len[key]:
                    gmm_thr = gmm_policy(np.array(value),given_gt_thr=cls_thr_list[ind][key])
                    cls_thr_list[ind][key] = np.clip(gmm_thr,min,max)
                    # cls_thr_list[ind][key] = np.clip(np.percentile(value,percent),min,max)
                else:
                    cls_thr_list[ind][key] = np.clip(cls_thr_list[ind][key],min,max)
        return class_metrix_list, cls_thr_list
    else:
        reg_level = len(class_metrix_list)
        for proposal, proposal_label, proposal_anchor_iou in zip(bbox_list, label_list, anchor_iou_list):
            for box, cls, iou in zip(proposal,proposal_label,proposal_anchor_iou):
                reg_level_idx = math.floor(iou.item()*reg_level)
                if reg_level_idx == reg_level:
                    reg_level_idx = reg_level - 1
                if len(class_metrix_list[reg_level_idx][cls.item()]) >= 200:
                    _ = class_metrix_list[reg_level_idx][cls.item()].pop(0)
                class_metrix_list[reg_level_idx][cls.item()].append(box[5].item())
        for ind in range(reg_level):
            for key,value in class_metrix_list[ind].items():
                if len(value) > vaild_len:
                    cls_thr_list[ind][key] = np.clip(np.percentile(value,percent),min,max)
                else:
                    cls_thr_list[ind][key] = np.clip(cls_thr_list[ind][key],min,max)
        return class_metrix_list, cls_thr_list


def get_adaptive_reg_cls_base_thr(class_metrix_list, cls_thr_list, bbox_list, label_list, anchor_iou_list, percent=35, min=0.5,max=0.9,vaild_len=100, keep_len = 200):
    preset_vaild_len = {0:100, 1:100, 2:100, 3:20, 4:100, 5:100, 6:20, 7:20, 8:20, 9:20, 10:100, 11:20, 12:20, 13:20, 14:20, 15:20, 16:20, 17:20, 18:20}
    reg_level = len(class_metrix_list)
    for proposal, proposal_label, proposal_anchor_iou in zip(bbox_list, label_list, anchor_iou_list):
        for box, cls, iou in zip(proposal,proposal_label,proposal_anchor_iou):
            reg_level_idx = math.floor(iou.item()*reg_level)
            if reg_level_idx == reg_level:
                reg_level_idx = reg_level - 1
            if keep_len == 'pre-set':
                if len(class_metrix_list[reg_level_idx][cls.item()]) >= (preset_vaild_len[cls.item()]*5):
                    _ = class_metrix_list[reg_level_idx][cls.item()].pop(0)
            else:
                if len(class_metrix_list[reg_level_idx][cls.item()]) >= keep_len:
                    _ = class_metrix_list[reg_level_idx][cls.item()].pop(0)
            class_metrix_list[reg_level_idx][cls.item()].append(box[5].item())

    for ind in range(reg_level):
        for key,value in class_metrix_list[ind].items():
            if vaild_len == 'pre-set':
                thr = thr_select_policy(np.array(value),given_gt_thr=cls_thr_list[ind][key],percent=percent,vaild_len=preset_vaild_len[key])
            else:
                thr = thr_select_policy(np.array(value),given_gt_thr=cls_thr_list[ind][key],percent=percent,vaild_len=vaild_len)
            cls_thr_list[ind][key] = np.clip(thr,min,max)
    return class_metrix_list, cls_thr_list



def filter_invalid_with_adaptive_reg_cls_thr(bbox, label, score, anchor_iou, cls_thr_list, min_size=0):
    reg_level = len(cls_thr_list)
    if bbox.size(0) == 0:
        return bbox,label
    else:
        valid = torch.tensor([sco > cls_thr_list[math.floor(iou.item()*reg_level)][cls.item()] for sco, cls, iou in zip(score,label,anchor_iou)])
        bbox = bbox[valid]
        label = label[valid]
        if min_size is not None:
            bw = bbox[:, 2]
            bh = bbox[:, 3]
            # bw = bbox[:, 2] - bbox[:, 0]
            # bh = bbox[:, 3] - bbox[:, 1]
            valid = (bw > min_size) & (bh > min_size)
            bbox = bbox[valid]
            label = label[valid]
        return bbox, label


def permute_to_N_HWA_K(tensor, K):
    """
    Transpose/reshape a tensor from (N, (A x K), H, W) to (N, (HxWxA), K)
    """
    if tensor.dim() == 3:
        tensor = tensor.unsqueeze(0)
    assert tensor.dim() == 4, tensor.shape
    N, _, H, W = tensor.shape
    tensor = tensor.view(N, -1, K, H, W)
    tensor = tensor.permute(0, 3, 4, 1, 2)
    tensor = tensor.reshape(N, -1, K)  # Size=(N,HWA,K)
    return tensor


def QFLv2(pred_sigmoid,          # (n, 80)
          teacher_sigmoid,         # (n) 0, 1-80: 0 is neg, 1-80 is positive
          weight=None,
          beta=2.0,
          reduction='mean'):
    # all goes to 0
    pt = pred_sigmoid
    zerolabel = pt.new_zeros(pt.shape)
    loss = F.binary_cross_entropy_with_logits(
        pred_sigmoid, zerolabel, reduction='none') * pt.pow(beta)
    # loss = torch.nn.BCEWithLogitsLoss(
    #     pred_sigmoid, zerolabel, reduction='none') * pt.pow(beta)
    pos = weight > 0

    # positive goes to bbox quality
    pt = teacher_sigmoid[pos] - pred_sigmoid[pos]
    loss[pos] = F.binary_cross_entropy_with_logits(
        pred_sigmoid[pos], teacher_sigmoid[pos], reduction='none') * pt.pow(beta)

    valid = weight >= 0
    if reduction == "mean":
        loss = loss[valid].mean()
    elif reduction == "sum":
        loss = loss[valid].sum()
    return loss