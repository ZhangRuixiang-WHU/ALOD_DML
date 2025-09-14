import pdb
import os
import random
import math
import json
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.nn import functional as F

import mmcv
from mmcv.runner.fp16_utils import force_fp32
from mmcv.ops import nms, nms_rotated

from mmdet.core import multi_apply, bbox2roi
from mmdet.core.bbox.iou_calculators import bbox_overlaps

from mmrotate import ROTATED_DETECTORS, build_detector
from mmrotate.core import rbbox2roi  # Supplement missing module import
from mmrotate.core.bbox.iou_calculators import rbbox_overlaps, build_iou_calculator
from mmrotate.core.bbox.transforms import (poly2obb_le90, obb2poly_le90, 
                                           hbb2obb, obb2hbb)

from src.utils import log_every_n
from src.utils.structure_utils import dict_split
from .multi_stream_detector import MultiSteamDetector
from .utils import (Transform2D, filter_invalid, 
                    filter_invalid_with_adaptive_reg_cls_thr,
                    get_adaptive_reg_cls_base_thr, get_adaptive_cls_base_thr,
                    filter_invalid_with_adaptive_thr_cls)


# Global Configuration
DEBUG = False  # Debug mode switch
DEFAULT_MASK_THR = 0.1  # Default mask threshold
DEFAULT_BURN_IN_ITER = 5000  # Default burn-in iteration count
DEFAULT_CUTMIX_ITER = 10000  # Default CutMix activation iteration count
DEFAULT_CLS_MAP_VERSION = 'dota1'  # Default class mapping version
DEFAULT_PERCENT_VERSION = '1ins'  # Default data percentage version


def save_json(save_path, data):
    """
    Save data to a JSON file
    
    Args:
        save_path: Path to save the JSON file (must end with .json)
        data: Dictionary data to be saved
    """
    assert save_path.split('.')[-1] == 'json', "Save path must end with .json"
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def load_json(file_path):
    """
    Load data from a JSON file
    
    Args:
        file_path: Path to the JSON file (must end with .json)
    
    Returns:
        Loaded dictionary data
    """
    assert file_path.split('.')[-1] == 'json', "File path must end with .json"
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def pointobb2thetaobb_180(pointobb):
    """
    Convert point-format bounding box (8 coordinates) to angle-format bounding box (x,y,w,h,theta)
    
    Args:
        pointobb: Point-format bounding box with shape (8,) or (1,8)
    
    Returns:
        Angle-format bounding box as a list [x, y, w, h, theta] (integer type)
    """
    # Adjust format: Reshape to 4x2 point matrix
    pointobb = np.int0(pointobb)
    pointobb = pointobb.reshape(4, 2)
    
    # Calculate minimum area rectangle
    rect = cv2.minAreaRect(pointobb)
    x, y = rect[0]  # Center coordinates
    w, h = rect[1]  # Width and height
    theta = rect[2]  # Rotation angle (degrees)
    
    return [int(x), int(y), int(w), int(h), int(theta)]


def hbb_le902hbb(hbb_le90):
    """
    Convert LE90-format horizontal bounding box (x,y,w,h,theta) to standard horizontal bounding box (x1,y1,x2,y2)
    
    Args:
        hbb_le90: LE90-format bounding box with shape (N,5), where theta=0 means normal orientation and theta=1 means 90-degree rotation
    
    Returns:
        Standard horizontal bounding box with shape (N,4)
    """
    # For theta=0: x1=x-w/2, y1=y-h/2, x2=x+w/2, y2=y+h/2
    hbb_0 = torch.stack([
        hbb_le90[:, 0] - hbb_le90[:, 2] / 2,
        hbb_le90[:, 1] - hbb_le90[:, 3] / 2,
        hbb_le90[:, 0] + hbb_le90[:, 2] / 2,
        hbb_le90[:, 1] + hbb_le90[:, 3] / 2
    ], dim=-1)
    
    # For theta=1 (90-degree rotation): x1=x-h/2, y1=y-w/2, x2=x+h/2, y2=y+w/2
    hbb_pi = torch.stack([
        hbb_le90[:, 0] - hbb_le90[:, 3] / 2,
        hbb_le90[:, 1] - hbb_le90[:, 2] / 2,
        hbb_le90[:, 0] + hbb_le90[:, 3] / 2,
        hbb_le90[:, 1] + hbb_le90[:, 2] / 2
    ], dim=-1)
    
    # Select corresponding format based on theta
    return torch.where((hbb_le90[:, 4] == 0).unsqueeze(-1), hbb_0, hbb_pi)


def find_matching_rectangles(rectangles, target_width, target_height):
    """
    Find rectangles in the list with width and height close to target values (within 10% error range)
    
    Args:
        rectangles: List of rectangle dictionaries, each containing 'width' and 'height' keys
        target_width: Target width
        target_height: Target height
    
    Returns:
        List of matching rectangle dictionaries
    """
    width_range = (target_width * 0.9, target_width * 1.1)
    height_range = (target_height * 0.9, target_height * 1.1)
    
    return [
        rect for rect in rectangles
        if (width_range[0] <= rect['width'] <= width_range[1] and
            height_range[0] <= rect['height'] <= height_range[1])
    ]


def find_matching_rectangles_large_range(rectangles, target_width, target_height):
    """
    Find rectangles in the list with width and height within a larger range of target values (50%-200% error range)
    
    Args:
        rectangles: List of rectangle dictionaries, each containing 'width' and 'height' keys
        target_width: Target width
        target_height: Target height
    
    Returns:
        List of matching rectangle dictionaries
    """
    width_range = (target_width * 0.5, target_width * 2.0)
    height_range = (target_height * 0.5, target_height * 2.0)
    
    return [
        rect for rect in rectangles
        if (width_range[0] <= rect['width'] <= width_range[1] and
            height_range[0] <= rect['height'] <= height_range[1])
    ]


def find_match_patch_ratio(supp_box_obb, supp_cls, patch_json, img_pseudo_bank=None, 
                          cls_map_version=DEFAULT_CLS_MAP_VERSION,
                          percent_version=DEFAULT_PERCENT_VERSION, zoom_ratio=1.0):
    """
    Match corresponding patch images based on target bounding box and class (added matching with high-confidence pseudo-labels from current image)
    
    Args:
        supp_box_obb: Target bounding box (LE90 format, containing x,y,w,h,theta)
        supp_cls: Target class (tensor format)
        patch_json: Global patch library JSON data (keys are class names, values are patch lists for that class)
        img_pseudo_bank: Temporary patch library built from high-confidence pseudo-labels of current image (optional)
        cls_map_version: Class mapping version (default 'dota1')
        percent_version: Data percentage version (default '1ins')
        zoom_ratio: Zoom ratio (default 1.0)
    
    Returns:
        Matched patch image (BGR format, numpy array), returns None if no match
    """
    # 1. Define DOTA1 dataset class mapping
    if cls_map_version == 'dota1':
        cls_map = {
            0: 'plane', 1: 'baseball-diamond', 2: 'bridge', 3: 'ground-track-field',
            4: 'small-vehicle', 5: 'large-vehicle', 6: 'ship', 7: 'tennis-court',
            8: 'basketball-court', 9: 'storage-tank', 10: 'soccer-ball-field',
            11: 'roundabout', 12: 'harbor', 13: 'swimming-pool', 14: 'helicopter'
        }
        # Define patch root directory mapping for different percentage versions
        patch_root_dir_map = {
            '1ins': 'data/dota1/train_obb/split_images/annfiles_1ins_cut_obb_patches',
            '2ins': 'data/dota1/train_obb/split_images/annfiles_2ins_cut_obb_patches',
            '3ins': 'data/dota1/train_obb/split_images/annfiles_3ins_cut_obb_patches'
        }
        if percent_version not in patch_root_dir_map:
            raise ValueError(f"Invalid percent_version: {percent_version}, available values are 1ins/2ins/3ins")
        patch_root_dir = patch_root_dir_map[percent_version]
    else:
        raise NotImplementedError(f"Unsupported cls_map_version: {cls_map_version}, please add dataset configuration")

    # 2. Get patch library corresponding to target class (prioritize using current image's high-confidence pseudo-label patch library)
    cls_name = cls_map[supp_cls.item()]
    # Merge current image's pseudo-label patch library with global patch library
    combined_patch_bank = []
    # Step 1: Add current image's high-confidence pseudo-label patches (if exist)
    if img_pseudo_bank is not None and cls_name in img_pseudo_bank:
        combined_patch_bank.extend(img_pseudo_bank[cls_name])
        log_every_n(f"Number of current image pseudo-label patches for class {cls_name}: {len(img_pseudo_bank[cls_name])}")
    # Step 2: Add global patch library patches (to ensure matching diversity)
    if cls_name in patch_json:
        combined_patch_bank.extend(patch_json[cls_name])
        log_every_n(f"Number of global patches for class {cls_name}: {len(patch_json[cls_name])}")
    
    # Return None if no patches after merging
    if not combined_patch_bank:
        log_every_n(f"Warning: No corresponding patches for class {cls_name} in both current image pseudo-label patch library and global patch library")
        return None
    patch_bank = combined_patch_bank

    # 3. Calculate target width and height (original size and scaled size)
    dst_w, dst_h = supp_box_obb[2].item(), supp_box_obb[3].item()
    dst_w_org, dst_h_org = dst_w / zoom_ratio, dst_h / zoom_ratio

    # 4. Multi-round patch matching (prioritize exact matching, then expand range)
    closest_rect = find_matching_rectangles(patch_bank, dst_w, dst_h)
    if not closest_rect:
        closest_rect = find_matching_rectangles_large_range(patch_bank, dst_w_org, dst_h_org)
    if not closest_rect:
        closest_rect = find_matching_rectangles_large_range(patch_bank, dst_w, dst_h)
    if not closest_rect:
        log_every_n(f"No matching patches: class {cls_name}, target dimensions ({dst_w:.1f},{dst_h:.1f})")
        return None

    # 5. Randomly select a patch and load with scaling
    select_patch = random.sample(closest_rect, 1)[0]
    # Distinguish patch sources: current image pseudo-labels (contain image data) vs global patch library (need file reading)
    if 'img_data' in select_patch:
        # Current image pseudo-label patch: directly extract image data
        patch_img = select_patch['img_data']
    else:
        # Global patch library: read file
        patch_filename = select_patch['name']
        patch_filepath = os.path.join(patch_root_dir, patch_filename)
        # Check if patch file exists
        if not os.path.exists(patch_filepath):
            log_every_n(f"Patch file does not exist: {patch_filepath}")
            return None
        # Load patch image
        patch_img = cv2.imread(patch_filepath)
        if patch_img is None:
            log_every_n(f"Failed to read patch file: {patch_filepath}")
            return None
    
    # Resize to target dimensions
    supplement_image = cv2.resize(patch_img, (int(dst_w), int(dst_h)))
    
    return supplement_image


@ROTATED_DETECTORS.register_module()
class DML_ALOD(MultiSteamDetector):
    """
    Attribute-aware Label-efficient Object Detection model (ALOD) based on Dynamic Multi-view Learning (DML)
    
    Core features:
    1. Teacher-student dual-model architecture, where teacher model generates pseudo-labels
    2. Supports fusion training with pseudo-labels and real labels
    3. Implements CutMix data augmentation (rotated target replacement based on patch library + current image high-confidence pseudo-labels)
    4. Cross-consistency loss (CCL) to optimize model consistency
    """
    def __init__(self, model: dict, train_cfg=None, test_cfg=None):
        super(DML_ALOD, self).__init__(
            dict(teacher=build_detector(model), student=build_detector(model)),
            train_cfg=train_cfg,
            test_cfg=test_cfg,
        )

        # Training configuration initialization
        if train_cfg is not None:
            self.current_iter = 0  # Current iteration count
            self.freeze("teacher")  # Freeze teacher model
            self.unsup_weight = train_cfg.unsup_weight  # Unsupervised loss weight
            self.ori_thr = 0.4  # Original threshold
            self.thr_after_cali = 0.7  # Threshold after calibration
            self.ori_do_merge = True  # Original label fusion switch
            self.ccl_weight = train_cfg.ccl_weight  # CCL loss weight
            
            # Load patch library JSON
            self.patch_json = load_json(train_cfg.patch_json_path)
            
            # Initialize save directory and log file
            self.save_dir = train_cfg.save_dir
            os.makedirs(self.save_dir, exist_ok=True)
            self.save_mask_img_dir = os.path.join(self.save_dir, 'save_mask_img')
            os.makedirs(self.save_mask_img_dir, exist_ok=True)
            self.save_txt_path = os.path.join(self.save_dir, 'thr_list.txt')
            self.save_txt_file = open(self.save_txt_path, 'w', encoding='utf-8')

            # Load optional configurations (use default values if not configured)
            self._load_optional_config(train_cfg)
        else:
            # No need to initialize training-related parameters in test mode
            self.train_cfg = None

    def _load_optional_config(self, train_cfg):
        """Load optional training configurations, use default values if not configured"""
        # Mask-related configurations
        self.use_mask = self._get_optional_cfg(train_cfg, 'use_mask', default=True,
                                              msg="use_mask not set, using default value True")
        self.mask_thr = self._get_optional_cfg(train_cfg, 'maske_thr', default=DEFAULT_MASK_THR,
                                              msg="mask_thr not set, using default value 0.1")
        
        # Iteration-related configurations
        self.burn_in_iter = self._get_optional_cfg(train_cfg, 'burn_in_iter', default=DEFAULT_BURN_IN_ITER,
                                                 msg="burn_in_iter not set, using default value 5000")
        self.cutmix_iter = self._get_optional_cfg(train_cfg, 'cutmix_iter', default=DEFAULT_CUTMIX_ITER,
                                                msg="cutmix_iter not set, using default value 10000")
        
        # Data-related configurations
        self.cls_map_version = self._get_optional_cfg(train_cfg, 'cls_map_version', default=DEFAULT_CLS_MAP_VERSION,
                                                    msg=f"cls_map_version not set, using default value {DEFAULT_CLS_MAP_VERSION}")
        self.percent_version = self._get_optional_cfg(train_cfg, 'percent_version', default=DEFAULT_PERCENT_VERSION,
                                                    msg=f"percent_version not set, using default value {DEFAULT_PERCENT_VERSION}")
        
        # Other training configurations
        self.mining_warmup = self._get_optional_cfg(train_cfg, 'mining_warmup', default=0,
                                                  msg="mining_warmup not set, using default value 0")
        self.use_stong_popsosals = self._get_optional_cfg(train_cfg, 'use_stong_popsosals', default=False,
                                                        msg="use_stong_popsosals not set, using default value False")
        self.use_stong_popsosals_for_CCL = self._get_optional_cfg(train_cfg, 'use_stong_popsosals_for_CCL', default=False,
                                                                msg="use_stong_popsosals_for_CCL not set, using default value False")
        self.mask_mode = self._get_optional_cfg(train_cfg, 'mask_mode', default='nomask',
                                              msg="mask_mode not set, using default value 'nomask'")
        self.mining_min_size = self._get_optional_cfg(train_cfg, 'mining_min_size', default=-1,
                                                    msg="mining_min_size not set, using default value -1 (no filtering)")
        self.mining_th_score = self._get_optional_cfg(train_cfg, 'mining_th_score', default=0.5,
                                                    msg="mining_th_score not set, using default value 0.5")
        self.max_iof = self._get_optional_cfg(train_cfg, 'max_iof', default=0.0,
                                            msg="max_iof not set, using default value 0.0")
        # New: Current image pseudo-label patch library configuration (high confidence threshold, same as calibrated threshold by default)
        self.img_pseudo_thr = self._get_optional_cfg(train_cfg, 'img_pseudo_thr', default=self.thr_after_cali,
                                                   msg=f"img_pseudo_thr not set, using default value {self.thr_after_cali}")
        
        # Calibration-related configurations (not enabled by default)
        self.start_fit_iter = self._get_optional_cfg(train_cfg, 'start_fit_iter', default=1e9,
                                                   msg="start_fit_iter not set, calibration not enabled by default")
        self.finished_fit = self._get_optional_cfg(train_cfg, 'finished_fit', default=False,
                                                  msg="finished_fit not set, calibration not enabled by default")

    def _get_optional_cfg(self, cfg, key, default, msg):
        """Get optional configuration, print prompt and return default value if not configured"""
        if hasattr(cfg, key):
            return getattr(cfg, key)
        else:
            print(f"Warning: {msg}")
            return default

    def get_distence(self, x, y):
        """Calculate Euclidean distance between two points"""
        return math.hypot(abs(x[0] - y[0]), abs(x[1] - y[1]))

    def _build_img_pseudo_bank(self, img_chw, img_meta, pseudo_bboxes, pseudo_labels, pseudo_scores):
        """
        Build temporary patch library from high-confidence pseudo-labels of current image
        
        Args:
            img_chw: Image tensor (C,H,W)
            img_meta: Image meta information (contains size, filename, etc.)
            pseudo_bboxes: Current image pseudo-label boxes (LE90 format, N×5)
            pseudo_labels: Current image pseudo-label classes (N)
            pseudo_scores: Current image pseudo-label confidence scores (N)
        
        Returns:
            Current image pseudo-label patch library dictionary, with keys as class names and values as lists of patch information (including width, height, image data)
        """
        # 1. Initialize patch library dictionary
        img_pseudo_bank = {}
        if self.cls_map_version == 'dota1':
            cls_map = {
                0: 'plane', 1: 'baseball-diamond', 2: 'bridge', 3: 'ground-track-field',
                4: 'small-vehicle', 5: 'large-vehicle', 6: 'ship', 7: 'tennis-court',
                8: 'basketball-court', 9: 'storage-tank', 10: 'soccer-ball-field',
                11: 'roundabout', 12: 'harbor', 13: 'swimming-pool', 14: 'helicopter'
            }
        else:
            raise NotImplementedError(f"Unsupported cls_map_version: {self.cls_map_version}")

        # 2. Filter high-confidence pseudo-labels (score >= img_pseudo_thr)
        high_conf_mask = pseudo_scores >= self.img_pseudo_thr
        if not high_conf_mask.any():
            log_every_n(f"No high-confidence pseudo-labels in current image (confidence threshold {self.img_pseudo_thr})")
            return img_pseudo_bank
        
        # Extract high-confidence pseudo-labels
        high_conf_boxes = pseudo_bboxes[high_conf_mask]
        high_conf_labels = pseudo_labels[high_conf_mask]
        high_conf_scores = pseudo_scores[high_conf_mask]

        # 3. Image format conversion (C,H,W -> H,W,C) and denormalization
        img_h, img_w, _ = img_meta['img_shape']
        img_norm = img_chw.permute(1, 2, 0).clone().detach()  # Avoid gradient propagation
        img_norm[:, :, 0] += 103.530
        img_norm[:, :, 1] += 116.280
        img_norm[:, :, 2] += 123.675
        img_np = img_norm.to('cpu').numpy().astype(np.uint8)
        img_np = np.ascontiguousarray(img_np)

        # 4. Iterate through high-confidence pseudo-labels to crop and generate patches
        for idx in range(high_conf_boxes.shape[0]):
            # Extract pseudo-label information
            obb = high_conf_boxes[idx, :5]  # LE90 format: x,y,w,h,theta
            cls_idx = high_conf_labels[idx].item()
            cls_name = cls_map[cls_idx]
            score = high_conf_scores[idx].item()

            # Convert pseudo-label to polygon format (for cropping)
            poly = obb2poly_le90(obb.unsqueeze(0)).squeeze(0).to('cpu').numpy()
            poly = np.clip(poly, 0, max(img_w-1, img_h-1))  # Prevent out-of-bounds
            poly = poly.reshape(4, 2).astype(np.int32)

            # Calculate bounding rectangle of pseudo-label area (for image cropping)
            x_min, y_min = np.min(poly[:, 0]), np.min(poly[:, 1])
            x_max, y_max = np.max(poly[:, 0]), np.max(poly[:, 1])
            patch_w = x_max - x_min + 1
            patch_h = y_max - y_min + 1

            # Filter too small patches (avoid invalid cropping)
            if patch_w < 5 or patch_h < 5:
                log_every_n(f"Skipping too small pseudo-label patch: width {patch_w}, height {patch_h}")
                continue

            # Crop pseudo-label area as patch image
            patch_img = img_np[y_min:y_max+1, x_min:x_max+1, :]

            # Add to patch library (grouped by class)
            if cls_name not in img_pseudo_bank:
                img_pseudo_bank[cls_name] = []
            img_pseudo_bank[cls_name].append({
                'name': f"img_pseudo_{idx}_cls{cls_idx}_score{score:.2f}.png",
                'width': patch_w,
                'height': patch_h,
                'score': score,
                'img_data': patch_img  # Store cropped image data to avoid repeated file reading
            })

        log_every_n(f"Current image high-confidence pseudo-label patch library built, containing {len(img_pseudo_bank.keys())} classes")
        return img_pseudo_bank

    def forward_train(self, img, img_metas, **kwargs):
        """
        Training forward propagation
        
        Args:
            img: Input image tensor
            img_metas: List of image meta information
            **kwargs: Other training parameters (such as gt_bboxes, gt_labels, tag, etc.)
        
        Returns:
            Loss dictionary containing supervised loss and optional unsupervised loss
        """
        # Call parent class training forward propagation (initialize teacher/student model inputs)
        super().forward_train(img, img_metas,** kwargs)
        self.current_iter += 1

        # 1. Control whether to enable label fusion during burn-in phase (no fusion during burn-in to avoid impact from low-quality pseudo-labels)
        if self.current_iter <= self.burn_in_iter:
            self.train_cfg.do_merge = False
        else:
            self.train_cfg.do_merge = self.ori_do_merge

        # 2. Data grouping: distinguish weak/strong augmented data by 'tag'
        kwargs.update({"img": img, "img_metas": img_metas, 
                      "tag": [meta["tag"] for meta in img_metas]})
        data_groups = dict_split(kwargs, "tag")
        for _, v in data_groups.items():
            v.pop("tag")  # Remove tag key to avoid errors in subsequent processing

        # 3. Initialize loss dictionary
        loss = {}

        # 4. Label mining and fusion (only enabled after burn-in)
        if self.current_iter >= self.mining_warmup:
            # Extract real labels from strong augmented data
            gt_bboxes = data_groups["strong"]["gt_bboxes"]
            gt_labels = data_groups["strong"]["gt_labels"]
            
            # Log: count average number of real labels per image
            avg_sup_gt_num = sum([len(bbox) for bbox in gt_bboxes]) / len(gt_bboxes)
            log_every_n({"sup_gt_num": avg_sup_gt_num})

            # Label fusion: supplement real labels with pseudo-labels generated by teacher model
            if self.train_cfg.do_merge:
                # Teacher model generates pseudo-labels (including filtered and all pseudo-labels)
                (pseudo_bboxes, pseudo_labels, 
                 all_pseudo_bboxes, all_pseudo_labels, 
                 teacher_info, student_info) = self.forward_teacher(data_groups["weak"], data_groups["strong"])
                
                # Merge real labels with pseudo-labels
                merge_bboxes, merge_labels = self.merge_rboxes(gt_bboxes, gt_labels, 
                                                              pseudo_bboxes, pseudo_labels)
                
                # Generate proposals for CCL if strong proposals are enabled
                strong_student_proposals = None
                if self.use_stong_popsosals:
                    merge_all_bboxes, merge_all_labels = self.merge_rboxes(gt_bboxes, gt_labels,
                                                                        all_pseudo_bboxes, all_pseudo_labels)
                    # Convert to horizontal bounding box format
                    strong_student_proposals = [
                        hbb_le902hbb(obb2hbb(boxes[:, :5], version='le90')) 
                        for boxes in merge_all_bboxes
                    ]

                # Update labels of strong data to merged labels
                data_groups["strong"]["gt_bboxes"] = [b[:, :5] for b in merge_bboxes]
                data_groups["strong"]["gt_labels"] = merge_labels

                # Enable CutMix augmentation (replace target areas with patch images + current image high-confidence pseudo-labels)
                if self.mask_mode == 'cutmix' and self.current_iter > self.cutmix_iter:
                    self._apply_cutmix_aug(data_groups, merge_bboxes, merge_labels,
                                          all_pseudo_bboxes, all_pseudo_labels,
                                          pseudo_bboxes, pseudo_labels)  

        # 5. Student model training (using strong augmented data)
        student_losses = self.student.forward_train(**data_groups["strong"])
        loss.update(** student_losses)

        # 6. Calculate cross-consistency loss (CCL)
        if self.train_cfg.do_merge and self.ccl_weight > 0:
            ccl_loss = self.unsup_cross_consistency_loss(teacher_info, student_info,
                                                       strong_student_proposals)
            loss.update(**ccl_loss)

        return loss

    def _apply_cutmix_aug(self, data_groups, merge_bboxes, merge_labels, all_pseudo_bboxes,
                         all_pseudo_labels, pseudo_bboxes, pseudo_labels):
        """
        Apply CutMix data augmentation (added current image high-confidence pseudo-labels as patch sources)
        
        Args:
            data_groups: Data grouping dictionary (contains strong data)
            merge_bboxes: List of merged label boxes
            merge_labels: List of merged labels
            all_pseudo_bboxes: List of all pseudo-label boxes (including low confidence)
            all_pseudo_labels: List of all pseudo-labels (including low confidence)
            pseudo_bboxes: List of filtered high-confidence pseudo-label boxes (newly added)
            pseudo_labels: List of filtered high-confidence pseudo-labels (newly added)
        """
        masked_imgs = []  # List of augmented images
        mix_aug_boxes = []  # List of augmented label boxes
        mix_aug_labels = []  # List of augmented labels

        # Iterate through parameters for each image (added pseudo_bboxes and pseudo_labels)
        iter_params = zip(all_pseudo_bboxes, all_pseudo_labels, 
                         pseudo_bboxes, pseudo_labels,  # New: high-confidence pseudo-labels
                         merge_bboxes, merge_labels,
                         data_groups["strong"]["img"], data_groups["strong"]["img_metas"])
        for (supp_boxes, supp_labels, high_conf_boxes, high_conf_labels,
             sup_boxes, sup_labels, img_chw, img_meta) in iter_params:
            # Extract basic image information
            img_filename = img_meta['filename'].split('/')[-1]
            img_h, img_w, _ = img_meta['img_shape']
            device = img_chw.device

            # Keep original image if no pseudo-labels
            if supp_boxes.size(0) == 0:
                masked_imgs.append(img_chw)
                mix_aug_boxes.append(supp_boxes)
                mix_aug_labels.append(supp_labels)
                continue

            # -------------------------- New: Build current image high-confidence pseudo-label patch library --------------------------
            # Extract confidence scores of high-confidence pseudo-labels (match from all_pseudo_bboxes)
            # Step 1: Get indices of high-confidence pseudo-labels in all_pseudo_bboxes (based on IOU matching)
            high_conf_indices = []
            if high_conf_boxes.size(0) > 0 and supp_boxes.size(0) > 0:
                # Calculate IOU between high-confidence pseudo-labels and all pseudo-labels
                iou_matrix = rbbox_overlaps(high_conf_boxes[:, :5], supp_boxes[:, :5])
                max_iou, max_idx = iou_matrix.max(dim=1)
                # Match pseudo-labels with IOU > 0.95 (ensure same target)
                valid_match = max_iou >= 0.95
                if valid_match.any():
                    high_conf_indices = max_idx[valid_match].unique()
            # Step 2: Extract confidence scores of high-confidence pseudo-labels
            high_conf_scores = torch.tensor([], device=device)
            if len(high_conf_indices) > 0:
                high_conf_scores = supp_boxes[high_conf_indices, 5]
                # Corresponding pseudo-label boxes and classes (ensure match with confidence scores)
                high_conf_boxes = supp_boxes[high_conf_indices, :5]
                high_conf_labels = supp_labels[high_conf_indices]
            # Step 3: Build current image pseudo-label patch library
            img_pseudo_bank = self._build_img_pseudo_bank(
                img_chw=img_chw,
                img_meta=img_meta,
                pseudo_bboxes=high_conf_boxes,
                pseudo_labels=high_conf_labels,
                pseudo_scores=high_conf_scores
            )
            # ---------------------------------------------------------------------------------------------------------------------

            # 1. Image format conversion: (C,H,W) -> (H,W,C), and denormalization
            img_norm = img_chw.permute(1, 2, 0).clone().detach()  # Avoid gradient propagation
            # Denormalization (based on MMDetection default mean values)
            img_norm[:, :, 0] += 103.530
            img_norm[:, :, 1] += 116.280
            img_norm[:, :, 2] += 123.675
            filled_image = img_norm.to('cpu').numpy().astype(np.uint8)
            filled_image = np.ascontiguousarray(filled_image)  # Ensure memory continuity

            # 2. Pseudo-label filtering: NMS deduplication + IOF filtering with real labels + class filtering
            supp_boxes_nms, keep_idx = nms_rotated(supp_boxes[:, :5], supp_boxes[:, 5], 0.05)
            supp_labels_nms = supp_labels[keep_idx]
            supp_scores_nms = supp_boxes[keep_idx, 5]

            # Filter pseudo-labels with too high overlap with real labels (IOF <= max_iof)
            if sup_boxes.size(0) > 0:
                iof_matrix = rbbox_overlaps(sup_boxes[:, :5], supp_boxes_nms[:, :5], mode='iof')
                max_iof, _ = iof_matrix.max(dim=0)
                valid_iof = max_iof <= self.max_iof
                # Filter pseudo-labels with inconsistent classes with real labels
                valid_cls = torch.isin(supp_labels_nms, sup_labels)
                valid_mask = valid_iof & valid_cls

                supp_boxes_nms = supp_boxes_nms[valid_mask]
                supp_labels_nms = supp_labels_nms[valid_mask]
                supp_scores_nms = supp_scores_nms[valid_mask]

            # 3. Convert pseudo-label format: LE90 -> polygon (for mask drawing)
            supp_boxes_poly = obb2poly_le90(supp_boxes_nms[:, :5])

            # 4. Iterate through each pseudo-label, replace with matching patch image (added img_pseudo_bank parameter)
            single_mix_boxes = []  # Augmented label boxes for single image
            single_mix_labels = []  # Augmented labels for single image
            single_mix_scores = []  # Augmented label confidence scores for single image

            for idx in range(supp_boxes_poly.shape[0]):
                supp_box_poly = supp_boxes_poly[idx]
                supp_cls = supp_labels_nms[idx]
                supp_score = supp_scores_nms[idx]

                # Low-confidence pseudo-labels: fill with gray (no patch replacement)
                if supp_score < self.mask_thr and self.use_mask:
                    supp_box_cpu = np.array(supp_box_poly.to('cpu'))
                    supp_box_cpu = np.clip(supp_box_cpu, 0, img_w - 1)  # Prevent out-of-bounds
                    cv2.fillPoly(filled_image, [supp_box_cpu.reshape(4, 2).astype(np.int64)],
                                color=(128, 128, 128))
                    continue

                # High-confidence pseudo-labels: match patch and replace (added img_pseudo_bank parameter)
                supp_box_obb = pointobb2thetaobb_180(supp_box_poly.to('cpu'))
                patch_img = find_match_patch_ratio(
                    supp_box_obb=torch.tensor(supp_box_obb, device=device),
                    supp_cls=supp_cls,
                    patch_json=self.patch_json,
                    img_pseudo_bank=img_pseudo_bank,  # New: pass current image pseudo-label patch library
                    cls_map_version=self.cls_map_version,
                    percent_version=self.percent_version,
                    zoom_ratio=img_w / 1024  # Scale based on 1024基准尺寸
                )

                # Skip if no matching patch
                if patch_img is None:
                    continue

                # Perspective transformation: map patch to pseudo-label area
                rect_center = (supp_box_obb[0], supp_box_obb[1])
                rect_size = (supp_box_obb[2], supp_box_obb[3])
                rect_theta = supp_box_obb[4]
                rect = (rect_center, rect_size, rect_theta)
                # Calculate four vertices of rectangle (for perspective transformation)
                box_pts = cv2.boxPoints(rect)
                box_pts = np.int0(box_pts)  # Convert to integer coordinates

                # Patch size alignment (ensure patch long side matches target area long side)
                patch_h, patch_w = patch_img.shape[:2]
                dst_pts = box_pts.astype(np.float32)
                # Calculate adjacent edge lengths of target area (determine long side direction)
                edge1_len = self.get_distence(dst_pts[0], dst_pts[1])
                edge2_len = self.get_distence(dst_pts[1], dst_pts[2])
                # Patch vertex coordinates (adjust based on long side direction)
                if (edge1_len - edge2_len) * (patch_w - patch_h) >= 0:
                    # Same long side direction: arrange vertices normally
                    src_pts = np.array([
                        [1, 1], [patch_w - 2, 1],
                        [patch_w - 2, patch_h - 2], [1, patch_h - 2]
                    ], dtype=np.float32)
                else:
                    # Opposite long side direction: rotate vertex arrangement
                    src_pts = np.array([
                        [1, patch_h - 2], [1, 1],
                        [patch_w - 2, 1], [patch_w - 2, patch_h - 2]
                    ], dtype=np.float32)

                # Calculate perspective transformation matrix and apply
                M = cv2.getPerspectiveTransform(src_pts, dst_pts)
                warped_patch = cv2.warpPerspective(patch_img, M, (filled_image.shape[1], filled_image.shape[0]))

                # Generate mask and replace image area
                mask = np.zeros_like(filled_image[:, :, 0])
                cv2.fillPoly(mask, [box_pts], 255)  # Label area mask
                mask = np.repeat(mask[:, :, np.newaxis], 3, axis=2)  # Convert to three channels
                # Image fusion: replace masked area with patch, keep original image in other areas
                filled_image = filled_image * (1 - mask / 255) + warped_patch * (mask / 255)
                filled_image = filled_image.astype(np.uint8)

                # Record augmented label information
                single_mix_boxes.append(box_pts.reshape(-1).tolist())
                single_mix_labels.append(supp_cls.item())
                single_mix_scores.append(supp_score.item())

            # 5. Periodically save augmented images (save every 100 iterations)
            if self.current_iter % 100 == 0 and sup_boxes.size(0) > 0:
                self._save_augmented_image(filled_image, sup_boxes, single_mix_boxes,
                                         img_filename)

            # 6. Process augmented label boxes (polygon -> LE90 format)
            if single_mix_boxes:
                mix_boxes_poly = torch.tensor(np.array(single_mix_boxes), device=device)
                mix_boxes_obb = poly2obb_le90(mix_boxes_poly.float())
                mix_scores = torch.tensor(single_mix_scores, device=device).unsqueeze(1)
                mix_boxes_obb = torch.cat([mix_boxes_obb, mix_scores], dim=-1)
                mix_aug_boxes.append(mix_boxes_obb)
                mix_aug_labels.append(torch.tensor(single_mix_labels, device=device))
            else:
                mix_aug_boxes.append(torch.empty((0, 6), device=device))
                mix_aug_labels.append(torch.empty(0, dtype=torch.long, device=device))

            # 7. Restore image format: (H,W,C) -> (C,H,W), and re-normalize
            filled_image = filled_image.astype(np.float32)
            filled_image[:, :, 0] -= 103.530
            filled_image[:, :, 1] -= 116.280
            filled_image[:, :, 2] -= 123.675
            filled_image_tensor = torch.tensor(filled_image, device=device)
            masked_imgs.append(filled_image_tensor.permute(2, 0, 1).contiguous())

        # 8. Update images and labels of strong data
        data_groups["strong"]["img"] = torch.cat([x.unsqueeze(0) for x in masked_imgs], dim=0)
        if mix_aug_boxes:
            mix_merge_bboxes, mix_merge_labels = self.merge_rboxes(merge_bboxes, merge_labels,
                                                                 mix_aug_boxes, mix_aug_labels)
            data_groups["strong"]["gt_bboxes"] = [b[:, :5] for b in mix_merge_bboxes]
            data_groups["strong"]["gt_labels"] = mix_merge_labels

    def _save_augmented_image(self, filled_image, sup_boxes, mix_boxes, img_filename):
        """Save augmented image (with visualization of real labels and augmented labels)"""
        # Draw real label boxes (blue)
        sup_boxes_poly = obb2poly_le90(sup_boxes[:, :5])
        sup_boxes_cpu = np.array(sup_boxes_poly.to('cpu'))
        for box in sup_boxes_cpu:
            box = box.reshape(4, 2).astype(np.int64)
            for idx in range(-1, 3):
                cv2.line(filled_image, (box[idx, 0], box[idx, 1]),
                        (box[idx + 1, 0], box[idx + 1, 1]),
                        color=(0, 35, 230), thickness=2)

        # Draw augmented label boxes (green)
        for box in mix_boxes:
            box = np.array(box).reshape(4, 2).astype(np.int64)
            for idx in range(-1, 3):
                cv2.line(filled_image, (box[idx, 0], box[idx, 1]),
                        (box[idx + 1, 0], box[idx + 1, 1]),
                        color=(230, 35, 0), thickness=2)

        # Save image
        save_path = os.path.join(
            self.save_mask_img_dir,
            f'iter_{self.current_iter}_proposals_large_mask_{img_filename}'
        )
        cv2.imwrite(save_path, filled_image)

    def forward_teacher(self, teacher_data, student_data):
        """
        Teacher model forward propagation: generate pseudo-labels and transform to student model's image space
        
        Args:
            teacher_data: Teacher model input data (weak augmentation)
            student_data: Student model input data (strong augmentation)
        
        Returns:
            pseudo_bboxes: Filtered pseudo-label boxes (list, one tensor per image)
            pseudo_labels: Filtered pseudo-labels (list, one tensor per image)
            all_pseudo_bboxes: All pseudo-label boxes (including low confidence, unfiltered)
            all_pseudo_labels: All pseudo-labels (including low confidence, unfiltered)
            teacher_info: Teacher model output information dictionary
            student_info: Student model input information dictionary
        """
        # 1. Align image order of teacher and student data (match by filename)
        t_filenames = [meta["filename"] for meta in teacher_data["img_metas"]]
        s_filenames = [meta["filename"] for meta in student_data["img_metas"]]
        t_idx = [t_filenames.index(name) for name in s_filenames]  # Indices of teacher data (matching student data order)

        # 2. Teacher model inference (no gradients)
        with torch.no_grad():
            teacher_info = self.extract_teacher_info(
                img=teacher_data["img"][torch.tensor(t_idx, device=teacher_data["img"].device).long()],
                img_metas=[teacher_data["img_metas"][idx] for idx in t_idx],
                proposals=[teacher_data["proposals"][idx] for idx in t_idx] 
                          if ("proposals" in teacher_data and teacher_data["proposals"] is not None) 
                          else None
            )

        # 3. Student model information initialization (for coordinate transformation)
        student_info = {
            "img_metas": student_data["img_metas"],
            "transform_matrix": [
                torch.from_numpy(meta["transform_matrix"]).float().to(student_data["img"].device)
                for meta in student_data["img_metas"]
            ]
        }

        # 4. Extract student model backbone features (for CCL loss)
        if self.ccl_weight > 0:
            student_feat = self.student.extract_feat(student_data["img"])
            student_info["backbone_feature"] = student_feat

        # 5. Calculate coordinate transformation matrix from teacher->student
        trans_mat = self._get_trans_mat(
            teacher_info["transform_matrix"], 
            student_info["transform_matrix"]
        )

        # 6. Transform pseudo-label coordinates (teacher image space -> student image space)
        # Filtered pseudo-labels
        pseudo_bboxes = self._transform_rbbox(
            teacher_info["det_bboxes"],
            trans_mat,
            [meta["img_shape"] for meta in student_info["img_metas"]]
        )
        pseudo_labels = teacher_info["det_labels"]

        # All pseudo-labels (including low confidence)
        all_pseudo_bboxes = self._transform_rbbox(
            teacher_info["det_bboxes_all"],
            trans_mat,
            [meta["img_shape"] for meta in student_info["img_metas"]]
        )
        all_pseudo_labels = teacher_info["det_labels_all"]

        return (pseudo_bboxes, pseudo_labels, all_pseudo_bboxes, 
                all_pseudo_labels, teacher_info, student_info)

    @force_fp32(apply_to=["bboxes", "trans_mat"])
    def _transform_rbbox(self, bboxes, trans_mat, max_shape):
        """
        Transform rotated bounding box coordinates (based on transformation matrix)
        
        Args:
            bboxes: List of rotated bounding boxes (one tensor per image)
            trans_mat: List of transformation matrices (one matrix per image)
            max_shape: List of maximum image dimensions (H,W,C for each image)
        
        Returns:
            List of transformed rotated bounding boxes
        """
        return Transform2D.transform_rbboxes(bboxes, trans_mat, max_shape)

    @force_fp32(apply_to=["bboxes", "trans_mat"])
    def _transform_bbox(self, bboxes, trans_mat, max_shape):
        """
        Transform horizontal bounding box coordinates (based on transformation matrix)
        
        Args:
            bboxes: List of horizontal bounding boxes (one tensor per image)
            trans_mat: List of transformation matrices (one matrix per image)
            max_shape: List of maximum image dimensions (H,W,C for each image)
        
        Returns:
            List of transformed horizontal bounding boxes
        """
        return Transform2D.transform_bboxes(bboxes, trans_mat, max_shape)

    @force_fp32(apply_to=["a", "b"])
    def _get_trans_mat(self, a, b):
        """
        Calculate transformation matrix from matrix a to matrix b (b @ a^{-1})
        
        Args:
            a: List of source transformation matrices
            b: List of target transformation matrices
        
        Returns:
            List of transformation matrices from a to b
        """
        return [bt @ at.inverse() for bt, at in zip(b, a)]

    def extract_teacher_info(self, img, img_metas, proposals=None, **kwargs):
        """
        Extract teacher model output information (proposals, pseudo-labels, features, etc.)
        
        Args:
            img: Teacher model input image
            img_metas: List of image meta information
            proposals: Pre-generated proposals (optional, generated by RPN if not provided)
        
        Returns:
            Teacher model information dictionary containing proposals, det_bboxes, det_labels, backbone_feature, etc.
        """
        teacher_info = {}

        # 1. Extract teacher model backbone features
        teacher_feat = self.teacher.extract_feat(img)
        teacher_info["backbone_feature"] = teacher_feat
        device = teacher_feat[0].device

        # 2. Generate Proposals (RPN output or pre-input)
        if proposals is None:
            # Get RPN parameters from configuration
            proposal_cfg = self.teacher.train_cfg.get("rpn_proposal", self.teacher.test_cfg.rpn)
            rpn_out = list(self.teacher.rpn_head(teacher_feat))
            proposals = self.teacher.rpn_head.get_bboxes(
                *rpn_out, img_metas=img_metas, cfg=proposal_cfg
            )
        teacher_info["proposals"] = proposals

        # 3. Teacher model ROI Head inference (generate pseudo-labels)
        # proposals, proposal_labels = self.teacher.roi_head.simple_test_bboxes(
        #     feats=teacher_feat,
        #     img_metas=img_metas,
        #     proposals=proposals,
        #     rcnn_cfg=self.teacher.test_cfg.rcnn,
        #     rescale=False
        # )
        proposals, proposal_labels = self.teacher.roi_head.simple_test_bboxes(
            teacher_feat, img_metas, proposals, self.teacher.test_cfg.rcnn, rescale=False
        )

        # 4. Format unification (ensure tensor devices are consistent)
        proposals = [p.to(device) for p in proposals]
        proposals = [p if p.shape[0] > 0 else p.new_zeros(0, 6) for p in proposals]
        proposal_labels = [l.to(device) for l in proposal_labels]

        # 5. Save all pseudo-labels (unfiltered, for subsequent CutMix augmentation)
        teacher_info["det_bboxes_all"] = proposals
        teacher_info["det_labels_all"] = proposal_labels

        # 6. Pseudo-label filtering (based on confidence and minimum size)
        # First round filtering: initial confidence threshold
        init_thr = self.train_cfg.pseudo_label_initial_score_thr
        if not isinstance(init_thr, float):
            raise NotImplementedError("Dynamic initial threshold not supported yet, must be set to a fixed float")
        proposals, proposal_labels, _ = zip(*[
            filter_invalid(
                bbox=prop,
                label=label,
                score=prop[:, -1],
                thr=init_thr,
                min_size=self.train_cfg.min_pseduo_box_size
            )
            for prop, label in zip(proposals, proposal_labels)
        ])

        # Second round filtering: actual confidence threshold (filter by -thr after negation, equivalent to score >= thr)
        real_thr = self.train_cfg.pseudo_label_real_score_thr
        if not isinstance(real_thr, float):
            raise NotImplementedError("Dynamic actual threshold not supported yet, must be set to a fixed float")
        proposals, proposal_labels, _ = zip(*[
            filter_invalid(
                bbox=prop,
                label=label,
                score=-prop[:, -1],  # Filter by -thr after negation
                thr=-real_thr,
                min_size=self.train_cfg.min_pseduo_box_size
            )
            for prop, label in zip(proposals, proposal_labels)
        ])

        # Third round filtering: final confidence threshold (ensure high-confidence pseudo-labels)
        proposals, proposal_labels, _ = zip(*[
            filter_invalid(
                bbox=prop,
                label=label,
                score=prop[:, -1],
                thr=real_thr,
                min_size=self.train_cfg.min_pseduo_box_size
            )
            for prop, label in zip(proposals, proposal_labels)
        ])

        # 7. Save filtered pseudo-labels
        teacher_info["det_bboxes"] = proposals
        teacher_info["det_labels"] = proposal_labels

        # 8. Save transformation matrices and image meta information
        teacher_info["transform_matrix"] = [
            torch.from_numpy(meta["transform_matrix"]).float().to(device)
            for meta in img_metas
        ]
        teacher_info["img_metas"] = img_metas
        teacher_info["det_bboxes_cali"] = teacher_info["det_bboxes"]  # Calibrated pseudo-labels (same as filtered by default)

        return teacher_info

    def unsup_cross_consistency_loss(self, teacher_info, student_info, strong_student_proposals=None):
        """
        Calculate cross-consistency loss (CCL): align ROI features of teacher and student models
        
        Args:
            teacher_info: Teacher model information dictionary
            student_info: Student model information dictionary
            strong_student_proposals: Student model's strong proposals (optional)
        
        Returns:
            Loss dictionary containing CCL loss
        """
        # 1. Extract backbone features of teacher and student models
        teacher_feat = teacher_info["backbone_feature"]
        student_feat = student_info["backbone_feature"]

        # 2. Determine Proposals (select strong proposals or teacher proposals based on configuration)
        if self.use_stong_popsosals and self.use_stong_popsosals_for_CCL and strong_student_proposals:
            # Use student's strong proposals, transform to teacher image space
            s_proposals = strong_student_proposals
            trans_mat_s2t = self._get_trans_mat(
                student_info["transform_matrix"],
                teacher_info["transform_matrix"]
            )
            t_proposals = self._transform_bbox(
                s_proposals,
                trans_mat_s2t,
                [meta["img_shape"] for meta in teacher_info["img_metas"]]
            )
        else:
            # Use teacher's proposals, transform to student image space
            t_proposals = teacher_info["proposals"]
            trans_mat_t2s = self._get_trans_mat(
                teacher_info["transform_matrix"],
                student_info["transform_matrix"]
            )
            s_proposals = self._transform_bbox(
                t_proposals,
                trans_mat_t2s,
                [meta["img_shape"] for meta in student_info["img_metas"]]
            )

        # 3. Return 0 loss if no valid Proposals
        total_proposal_num = sum([len(bbox) for bbox in t_proposals])
        if total_proposal_num == 0:
            return {"loss_ccl": torch.tensor(0.0, device=teacher_feat[0].device)}

        # 4. Extract teacher model ROI features (no gradients)
        with torch.no_grad():
            t_roi = bbox2roi(t_proposals)
            t_roi_feat = self.teacher.roi_head.bbox_roi_extractor(
                feats=teacher_feat[:self.teacher.roi_head.bbox_roi_extractor.num_inputs],
                rois=t_roi
            )

        # 5. Extract student model ROI features
        s_roi = bbox2roi(s_proposals)
        s_roi_feat = self.student.roi_head.bbox_roi_extractor(
            feats=student_feat[:self.student.roi_head.bbox_roi_extractor.num_inputs],
            rois=s_roi
        )

        # 6. Calculate cosine similarity loss (1 - average cosine similarity)
        cos_sim = F.cosine_similarity(t_roi_feat, s_roi_feat, dim=1)
        loss_ccl = (1 - cos_sim.mean()) * self.ccl_weight

        return {"loss_ccl": loss_ccl}

    def merge_rboxes(self, gt_bboxes, gt_labels, dt_boxes, dt_labels, dt_cali=None):
        """
        Merge real label boxes (gt) and detection boxes (dt, pseudo-labels), remove duplicate boxes
        
        Args:
            gt_bboxes: List of real label boxes (one tensor per image)
            gt_labels: List of real labels (one tensor per image)
            dt_boxes: List of detection boxes (one tensor per image)
            dt_labels: List of detection labels (one tensor per image)
            dt_cali: Detection box calibration information (optional, used for post-calibration filtering)
        
        Returns:
            Lists of merged label boxes and labels
        """
        # Calibration mode judgment (whether to enable post-calibration filtering)
        use_cali_filter = (dt_cali is not None and 
                          self.current_iter > self.start_fit_iter and 
                          self.finished_fit)

        new_gt_bboxes = []
        new_gt_labels = []

        # Iterate through labels for each image
        iter_args = zip(gt_bboxes, gt_labels, dt_boxes, dt_labels)
        if use_cali_filter:
            iter_args = zip(gt_bboxes, gt_labels, dt_boxes, dt_labels, dt_cali)

        for args in iter_args:
            if use_cali_filter:
                gt_boxes_per_img, gt_labels_per_img, dt_boxes_per_img, dt_labels_per_img, dt_cali_per_img = args
            else:
                gt_boxes_per_img, gt_labels_per_img, dt_boxes_per_img, dt_labels_per_img = args
                dt_cali_per_img = None

            device = gt_boxes_per_img.device if gt_boxes_per_img.size(0) > 0 else dt_boxes_per_img.device

            # 1. Filter small-sized detection boxes
            if self.mining_min_size >= 0 and dt_boxes_per_img.size(0) > 0:
                # Extract width and height (LE90 format: x,y,w,h,theta)
                if dt_boxes_per_img.shape[1] < 5:
                    raise ValueError(f"Invalid detection box format, must have at least 5 dimensions (current {dt_boxes_per_img.shape[1]} dimensions)")
                w = dt_boxes_per_img[:, 2]
                h = dt_boxes_per_img[:, 3]
                valid_mask = (w > self.mining_min_size) & (h > self.mining_min_size)
                
                # Apply filtering
                if not valid_mask.all():
                    dt_boxes_per_img = dt_boxes_per_img[valid_mask]
                    dt_labels_per_img = dt_labels_per_img[valid_mask]
                    if use_cali_filter:
                        dt_cali_per_img = dt_cali_per_img[valid_mask]

            # 2. Class matching filtering (only keep detection boxes with classes consistent with real labels)
            if gt_labels_per_img.size(0) > 0 and dt_labels_per_img.size(0) > 0:
                # Build class matching matrix (gt classes × dt classes)
                gt_cls_mat = gt_labels_per_img.reshape(-1, 1)
                dt_cls_mat = dt_labels_per_img.reshape(1, -1)
                class_filter = (gt_cls_mat == dt_cls_mat)

                # Calculate IOU, IOF, IOB matrices (for duplicate box judgment)
                if use_cali_filter:
                    # Calibration mode: calculate overlap using horizontal boxes (dt_boxes_per_img is 4-dimensional x1y1x2y2)
                    if dt_boxes_per_img.shape[1] != 4:
                        raise ValueError(f"Detection boxes in calibration mode must be 4-dimensional (x1y1x2y2), current {dt_boxes_per_img.shape[1]} dimensions")
                    iob_matrix = rbbox_overlaps(dt_boxes_per_img[:, :4], gt_boxes_per_img, mode='iof').T
                    iof_matrix = rbbox_overlaps(gt_boxes_per_img, dt_boxes_per_img[:, :4], mode='iof')
                    iou_matrix = rbbox_overlaps(gt_boxes_per_img, dt_boxes_per_img[:, :4])
                else:
                    # Non-calibration mode: calculate overlap using rotated boxes (dt_boxes_per_img is 5-dimensional LE90)
                    if dt_boxes_per_img.shape[1] < 5:
                        raise ValueError(f"Detection boxes in non-calibration mode must have at least 5 dimensions (LE90), current {dt_boxes_per_img.shape[1]} dimensions")
                    iob_matrix = rbbox_overlaps(dt_boxes_per_img[:, :5], gt_boxes_per_img[:, :5], mode='iof').T
                    iof_matrix = rbbox_overlaps(gt_boxes_per_img[:, :5], dt_boxes_per_img[:, :5], mode='iof')
                    iou_matrix = rbbox_overlaps(gt_boxes_per_img[:, :5], dt_boxes_per_img[:, :5])

                # Build duplicate box filtering matrix (considered duplicate if any overlap condition is met)
                iob_filter = (iob_matrix > 0.8) & class_filter
                iof_filter = (iof_matrix > 0.8) & class_filter
                iou_filter = (iou_matrix > self.mining_th_score) & class_filter
                repeat_filter = iou_filter | (iou_matrix > 0.75) | iof_filter | iob_filter

                # Indices of non-duplicate detection boxes (not matched by any real label)
                unlabel_idxs = (repeat_filter.sum(dim=0) == 0)

                # Calibration mode: additional filtering of low-confidence detection boxes
                if use_cali_filter:
                    if dt_cali_per_img is None or dt_cali_per_img.size(0) != dt_boxes_per_img.size(0):
                        raise ValueError("Calibration information matching the number of detection boxes is required in calibration mode")
                    # Extract calibrated confidence (last dimension)
                    cali_conf = dt_cali_per_img[:, -1]
                    confident_idxs = (cali_conf >= self.thr_after_cali)
                    # Merge filtering conditions (non-duplicate and high confidence)
                    unlabel_idxs = unlabel_idxs & confident_idxs
            else:
                # All detection boxes are non-duplicate when there are no real labels or no detection boxes
                unlabel_idxs = torch.ones_like(dt_labels_per_img, dtype=torch.bool)

            # 3. Format alignment (ensure real label boxes and detection boxes have consistent dimensions)
            if gt_boxes_per_img.size(0) == 0:
                # Directly use detection box format when there are no real labels
                gt_boxes_per_img = torch.empty(
                    (0, dt_boxes_per_img.shape[1]),
                    dtype=dt_boxes_per_img.dtype,
                    layout=dt_boxes_per_img.layout,
                    device=device
                )
            else:
                # Pad dimensions (when detection box dimensions > real label box dimensions)
                pad_dim = dt_boxes_per_img.shape[1] - gt_boxes_per_img.shape[1]
                if pad_dim > 0:
                    gt_boxes_per_img = F.pad(gt_boxes_per_img, (0, pad_dim, 0, 0), value=0)
                    # Set real label confidence to 1 (last dimension of detection boxes is confidence)
                    if pad_dim >= 1:
                        gt_boxes_per_img[:, -1] = 1.0

            # 4. Merge labels (real labels + non-duplicate detection boxes)
            merged_boxes = torch.cat([gt_boxes_per_img, dt_boxes_per_img[unlabel_idxs]])
            merged_labels = torch.cat([gt_labels_per_img, dt_labels_per_img[unlabel_idxs]])

            # Debug log: print merge information
            if DEBUG and unlabel_idxs.sum() > 0:
                print(f"Number of newly added labels after merging: {unlabel_idxs.sum().int()}, classes: {dt_labels_per_img[unlabel_idxs]}")
                if gt_labels_per_img.size(0) > 0:
                    print(f"Maximum IOF value: {((iof_matrix * class_filter).T)[unlabel_idxs].max(1)[0]}")
                    print(f"Maximum IOB value: {((iob_matrix * class_filter).T)[unlabel_idxs].max(1)[0]}")
                    print(f"Maximum IOU value: {((iou_matrix * class_filter).T)[unlabel_idxs].max(1)[0]}")
                print(f"Confidence scores of newly added labels: {dt_boxes_per_img[unlabel_idxs][:, -1]}")

            new_gt_bboxes.append(merged_boxes)
            new_gt_labels.append(merged_labels)

        return new_gt_bboxes, new_gt_labels

    def __del__(self):
        """Destructor: close log file"""
        if hasattr(self, 'save_txt_file') and self.save_txt_file:
            self.save_txt_file.close()
