_base_ = "base_ro_faster.py"

model = dict(
    backbone=dict(
        depth=50,
        norm_cfg=dict(requires_grad=False),
        norm_eval=True,
        style="caffe",
        init_cfg=dict(
            type="Pretrained", checkpoint="open-mmlab://detectron2/resnet50_caffe"
        ),
    ),
)


data_root = 'data/dota1/'
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        ann_file=data_root + 'train_obb/split_images/annfiles_1ins',
        img_prefix=data_root + 'train_obb/split_images/images/',
    ),
    val=dict(
        ann_file=data_root + 'val_obb/split_images/annfiles',
        img_prefix=data_root + 'val_obb/split_images/images/',
    ),
    test=dict(
        ann_file=data_root + 'val_obb/split_images/annfiles',
        img_prefix=data_root + 'val_obb/split_images/images/',
    ),
    
)

semi_wrapper = dict(
    type="DML_ALOD",
    model="${model}",
    train_cfg=dict(
        use_teacher_proposal=False,
        pseudo_label_real_score_thr = 0.9,
        pseudo_label_initial_score_thr=0.00,
        burn_in_iter = 0,
        cutmix_iter = 4500,
        min_pseduo_box_size=0,
        mining_th_score=0.6,
        mining_warmup=0,
        mining_min_size=10,
        do_merge=True,
        save_dir="${work_dir}/save",
        mask_mode = 'cutmix', # nomask
        ccl_weight = 1.0,
        use_stong_popsosals = True,
        use_stong_popsosals_for_CCL = True,
        patch_json_path = 'CUT_PATCHES_JSON',
        mask_thr = 0.2,
        use_mask = True,
        max_iof = 0.05,
        percent_version = '1ins'
    ),
    test_cfg=dict(inference_on="teacher"),
)

mul = 1
optimizer = dict(type="SGD", lr=0.005, momentum=0.9, weight_decay=0.0001,paramwise_cfg=dict(bias_lr_mult=2.0, bias_decay_mult=0.0))
runner = dict(_delete_=True, type="IterBasedRunner", max_iters=6000)
evaluation = dict(type="SubModulesDistEvalHook", interval=500,save_best='mAP')
checkpoint_config = dict(by_epoch=False, interval=500, max_keep_ckpts=2)

load_from = 'PATH_OF_PRETRAINED_MODEL_EXPAND_FOR_T-S_MODEL'