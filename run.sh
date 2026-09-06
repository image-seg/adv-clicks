#!/usr/bin/env bash
# Запускайте из корня репозитория после настройки config.yml и загрузки весов.

# Минимизация IoU
python3 scripts/evaluate_model_sam.py NoBRS --checkpoint=./MODEL_CHECKPOINTS/SAM/sam_vit_b_01ec64.pth --deterministic --print-ious --save-ious --datasets=GrabCut,Berkeley,DAVIS,COCO-MVal --n-clicks=10 --n_opt_steps=11 --lr_mult=1 --n_workers=1 --iou-analysis --thresh=0.5 --optim_min
python3 scripts/evaluate_model_sam.py NoBRS --checkpoint=./MODEL_CHECKPOINTS/SAM/sam_vit_l_0b3195.pth --deterministic --print-ious --save-ious --datasets=GrabCut,Berkeley,DAVIS,COCO-MVal --n-clicks=10 --n_opt_steps=11 --lr_mult=1 --n_workers=1 --iou-analysis --thresh=0.5 --optim_min
python3 scripts/evaluate_model_sam.py NoBRS --checkpoint=./MODEL_CHECKPOINTS/SAM/sam_vit_h_4b8939.pth --deterministic --print-ious --save-ious --datasets=GrabCut,Berkeley,DAVIS,COCO-MVal --n-clicks=10 --n_opt_steps=11 --lr_mult=1 --n_workers=1 --iou-analysis --thresh=0.5 --optim_min
# Максимизация IoU
python3 scripts/evaluate_model_sam.py NoBRS --checkpoint=./MODEL_CHECKPOINTS/SAM/sam_vit_b_01ec64.pth --deterministic --print-ious --save-ious --datasets=GrabCut,Berkeley,DAVIS,COCO-MVal --n-clicks=10 --n_opt_steps=11 --lr_mult=1 --n_workers=1 --iou-analysis --thresh=0.5
python3 scripts/evaluate_model_sam.py NoBRS --checkpoint=./MODEL_CHECKPOINTS/SAM/sam_vit_l_0b3195.pth --deterministic --print-ious --save-ious --datasets=GrabCut,Berkeley,DAVIS,COCO-MVal --n-clicks=10 --n_opt_steps=11 --lr_mult=1 --n_workers=1 --iou-analysis --thresh=0.5
python3 scripts/evaluate_model_sam.py NoBRS --checkpoint=./MODEL_CHECKPOINTS/SAM/sam_vit_h_4b8939.pth --deterministic --print-ious --save-ious --datasets=GrabCut,Berkeley,DAVIS,COCO-MVal --n-clicks=10 --n_opt_steps=11 --lr_mult=1 --n_workers=1 --iou-analysis --thresh=0.5


# Минимизация IoU
python3 scripts/evaluate_model_mobilesam.py NoBRS --checkpoint=./MODEL_CHECKPOINTS/MobileSAM/mobile_sam.pt --deterministic --print-ious --save-ious --datasets=GrabCut,Berkeley,DAVIS,COCO-MVal --n-clicks=10 --n_opt_steps=11 --n_workers=3 --lr_mult=1 --iou-analysis --thresh=0.5 --optim_min
# Максимизация IoU
python3 scripts/evaluate_model_mobilesam.py NoBRS --checkpoint=./MODEL_CHECKPOINTS/MobileSAM/mobile_sam.pt --deterministic --print-ious --save-ious --datasets=GrabCut,Berkeley,DAVIS,COCO-MVal --n-clicks=10 --n_opt_steps=11 --n_workers=3 --lr_mult=1 --iou-analysis --thresh=0.5


# Минимизация IoU
python3 scripts/evaluate_model_samhq.py NoBRS --checkpoint=./MODEL_CHECKPOINTS/SAM-HQ/sam_hq_vit_b.pth --deterministic --print-ious --save-ious --datasets=GrabCut,Berkeley,DAVIS,COCO-MVal --n-clicks=10 --n_opt_steps=11 --n_workers=1 --lr_mult=1 --iou-analysis --thresh=0.5 --optim_min
python3 scripts/evaluate_model_samhq.py NoBRS --checkpoint=./MODEL_CHECKPOINTS/SAM-HQ/sam_hq_vit_l.pth --deterministic --print-ious --save-ious --datasets=GrabCut,Berkeley,DAVIS,COCO-MVal --n-clicks=10 --n_opt_steps=11 --n_workers=1 --lr_mult=1 --iou-analysis --thresh=0.5 --optim_min
python3 scripts/evaluate_model_samhq.py NoBRS --checkpoint=./MODEL_CHECKPOINTS/SAM-HQ/sam_hq_vit_h.pth --deterministic --print-ious --save-ious --datasets=GrabCut,Berkeley,DAVIS,COCO-MVal --n-clicks=10 --n_opt_steps=11 --n_workers=1 --lr_mult=1 --iou-analysis --thresh=0.5 --optim_min
# Максимизация IoU
python3 scripts/evaluate_model_samhq.py NoBRS --checkpoint=./MODEL_CHECKPOINTS/SAM-HQ/sam_hq_vit_b.pth --deterministic --print-ious --save-ious --datasets=GrabCut,Berkeley,DAVIS,COCO-MVal --n-clicks=10 --n_opt_steps=11 --n_workers=1 --lr_mult=1 --iou-analysis --thresh=0.5
python3 scripts/evaluate_model_samhq.py NoBRS --checkpoint=./MODEL_CHECKPOINTS/SAM-HQ/sam_hq_vit_l.pth --deterministic --print-ious --save-ious --datasets=GrabCut,Berkeley,DAVIS,COCO-MVal --n-clicks=10 --n_opt_steps=11 --n_workers=1 --lr_mult=1 --iou-analysis --thresh=0.5
python3 scripts/evaluate_model_samhq.py NoBRS --checkpoint=./MODEL_CHECKPOINTS/SAM-HQ/sam_hq_vit_h.pth --deterministic --print-ious --save-ious --datasets=GrabCut,Berkeley,DAVIS,COCO-MVal --n-clicks=10 --n_opt_steps=11 --n_workers=1 --lr_mult=1 --iou-analysis --thresh=0.5


# Минимизация IoU
python3 scripts/evaluate_model_ritm.py NoBRS --checkpoint=./MODEL_CHECKPOINTS/RITM/coco_lvis_h18_itermask.pth --deterministic --print-ious --save-ious --datasets=GrabCut,Berkeley,DAVIS,COCO-MVal --n-clicks=10 --n_opt_steps=11 --n_workers=2 --lr_mult=1 --iou-analysis --thresh=0.49 --optim_min
python3 scripts/evaluate_model_ritm.py NoBRS --checkpoint=./MODEL_CHECKPOINTS/RITM/coco_lvis_h18_baseline.pth --deterministic --print-ious --save-ious --datasets=GrabCut,Berkeley,DAVIS,COCO-MVal --n-clicks=10 --n_opt_steps=11 --n_workers=2 --lr_mult=1 --iou-analysis --thresh=0.49 --optim_min
python3 scripts/evaluate_model_ritm.py NoBRS --checkpoint=./MODEL_CHECKPOINTS/RITM/coco_lvis_h18s_itermask.pth --deterministic --print-ious --save-ious --datasets=GrabCut,Berkeley,DAVIS,COCO-MVal --n-clicks=10 --n_opt_steps=11 --n_workers=2 --lr_mult=1 --iou-analysis --thresh=0.49 --optim_min
python3 scripts/evaluate_model_ritm.py NoBRS --checkpoint=./MODEL_CHECKPOINTS/RITM/coco_lvis_h32_itermask.pth --deterministic --print-ious --save-ious --datasets=GrabCut,Berkeley,DAVIS,COCO-MVal --n-clicks=10 --n_opt_steps=11 --n_workers=2 --lr_mult=1 --iou-analysis --thresh=0.49 --optim_min
python3 scripts/evaluate_model_ritm.py NoBRS --checkpoint=./MODEL_CHECKPOINTS/RITM/sbd_h18_itermask.pth --deterministic --print-ious --save-ious --datasets=GrabCut,Berkeley,DAVIS,COCO-MVal --n-clicks=10 --n_opt_steps=11 --n_workers=2 --lr_mult=1 --iou-analysis --thresh=0.49 --optim_min
# Максимизация IoU
python3 scripts/evaluate_model_ritm.py NoBRS --checkpoint=./MODEL_CHECKPOINTS/RITM/coco_lvis_h18_itermask.pth --deterministic --print-ious --save-ious --datasets=GrabCut,Berkeley,DAVIS,COCO-MVal --n-clicks=10 --n_opt_steps=11 --n_workers=2 --lr_mult=1 --iou-analysis --thresh=0.49
python3 scripts/evaluate_model_ritm.py NoBRS --checkpoint=./MODEL_CHECKPOINTS/RITM/coco_lvis_h18_baseline.pth --deterministic --print-ious --save-ious --datasets=GrabCut,Berkeley,DAVIS,COCO-MVal --n-clicks=10 --n_opt_steps=11 --n_workers=2 --lr_mult=1 --iou-analysis --thresh=0.49
python3 scripts/evaluate_model_ritm.py NoBRS --checkpoint=./MODEL_CHECKPOINTS/RITM/coco_lvis_h18s_itermask.pth --deterministic --print-ious --save-ious --datasets=GrabCut,Berkeley,DAVIS,COCO-MVal --n-clicks=10 --n_opt_steps=11 --n_workers=2 --lr_mult=1 --iou-analysis --thresh=0.49
python3 scripts/evaluate_model_ritm.py NoBRS --checkpoint=./MODEL_CHECKPOINTS/RITM/coco_lvis_h32_itermask.pth --deterministic --print-ious --save-ious --datasets=GrabCut,Berkeley,DAVIS,COCO-MVal --n-clicks=10 --n_opt_steps=11 --n_workers=2 --lr_mult=1 --iou-analysis --thresh=0.49
python3 scripts/evaluate_model_ritm.py NoBRS --checkpoint=./MODEL_CHECKPOINTS/RITM/sbd_h18_itermask.pth --deterministic --print-ious --save-ious --datasets=GrabCut,Berkeley,DAVIS,COCO-MVal --n-clicks=10 --n_opt_steps=11 --n_workers=2 --lr_mult=1 --iou-analysis --thresh=0.49


# Минимизация IoU
python3 scripts/evaluate_model_simpleclick.py NoBRS --checkpoint=./MODEL_CHECKPOINTS/SimpleClick/cocolvis_vit_base.pth --deterministic --print-ious --save-ious --datasets=GrabCut,Berkeley,DAVIS,COCO-MVal --n-clicks=10 --n_opt_steps=11 --n_workers=2 --lr_mult=1 --iou-analysis --thresh=0.49 --optim_min
python3 scripts/evaluate_model_simpleclick.py NoBRS --checkpoint=./MODEL_CHECKPOINTS/SimpleClick/cocolvis_vit_large.pth --deterministic --print-ious --save-ious --datasets=GrabCut,Berkeley,DAVIS,COCO-MVal --n-clicks=10 --n_opt_steps=11 --n_workers=2 --lr_mult=1 --iou-analysis --thresh=0.49 --optim_min
python3 scripts/evaluate_model_simpleclick.py NoBRS --checkpoint=./MODEL_CHECKPOINTS/SimpleClick/cocolvis_vit_huge.pth --deterministic --print-ious --save-ious --datasets=GrabCut,Berkeley,DAVIS,COCO-MVal --n-clicks=10 --n_opt_steps=11 --n_workers=1 --lr_mult=1 --iou-analysis --thresh=0.49 --optim_min
python3 scripts/evaluate_model_simpleclick.py NoBRS --checkpoint=./MODEL_CHECKPOINTS/SimpleClick/sbd_vit_base.pth --deterministic --print-ious --save-ious --datasets=GrabCut,Berkeley,DAVIS,COCO-MVal --n-clicks=10 --n_opt_steps=11 --n_workers=2 --lr_mult=1 --iou-analysis --thresh=0.49 --optim_min
python3 scripts/evaluate_model_simpleclick.py NoBRS --checkpoint=./MODEL_CHECKPOINTS/SimpleClick/sbd_vit_large.pth --deterministic --print-ious --save-ious --datasets=GrabCut,Berkeley,DAVIS,COCO-MVal --n-clicks=10 --n_opt_steps=11 --n_workers=2 --lr_mult=1 --iou-analysis --thresh=0.49 --optim_min
python3 scripts/evaluate_model_simpleclick.py NoBRS --checkpoint=./MODEL_CHECKPOINTS/SimpleClick/sbd_vit_huge.pth --deterministic --print-ious --save-ious --datasets=GrabCut,Berkeley,DAVIS,COCO-MVal --n-clicks=10 --n_opt_steps=11 --n_workers=1 --lr_mult=1 --iou-analysis --thresh=0.49 --optim_min
python3 scripts/evaluate_model_simpleclick.py NoBRS --checkpoint=./MODEL_CHECKPOINTS/SimpleClick/sbd_vit_xtiny.pth --deterministic --print-ious --save-ious --datasets=GrabCut,Berkeley,DAVIS,COCO-MVal --n-clicks=10 --n_opt_steps=11 --n_workers=3 --lr_mult=1 --iou-analysis --thresh=0.49 --optim_min
# Максимизация IoU
python3 scripts/evaluate_model_simpleclick.py NoBRS --checkpoint=./MODEL_CHECKPOINTS/SimpleClick/cocolvis_vit_base.pth --deterministic --print-ious --save-ious --datasets=GrabCut,Berkeley,DAVIS,COCO-MVal --n-clicks=10 --n_opt_steps=11 --n_workers=2 --lr_mult=1 --iou-analysis --thresh=0.49
python3 scripts/evaluate_model_simpleclick.py NoBRS --checkpoint=./MODEL_CHECKPOINTS/SimpleClick/cocolvis_vit_large.pth --deterministic --print-ious --save-ious --datasets=GrabCut,Berkeley,DAVIS,COCO-MVal --n-clicks=10 --n_opt_steps=11 --n_workers=2 --lr_mult=1 --iou-analysis --thresh=0.49
python3 scripts/evaluate_model_simpleclick.py NoBRS --checkpoint=./MODEL_CHECKPOINTS/SimpleClick/cocolvis_vit_huge.pth --deterministic --print-ious --save-ious --datasets=GrabCut,Berkeley,DAVIS,COCO-MVal --n-clicks=10 --n_opt_steps=11 --n_workers=1 --lr_mult=1 --iou-analysis --thresh=0.49
python3 scripts/evaluate_model_simpleclick.py NoBRS --checkpoint=./MODEL_CHECKPOINTS/SimpleClick/sbd_vit_base.pth --deterministic --print-ious --save-ious --datasets=GrabCut,Berkeley,DAVIS,COCO-MVal --n-clicks=10 --n_opt_steps=11 --n_workers=2 --lr_mult=1 --iou-analysis --thresh=0.49
python3 scripts/evaluate_model_simpleclick.py NoBRS --checkpoint=./MODEL_CHECKPOINTS/SimpleClick/sbd_vit_large.pth --deterministic --print-ious --save-ious --datasets=GrabCut,Berkeley,DAVIS,COCO-MVal --n-clicks=10 --n_opt_steps=11 --n_workers=2 --lr_mult=1 --iou-analysis --thresh=0.49
python3 scripts/evaluate_model_simpleclick.py NoBRS --checkpoint=./MODEL_CHECKPOINTS/SimpleClick/sbd_vit_huge.pth --deterministic --print-ious --save-ious --datasets=GrabCut,Berkeley,DAVIS,COCO-MVal --n-clicks=10 --n_opt_steps=11 --n_workers=1 --lr_mult=1 --iou-analysis --thresh=0.49
python3 scripts/evaluate_model_simpleclick.py NoBRS --checkpoint=./MODEL_CHECKPOINTS/SimpleClick/sbd_vit_xtiny.pth --deterministic --print-ious --save-ious --datasets=GrabCut,Berkeley,DAVIS,COCO-MVal --n-clicks=10 --n_opt_steps=11 --n_workers=3 --lr_mult=1 --iou-analysis --thresh=0.49


# Минимизация IoU
python3 scripts/evaluate_model_gpcis.py Baseline --checkpoint=./MODEL_CHECKPOINTS/GPCIS/GPCIS_Resnet50.pth --deterministic --print-ious --save-ious --datasets=GrabCut,Berkeley,DAVIS,COCO-MVal --n-clicks=10 --n_opt_steps=11 --n_workers=3 --lr_mult=1 --iou-analysis --thresh=0.49 --optim_min
# Максимизация IoU
python3 scripts/evaluate_model_gpcis.py Baseline --checkpoint=./MODEL_CHECKPOINTS/GPCIS/GPCIS_Resnet50.pth --deterministic --print-ious --save-ious --datasets=GrabCut,Berkeley,DAVIS,COCO-MVal --n-clicks=10 --n_opt_steps=11 --n_workers=3 --lr_mult=1 --iou-analysis --thresh=0.49


# Минимизация IoU
python3 scripts/evaluate_model_cdnet.py CDNet --checkpoint=./MODEL_CHECKPOINTS/CDNet/cclvs_last_checkpoint.pth --deterministic --print-ious --save-ious --datasets=GrabCut,Berkeley,DAVIS,COCO-MVal --n-clicks=10 --n_opt_steps=11 --n_workers=3 --lr_mult=1 --iou-analysis --thresh=0.49 --optim_min
python3 scripts/evaluate_model_cdnet.py CDNet --checkpoint=./MODEL_CHECKPOINTS/CDNet/sbd_last_checkpoint.pth --deterministic --print-ious --save-ious --datasets=GrabCut,Berkeley,DAVIS,COCO-MVal --n-clicks=10 --n_opt_steps=11 --n_workers=3 --lr_mult=1 --iou-analysis --thresh=0.49 --optim_min
# Максимизация IoU
python3 scripts/evaluate_model_cdnet.py CDNet --checkpoint=./MODEL_CHECKPOINTS/CDNet/cclvs_last_checkpoint.pth --deterministic --print-ious --save-ious --datasets=GrabCut,Berkeley,DAVIS,COCO-MVal --n-clicks=10 --n_opt_steps=11 --n_workers=3 --lr_mult=1 --iou-analysis --thresh=0.49
python3 scripts/evaluate_model_cdnet.py CDNet --checkpoint=./MODEL_CHECKPOINTS/CDNet/sbd_last_checkpoint.pth --deterministic --print-ious --save-ious --datasets=GrabCut,Berkeley,DAVIS,COCO-MVal --n-clicks=10 --n_opt_steps=11 --n_workers=3 --lr_mult=1 --iou-analysis --thresh=0.49


# Минимизация IoU
python3 scripts/evaluate_model_cfr.py NoBRS --checkpoint=./MODEL_CHECKPOINTS/CFR-ICL/cocolvis_icl_vit_huge.pth --deterministic --print-ious --save-ious --datasets=GrabCut,Berkeley,DAVIS,COCO-MVal --cf-n=4 --acf --n-clicks=10 --n_opt_steps=11 --n_workers=1 --lr_mult=1 --iou-analysis --thresh=0.49 --optim_min
# Максимизация IoU
python3 scripts/evaluate_model_cfr.py NoBRS --checkpoint=./MODEL_CHECKPOINTS/CFR-ICL/cocolvis_icl_vit_huge.pth --deterministic --print-ious --save-ious --datasets=GrabCut,Berkeley,DAVIS,COCO-MVal --cf-n=4 --acf --n-clicks=10 --n_opt_steps=11 --n_workers=1 --lr_mult=1 --iou-analysis --thresh=0.49