EXP_NAME=$1
python lazyconfig_train_net.py --config-file configs/ViT/COCO/mask_rcnn_vitdet_b_100ep.py \
    --num-gpus 8 --num-machines 1 \
    train.init_checkpoint=../../ckpts/${EXP_NAME}_d2.pth train.output_dir=../../output/COCO_mask_rcnn_vitdet_b_100ep_${EXP_NAME}