#!/bin/bash

set -e
set -x

data_dir="datasets/imglists/ego4d_1.28m.txt"
output_dir="output/slotmim_ego1.28m_vitb16_np1024_400ep_ctr8192_cross_pred0.3"

MASTER_PORT=$((RANDOM % 101 + 20001))
torchrun --master_port $MASTER_PORT --nproc_per_node=4 \
    main_pretrain_mim.py \
    --dataset ImgList \
    --data-dir ${data_dir} \
    --output-dir ${output_dir} \
    \
    --arch vit_base_patch16 \
    --dim-hidden 2048 \
    --dim-hidden-slot 4096 \
    --dim-out 256 \
    --use-bn-in-head False \
    --num-prototypes 1024 \
    --teacher-momentum 0.996 \
    --warmup-teacher-temp 0.04 \
    --teacher-temp 0.07 \
    --warmup-teacher-temp-epochs 50 \
    --drop-path-rate 0.1 \
    --group-loss-weight 0.5 \
    --use-sinkhorn False \
    \
    --pred-ratio 0.3 \
    --pred-ratio-var 0.2 \
    --pred-shape block \
    --pred-start-epoch 0 \
    --use-cross-patch-loss True \
    --use-masked-patch-loss True \
    --num-prototypes-slot 8192 \
    --warmup-teacher-slot-temp 0.04 \
    --teacher-slot-temp 0.04 \
    \
    --batch-size 1024 \
    --optimizer adamw \
    --base-lr 1.5e-4 \
    --min-lr 2e-6 \
    --weight-decay 0.04 \
    --weight-decay-end 0.4 \
    --warmup-epoch 10 \
    --clip-grad 0.3 \
    --epochs 400 \
    --freeze_slots 3 \
    \
    --print-freq 10 \
    --eval-freq 50 \
    --save-freq 100 \
    --auto-resume \
    --num-workers 8 \
    --fp16 \
    --compile