PRETRAIN_CHKPT=$1
MASTER_PORT=$((RANDOM % 101 + 20000))
FNAME=$(basename ${PRETRAIN_CHKPT})
FNAME="${FNAME%.*}"
OMP_NUM_THREADS=1 torchrun --nproc_per_node=4 --master_port=$MASTER_PORT main_finetune.py \
    --accum_iter 1 \
    --batch_size 256 \
    --model vit_base_patch16 \
    --finetune ${PRETRAIN_CHKPT} \
    --epochs 100 \
    --blr 1e-3 --layer_decay 0.75 \
    --weight_decay 0.05 --drop_path 0.1 --mixup 0.8 --cutmix 1.0 --reprob 0.25 \
    --dist_eval \
    --num_workers 16 \
    --output_dir ./output_dir/ft/$FNAME \
    --log_dir    ./output_dir/ft/$FNAME