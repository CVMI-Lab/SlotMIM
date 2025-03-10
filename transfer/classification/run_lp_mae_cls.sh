PRETRAIN_CHKPT=$1
MASTER_PORT=$((RANDOM % 101 + 20000))
FNAME=$(basename ${PRETRAIN_CHKPT})
FNAME="${FNAME%.*}"
OMP_NUM_THREADS=1 torchrun --nproc_per_node=4 --master_port=$MASTER_PORT main_linprobe.py \
    --accum_iter 1 \
    --batch_size 4096 \
    --model vit_base_patch16 --cls_token \
    --finetune ${PRETRAIN_CHKPT} \
    --epochs 90 \
    --blr 0.1 \
    --weight_decay 0.0 \
    --dist_eval \
    --num_workers 10 \
    --output_dir ./output_dir/lp/${FNAME}_cls \
    --log_dir    ./output_dir/lp/${FNAME}_cls