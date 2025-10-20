#!/bin/bash

export TOKENIZERS_PARALLELISM=false

python train_scaled_optimized.py \
    --resume checkpoints_deep/latest_checkpoint.pt \
    --dataset wikitext \
    --max-train 50000 \
    --max-val 1743 \
    --batch-size 2 \
    --grad-accum-steps 8 \
    --num-epochs 10 \
    --lr 3e-4 \
    --num-workers 8 \
    --prefetch-factor 4 \
    --eval-interval 3125 \
    --save-interval 3125 \
    --log-interval 50 \
    --out-dir checkpoints_deep \
    --ckpt-name best_model.pt \
    --d-model 768 \
    --num-heads 12 \
    --max-seq-len 1024
