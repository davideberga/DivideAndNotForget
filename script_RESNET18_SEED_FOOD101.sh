#!/bin/bash
python3 src/main_incremental.py \
    --approach seed \
    --gmms 1 \
    --max-experts 5 \
    --use-multivariate \
    --nepochs 50 \
    --batch-size 128 \
    --datasets food101 \
    --num-tasks 10 \
    --lr 0.05 \
    --weight-decay 5e-4 \
    --clipping 1 \
    --alpha 0.99 \
    --use-test-as-val \
    --network resnet18 \
    --pretrained \
    --momentum 0.9 \
    --exp-name exp_SEED_res18_food101