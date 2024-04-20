#!/bin/bash
python3 src/main_incremental.py \
    --approach seed \
    --gmms 1 \
    --max-experts 1 \
    --use-multivariate \
    --nepochs 200 \
    --batch-size 64 \
    --datasets food101 \
    --num-tasks 1 \
    --lr 0.05 \
    --weight-decay 5e-4 \
    --clipping 1 \
    --alpha 0.99 \
    --use-test-as-val \
    --network resnet50 \
    --pretrained \
    --momentum 0.9 \
    --exp-name exp_BASELINE_res50_food101