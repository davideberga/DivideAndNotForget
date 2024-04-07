#!/bin/bash
python src/main_incremental.py \
    --approach seed \
    --gmms 1 \
    --max-experts 5 \
    --use-multivariate \
    --nepochs 100 \
    --batch-size 128 \
    --num-workers 4 \
    --datasets car_parts \
    --num-tasks 10 \
    --nc-first-task 10 \
    --lr 0.05 \
    --weight-decay 5e-4 \
    --clipping 1 \
    --alpha 0.99 \
    --use-test-as-val \
    --network resnet50 \
    --pretrained \
    --momentum 0.9 \
    --exp-name exp_imagenette_res50 \
    --seed 0



