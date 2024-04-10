#!/bin/bash
python src/main_incremental.py \
    --approach seed \
    --gmms 1 \
    --max-experts 5 \
    --use-multivariate \
    --nepochs 10 \
    --batch-size 224 \
    --datasets cifar100 \
    --num-tasks 10 \
    --lr 0.05 \
    --weight-decay 5e-4 \
    --clipping 1 \
    --alpha 0.99 \
    --use-test-as-val \
    --network resnet18 \
    --pretrained \
    --momentum 0.9 \
    --exp-name exp_imagenette_res50



