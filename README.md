# Divide and Not Forget
**ResNet with SEED: comparison of different ResNet in the class incremental setting**

Bergamasco Davide, Righetti Alberto

## :flashlight: Overview

This repository contains the code we used to perfom all the tests with the different Resnet. We take some of the code from the authors of the original paper and built our solution on top of it:

- Paper: [Divide and not forget: Ensemble of selectively trained experts in Continual Learning](https://arxiv.org/pdf/2401.10191.pdf)
- Paper Code: [Github](https://github.com/grypesc/SEED)

It is actually strongly based on the FACIL framework, a popular benchmark for Class-Incremental learning techniques. [FACIL](https://github.com/mmasana/FACIL)

## :construction_worker: Setup

To get started use conda to create the environment:

```bash
conda env create -f environment.yml
```

## :blue_book: Project structure

`src/`
- `approach/` : contains the actual implementation of SEED, the ensembling method proposed by the authors. It also contains a Pytorch implementation of Gaussian Mixture Models.
- `datasets/` : contains the code to load the datasets and split them in multiple loaders (one for each task).
    - `dataset_config.py`: list of trasfomations required for each dataset
    - `base_dataset.py`: class to load a directory dataset
    - `memory_dataset.py`: class to load a dataset that has already the images in numpy format
    - `data_loader.py`: create a train, val and test Pytorch dataloader for each task. It also split classes in different tasks.
- `loggers/` : a simple disk logger, we used it to save figures as png and data as Numpy text files, in order to be postprocessed lately.
- `networks/` : contains the modified Resnet Models as describe in the reference paper.

`main_incremental.py`: the main file of the project used to train and to produce results of the different experiments we prepared.

`script_{MODEL}_{METHOD}_{DATASET}.py`: all experiments we performed.

`create_plot.py`: allows to create plots of comparison between ResNet18, Resnet50 and ResNet101, in terms of training and validation accuracy and loss;
futhermore there are even test accuracy aware, agnostic and precision-recall metrics. The plot are also performed in comparison between CIFAR100 and FOOD101 datasets.

## :arrow_forward: Run

Use the following example as base to reproduce results:

```bash
python3 src/main_incremental.py \
    --approach seed \           # Method
    --gmms 1 \
    --max-experts 5 \           # Number of experts
    --use-multivariate \
    --nepochs 500 \             # Number of epoch
    --batch-size 128 \          # Batch size
    --datasets cifar100 \       # Dataset
    --num-tasks 10 \            # Number of tasks
    --lr 0.05 \                 # Learning rate
    --weight-decay 5e-4 \       # Weight decay
    --clipping 1 \              # If apply clipping to gradients
    --alpha 0.99 \              # Weight for the loss, in finetuning mode
    --network resnet18 \        # Network type
    --momentum 0.9 \
    --exp-name exp_name 
```