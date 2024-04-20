import os, glob
from itertools import compress

import numpy as np
import cv2
from torch.utils import data
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR100 as TorchVisionCIFAR100
from torchvision.datasets import Food101 as TorchVisionFOOD101
from . import base_dataset as basedat
from . import memory_dataset as memd
from .dataset_config import dataset_config
from .autoaugment import CIFAR10Policy, ImageNetPolicy
from .ops import Cutout
from PIL import Image
import shutil
import re


def get_loaders(datasets, num_tasks, nc_first_task, batch_size, validation=.1,
                extra_aug=""):
    """Apply transformations to Datasets and create the DataLoaders for each task"""

    trn_load, val_load, tst_load = [], [], []
    taskcla = []
    dataset_offset = 0
    for idx_dataset, cur_dataset in enumerate(datasets, 0):
        # get configuration for current dataset
        dc = dataset_config[cur_dataset]

        # transformations
        trn_transform, tst_transform = get_transforms(resize=dc['resize'],
                                                      test_resize=dc['test_resize'],
                                                      pad=dc['pad'],
                                                      crop=dc['crop'],
                                                      flip=dc['flip'],
                                                      normalize=dc['normalize'],
                                                      extend_channel=dc['extend_channel'],
                                                      extra_aug=extra_aug, ds_name=cur_dataset)

        # datasets
        trn_dset, val_dset, tst_dset, curtaskcla, txt_classes = get_datasets(cur_dataset, dc['path'], num_tasks, nc_first_task,
                                                                validation=validation,
                                                                trn_transform=trn_transform,
                                                                tst_transform=tst_transform,
                                                                class_order=dc['class_order'])

        # apply offsets in case of multiple datasets
        if idx_dataset > 0:
            for tt in range(num_tasks):
                trn_dset[tt].labels = [elem + dataset_offset for elem in trn_dset[tt].labels]
                val_dset[tt].labels = [elem + dataset_offset for elem in val_dset[tt].labels]
                tst_dset[tt].labels = [elem + dataset_offset for elem in tst_dset[tt].labels]
        dataset_offset = dataset_offset + sum([tc[1] for tc in curtaskcla])

        # reassign class idx for multiple dataset case
        curtaskcla = [(tc[0] + idx_dataset * num_tasks, tc[1]) for tc in curtaskcla]

        # extend final taskcla list
        taskcla.extend(curtaskcla)

        # loaders
        for tt in range(num_tasks):
            trn_load.append(data.DataLoader(trn_dset[tt], batch_size=batch_size, shuffle=True, num_workers=25 ))
            val_load.append(data.DataLoader(val_dset[tt], batch_size=batch_size, shuffle=False, num_workers=25  ))
            tst_load.append(data.DataLoader(tst_dset[tt], batch_size=batch_size, shuffle=False, num_workers=25  ))
    return trn_load, val_load, tst_load, taskcla, txt_classes


def get_datasets(dataset, path, num_tasks, nc_first_task, validation, trn_transform, tst_transform, class_order=None):
    """Extract datasets and create Dataset class"""

    trn_dset, val_dset, tst_dset = [], [], []
    txt_classes = []

    if 'food101' == dataset:

        trn = TorchVisionFOOD101(path, split="train", download=True)
        tst = TorchVisionFOOD101(path, split="test", download=True)

        _ensure_food(path, trn.class_to_idx)

        txt_classes = trn.classes

        all_data, taskcla, class_indices = basedat.get_data(path, validation=validation,
                                                         num_tasks=num_tasks, nc_first_task=nc_first_task,
                                                         shuffle_classes=class_order is None, class_order=class_order)
        # set dataset type
        Dataset = basedat.BaseDataset

    elif 'cifar100' in dataset:
        tvcifar_trn = TorchVisionCIFAR100(path, train=True, download=True)
        tvcifar_tst = TorchVisionCIFAR100(path, train=False, download=True)
        trn_data = {'x': tvcifar_trn.data, 'y': tvcifar_trn.targets}
        tst_data = {'x': tvcifar_tst.data, 'y': tvcifar_tst.targets}
        # compute splits
        all_data, taskcla, class_indices = memd.get_data(trn_data, tst_data, validation=validation,
                                                         num_tasks=num_tasks, nc_first_task=nc_first_task,
                                                         shuffle_classes=class_order is None, class_order=class_order)
        txt_classes = tvcifar_trn.classes
        # set dataset type
        Dataset = memd.MemoryDataset

    offset = 0
    for task in range(num_tasks):
        all_data[task]['trn']['y'] = [label + offset for label in all_data[task]['trn']['y']]
        all_data[task]['val']['y'] = [label + offset for label in all_data[task]['val']['y']]
        all_data[task]['tst']['y'] = [label + offset for label in all_data[task]['tst']['y']]
        trn_dset.append(Dataset(all_data[task]['trn'], trn_transform, class_indices))
        val_dset.append(Dataset(all_data[task]['val'], tst_transform, class_indices))
        tst_dset.append(Dataset(all_data[task]['tst'], tst_transform, class_indices))
        offset += taskcla[task][1]

    return trn_dset, val_dset, tst_dset, taskcla, txt_classes


def get_transforms(resize, test_resize, pad, crop, flip, normalize, extend_channel, extra_aug="", ds_name=""):
    """Unpack transformations and apply to train or test splits"""

    trn_transform_list = []
    tst_transform_list = []
    
    # resize
    if resize is not None:
        trn_transform_list.append(transforms.Resize(resize))
        tst_transform_list.append(transforms.Resize(resize))

    # padding
    if pad is not None:
        trn_transform_list.append(transforms.Pad(pad))
        tst_transform_list.append(transforms.Pad(pad))

    # test only resize
    if test_resize is not None:
        tst_transform_list.append(transforms.Resize(test_resize))

    # crop
    if crop is not None:
        if 'cifar' in ds_name.lower():
            trn_transform_list.append(transforms.RandomCrop(crop))
        else:
            trn_transform_list.append(transforms.RandomResizedCrop(crop))
        tst_transform_list.append(transforms.CenterCrop(crop))

    # flips
    if flip:
        trn_transform_list.append(transforms.RandomHorizontalFlip())

    trn_transform_list.append(transforms.ColorJitter(brightness=63 / 255))
    if extra_aug == 'fetril':  # Similar as in PyCIL
        if 'cifar' in ds_name.lower():
            trn_transform_list.append(CIFAR10Policy())
        elif 'imagenet' in ds_name.lower():
            trn_transform_list.append(ImageNetPolicy())
        elif 'domainnet' in ds_name.lower():
            trn_transform_list.append(ImageNetPolicy())
        else:
            raise RuntimeError(f'Please check and update the data agumentation code for your dataset: {ds_name}')
      
    # to tensor
    trn_transform_list.append(transforms.ToTensor())
    tst_transform_list.append(transforms.ToTensor())
    
    if extra_aug == 'fetril':  # Similar as in PyCIL
        trn_transform_list.append(Cutout(n_holes=1, length=16))
   
    # normalization
    if normalize is not None:
        trn_transform_list.append(transforms.Normalize(mean=normalize[0], std=normalize[1]))
        tst_transform_list.append(transforms.Normalize(mean=normalize[0], std=normalize[1]))

    # gray to rgb
    if extend_channel is not None:
        trn_transform_list.append(transforms.Lambda(lambda x: x.repeat(extend_channel, 1, 1)))
        tst_transform_list.append(transforms.Lambda(lambda x: x.repeat(extend_channel, 1, 1)))

    return transforms.Compose(trn_transform_list), \
           transforms.Compose(tst_transform_list)



def _ensure_food(path, class_to_idx):
    train_txt_path = os.path.join(path, 'food-101', 'meta', 'train.txt')
    train_txt_path_new = os.path.join(path, 'train.txt')

    if(os.path.isfile(train_txt_path_new)): return

    result_lines = _extract_lines(train_txt_path, class_to_idx)

    with open(train_txt_path_new, 'w+') as file:
        file.writelines(result_lines)

    test_txt_path = os.path.join(path, 'food-101', 'meta', 'test.txt')
    test_txt_path_new = os.path.join(path, 'test.txt')

    result_lines = _extract_lines(test_txt_path, class_to_idx)

    with open(test_txt_path_new, 'w+') as file:
        file.writelines(result_lines)

    

def _extract_lines(path, class_to_idx):
    result_lines = []
    with open(path, 'r') as file:
        lines = file.readlines()
        for path in lines:
            path = path.strip()
            txt_class = re.match(r'(.+)\/\d+', path).group(1)
            class_idx = class_to_idx[txt_class]
            result_lines.append(f'food-101/images/{path}.jpg {class_idx}\n')
    return result_lines
    

