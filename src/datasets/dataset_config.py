from os.path import join

_BASE_DATA_PATH = "./data" #"/raid/NFS_SHARE/datasets/"

dataset_config = {
    'food101': {
        'path': join(_BASE_DATA_PATH, 'food101'),
        'resize': (256,256),
        'pad': 4,
        'flip': True,
        'normalize': ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    },
    'cifar100': {
        'path': join(_BASE_DATA_PATH, 'cifar100'),
        'resize': None,
        'pad': 4,
        'crop': 32,
        'flip': True,
        'normalize': ((0.5071, 0.4866, 0.4409), (0.2009, 0.1984, 0.2023))
    },
}

# Add missing keys:
for dset in dataset_config.keys():
    for k in ['test_resize', 'resize', 'pad', 'crop', 'normalize', 'class_order', 'extend_channel']:
        if k not in dataset_config[dset].keys():
            dataset_config[dset][k] = None
    if 'flip' not in dataset_config[dset].keys():
        dataset_config[dset]['flip'] = False
