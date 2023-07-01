import os
import torch

from .cifar10 import load_cifar10

DATASETS = ['cifar10']

_LOAD_DATASET_FN = {
    'cifar10': load_cifar10,
}


def get_data_info(data_dir):
    """
    Returns dataset information.
    Arguments:
        data_dir (str): path to data directory.
    """
    dataset = os.path.basename(os.path.normpath(data_dir))
  
    if 'cifar10' in data_dir:
        from .cifar10 import DATA_DESC
    else:
        raise ValueError(f'Only data in {DATASETS} are supported!')
    DATA_DESC['data'] = dataset
    return DATA_DESC


def load_data(data_dir, batch_size=128, batch_size_test=256, num_workers=4, use_augmentation='base', use_consistency=False, shuffle_train=True, 
             validation=False):
    """
    Returns train, test datasets and dataloaders.
    Arguments:
        data_dir (str): path to data directory.
        batch_size (int): batch size for training.
        batch_size_test (int): batch size for validation.
        num_workers (int): number of workers for loading the data.
        use_augmentation (base/none): whether to use augmentations for training set.
        shuffle_train (bool): whether to shuffle training set.
        aux_data_filename (str): path to unlabelled data.
        unsup_fraction (float): fraction of unlabelled data per batch.
        validation (bool): if True, also returns a validation dataloader for unspervised cifar10 (as in Gowal et al, 2020).
    """
    dataset = os.path.basename(os.path.normpath(data_dir))
    load_dataset_fn = _LOAD_DATASET_FN[dataset]
    
    train_dataset, test_dataset = load_dataset_fn(data_dir=data_dir, use_augmentation=use_augmentation)
           
    #pin_memory = torch.cuda.is_available()
    pin_memory = False
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle_train, 
                                                       num_workers=num_workers, pin_memory=pin_memory)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size_test, shuffle=False, 
                                                      num_workers=num_workers, pin_memory=pin_memory)
    
    return train_dataset, test_dataset, train_dataloader, test_dataloader
