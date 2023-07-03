"""
Evaluation with AutoAttack.
"""

import json
import time
import argparse
import shutil

import os
import numpy as np
import pandas as pd

import torch
import torch.nn as nn

from autoattack import AutoAttack
    
from core.data import get_data_info
from core.data import load_data
from core.models import create_model

from core.utils import Logger
from core.utils import parser_eval
from core.utils import seed

import torchvision
from torchvision import datasets, transforms



# Setup

parse = parser_eval()
args = parse.parse_args()

LOG_DIR = args.log_dir + '/' + args.desc
with open(LOG_DIR+'/args.txt', 'r') as f:
    old = json.load(f)
    args.__dict__ = dict(vars(args), **old)

if args.data in ['cifar10']:
    da = '/cifar10/'


DATA_DIR = args.data_dir + da
WEIGHTS = LOG_DIR + '/weights-best.pt'

log_path = LOG_DIR + '/log-aa.log'
logger = Logger(log_path)

info = get_data_info(DATA_DIR)

BATCH_SIZE = 128
BATCH_SIZE_VALIDATION = 256

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
kwargs = {'num_workers': 8, 'pin_memory': True} if use_cuda else {}

logger.log('Using device: {}'.format(device))


# Load data

seed(args.seed)
CIFAR_MEAN = [0.49139968, 0.48215827, 0.44653124]         
CIFAR_STD = [0.24703233, 0.24348505, 0.26158768]
test_transform = transforms.Compose([
                transforms.ToTensor(),
               transforms.Normalize(CIFAR_MEAN, CIFAR_STD)])

testset = torchvision.datasets.CIFAR10(root='./cifar-data', train=False, download=True, transform=test_transform)


test_dataloader = torch.utils.data.DataLoader(testset, batch_size=256, shuffle=False, **kwargs)

if args.train:
    logger.log('Evaluating on training set.')
    l = [x for (x, y) in train_dataloader]
    x_test = torch.cat(l, 0)
    l = [y for (x, y) in train_dataloader]
    y_test = torch.cat(l, 0)
else:
    l = [x for (x, y) in test_dataloader]
    x_test = torch.cat(l, 0)
    l = [y for (x, y) in test_dataloader]
    y_test = torch.cat(l, 0)



# Model
print(args.model)
model = create_model(args.model, args.normalize, info, device)
checkpoint = torch.load(WEIGHTS)
if 'tau' in args and args.tau:
    print ('Using WA model.')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
del checkpoint



# AA Evaluation

seed(args.seed)
norm = 'Linf' if args.attack in ['fgsm', 'linf-pgd', 'linf-df'] else 'L2'
adversary = AutoAttack(model, norm=norm, eps=args.attack_eps, log_path=log_path, version=args.version, seed=args.seed)

if args.version == 'custom':
    adversary.attacks_to_run = ['apgd-ce', 'apgd-t']
    adversary.apgd.n_restarts = 1
    adversary.apgd_targeted.n_restarts = 1

with torch.no_grad():
    x_adv = adversary.run_standard_evaluation(x_test, y_test, bs=BATCH_SIZE_VALIDATION)

print ('Script Completed.')
