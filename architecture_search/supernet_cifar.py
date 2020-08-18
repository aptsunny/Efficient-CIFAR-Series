# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import argparse
import logging
import random
import numpy as np
import torch
import torch.nn as nn
from nni.nas.pytorch.callbacks import LRSchedulerCallback
from nni.nas.pytorch.callbacks import ModelCheckpoint

from cifar_spos import SPOSSupernetTrainingMutator, SPOSSupernetTrainer
from network import Superresnet, load_and_parse_state_dict
from utils import *
from dataset_cifar import *

logger = logging.getLogger("nni.spos.supernet")

def fast_hpo_lr_parameters(model, lr_group=[], arch_search=None):
    # lr_group -> initial lr group
    rest_name = []
    figure_ = []
    for name, param in model.named_parameters():
        rest_name.append(param)
        figure_.append(name)

    figure_choice=[]
    choice = []
    # 13/12/12/6/1 = 44
    if len(rest_name) == 44:
        loc = 0
        # 15
        # split_list = [1, 3, 3, 3, 3, 3, 3, 3, 1]
        split_list = [13, 12, 12, 6, 1]
        for i in range(0, len(split_list), 1):
            st = loc
            en = loc + split_list[i]
            # print(loc, loc + split_list[i])
            b = figure_[st:en]
            a = rest_name[st:en]
            loc = en
            figure_choice.append(b)
            choice.append(a)
    groups = [dict(params=choice[x], lr=lr_group[x]) for x in range(len(choice))]
    return groups

def fast_hpo_lr_parameters_freeze(model, lr_group=[], arch_search=None):
    #
    kkk = []
    for m in model.modules():
        if hasattr(m, 'active'):
            temp = {'params': m.parameters(), 'lr': m.lr, 'layer_index': m.layer_index}

            # if m.layer_index==4 or m.layer_index==3:
            # if m.layer_index<5:
            #     temp = {'params': m.parameters(), 'lr': 4e-5, 'layer_index': m.layer_index}
            # else:
            #     temp = {'params': m.parameters(), 'lr': m.lr, 'layer_index': m.layer_index}
            kkk.append(temp)
            # print(kkk)
    return kkk

if __name__ == "__main__":
    parser = argparse.ArgumentParser("SPOS Supernet Training")
    parser.add_argument("--spos-preprocessing", action="store_true", default=False,
                        help="When true, image values will range from 0 to 255 and use BGR (as in original repo).")
    parser.add_argument("--load-checkpoint", action="store_true", default=False)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=0.4)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--log-frequency", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--label-smoothing", type=float, default=0.1)
    parser.add_argument("--classes", type=int, default=100)

    # Modification
    parser.add_argument("--batch-size", type=int, default=512) # 512
    parser.add_argument("--epochs", type=int, default=200) #24 # 40 #
    parser.add_argument("--lr-scheduler", type=str, default='linear', help="linear， step")
    parser.add_argument("--mode", type=str, default='', help="normal")
    parser.add_argument("--layer-wise", type=str, default='inner_decay',
                        help="design/ decay / origin: # design -> [13, 12, 12, 6, 1], decay -> [0, 1, 2, 3, 4] , origin None")
    parser.add_argument("--tag", type=str, default='0.8 linear layerwise2 baseline 200', help="normal")

    # plotter = VisdomLinePlotter(env_name='40epoch R23 C12 e5fixed+ multi-path loss')
    # plotter = VisdomLinePlotter(env_name='200epoch R23 C12345678 e2fixed+ multi-path loss')
    # plotter = VisdomLinePlotter(env_name='40epoch R23 C12345678 e2fixed+ multi-path loss')
    # plotter = VisdomLinePlotter(env_name='400epoch R23 C12345678 e2fixed+ multi-path loss')
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.deterministic = True

    lr = args.learning_rate

    # Derive from Hyperparameter tuning
    # 78.01_cifar100_7_7_7_2_e24_t210.49_logs.tsv
    RCV_CONFIG = {
        "peak_lr": 0.6499631190592446,
        "prep": 64,
        "layer1": 112,
        "layer2": 256,
        "layer3": 512,
        "extra_prep": 1,
        "extra_layer1": 0,
        "extra_layer2": 0,
        "extra_layer3": 0,
        "res_prep": 2,
        "res_layer1": 3,
        "res_layer2": 3,
        "res_layer3": 1
    }

    # Network Configuration
    channels = {'prep': RCV_CONFIG['prep'], 'layer1': RCV_CONFIG['layer1'], 'layer2': RCV_CONFIG['layer2'],
                'layer3': RCV_CONFIG['layer3']} if 'prep' in RCV_CONFIG \
        else {'prep': 48, 'layer1': 112, 'layer2': 256, 'layer3': 384}
    extra_layers = {'prep': RCV_CONFIG['extra_prep'], 'layer1': RCV_CONFIG['extra_layer1'],
                    'layer2': RCV_CONFIG['extra_layer2'],
                    'layer3': RCV_CONFIG['extra_layer3']} if 'extra_prep' in RCV_CONFIG \
        else {'prep': 0, 'layer1': 0, 'layer2': 0, 'layer3': 0}
    res_layers = {'prep': RCV_CONFIG['res_prep'], 'layer1': RCV_CONFIG['res_layer1'],
                  'layer2': RCV_CONFIG['res_layer2'], 'layer3': RCV_CONFIG['res_layer3']} if 'res_prep' in RCV_CONFIG \
        else {'prep': 0, 'layer1': 1, 'layer2': 0, 'layer3': 1}

    # Train supernet 30% Initial learning rate
    lr = RCV_CONFIG['peak_lr']*0.3

    timer = Timer()
    print('Preprocessing training data')
    dataset_train, dataset_valid = get_dataset("cifar100", cutout_length=8)
    train_loader = torch.utils.data.DataLoader(dataset_train,
                                               shuffle=True,
                                               batch_size=args.batch_size,
                                               num_workers=args.workers)

    print(f'Finished in {timer():.2} seconds')

    print('Preprocessing test data')
    valid_loader = torch.utils.data.DataLoader(dataset_valid,
                                               shuffle=False,
                                               batch_size=args.batch_size,
                                               num_workers=args.workers)
    print(f'Finished in {timer():.2} seconds')

    print('Preprocessing training')
    model = Superresnet(mode=args.mode, channels=channels, extra_layers=extra_layers, res_layers=res_layers, n_classes=args.classes, epoch=args.epochs)

    if args.load_checkpoint:
        if not args.spos_preprocessing:
            logger.warning("You might want to use SPOS preprocessing if you are loading their checkpoints.")
        model.load_state_dict(load_and_parse_state_dict("../checkpoints/epoch_29.pth.tar"))
        # update optimizer undone

    model.cuda()
    if torch.cuda.device_count() > 1:  # exclude last gpu, saving for data preprocessing on gpu
        model = nn.DataParallel(model, device_ids=list(range(0, torch.cuda.device_count() - 1)))

    # Get mutator from search space design
    mutator = SPOSSupernetTrainingMutator(model, flops_func=model.get_candidate_flops, flops_lb=0, flops_ub=360E6, starting_line=2)

    criterion = nn.CrossEntropyLoss()


    # Optimizer
    if args.layer_wise == 'origin':
        optimizer = torch.optim.SGD(model.parameters(),
                                    lr=lr,
                                    momentum=args.momentum,
                                    weight_decay=args.weight_decay)

    elif args.layer_wise == 'design':
        optimizer = torch.optim.SGD(fast_hpo_lr_parameters(model, lr_group=[lr, lr, lr, lr, lr]),
                                    lr=lr,
                                    momentum=args.momentum,
                                    weight_decay=args.weight_decay)

    elif args.layer_wise == 'decay':
        optimizer = torch.optim.SGD(fast_hpo_lr_parameters_freeze(model),
                                        nesterov=True,
                                        momentum=args.momentum,
                                        weight_decay=args.weight_decay)

    elif args.layer_wise == 'inner_decay':
        optimizer = model.optim


    if args.layer_wise != 'inner_decay':
        # Warm-up && LR scheduler
        from warmup_scheduler import GradualWarmupScheduler
        scheduler_linear = LinearLR(optimizer, args.epochs-6)
        scheduler_step = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.8)
        # test constant learning rate
        # scheduler_step = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.99)

        if args.lr_scheduler == 'linear': # scheduler_linear # scheduler_steplr
            scheduler = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=5, after_scheduler=scheduler_linear)
        elif args.lr_scheduler == 'step':
            scheduler = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=5, after_scheduler=scheduler_step)
        elif args.lr_scheduler == 'lambda':
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: (1.0 - step / args.epochs) if step <= args.epochs else 0, last_epoch=-1)

        callbacks = [LRSchedulerCallback(scheduler), ModelCheckpoint("../checkpoints")], # update learning rate

    else:
        callbacks = None


    # Mixed Precision Training
    scaler = torch.cuda.amp.GradScaler()

    print(f'Finished load model in {timer():.2} seconds')

    trainer = SPOSSupernetTrainer(model, criterion, accuracy, optimizer, args.epochs, train_loader, valid_loader,
                                  mutator=mutator,
                                  batch_size=args.batch_size,
                                  log_frequency=args.log_frequency,
                                  workers=args.workers,
                                  callbacks= callbacks,
                                  # callbacks=[LRSchedulerCallback(model), ModelCheckpoint("../checkpoints")], # update learning rate 按照epoch更新
                                  scaler=scaler,
                                  env_name=args.tag,
                                  timer = timer)

    trainer.train()
    print(f'Finished in {timer():.2} seconds')


