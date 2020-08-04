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
from network import CIFAR100_OneShot, load_and_parse_state_dict
from utils import *
from dataset_cifar import *

# from apex import amp # old fp16

logger = logging.getLogger("nni.spos.supernet")

if __name__ == "__main__":
    parser = argparse.ArgumentParser("SPOS Supernet Training")
    parser.add_argument("--spos-preprocessing", action="store_true", default=False,
                        help="When true, image values will range from 0 to 255 and use BGR (as in original repo).")
    parser.add_argument("--load-checkpoint", action="store_true", default=False)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=100) #24
    parser.add_argument("--learning-rate", type=float, default=0.4)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--log-frequency", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--label-smoothing", type=float, default=0.1)
    parser.add_argument("--classes", type=int, default=100)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.deterministic = True

    lr = args.learning_rate

    hp_result ={
        "peak_lr": 0.36057638714284174,
        "prep": 48,
        "layer1": 84,
        "layer2": 256,
        "layer3": 512
    }

    c_prep = hp_result['prep']
    c_layer1 = hp_result['layer1']
    c_layer2 = hp_result['layer2']
    c_layer3 = hp_result['layer3']
    channels = [c_prep, c_layer1, c_layer2, c_layer3]
    lr = hp_result['peak_lr']

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
    model = CIFAR100_OneShot(channels=channels, n_classes=args.classes)

    if args.load_checkpoint:
        if not args.spos_preprocessing:
            logger.warning("You might want to use SPOS preprocessing if you are loading their checkpoints.")
        model.load_state_dict(load_and_parse_state_dict("../checkpoints/epoch_29.pth.tar"))

    model.cuda()
    if torch.cuda.device_count() > 1:  # exclude last gpu, saving for data preprocessing on gpu
        model = nn.DataParallel(model, device_ids=list(range(0, torch.cuda.device_count() - 1)))

    # mutator
    mutator = SPOSSupernetTrainingMutator(model, flops_func=model.get_candidate_flops, flops_lb=0, flops_ub=360E6)

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.SGD(model.parameters(), lr=lr,
                                momentum=args.momentum, weight_decay=args.weight_decay)

    scaler = torch.cuda.amp.GradScaler()

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,
                                                  lambda step: (1.0 - step / args.epochs)
                                                  if step <= args.epochs else 0,
                                                  last_epoch=-1)

    trainer = SPOSSupernetTrainer(model, criterion, accuracy, optimizer, args.epochs, train_loader, valid_loader,
                                  mutator=mutator,
                                  batch_size=args.batch_size,
                                  log_frequency=args.log_frequency,
                                  workers=args.workers,
                                  callbacks=[LRSchedulerCallback(scheduler),
                                             ModelCheckpoint("../checkpoints")],
                                  scaler=scaler)

    trainer.train()
    print(f'Finished in {timer():.2} seconds')


