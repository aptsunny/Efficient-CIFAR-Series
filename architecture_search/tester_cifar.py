# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import argparse
import logging
import random
import time
from itertools import cycle

import nni
import numpy as np
import torch
import torch.nn as nn
from nni.nas.pytorch.classic_nas import get_and_apply_next_architecture
from nni.nas.pytorch.utils import AverageMeterGroup

from network import CIFAR100_OneShot, load_and_parse_state_dict
from utils import CrossEntropyLabelSmooth, accuracy
from dataset_cifar import *

logger = logging.getLogger("nni.spos.tester")


def retrain_bn(model, criterion, max_iters, log_freq, loader):
    with torch.no_grad():
        logger.info("Clear BN statistics...")
        for m in model.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.running_mean = torch.zeros_like(m.running_mean)
                m.running_var = torch.ones_like(m.running_var)

        logger.info("Train BN with training set (BN sanitize)...")
        model.train()
        meters = AverageMeterGroup()
        for step in range(max_iters):
            inputs, targets = next(loader)
            logits = model(inputs)
            loss = criterion(logits, targets)
            metrics = accuracy(logits, targets)
            metrics["loss"] = loss.item()
            meters.update(metrics)
            if step % log_freq == 0 or step + 1 == max_iters:
                logger.info("Train Step [%d/%d] %s", step + 1, max_iters, meters)


def test_acc(model, criterion, log_freq, loader):
    logger.info("Start testing...")
    model.eval()
    meters = AverageMeterGroup()
    start_time = time.time()
    with torch.no_grad():
        for step, (inputs, targets) in enumerate(loader):
            logits = model(inputs)
            loss = criterion(logits, targets)
            metrics = accuracy(logits, targets)
            metrics["loss"] = loss.item()
            meters.update(metrics)
            if step % log_freq == 0 or step + 1 == len(loader):
                logger.info("Valid Step [%d/%d] time %.3fs acc1 %.4f acc5 %.4f loss %.4f",
                            step + 1, len(loader), time.time() - start_time,
                            meters.acc1.avg, meters.acc5.avg, meters.loss.avg)
    return meters.acc1.avg


def evaluate_acc(model, criterion, args, loader_train, loader_test):
    acc_before = test_acc(model, criterion, args.log_frequency, loader_test)
    nni.report_intermediate_result(acc_before)

    retrain_bn(model, criterion, args.train_iters, args.log_frequency, loader_train)
    acc = test_acc(model, criterion, args.log_frequency, loader_test)
    assert isinstance(acc, float)
    nni.report_intermediate_result(acc)
    nni.report_final_result(acc)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("SPOS Candidate Tester")

    parser.add_argument("--checkpoint", type=str, default="../checkpoints/epoch_39.pth.tar")
    # parser.add_argument("--checkpoint", type=str, default="../checkpoints/epoch_0.pth.tar")
    # parser.add_argument("--spos-preprocessing", action="store_true", default=False,
    #                     help="When true, image values will range from 0 to 255 and use BGR "
    #                          "(as in original repo).")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--train-iters", type=int, default=200) # bn
    parser.add_argument("--test-batch-size", type=int, default=512)
    parser.add_argument("--log-frequency", type=int, default=10)
    parser.add_argument("--classes", type=int, default=100)

    args = parser.parse_args()

    # use a fixed set of image will improve the performance
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.deterministic = True

    assert torch.cuda.is_available()


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

    model = CIFAR100_OneShot(channels=channels, n_classes=args.classes)

    criterion = nn.CrossEntropyLoss()
    # criterion = CrossEntropyLabelSmooth(10, 0.1)

    get_and_apply_next_architecture(model)

    model.load_state_dict(load_and_parse_state_dict(filepath=args.checkpoint))
    # model.cuda()

    dataset_train, dataset_valid = get_dataset("cifar100", cutout_length=8)

    train_loader = torch.utils.data.DataLoader(dataset_train,
                                                    batch_size=args.batch_size,
                                                    num_workers=args.workers)

    val_loader = torch.utils.data.DataLoader(dataset_valid,
                                                   batch_size=args.batch_size,
                                                   num_workers=args.workers)

    train_loader = cycle(train_loader)

    evaluate_acc(model, criterion, args, train_loader, val_loader)
