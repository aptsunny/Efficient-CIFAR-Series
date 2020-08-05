# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import logging
import torch
from nni.nas.pytorch.trainer import Trainer
from nni.nas.pytorch.utils import AverageMeterGroup

from .mutator_cifar import SPOSSupernetTrainingMutator

logger = logging.getLogger(__name__)

from .plot_utils import VisdomLinePlotter

class SPOSSupernetTrainer(Trainer):
    """
    This trainer trains a supernet that can be used for evolution search.

    Parameters
    ----------
    model : nn.Module
        Model with mutables.
    mutator : Mutator
        A mutator object that has been initialized with the model.
    loss : callable
        Called with logits and targets. Returns a loss tensor.
    metrics : callable
        Returns a dict that maps metrics keys to metrics data.
    optimizer : Optimizer
        Optimizer that optimizes the model.
    num_epochs : int
        Number of epochs of training.
    train_loader : iterable
        Data loader of training. Raise ``StopIteration`` when one epoch is exhausted.
    dataset_valid : iterable
        Data loader of validation. Raise ``StopIteration`` when one epoch is exhausted.
    batch_size : int
        Batch size.
    workers: int
        Number of threads for data preprocessing. Not used for this trainer. Maybe removed in future.
    device : torch.device
        Device object. Either ``torch.device("cuda")`` or ``torch.device("cpu")``. When ``None``, trainer will
        automatic detects GPU and selects GPU first.
    log_frequency : int
        Number of mini-batches to log metrics.
    callbacks : list of Callback
        Callbacks to plug into the trainer. See Callbacks.
    """

    def __init__(self, model, loss, metrics, optimizer, num_epochs, train_loader, valid_loader,
                 mutator=None, batch_size=64, workers=4, device=None, log_frequency=None,
                 callbacks=None, scaler=None):
        assert torch.cuda.is_available()
        super().__init__(model, mutator if mutator is not None else SPOSSupernetTrainingMutator(model),
                         loss, metrics, optimizer, num_epochs, None, None,
                         batch_size, workers, device, log_frequency, callbacks)

        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.scaler = scaler

        self.train_acc = []
        self.val_acc = []
        self.record_epoch = []
        global plotter
        plotter = VisdomLinePlotter(env_name='Tutorial Plots')


    def train_one_epoch(self, epoch):
        self.model.train()
        meters = AverageMeterGroup()

        for step, (x, y) in enumerate(self.train_loader):
            x = x.cuda()
            y = y.cuda()

            self.optimizer.zero_grad()
            self.mutator.reset() # sample_search nni_sy/examples/nas/spos_randa_fair/cifar_spos/mutator_cifar.py

            with torch.cuda.amp.autocast():
                logits = self.model(x).squeeze()
                loss = self.loss(logits, y)

            self.scaler.scale(loss).backward() # loss.backward()
            self.scaler.step(self.optimizer) # self.optimizer.step()
            self.scaler.update()


            metrics = self.metrics(logits, y)
            metrics["loss"] = loss.item()
            meters.update(metrics)
            # if self.log_frequency is not None and step % self.log_frequency == 0:
            #     logger.info("Epoch [%s/%s] Step [%s/%s]  %s", epoch + 1, self.num_epochs, step + 1, len(self.train_loader), meters)
                # print('Epoch [{}/{}] Step [{}/{}]  {}'.format(epoch + 1,
                #             self.num_epochs, step + 1, len(self.train_loader), meters))

        # print('train acc:', epoch, meters['acc1'])
        # self.train_acc.append(meters['acc1'])
        # self.record_epoch.append(epoch)
        # print()
        plotter.plot('loss', 'train', 'Class Loss', epoch, metrics["loss"])
        plotter.plot('acc', 'train', 'Class Accuracy', epoch, meters['acc1'].avg)



    def validate_one_epoch(self, epoch):
        self.model.eval()
        meters = AverageMeterGroup()
        with torch.no_grad():
            for step, (x, y) in enumerate(self.valid_loader):
                x = x.cuda()
                y = y.cuda()

                self.mutator.reset()
                logits = self.model(x)
                loss = self.loss(logits, y)
                metrics = self.metrics(logits, y)
                metrics["loss"] = loss.item()
                meters.update(metrics)
                # if self.log_frequency is not None and step % self.log_frequency == 0:
                #     logger.info("Epoch [%s/%s] Validation Step [%s/%s]  %s", epoch + 1, self.num_epochs, step + 1, len(self.valid_loader), meters)
                    # print('Epoch [{}/{}] Validation Step [{}/{}]  {}'.format(epoch + 1,
                    #             self.num_epochs, step + 1, len(self.valid_loader), meters))

            print('valid acc:', epoch, meters['acc1'])
            # self.val_acc.append(meters['acc1'])

            # Plot validation results
            plotter.plot('loss', 'val', 'Class Loss', epoch, metrics["loss"])
            plotter.plot('acc', 'val', 'Class Accuracy', epoch, meters['acc1'].avg)


    """
    def plot_curve(self, name='24epoch'):
        import matplotlib.pyplot as plt
        import numpy as np
        # T = np.arctan2(Y_axis, X_axis)
        # plt.scatter(X_axis, Y_axis, s=2, c=T, alpha=0.5)
        plt.scatter(self.record_epoch, self.train_acc, c=1, s=20, alpha=0.5)
        plt.scatter(self.record_epoch, self.val_acc, c=2, s=20, alpha=0.5)
        plt.savefig('{}.png'.format(name), dpi=600)
        print('save {}.png'.format(name))
        plt.show()
    """