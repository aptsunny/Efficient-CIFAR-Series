# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import logging
import torch
from nni.nas.pytorch.trainer import Trainer
from nni.nas.pytorch.utils import AverageMeterGroup

from .mutator_cifar import SPOSSupernetTrainingMutator

# from apex import amp # old fp16

logger = logging.getLogger(__name__)

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

    def train_one_epoch(self, epoch):
        self.model.train()
        meters = AverageMeterGroup()

        for step, (x, y) in enumerate(self.train_loader):
            x = x.cuda()
            y = y.cuda()

            self.optimizer.zero_grad()
            self.mutator.reset() # sample_search nni_sy/examples/nas/spos_randa_fair/cifar_spos/mutator_cifar.py

            # NEW
            with torch.cuda.amp.autocast():
                logits = self.model(x).squeeze()
                loss = self.loss(logits, y)


            # logits = self.model(x)
            # loss = self.loss(logits, y)



            # old fp16
            # with amp.scale_loss(loss, self.optimizer) as scaled_loss:
            #     scaled_loss.backward()

            # new
            self.scaler.scale(loss).backward() # loss.backward()
            self.scaler.step(self.optimizer) # self.optimizer.step()
            self.scaler.update()


            metrics = self.metrics(logits, y)
            metrics["loss"] = loss.item()
            meters.update(metrics)
            if self.log_frequency is not None and step % self.log_frequency == 0:
                logger.info("Epoch [%s/%s] Step [%s/%s]  %s", epoch + 1,
                            self.num_epochs, step + 1, len(self.train_loader), meters)
                # print('Epoch [{}/{}] Step [{}/{}]  {}'.format(epoch + 1,
                #             self.num_epochs, step + 1, len(self.train_loader), meters))
        print('train acc:', epoch, meters['acc1'])



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
                if self.log_frequency is not None and step % self.log_frequency == 0:
                    logger.info("Epoch [%s/%s] Validation Step [%s/%s]  %s", epoch + 1,
                                self.num_epochs, step + 1, len(self.valid_loader), meters)
                    # print('Epoch [{}/{}] Validation Step [{}/{}]  {}'.format(epoch + 1,
                    #             self.num_epochs, step + 1, len(self.valid_loader), meters))
            print('valid acc:', epoch, meters['acc1'])