# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import logging
import torch
import numpy as np

from nni.nas.pytorch.trainer import Trainer
from nni.nas.pytorch.utils import AverageMeterGroup

from .mutator_cifar import SPOSSupernetTrainingMutator
from .plot_utils import VisdomLinePlotter

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
                 callbacks=None, scaler=None, env_name='', timer=None):
        assert torch.cuda.is_available()
        super().__init__(model, mutator if mutator is not None else SPOSSupernetTrainingMutator(model),
                         loss, metrics, optimizer, num_epochs, None, None,
                         batch_size, workers, device, log_frequency, callbacks)

        self.num_epochs = num_epochs
        self.train_loader = train_loader
        self.valid_loader = valid_loader

        self.scaler = scaler
        # self.scheduler  = self.callbacks[0].scheduler

        global plotter
        plotter = VisdomLinePlotter(env_name=env_name)

        # Initial choice
        self.ccc = {'conv1': np.array([0., 0.]),
                    'conv2': np.array([0., 0.]),
                    'conv3': np.array([0., 0.]),
                    'conv4': np.array([0., 0.]),
                    'skip':  np.array([0., 0.])}
        self.timer = timer

    def train_one_epoch(self, epoch):
        self.model.train()
        meters = AverageMeterGroup()

        for step, (x, y) in enumerate(self.train_loader):
            x = x.cuda()
            y = y.cuda()

            self.optimizer.zero_grad()
            self.mutator.update(epoch)
            self.mutator.reset()

            # check one choice
            mutator = self.mutator.status()
            for name in mutator:
                if len(mutator[name])==1:
                    mutator[name].append(0)
                # print(name, len(mutator[name]))
            for i in mutator:
                self.ccc[i] = np.sum([self.ccc[i], mutator[i]], axis=0)

            with torch.cuda.amp.autocast():
                logits = self.model(x).squeeze()
                loss = self.loss(logits, y)

            # Mixed Precision Training
            self.scaler.scale(loss).backward() # loss.backward()
            self.scaler.step(self.optimizer) # self.optimizer.step()
            self.scaler.update()

            metrics = self.metrics(logits, y)
            metrics["loss"] = loss.item()
            meters.update(metrics)

            # instead callback
            self.model.step()


            # print(mutator)
            # {'conv1': [1.0, 0.0], 'conv2': [1.0, 0], 'conv3': [1.0, 0], 'conv4': [1.0, 0], 'skip': [0.0, 1.0]}
            # only two choices
            net_index = []
            for i in mutator:
                net_index.append(i+"_"+str(mutator[i][0]))
            choice_sha = "".join(net_index)
            # print(choice_sha)
            # plotter.plot('Childmodel train loss', 'train path-{}'.format(choice_sha), 'Multi-path Loss', epoch*1000+step, metrics["loss"])


            # aaa = self.scheduler.record_lr()

            # 太耗时
            # aaa = self.model.record_lr()
            # print(aaa)
            # plotter.plot('layer-wise learning rate', 'prep', 'layer-wise learning rate scheduler', epoch, aaa['prep'])
            # plotter.plot('layer-wise learning rate', 'layer1', 'layer-wise learning rate scheduler', epoch, aaa['layer1'])
            # plotter.plot('layer-wise learning rate', 'layer2', 'layer-wise learning rate scheduler', epoch, aaa['layer2'])
            # plotter.plot('layer-wise learning rate', 'layer3', 'layer-wise learning rate scheduler', epoch, aaa['layer3'])
            # plotter.plot('layer-wise learning rate', 'fc', 'layer-wise learning rate scheduler', epoch, aaa['classifier'])



            # if self.log_frequency is not None and step % self.log_frequency == 0:
            #     logger.info("Epoch [%s/%s] Step [%s/%s]  %s", epoch + 1, self.num_epochs, step + 1, len(self.train_loader), meters)
                # print('Epoch [{}/{}] Step [{}/{}]  {}'.format(epoch + 1,
                #             self.num_epochs, step + 1, len(self.train_loader), meters))

        # train 各个分支的曲线
        plotter.plot('loss', 'train', 'Class Loss', epoch, metrics["loss"])
        plotter.plot('acc', 'train', 'Class Accuracy', epoch, meters['acc1'].avg)

        print('train acc:', epoch, meters['acc1'].avg)
        #
        # plotter.plot('learning rate', 'train', 'learning rate scheduler', epoch, self.scheduler.get_lr()[0])


        aaa = self.model.record_lr()
        """
        {'prep': 0.1552, 'layer1': None, 'layer2': None, 'layer3': None, 'classifier': 0.1552,
         'layer1.conv1': 0.5920410156249998, 
         'layer1.feature.0': 0.4115078863161405,
         'layer1.feature.1': 0.29203618087426436, 
         'layer1.feature.2': 0.21113006683836988,
         
         'layer2.conv1': 0.5920410156249998, 
         'layer2.feature.0': 0.4115078863161405,
         'layer2.feature.1': 0.29203618087426436, 
         'layer2.feature.2': 0.21113006683836988,
         
         'layer3.conv1': 0.5920410156249998, 
         'layer3.feature.0': 0.4115078863161405}
         
         
         {'prep': 0.1552, 'layer1': None, 'layer2': None, 'layer3': None, 'classifier': 0.1552, 
         'layer1.conv1': 0.5920410156249998, 
         'layer1.conv2': 0.4115078863161405, 
         'layer1.conv3': 0.29203618087426436, 
         'layer1.conv4': 0.21113006683836988, 
         
         'layer2.conv1': 0.5920410156249998, 
         'layer2.conv2': 0.4115078863161405, 
         'layer2.conv3': 0.29203618087426436, 
         'layer2.conv4': 0.21113006683836988, 
         
         'layer3.conv1': 0.5920410156249998, 
         'layer3.conv2': 0.4115078863161405, 
         'layer3.conv3': 0.29203618087426436, 
         'layer3.conv4': 0.21113006683836988}
        """
        # print(aaa)
        for layer_name in aaa.keys():
            if aaa[layer_name]:
                if layer_name.startswith('layer1'):
                    plotter.plot('layer1 learning rate', '{}'.format(layer_name), 'layer-wise learning rate scheduler', epoch, aaa[layer_name])
                elif layer_name.startswith('layer2'):
                    plotter.plot('layer2 learning rate', '{}'.format(layer_name), 'layer-wise learning rate scheduler', epoch, aaa[layer_name])
                elif layer_name.startswith('layer3'):
                    plotter.plot('layer3 learning rate', '{}'.format(layer_name), 'layer-wise learning rate scheduler', epoch, aaa[layer_name])
                else:
                    plotter.plot('prep/fc learning rate', '{}'.format(layer_name), 'layer-wise learning rate scheduler', epoch, aaa[layer_name])


        # plotter.plot('layer-wise learning rate', 'prep', 'layer-wise learning rate scheduler', epoch, aaa['prep'])
        # plotter.plot('layer-wise learning rate', 'layer1', 'layer-wise learning rate scheduler', epoch, aaa['layer1'])
        # plotter.plot('layer-wise learning rate', 'layer2', 'layer-wise learning rate scheduler', epoch, aaa['layer2'])
        # plotter.plot('layer-wise learning rate', 'layer3', 'layer-wise learning rate scheduler', epoch, aaa['layer3'])
        # plotter.plot('layer-wise learning rate', 'fc', 'layer-wise learning rate scheduler', epoch, aaa['classifier'])

        # plotter.plot('layer-wise learning rate', 'prep', 'layer-wise learning rate scheduler', epoch, self.scheduler.get_lr()[0])
        # plotter.plot('layer-wise learning rate', 'layer1', 'layer-wise learning rate scheduler', epoch, self.scheduler.get_lr()[1])
        # plotter.plot('layer-wise learning rate', 'layer2', 'layer-wise learning rate scheduler', epoch, self.scheduler.get_lr()[2])
        # plotter.plot('layer-wise learning rate', 'layer3', 'layer-wise learning rate scheduler', epoch, self.scheduler.get_lr()[3])
        # plotter.plot('layer-wise learning rate', 'fc', 'layer-wise learning rate scheduler', epoch, self.scheduler.get_lr()[4])
        #

        # if epoch == 39 or epoch == 199:
        if epoch == 39 or epoch == 199:
            plotter.plot_bar('{} epoch, sample choice counts'.format(epoch), 'train', 'choice', epoch, self.ccc)

        print(f'Finished Training one epoch in {self.timer():.2} seconds')


    def validate_one_epoch(self, epoch):
        self.model.eval()
        meters = AverageMeterGroup()
        with torch.no_grad():
            for step, (x, y) in enumerate(self.valid_loader):
                x = x.cuda()
                y = y.cuda()

                self.mutator.update(epoch)
                self.mutator.reset()
                # print('val mutator:', self.mutator.status())

                logits = self.model(x)
                loss = self.loss(logits, y)
                metrics = self.metrics(logits, y)
                metrics["loss"] = loss.item()
                meters.update(metrics)

                # print(mutator)
                # {'conv1': [1.0, 0.0], 'conv2': [1.0, 0], 'conv3': [1.0, 0], 'conv4': [1.0, 0], 'skip': [0.0, 1.0]}
                # only two choices
                mutator = self.mutator.status()
                net_index = []
                for i in mutator:
                    net_index.append(i + "_" + str(mutator[i][0]))
                choice_sha = "".join(net_index)
                # print(choice_sha)
                #
                # plotter.plot('Childmodel val loss', 'val path-{}'.format(choice_sha), 'Multi-path Loss', epoch * 1000 + step, metrics["loss"])

                # if self.log_frequency is not None and step % self.log_frequency == 0:
                #     logger.info("Epoch [%s/%s] Validation Step [%s/%s]  %s", epoch + 1, self.num_epochs, step + 1, len(self.valid_loader), meters)
                    # print('Epoch [{}/{}] Validation Step [{}/{}]  {}'.format(epoch + 1,
                    #             self.num_epochs, step + 1, len(self.valid_loader), meters))

            print('valid acc:', epoch, meters['acc1'])

            # Plot validation results
            plotter.plot('loss', 'val', 'Class Loss', epoch, metrics["loss"])
            plotter.plot('acc', 'val', 'Class Accuracy', epoch, meters['acc1'].avg)

        print(f'Finished Validating one epoch in {self.timer():.2} seconds')
