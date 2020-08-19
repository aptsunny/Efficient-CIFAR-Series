# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
import pickle
import re

import torch
import torch.nn as nn
import numpy as np

from nni.nas.pytorch.mutables import LayerChoice, InputChoice

from blocks import ShuffleNetBlock, ShuffleXceptionBlock, BasicBlock, Bottleneck, ConvBnRelu, \
    ConvBnReluPool, Mul, DynamicBasicBlock, ConvBnReluPool_detail, DynamicResidualBlock, Residual, DynamicResidualBlock_fix


# scale_fn = lambda x: x**3

class Superresnet(nn.Module):
    def __init__(self,
                 mode='',
                 input_size=32,
                 n_classes=100,
                 channels=None,
                 extra_layers=None,
                 res_layers=None,
                 op_flops_path="/home/ubuntu/0_datasets/op_flops_dict.pkl",
                 epoch=None,
                 weight=0.125):
        super().__init__()

        assert input_size % 32 == 0
        # with open(os.path.join(os.path.dirname(__file__), op_flops_path), "rb") as fp:
        #     self._op_flops_dict = pickle.load(fp)
        # self.stage_blocks = [3, 1, 3]


        # base lr 1e-1
        self.base_lr = 0.8
        self.epoch = epoch

        # layer1:32->16, layer2:16->8, layer3:8->4
        self.extra_layers = extra_layers or {'prep': 0, 'layer1': 0, 'layer2': 0, 'layer3': 0}
        self.res_layers = res_layers or {'prep': 0, 'layer1': 0, 'layer2': 0, 'layer3': 0}
        self.stage_channels = channels or {'prep': 64, 'layer1': 128, 'layer2': 256, 'layer3': 512}

        # first_conv_channels = self.stage_channels[0]
        # layers_channels = self.stage_channels[1:]
        # self._first_conv_channels = first_conv_channels
        # self._last_conv_channels = last_conv_channels

        self._parsed_flops = dict()
        self._input_size = input_size
        self._feature_map_size = input_size
        self._n_classes = n_classes


        # self.pool = nn.MaxPool2d(2)
        # self._feature_map_size //= 2
        # p_channels = first_conv_channels
        # features = []
        # for num_blocks, channels in zip(self.stage_blocks, self.stage_channels):
        #     features.extend(self._make_blocks(num_blocks, p_channels, channels))
        #     p_channels = channels
        # self.features = nn.Sequential(*features)

        """
        # struct 1
        p_channels = first_conv_channels
        for num_blocks, channels in zip(self.stage_blocks, layers_channels):
            if channels == 84:#32:#
                layer1 = self._make_blocks(num_blocks, p_channels, channels)
                self.layer1 = nn.Sequential(*layer1)
            elif channels == 256:#48:#
                layer2 = self._make_blocks(num_blocks, p_channels, channels)
                self.layer2 = nn.Sequential(*layer2)
            elif channels == 384:#96:#
                layer3 = self._make_blocks(num_blocks, p_channels, channels)
                self.layer3 = nn.Sequential(*layer3)
            p_channels = channels
        
        # struct 2
        # self.layer1 = DynamicBasicBlock(first_conv_channels, layers_channels[0])
        # self.layer2 = DynamicBasicBlock(layers_channels[0], layers_channels[1])
        # self.layer3 = DynamicBasicBlock(layers_channels[1], layers_channels[2])
        
        # struct 3
        p_channels = first_conv_channels
        for num_blocks, channels in zip(self.stage_blocks, self.stage_channels):
            if channels == 128:
                layer1 = self._make_blocks(num_blocks, p_channels, channels)
                self.layer1 = nn.Sequential(*layer1)
            elif channels == 256:
                layer2 = self._make_blocks(num_blocks, p_channels, channels)
                self.layer2 = nn.Sequential(*layer2)
            elif channels == 512:
                layer3 = self._make_blocks(num_blocks, p_channels, channels)
                self.layer3 = nn.Sequential(*layer3)
            p_channels = channels
        """

        # struct 4
        if mode == 'normal':
            # building first layer
            # print(self.extra_layers)
            # print(self.extra_layers['prep'])
            prep_extra = self.extra_layers['prep']
            prep_res = self.res_layers['prep']

            www = [nn.Conv2d(3, self.stage_channels['prep'], 3, 1, 1, bias=False),
                nn.BatchNorm2d(self.stage_channels['prep'], affine=False),
                nn.ReLU(inplace=True)]
            if prep_extra == 0:
                pass
            else:
                for i in range(prep_extra):
                    www.append(Residual(self.stage_channels['prep'], self.stage_channels['prep']))
            if prep_res == 0:
                pass
            else:
                for i in range(prep_res):
                    www.append(ConvBnRelu(self.stage_channels['prep'], self.stage_channels['prep'], stride=1, k=3))

            self.prep = nn.Sequential(*www)
            self.prep.active = True
            # self.prep.layer_index = 0
            self.prep.layer_index = 4

            self.layer1 = DynamicResidualBlock_fix(self.stage_channels['prep'],   self.stage_channels['layer1'],
                                               extra=self.extra_layers['layer1'], res=self.res_layers['layer1'])
            # self.layer1.active = True
            # self.layer1.layer_index = 1

            self.layer2 = DynamicResidualBlock_fix(self.stage_channels['layer1'], self.stage_channels['layer2'],
                                               extra=self.extra_layers['layer2'], res=self.res_layers['layer2'])
            # self.layer2.active = True
            # self.layer2.layer_index = 2

            self.layer3 = DynamicResidualBlock_fix(self.stage_channels['layer2'], self.stage_channels['layer3'],
                                               extra=self.extra_layers['layer3'], res=self.res_layers['layer3'])

            # self.layer3.active = True
            # self.layer3.layer_index = 3

            ## 按squential 设计 -> 每个block的squential设计


        else:
            # building first layer
            self.prep = nn.Sequential(
                nn.Conv2d(3, self.stage_channels['prep'], 3, 1, 1, bias=False),
                nn.BatchNorm2d(self.stage_channels['prep'], affine=False),
                nn.ReLU(inplace=True),
            )

            self.prep.active = True
            # self.prep.layer_index = 0
            self.prep.layer_index = 4

            self.layer1 = DynamicResidualBlock(self.stage_channels['prep'], self.stage_channels['layer1'])
            # self.layer1.active = True
            # self.layer1.layer_index = 1

            self.layer2 = DynamicResidualBlock(self.stage_channels['layer1'], self.stage_channels['layer2'])
            # self.layer2.active = True
            # self.layer2.layer_index = 2

            self.layer3 = DynamicResidualBlock(self.stage_channels['layer2'], self.stage_channels['layer3'])
            # self.layer3.active = True
            # self.layer3.layer_index = 3

        self.globalpool = nn.MaxPool2d(4)
        self.classifier = nn.Sequential(
            nn.Linear(self.stage_channels['layer3'], n_classes, bias=False),
            Mul(weight))

        self.classifier.active = True
        self.classifier.layer_index = 4

        self._initialize_weights() # Initial layer-wise learning rate

        # Optimizer
        # self.optim = torch.optim.SGD([{'params':m.parameters(),
        #                                  'lr':m.lr,
        #                                  'layer_index':m.layer_index} for m in self.modules() if hasattr(m,'active')],
        #                                 nesterov=True, momentum=0.9, weight_decay=1e-4)

        self.optim = torch.optim.SGD(self.fast_hpo_lr_parameters_freeze(), nesterov=True, momentum=0.9, weight_decay=1e-4)


        # self.t_0 = 0.5
        # Iteration Counter
        self.j = 0
        # A simple dummy variable that indicates we are using an iteration-wise
        # annealing scheme as opposed to epoch-wise.
        self.lr_sched = {'itr': 0}

        """"""

    def fast_hpo_lr_parameters_freeze(self):
        #
        kkk = []
        for m in self.modules():

            # 所有的batchnorm跟着index=4update
            # if isinstance(m, nn.BatchNorm2d):  # m.weight.shape
            #     m.active = True
            #     m.layer_index = 4
            #     m.lr = 0.01
            # 这样填进去还要和 初始化的lr契合

            # if isinstance(m, nn.BatchNorm2d):  # m.weight.shape
            #     m.active = True
            #     m.layer_index = 4
            #     m.max_j=40000
            #     m.warm_j=5000
            #     m.lr_ratio=1

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

    def _make_blocks(self, blocks, in_channels, channels):
        result = []
        for i in range(blocks):
            # stride, inp, oup
            stride = 2 if i == 0 else 1
            inp = in_channels if i == 0 else channels
            oup = channels

            base_mid_channels = channels // 2
            mid_channels = int(base_mid_channels)  # prepare for scale

            choice_block = LayerChoice([
                ConvBnReluPool(inp, oup, stride=stride, k=3),
                # ConvBnReluPool(inp, oup, stride=stride, k=5),
                # ConvBnRelu(inp, oup, stride=stride, k=3),
                # ConvBnRelu(inp, oup, stride=stride, k=5)

                # ConvBnRelu(inp, oup, stride=stride, k=5),
                # ConvBnRelu(inp, oup, stride=stride, k=7)
                # BasicBlock(inp, oup, stride=stride),
                # Bottleneck(inp, oup, stride=stride),
                # ShuffleNetBlock(inp, oup, mid_channels=mid_channels, ksize=3, stride=stride),
                # ShuffleNetBlock(inp, oup, mid_channels=mid_channels, ksize=5, stride=stride),
                # ShuffleNetBlock(inp, oup, mid_channels=mid_channels, ksize=7, stride=stride),
                # ShuffleXceptionBlock(inp, oup, mid_channels=mid_channels, stride=stride)
            ])
            result.append(choice_block)

            # find the corresponding flops
            # flop_key = (inp, oup, mid_channels, self._feature_map_size, self._feature_map_size, stride)
            # flop_key = (64, 64, 38, 56, 56, 1)
            # self._parsed_flops[choice_block.key] = [
            #     self._op_flops_dict["{}_stride_{}".format(k, stride)][flop_key] for k in self.block_keys
            # ]
            # self._parsed_flops[choice_block.key] = [int(k+1) for k in self.block_keys ]

            if stride == 2:
                self._feature_map_size //= 2
        return result

    def _make_blocks_detail(self, blocks, in_channels, channels):
        # result = []
        for i in range(blocks):
            stride = 2 if i == 0 else 1
            inp = in_channels if i == 0 else channels
            oup = channels
            choice_block = ConvBnReluPool_detail(inp, oup, stride=stride, k=3)
            # result.append(choice_block)

            if stride == 2:
                self._feature_map_size //= 2
        return choice_block

    def forward(self, x):
        bs = x.size(0)
        x = self.prep(x)
        # x = self.pool(x)
        # x = self.features(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.globalpool(x)
        x = x.contiguous().view(bs, -1)
        x = self.classifier(x)
        return x

    def get_candidate_flops(self, candidate):
        """
        conv1_flops = self._op_flops_dict["conv1"][(3, self._first_conv_channels,
                                                    self._input_size, self._input_size, 2)]
        # Should use `last_conv_channels` here, but megvii insists that it's `n_classes`. Keeping it.

        # https://github.com/megvii-model/SinglePathOneShot/blob/36eed6cf083497ffa9cfe7b8da25bb0b6ba5a452/src/Supernet/flops.py#L313
        if self._n_classes == 1000:
            rest_flops = self._op_flops_dict["rest_operation"][(self.stage_channels[-1], self._n_classes,
                                                                self._feature_map_size, self._feature_map_size, 1)]
        # 33136640 = (7*7*640 + 1000)*1024
        elif self._n_classes == 10:
            rest_flops = self._op_flops_dict["rest_operation"][(self.stage_channels[-1], self._n_classes * 100,
                                                                self._feature_map_size, self._feature_map_size, 1)]
            # cifar 实际会调整 但是这里搜按照计算量sample，所以按照1000类算
            # rest_flops = rest_flops / 102.4

        total_flops = conv1_flops + rest_flops

        for k, m in candidate.items():
            parsed_flops_dict = self._parsed_flops[k]
            if isinstance(m, dict):  # to be compatible with classical nas format
                total_flops += parsed_flops_dict[m["_idx"]]
            else:
                total_flops += parsed_flops_dict[torch.max(m, 0)[1]]
        return total_flops
        """
        # for k, m in candidate.items():
        #     print(k, m, torch.max(m, 0)[1])

        return 50

    def _initialize_weights(self):
        for name, m in self.named_modules():
            if isinstance(m, nn.Conv2d):
                if 'first' in name:
                    nn.init.normal_(m.weight, 0, 0.01)
                else:
                    nn.init.normal_(m.weight, 0, 1.0 / m.weight.shape[1])
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.BatchNorm2d):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0001)
                nn.init.constant_(m.running_mean, 0)



            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0001)
                nn.init.constant_(m.running_mean, 0)



            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

            # Set the layerwise scaling and annealing parameters
            if hasattr(m,'active'): # 假如active lr_ratio = (0.8 + 0.2 * [0/1/2/3/4 -> 39 ]/39 )^3
                # m.lr_ratio = (0.5 + 0.5 * float(m.layer_index) / 4)**3
                # m.lr_ratio = (0.8 + 0.2 * float(m.layer_index) / 4)**3
                # m.lr_ratio = (0.8 + 0.2 * float(m.layer_index) / 4)**2
                m.lr_ratio = (0.8 + 0.2 * float(m.layer_index) / 4)
                # m.lr_ratio = (0.8 + 0.2 * float(4 - m.layer_index) / 4)
                # m.lr_ratio = (0.5 + 0.5 * float(m.layer_index) / 4)
                # m.lr_ratio = (0.5 + 0.5 * (4 - float(m.layer_index) )/ 4)**3

                # batchsize = 512, epoch=40
                m.max_j = self.epoch * 100 * m.lr_ratio ## 50->1000, 400->125 , 512-> 100
                m.warm_j = 5 * 100 * m.lr_ratio

                # Optionally scale the learning rates to have the same total
                # distance traveled (modulo the gradients).
                m.lr = self.base_lr / m.lr_ratio

        # for name, m in self.named_modules():
        #     print(name)

    # def -> update_lr
    def step(self):
        # Loop over all modules
        for m in self.modules():
            # If a module is active:
            # if isinstance(m, nn.BatchNorm2d):  # m.weight.shape
            #     print(m.bias, m.weight)



            if hasattr(m, 'active') and m.active:
                # If we've passed this layer's freezing point, deactivate it.

                if self.j > m.max_j:
                # if self.j < m.max_j:
                    m.active = False
                    # Also make sure we remove all this layer from the optimizer
                    for i, group in enumerate(self.optim.param_groups):
                        if group['layer_index'] == m.layer_index:
                            # self.optim.param_groups.remove(group)
                            self.optim.param_groups[i]['lr'] = 1e-6




                # If not, update the LR
                elif self.j <= m.max_j and self.j > m.warm_j:
                    for i, group in enumerate(self.optim.param_groups):
                        if group['layer_index'] == m.layer_index:  # 0.05/ 0.512
                            # self.base_lr
                            # step
                            self.optim.param_groups[i]['lr'] = (self.base_lr/ 2 / m.lr_ratio) * (1 + np.cos(np.pi * (self.j - m.warm_j) / m.max_j))
                            # self.optim.param_groups[i]['lr'] = (self.base_lr/ 2 / m.lr_ratio) * (1 + np.cos(np.pi * self.j / m.max_j))
                            # self.optim.param_groups[i]['lr'] = (0.05 / m.lr_ratio) * (1 + np.cos(np.pi * self.j / m.max_j))

                            # linear
                            # self.optim.param_groups[i]['lr'] = 0.1 / m.lr_ratio * (1 - self.j / m.max_j)

                            # print(i, (0.05 / m.lr_ratio) * (1 + np.cos(np.pi * self.j / m.max_j)))
                            # plotter.plot('learning rate', '{} train'.format(m.layer_index), 'layer-wise learning rate', epoch, self.optim.param_groups[i]['lr'])  # visdom
                            # plotter.plot('learning rate', '{} train'.format(m.layer_index), 'layer-wise learning rate', iteration, self.optim.param_groups[i]['lr'])  # visdom
                else:
                    for i, group in enumerate(self.optim.param_groups):
                        if group['layer_index'] == m.layer_index:  # 0.05/ 0.512
                            self.optim.param_groups[i]['lr'] = self.base_lr / m.lr_ratio * (self.j / m.warm_j)




                            # print("{}, {:.3f}, {:.3f}, {:.3f}".format(m.layer_index, m.lr, m.lr_ratio, m.max_j))


            # if isinstance(m, nn.Conv2d): # m.weight.shape
            #     print("{}, {:.3f}, {}, {:.3f}, {:.3f}".format(m.layer_index, m.lr, m.weight.shape, m.lr_ratio, m.max_j))
            # elif isinstance(m, nn.BatchNorm2d): # m.weight.shape
            #     print("{}, {:.3f}, {}, {:.3f}, {:.3f}".format(m.layer_index, m.lr, m.weight.shape, m.lr_ratio, m.max_j))
            # elif isinstance(m, nn.Linear): # m.bias.shape
            #     print("{}, {:.3f}, {}, {:.3f}, {:.3f}".format(m.layer_index, m.lr, m.bias.shape, m.lr_ratio, m.max_j))
            # print(m.lr, m.weight.shape, m.lr_ratio, m.max_j)

            # if isinstance(m, nn.BatchNorm2d):
            #     if m.weight is not None:
            #         nn.init.constant_(m.weight, 1)
            #     if m.bias is not None:
            #         nn.init.constant_(m.bias, 0.0001)
            #     nn.init.constant_(m.running_mean, 0)



        # Update the iteration counter
        self.j += 1
        # print(self.j)

    def record_lr(self):
        temp_lr = {'prep': None, 'layer1': None, 'layer2': None, 'layer3': None, 'classifier': None}

        # for i, group in enumerate(self.optim.param_groups):
        #     print(i, group['layer_index'], group['lr'])

        # for name, m in self.named_modules():
        #     if hasattr(m, 'active') and m.active:
        #         print('active:', name, m.lr)
        #     else:
        #         if hasattr(m, 'lr'):
        #             print('XXtive:', name, m.lr)
        #         else:
        #             print('NO m.lr')

        # for m in self.modules():
        for name, m in self.named_modules():
            if hasattr(m, 'active') and m.active:
                for i, group in enumerate(self.optim.param_groups):
                    if group['layer_index'] == m.layer_index:  # 0.05/ 0.512
                        temp_lr[name] = self.optim.param_groups[i]['lr']
                        # print(self.optim.param_groups[i]['lr'])

        # for name, m in self.named_modules():
        #     if hasattr(m, 'active'):
                # print(name, m, m.lr_ratio, m.max_j, m.lr)
                # print("{},\t\t lr_ratio: {:.3f},\t max_iteration: {:.1f},\t Initial learning rate: {:.3f}.".format(name, m.lr_ratio, m.max_j, m.lr))
                # temp_lr.append(m.lr)
                # temp_lr[name] = m.lr


        return temp_lr




class ShuffleNetV2OneShot(nn.Module):
    block_keys = [
        'shufflenet_3x3',
        'shufflenet_5x5',
        'shufflenet_7x7',
        'xception_3x3',
    ]

    def __init__(self,
                 input_size=224,
                 first_conv_channels=16,
                 last_conv_channels=512,
                 n_classes=1000,
                 op_flops_path="/home/ubuntu/0_datasets/op_flops_dict.pkl"):
        super().__init__()

        assert input_size % 32 == 0
        with open(os.path.join(os.path.dirname(__file__), op_flops_path), "rb") as fp:
            self._op_flops_dict = pickle.load(fp)

        self.stage_blocks = [4, 4, 8, 4]
        self.stage_channels = [64, 160, 320, 640]
        # self.stage_channels = [32, 80, 160, 320]

        self._parsed_flops = dict()
        self._input_size = input_size
        self._feature_map_size = input_size
        self._first_conv_channels = first_conv_channels
        self._last_conv_channels = last_conv_channels
        self._n_classes = n_classes

        # building first layer
        self.first_conv = nn.Sequential(
            nn.Conv2d(3, first_conv_channels, 3, 2, 1, bias=False),
            nn.BatchNorm2d(first_conv_channels, affine=False),
            nn.ReLU(inplace=True),
        )
        self._feature_map_size //= 2

        p_channels = first_conv_channels
        features = []

        for num_blocks, channels in zip(self.stage_blocks, self.stage_channels):
            features.extend(self._make_blocks(num_blocks, p_channels, channels))
            p_channels = channels

        self.features = nn.Sequential(*features)

        self.conv_last = nn.Sequential(
            nn.Conv2d(p_channels, last_conv_channels, 1, 1, 0, bias=False),
            nn.BatchNorm2d(last_conv_channels, affine=False),
            nn.ReLU(inplace=True),
        )

        self.globalpool = nn.AvgPool2d(self._feature_map_size)
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Sequential(
            nn.Linear(last_conv_channels, n_classes, bias=False),
        )

        self._initialize_weights()

    def _make_blocks(self, blocks, in_channels, channels):
        result = []
        for i in range(blocks):
            stride = 2 if i == 0 else 1

            inp = in_channels if i == 0 else channels
            oup = channels

            base_mid_channels = channels // 2
            mid_channels = int(base_mid_channels)  # prepare for scale
            choice_block = LayerChoice([
                ShuffleNetBlock(inp, oup, mid_channels=mid_channels, ksize=3, stride=stride),
                ShuffleNetBlock(inp, oup, mid_channels=mid_channels, ksize=5, stride=stride),
                ShuffleNetBlock(inp, oup, mid_channels=mid_channels, ksize=7, stride=stride),
                ShuffleXceptionBlock(inp, oup, mid_channels=mid_channels, stride=stride)
            ])
            result.append(choice_block)

            # find the corresponding flops
            flop_key = (inp, oup, mid_channels, self._feature_map_size, self._feature_map_size, stride)
            self._parsed_flops[choice_block.key] = [
                self._op_flops_dict["{}_stride_{}".format(k, stride)][flop_key] for k in self.block_keys
            ]

            if stride == 2:
                self._feature_map_size //= 2
        return result

    def forward(self, x):
        bs = x.size(0)
        x = self.first_conv(x)
        x = self.features(x)
        x = self.conv_last(x)
        x = self.globalpool(x)

        x = self.dropout(x)
        x = x.contiguous().view(bs, -1)
        x = self.classifier(x)
        return x

    def get_candidate_flops(self, candidate):
        conv1_flops = self._op_flops_dict["conv1"][(3, self._first_conv_channels,
                                                    self._input_size, self._input_size, 2)]
        # Should use `last_conv_channels` here, but megvii insists that it's `n_classes`. Keeping it.

        # https://github.com/megvii-model/SinglePathOneShot/blob/36eed6cf083497ffa9cfe7b8da25bb0b6ba5a452/src/Supernet/flops.py#L313
        if self._n_classes == 1000:
            rest_flops = self._op_flops_dict["rest_operation"][(self.stage_channels[-1], self._n_classes,
                                                                self._feature_map_size, self._feature_map_size, 1)]
        # 33136640 = (7*7*640 + 1000)*1024
        elif self._n_classes == 10:
            rest_flops = self._op_flops_dict["rest_operation"][(self.stage_channels[-1], self._n_classes * 100,
                                                                self._feature_map_size, self._feature_map_size, 1)]
            # cifar 实际会调整 但是这里搜按照计算量sample，所以按照1000类算
            # rest_flops = rest_flops / 102.4

        total_flops = conv1_flops + rest_flops
        for k, m in candidate.items():
            parsed_flops_dict = self._parsed_flops[k]
            if isinstance(m, dict):  # to be compatible with classical nas format
                total_flops += parsed_flops_dict[m["_idx"]]
            else:
                total_flops += parsed_flops_dict[torch.max(m, 0)[1]]
        return total_flops

    def _initialize_weights(self):
        for name, m in self.named_modules():
            if isinstance(m, nn.Conv2d):
                if 'first' in name:
                    nn.init.normal_(m.weight, 0, 0.01)
                else:
                    nn.init.normal_(m.weight, 0, 1.0 / m.weight.shape[1])
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0001)
                nn.init.constant_(m.running_mean, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0001)
                nn.init.constant_(m.running_mean, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

def load_and_parse_state_dict(filepath="./data/checkpoint-150000.pth.tar"):
    checkpoint = torch.load(filepath, map_location=torch.device("cpu"))
    result = dict()
    # aaa = []
    # for key, values in checkpoint.items():
    #     aaa.append(key)
        # print(key)
    # print(aaa)

    # for k, v in checkpoint["state_dict"].items():
    for k, v in checkpoint.items():
        if k.startswith("module."):
            k = k[len("module."):]
        # k = re.sub(r"^(features.\d+).(\d+)", "\\1.choices.\\2", k)
        result[k] = v
    return result
