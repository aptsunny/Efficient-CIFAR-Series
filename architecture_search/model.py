# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import torch
import torch.nn as nn

import op
from nni.nas.pytorch import mutables

import torch.nn.functional as F

from nni.nas.pytorch.mutables import LayerChoice, InputChoice

class CNN_naive(nn.Module):
    def __init__(self):
        super(CNN_naive, self).__init__()
        self.conv1 = LayerChoice([nn.Conv2d(3, 6, 3, padding=1), nn.Conv2d(3, 6, 5, padding=2)])
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = LayerChoice([nn.Conv2d(6, 16, 3, padding=1), nn.Conv2d(6, 16, 5, padding=2)])
        self.conv3 = nn.Conv2d(16, 16, 1)

        self.skipconnect = InputChoice(n_candidates=1)
        self.bn = nn.BatchNorm2d(16)

        self.gap = nn.AdaptiveAvgPool2d(4)
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        bs = x.size(0)

        x = self.pool(F.relu(self.conv1(x)))
        x0 = F.relu(self.conv2(x))
        x1 = F.relu(self.conv3(x0))

        x0 = self.skipconnect([x0])
        if x0 is not None:
            x1 += x0
        x = self.pool(self.bn(x1))

        x = self.gap(x).view(bs, -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class AuxiliaryHead(nn.Module):
    """ Auxiliary head in 2/3 place of network to let the gradient flow well """

    def __init__(self, input_size, C, n_classes):
        """ assuming input size 7x7 or 8x8 """
        assert input_size in [7, 8]
        super().__init__()
        self.net = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.AvgPool2d(5, stride=input_size - 5, padding=0, count_include_pad=False),  # 2x2 out
            nn.Conv2d(C, 128, kernel_size=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 768, kernel_size=2, bias=False),  # 1x1 out
            nn.BatchNorm2d(768),
            nn.ReLU(inplace=True)
        )
        self.linear = nn.Linear(768, n_classes)

    def forward(self, x):
        out = self.net(x)
        out = out.view(out.size(0), -1)  # flatten
        logits = self.linear(out)
        return logits

class Node(nn.Module):
    def __init__(self, node_id, num_prev_nodes, channels, num_downsample_connect):
        super().__init__()
        self.ops = nn.ModuleList()
        choice_keys = []
        for i in range(num_prev_nodes):
            stride = 2 if i < num_downsample_connect else 1
            choice_keys.append("{}_p{}".format(node_id, i))
            self.ops.append(
                mutables.LayerChoice( # 多种OP操作
                    [
                        op.PoolBN('max', channels, 3, stride, 1, affine=False),
                        op.PoolBN('avg', channels, 3, stride, 1, affine=False),
                        nn.Identity() if stride == 1 else op.FactorizedReduce(channels, channels, affine=False),
                        op.SepConv(channels, channels, 3, stride, 1, affine=False),
                        op.SepConv(channels, channels, 5, stride, 2, affine=False),
                        op.DilConv(channels, channels, 3, stride, 2, 2, affine=False),
                        op.DilConv(channels, channels, 5, stride, 4, 2, affine=False)
                    ],
                    key=choice_keys[-1]))
        self.drop_path = op.DropPath()
        self.input_switch = mutables.InputChoice(choose_from=choice_keys, n_chosen=2, key="{}_switch".format(node_id))

    def forward(self, prev_nodes):
        assert len(self.ops) == len(prev_nodes)
        out = [op(node) for op, node in zip(self.ops, prev_nodes)]
        out = [self.drop_path(o) if o is not None else None for o in out]
        return self.input_switch(out)

class Cell(nn.Module):

    def __init__(self, n_nodes, channels_pp, channels_p, channels, reduction_p, reduction):
        super().__init__()
        self.reduction = reduction
        self.n_nodes = n_nodes

        # If previous cell is reduction cell, current input size does not match with
        # output size of cell[k-2]. So the output[k-2] should be reduced by preprocessing.
        if reduction_p: # 前两层固定不参与搜索，然和根据之前所连cell类型进行设定
            self.preproc0 = op.FactorizedReduce(channels_pp, channels, affine=False)
        else:
            self.preproc0 = op.StdConv(channels_pp, channels, 1, 1, 0, affine=False)
        self.preproc1 = op.StdConv(channels_p, channels, 1, 1, 0, affine=False)

        # generate dag
        self.mutable_ops = nn.ModuleList()
        for depth in range(2, self.n_nodes + 2):
            self.mutable_ops.append(Node("{}_n{}".format("reduce" if reduction else "normal", depth),
                                         depth, channels, 2 if reduction else 0))

    def forward(self, s0, s1):
        # s0, s1 are the outputs of previous previous cell and previous cell, respectively.
        tensors = [self.preproc0(s0), self.preproc1(s1)]
        for node in self.mutable_ops:
            cur_tensor = node(tensors)
            tensors.append(cur_tensor)

        output = torch.cat(tensors[2:], dim=1)
        return output

class CNN(nn.Module):
    # defined
    def __init__(self, input_size, in_channels, channels, n_classes, n_layers, n_nodes=4,
                 stem_multiplier=3, auxiliary=False):
        super().__init__()
        self.in_channels = in_channels
        self.channels = channels
        self.n_classes = n_classes
        self.n_layers = n_layers
        self.aux_pos = 2 * n_layers // 3 if auxiliary else -1

        c_cur = stem_multiplier * self.channels # (3=stem_multiplier)*16=48
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, c_cur, 3, 1, 1, bias=False),
            nn.BatchNorm2d(c_cur)
        )

        # for the first cell, stem is used for both s0 and s1
        # [!] channels_pp and channels_p is output channel size, but c_cur is input channel size.
        channels_pp, channels_p, c_cur = c_cur, c_cur, channels
        # 48,48,16
        self.cells = nn.ModuleList()
        reduction_p, reduction = False, False
        for i in range(n_layers):
            reduction_p, reduction = reduction, False
            # Reduce featuremap size and double channels in 1/3 and 2/3 layer.
            if i in [n_layers // 3, 2 * n_layers // 3]: # 只在特定的层使用缩减单元，缩减单元增大通道数，减小尺寸
                c_cur *= 2 # 缩减一次通道数乘2
                reduction = True

            cell = Cell(n_nodes, channels_pp, channels_p, c_cur, reduction_p, reduction)
            self.cells.append(cell)
            c_cur_out = c_cur * n_nodes
            channels_pp, channels_p = channels_p, c_cur_out

            if i == self.aux_pos:
                self.aux_head = AuxiliaryHead(input_size // 4, channels_p, n_classes)
        # layers*8 finished
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.linear = nn.Linear(channels_p, n_classes)

    def forward(self, x):
        s0 = s1 = self.stem(x)

        aux_logits = None
        for i, cell in enumerate(self.cells):
            s0, s1 = s1, cell(s0, s1)
            if i == self.aux_pos and self.training:
                aux_logits = self.aux_head(s1)

        out = self.gap(s1)
        out = out.view(out.size(0), -1)  # flatten
        logits = self.linear(out)

        if aux_logits is not None:
            return logits, aux_logits
        return logits

    def drop_path_prob(self, p):
        for module in self.modules():
            if isinstance(module, op.DropPath):
                module.p = p
