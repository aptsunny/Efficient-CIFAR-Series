# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import torch
import torch.nn as nn
import torch.nn.functional as F
from nni.nas.pytorch.mutables import LayerChoice, InputChoice

class Mul(nn.Module):
    def __init__(self, weight):
        super().__init__()
        self.weight = weight
    def __call__(self, x):
        return x*self.weight

class ConvBnReluPool_detail(nn.Module):
    def __init__(self, inplanes, outplanes, stride, k):
        super(ConvBnReluPool_detail, self).__init__()

        base_conv = LayerChoice([
            nn.Conv2d(inplanes, outplanes, kernel_size=3, stride=1, padding=1, bias=False),
            nn.Conv2d(inplanes, outplanes, kernel_size=5, stride=1, padding=2, bias=False),

        ])
        if stride == 2:
            self.op = nn.Sequential(
                base_conv,
                nn.BatchNorm2d(outplanes),
                nn.ReLU(),
                nn.MaxPool2d(2)
            )
        else:
            self.op = nn.Sequential(
                base_conv,
                nn.BatchNorm2d(outplanes),
                nn.ReLU()
            )
    def forward(self, x):
        return self.op(x)

class ConvBnReluPool(nn.Module):
    def __init__(self, inplanes, outplanes, stride, k):
        super(ConvBnReluPool, self).__init__()
        if stride == 2:
            self.op = nn.Sequential(
                nn.Conv2d(inplanes, outplanes, kernel_size=k, stride=1, padding=k // 2, bias=False),
                nn.BatchNorm2d(outplanes),
                nn.ReLU(),
                nn.MaxPool2d(2)
            )
        else:
            self.op = nn.Sequential(
                nn.Conv2d(inplanes, outplanes, kernel_size=k, stride=1, padding=k // 2, bias=False),
                nn.BatchNorm2d(outplanes),
                nn.ReLU()
            )
    def forward(self, x):
        return self.op(x)

class ConvBnRelu(nn.Module):
    def __init__(self, inplanes, outplanes, stride, k):
        super(ConvBnRelu, self).__init__()
        self.op = nn.Sequential(
            nn.Conv2d(inplanes, outplanes, kernel_size=k, stride=stride, padding=k // 2, bias=False),
            nn.BatchNorm2d(outplanes),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.op(x)

class DynamicBasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_planes, planes):
        super(DynamicBasicBlock, self).__init__()

        self.conv1 = nn.Sequential(
                                    nn.Conv2d(in_planes, planes, kernel_size=3, stride=1, padding=1, bias=False),
                                    nn.BatchNorm2d(planes),
                                    nn.ReLU(),
                                    nn.MaxPool2d(2)
        )


        self.conv2 = nn.Sequential(
            # nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False),
            LayerChoice([
                nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False),
                nn.Conv2d(planes, planes, kernel_size=5, stride=1, padding=2, bias=False)
            ], key='conv2'),
            nn.BatchNorm2d(planes),
            nn.ReLU(),
        )
        self.conv3 = nn.Sequential(
            # nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False),
            LayerChoice([
                nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False),
                nn.Conv2d(planes, planes, kernel_size=5, stride=1, padding=2, bias=False)
            ], key='conv3'),
            nn.BatchNorm2d(planes),
            nn.ReLU(),
        )

        self.conv4 = nn.Sequential(
            # nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False),
            LayerChoice([
                nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False),
                nn.Conv2d(planes, planes, kernel_size=5, stride=1, padding=2, bias=False)
            ], key='conv4'),
            nn.BatchNorm2d(planes),
            nn.ReLU(),
        )

        """
        self.shortcut = nn.Sequential(
            nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(planes),
            nn.ReLU(),
            nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(planes),
            nn.ReLU()
        )
        
        forward:
        out = self.shortcut(out)
        
        """
        self.input_switch = InputChoice(
                                        choose_from=["conv2", "conv3", "conv4"],
                                        # choose_from=[InputChoice.NO_KEY, "conv2", "conv3", "conv4"],
                                        # n_candidates=4,
                                        n_candidates=3,
                                        n_chosen=1,
                                        key='skip')

    def forward(self, x):
        # out = F.relu(self.bn1(self.conv1(x)))
        # out = self.pooling(out)

        conv2_input = self.conv1(x)  #

        conv3_input = self.conv2(conv2_input)

        conv4_input = self.conv3(conv3_input)

        conv4_output = self.conv4(conv4_input)

        zero_x = torch.zeros_like(conv2_input)

        # skip_x = self.input_switch([zero_x, conv3_input, conv4_input, conv4_output])
        skip_x = self.input_switch([conv3_input, conv4_input, conv4_output])

        # print(out.shape, skip_x.shape)
        out = torch.add(conv2_input, skip_x)
        return out

class Residual(nn.Module):
    def __init__(self, in_planes, planes, stride=1):
        super(Residual, self).__init__()
        self.shortcut = nn.Sequential(
            nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(planes),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(planes),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = x + self.shortcut(x)
        return x

class DynamicResidualBlock_fix(nn.Module):
    def __init__(self, in_planes, planes, extra=None, res=None):
        super(DynamicResidualBlock_fix, self).__init__()
        self.conv1 = ConvBnReluPool(in_planes, planes, stride=2, k=3)

        self.conv1.active = True
        self.conv1.layer_index = 0

        www = []
        if extra==0:
            pass
        else:
            for i in range(extra):
                www.append(Residual(planes, planes))
        if res==0:
            pass
        else:
            for i in range(res):
                www.append(ConvBnRelu(planes, planes, stride=1, k=3))

        # add layer_index
        index_=len(www)
        for i in range(index_):
            www[i].active = True
            www[i].layer_index = i+1 # start from 1

        self.feature = nn.Sequential(*www)

    def forward(self, x):
        x = self.conv1(x)
        out = self.feature(x)
        return out

class DynamicResidualBlock(nn.Module):
    def __init__(self, in_planes, planes, extra=None, res=None):
        super(DynamicResidualBlock, self).__init__()
        # a
        # self.conv1 = LayerChoice([ConvBnReluPool(in_planes, planes, stride=2, k=3)] ,key = 'conv1')
        # b
        self.conv1 = LayerChoice([ConvBnReluPool(in_planes, planes, stride=2, k=3),
                                  ConvBnRelu(in_planes, planes, stride=2, k=3)], key='conv1')
        self.conv1.active = True
        self.conv1.layer_index = 0

        self.conv2 = LayerChoice([ConvBnRelu(planes, planes, stride=1, k=3),
                                  Residual(planes, planes)], key='conv2')
        self.conv2.active = True
        self.conv2.layer_index = 1

        self.conv3 = LayerChoice([ConvBnRelu(planes, planes, stride=1, k=3),
                                  Residual(planes, planes)], key='conv3')
        self.conv3.active = True
        self.conv3.layer_index = 2

        self.conv4 = LayerChoice([ConvBnRelu(planes, planes, stride=1, k=3),
                                  Residual(planes, planes)], key='conv4')
        self.conv4.active = True
        self.conv4.layer_index = 3

        # self.conv2 = LayerChoice([Residual(planes, planes)], key='conv2')
        # self.conv3 = LayerChoice([Residual(planes, planes)], key='conv3')
        # self.conv4 = LayerChoice([Residual(planes, planes)], key='conv4')

        # self.input_switch = InputChoice(
        #     choose_from=["conv1", "conv2", "conv3", "conv4"],
        #     n_candidates=4,
        #     n_chosen=1,
        #     key='skip')
        #
        self.input_switch = InputChoice(
            choose_from=["conv3", "conv4"],
            n_candidates=2,
            n_chosen=1,
            key='skip')

    def forward(self, x):
        # a/b
        conv2_input = self.conv1(x)
        conv3_input = self.conv2(conv2_input)
        conv4_input = self.conv3(conv3_input)
        conv4_output = self.conv4(conv4_input)
        # out = self.input_switch([conv2_input, conv3_input, conv4_input, conv4_output])
        out = self.input_switch([conv4_input, conv4_output])
        return out

class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ShuffleNetBlock(nn.Module):
    """
    When stride = 1, the block receives input with 2 * inp channels. Otherwise inp channels.
    """

    def __init__(self, inp, oup, mid_channels, ksize, stride, sequence="pdp"):
        super().__init__()
        assert stride in [1, 2]
        assert ksize in [3, 5, 7]
        self.channels = inp // 2 if stride == 1 else inp
        self.inp = inp
        self.oup = oup
        self.mid_channels = mid_channels
        self.ksize = ksize
        self.stride = stride
        self.pad = ksize // 2
        self.oup_main = oup - self.channels
        assert self.oup_main > 0

        self.branch_main = nn.Sequential(*self._decode_point_depth_conv(sequence))

        if stride == 2:
            self.branch_proj = nn.Sequential(
                # dw
                nn.Conv2d(self.channels, self.channels, ksize, stride, self.pad,
                          groups=self.channels, bias=False),
                nn.BatchNorm2d(self.channels, affine=False),
                # pw-linear
                nn.Conv2d(self.channels, self.channels, 1, 1, 0, bias=False),
                nn.BatchNorm2d(self.channels, affine=False),
                nn.ReLU(inplace=True)
            )

    def forward(self, x):
        if self.stride == 2:
            x_proj, x = self.branch_proj(x), x
        else:
            x_proj, x = self._channel_shuffle(x)
        return torch.cat((x_proj, self.branch_main(x)), 1)

    def _decode_point_depth_conv(self, sequence):
        result = []
        first_depth = first_point = True
        pc = c = self.channels
        for i, token in enumerate(sequence):
            # compute output channels of this conv
            if i + 1 == len(sequence):
                assert token == "p", "Last conv must be point-wise conv."
                c = self.oup_main
            elif token == "p" and first_point:
                c = self.mid_channels
            if token == "d":
                # depth-wise conv
                assert pc == c, "Depth-wise conv must not change channels."
                result.append(nn.Conv2d(pc, c, self.ksize, self.stride if first_depth else 1, self.pad,
                                        groups=c, bias=False))
                result.append(nn.BatchNorm2d(c, affine=False))
                first_depth = False
            elif token == "p":
                # point-wise conv
                result.append(nn.Conv2d(pc, c, 1, 1, 0, bias=False))
                result.append(nn.BatchNorm2d(c, affine=False))
                result.append(nn.ReLU(inplace=True))
                first_point = False
            else:
                raise ValueError("Conv sequence must be d and p.")
            pc = c
        return result

    def _channel_shuffle(self, x):
        bs, num_channels, height, width = x.data.size()
        assert (num_channels % 4 == 0)
        x = x.reshape(bs * num_channels // 2, 2, height * width)
        x = x.permute(1, 0, 2)
        x = x.reshape(2, -1, num_channels // 2, height, width)
        return x[0], x[1]

class ShuffleXceptionBlock(ShuffleNetBlock):
    def __init__(self, inp, oup, mid_channels, stride):
        super().__init__(inp, oup, mid_channels, 3, stride, "dpdpdp")
