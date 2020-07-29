# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
import pickle
import re

import torch
import torch.nn as nn

from nni.nas.pytorch.mutables import LayerChoice, InputChoice

from blocks import ShuffleNetBlock, ShuffleXceptionBlock, BasicBlock, Bottleneck, ConvBnRelu, ConvBnReluPool, Mul, DynamicBasicBlock, ConvBnReluPool_detail

class CIFAR100_OneShot(nn.Module):
    def __init__(self,
                 input_size=32,
                 n_classes=1000,
                 channels = None,
                 weight=0.125,
                 op_flops_path="/home/ubuntu/0_datasets/op_flops_dict.pkl"):
        super().__init__()

        assert input_size % 32 == 0
        with open(os.path.join(os.path.dirname(__file__), op_flops_path), "rb") as fp:
            self._op_flops_dict = pickle.load(fp)

        # layer
        self.stage_blocks = [3, 1, 3]
        # self.stage_channels = [112, 256, 384]
        # self.stage_channels = [128, 256, 512]
        self.stage_channels = channels or [64, 128, 256, 512]
        first_conv_channels = self.stage_channels[0]
        layers_channels = self.stage_channels[1:]

        self._parsed_flops = dict()
        self._input_size = input_size
        self._feature_map_size = input_size
        self._first_conv_channels = first_conv_channels
        # self._last_conv_channels = last_conv_channels
        self._n_classes = n_classes

        # building first layer
        self.prep = nn.Sequential(
            nn.Conv2d(3, first_conv_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(first_conv_channels, affine=False),
            nn.ReLU(inplace=True),
        )

        # self.pool = nn.MaxPool2d(2)
        # self._feature_map_size //= 2

        # build feature layer1:32->16, layer2:16->8, layer3:8->4
        # p_channels = first_conv_channels
        # features = []
        # for num_blocks, channels in zip(self.stage_blocks, self.stage_channels):
        #     features.extend(self._make_blocks(num_blocks, p_channels, channels))
        #     p_channels = channels
        # self.features = nn.Sequential(*features)


        # struct 1
        p_channels = first_conv_channels
        for num_blocks, channels in zip(self.stage_blocks, layers_channels):
            if channels == 84:
                layer1 = self._make_blocks(num_blocks, p_channels, channels)
                self.layer1 = nn.Sequential(*layer1)
            
            elif channels == 256:
                layer2 = self._make_blocks(num_blocks, p_channels, channels)
                self.layer2 = nn.Sequential(*layer2)

            elif channels == 384:
                layer3 = self._make_blocks(num_blocks, p_channels, channels)
                self.layer3 = nn.Sequential(*layer3)

            p_channels = channels
        """
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
       


        # struct 2
        self.layer1 = DynamicBasicBlock(first_conv_channels, layers_channels[0])
        self.layer2 = DynamicBasicBlock(layers_channels[0], layers_channels[1])
        self.layer3 = DynamicBasicBlock(layers_channels[1], layers_channels[2])
        """



        self.globalpool = nn.MaxPool2d(4)
        self.classifier = nn.Sequential(
            nn.Linear(layers_channels[-1], n_classes, bias=False),
            Mul(weight)
        )
        self._initialize_weights()

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
            choice_block = mutables.LayerChoice([
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
