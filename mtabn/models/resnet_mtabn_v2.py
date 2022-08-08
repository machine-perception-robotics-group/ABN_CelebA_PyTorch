#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""resnet_mtabn_v2.py

ABN with Multitask Learning.
This network model is originally developed by Masahiro Mitsuhara,
which is different from the original implementation described in our CVPR paper.
"""


import copy
import math
import torch
import torch.nn as nn

import torch.utils.model_zoo as model_zoo
try:
    from torch.hub import load_state_dict_from_url
except ImportError:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url


model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class MtABNResNetV2(nn.Module):

    def __init__(self, block, layers, num_classes=40, residual_attention=True):
        self.inplanes = 64
        super().__init__()

        self.num_classes = num_classes
        self.residual_attention = residual_attention

        # feature extractor -----------
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0], stride=1, down_size=True)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, down_size=True)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, down_size=True)

        # perception branch -----------
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, down_size=True)
        _output_per = [nn.Linear(512*block.expansion, 2) for _ in range(num_classes)]
        self.output_per = nn.ModuleList(_output_per)

        # attention branch ------------
        self.inplanes = int(self.inplanes / 2)
        self.layer4_att = self._make_layer(block, 512, layers[3], stride=1, down_size=False)
        self.att_conv = nn.Conv2d(512 * block.expansion, num_classes, kernel_size=1, padding=0, bias=False)
        self.depth_conv = nn.Conv2d(in_channels=num_classes, out_channels=num_classes, kernel_size=1, stride=1, padding=0, groups=num_classes)

        # pooling and activation ------
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)

        # initialize weights ----------
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # feature extractor -----------
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        # attention branch ------------
        x_att = self.layer4_att(x)
        x_att = self.att_conv(x_att)
        x_att = self.depth_conv(x_att)

        # attention map [batch, num_classes, 14, 14]
        attention_map = x_att

        # output of att. branch
        out_att = self.global_avg_pool(x_att)
        out_att = out_att.view(out_att.size(0), -1)

        # perception branch -----------
        out_per = []
        for i in range(self.num_classes):

            # task-wise att. mechanism
            _task_att = attention_map[:, i:i+1, :, :]
            if self.residual_attention:
                x_att_mechanism = x * torch.add(_task_att, 1)
            else:
                x_att_mechanism = x * _task_att

            # perception branch for each task
            x_per = self.layer4(x_att_mechanism)
            x_per = self.global_avg_pool(x_per)
            x_per = x_per.view(x_per.size(0), -1)
            x_per = self.output_per[i](x_per)
            out_per.append(x_per.squeeze())

        out_per = torch.stack(out_per).permute(1, 2, 0)

        return out_per, out_att, attention_map


    def _make_layer(self, block, planes, blocks, stride=1, down_size=True):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))

        if down_size:
            self.inplanes = planes * block.expansion
            for i in range(1, blocks):
                layers.append(block(self.inplanes, planes))

            return nn.Sequential(*layers)
        else:
            inplanes = planes * block.expansion
            for i in range(1, blocks):
                layers.append(block(inplanes, planes))

            return nn.Sequential(*layers)


def mtabn_v2_resnet18(pretrained=False, **kwargs):
    """Constructs an Attention Branch Network with 3D ResNet-18 model."""
    model = MtABNResNetV2(BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        model.load_state_dict(load_state_dict_from_url(model_urls['resnet18'], progress=True), strict=False)
    return model


def mtabn_v2_resnet34(pretrained=False, **kwargs):
    """Constructs an Attention Branch Network with 3D ResNet-34 model."""
    model = MtABNResNetV2(BasicBlock, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(load_state_dict_from_url(model_urls['resnet34'], progress=True), strict=False)
    return model


def mtabn_v2_resnet50(pretrained=False, **kwargs):
    """Constructs an Attention Branch Network with 3D ResNet-50 model."""
    model = MtABNResNetV2(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(load_state_dict_from_url(model_urls['resnet50'], progress=True), strict=False)
    return model


def mtabn_v2_resnet101(pretrained=False, **kwargs):
    """Constructs an Attention Branch Network with 3D ResNet-101 model."""
    model = MtABNResNetV2(Bottleneck, [3, 4, 23, 3], **kwargs)
    if pretrained:
        model.load_state_dict(load_state_dict_from_url(model_urls['resnet101'], progress=True), strict=False)
    return model


def mtabn_v2_resnet152(pretrained=False, **kwargs):
    """Constructs an Attention Branch Network with 3D ResNet-152 model."""
    model = MtABNResNetV2(Bottleneck, [3, 8, 36, 3], **kwargs)
    if pretrained:
        model.load_state_dict(load_state_dict_from_url(model_urls['resnet152'], progress=True), strict=False)
    return model


if __name__ == '__main__':

    ### CelebA setting
    batch_size = 5
    n_class = 40
    image_size = 224

    print("debug ...")
    print("    batch size:", batch_size)
    print("    number of classes:", n_class)
    print("    image size:", image_size)

    input = torch.zeros([batch_size, 3, 224, 224], dtype=torch.float32)

    model18 = mtabn_v2_resnet18(pretrained=True, num_classes=n_class, residual_attention=True)
    output18 = model18(input)
    print("output of resnet-18", output18[0].size(), output18[1].size(), output18[2].size())

    model34 = mtabn_v2_resnet34(pretrained=True, num_classes=n_class, residual_attention=True)
    output34 = model34(input)
    print("output of resnet-34", output34[0].size(), output34[1].size(), output34[2].size())

    model50 = mtabn_v2_resnet50(pretrained=True, num_classes=n_class, residual_attention=True)
    output50 = model50(input)
    print("output of resnet-50", output50[0].size(), output50[1].size(), output50[2].size())

    model101 = mtabn_v2_resnet101(pretrained=True, num_classes=n_class, residual_attention=True)
    output101 = model101(input)
    print("output of resnet-101", output101[0].size(), output101[1].size(), output101[2].size())

    model152 = mtabn_v2_resnet152(pretrained=True, num_classes=n_class, residual_attention=True)
    output152 = model152(input)
    print("output of resnet-152", output152[0].size(), output152[1].size(), output152[2].size())
