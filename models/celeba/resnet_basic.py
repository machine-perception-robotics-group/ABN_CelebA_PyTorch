import torch
import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo


__all__ = ['ResNet_basic', 'resnet18', 'resnet34']


model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
}


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
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

class make_mtl_block(nn.Module):

    def __init__(self, block, layers, num_tasks):
        self.num_tasks = num_tasks
        super(make_mtl_block, self).__init__()
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, down_size=True)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.sigmoid = nn.Sigmoid()

        output = [nn.Linear(512 * block.expansion, 1) for _ in range(self.num_tasks)]
        # att_conv = [nn.Conv2d(512 * block.expansion, 1, kernel_size=1, padding=0, bias=True) for _ in range(num_tasks)]
        # att_bn = [nn.BatchNorm2d(1) for _ in range(num_tasks)]
        self.output = nn.ModuleList(output)
        # self.att_conv = nn.ModuleList(att_conv)
        # self.att_bn = nn.ModuleList(att_bn)

    def _make_layer(self, block, planes, blocks, stride=1, down_size=True):
        downsample = None
        self.inplanes = 256
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

    def forward(self, x, att_elem):
        pred = []
        attention = []
        for i in range(self.num_tasks):
            bs, cs, ys, xs = att_elem.shape
            # item_att_elem = att_elem # [:, i].view(bs, 1, ys, xs)
            item_att = att_elem[:, i].view(bs, 1, ys, xs)
            # item_att = self.att_conv[i](item_att_elem)
            # item_att = self.sigmoid(self.att_bn[i](item_att))
            # item_att = self.sigmoid(item_att)
            attention.append(item_att)

            sh = item_att * x
            sh += x
            sh = self.layer4(sh)
            sh = self.avgpool(sh)
            sh = sh.view(sh.size(0), -1)
            sh = self.sigmoid(self.output[i](sh))
            pred.append(sh)

        return pred, attention


class ResNet_basic(nn.Module):

    def __init__(self, block, layers, num_classes=40):
        self.inplanes = 64
        super(ResNet_basic, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], down_size=True)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, down_size=True)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, down_size=True)

        self.att_layer4 = self._make_layer(block, 512, layers[3], stride=1, down_size=False)
        # self.bn_att = nn.BatchNorm2d(512 * block.expansion)
        self.att_conv = nn.Conv2d(512 * block.expansion, num_classes, kernel_size=1, padding=0,
                               bias=False)
        # self.bn_att2 = nn.BatchNorm2d(num_classes)

        self.att_conv2 = nn.Conv2d(num_classes, num_classes, kernel_size=3, padding=1,
                               bias=False)
        self.att_gap = nn.MaxPool2d(14)
        self.sigmoid = nn.Sigmoid()
        self.mtl = make_mtl_block(block, layers, 40)
        self.depth_conv = nn.Conv2d(
            in_channels=40,
            out_channels=40,
            kernel_size=1,
            stride=1,
            padding=0,
            groups=40
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


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


    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        ax = self.att_layer4(x)
        ax = self.att_conv(ax)
        ax = self.depth_conv(ax)
        self.att = self.sigmoid(ax)
        bs, cs, ys, xs = ax.shape

        ax = self.att_gap(ax)
        ax = self.sigmoid(ax)
        ax = ax.view(ax.size(0), -1)

        rx, attention = self.mtl(x, self.att)

        return ax, rx, attention


def resnet18(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet_basic(BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18']))
    return model


def resnet34(pretrained=False, **kwargs):
    """Constructs a ResNet-34 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet_basic(BasicBlock, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet34']))
    return model

