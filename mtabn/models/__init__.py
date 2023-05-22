#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import torch.nn as nn
from .resnet_mtabn_v1 import mtabn_v1_resnet18, mtabn_v1_resnet34, mtabn_v1_resnet50, mtabn_v1_resnet101, mtabn_v1_resnet152
from .resnet_mtabn_v2 import mtabn_v2_resnet18, mtabn_v2_resnet34, mtabn_v2_resnet50, mtabn_v2_resnet101, mtabn_v2_resnet152
from torchvision.models import resnet18, resnet34, resnet50, resnet101, resnet152


MODEL_NAMES_RESNET = ('resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152')
MODEL_NAMES_MTABN_V1_RESNET = ('mtabn_v1_resnet18', 'mtabn_v1_resnet34', 'mtabn_v1_resnet50', 'mtabn_v1_resnet101', 'mtabn_v1_resnet152')
MODEL_NAMES_MTABN_V2_RESNET = ('mtabn_v2_resnet18', 'mtabn_v2_resnet34', 'mtabn_v2_resnet50', 'mtabn_v2_resnet101', 'mtabn_v2_resnet152')


MODEL_NAMES = MODEL_NAMES_RESNET + MODEL_NAMES_MTABN_V1_RESNET + MODEL_NAMES_MTABN_V2_RESNET


def load_model(model_name, num_classes, residual_attention, pretrained=True):
    assert model_name in MODEL_NAMES, "ERROR: model name %s does not exists." % model_name
    print("build network model: %s" % model_name)
    print("    number of classes:", num_classes)
    print("    use residual attention:", residual_attention, "(ignored for resnet)")
    print("    use pre-trained model:", pretrained)
    print("")

    ### ResNet
    if model_name == 'resnet18':
        model = resnet18(pretrained=pretrained)
        _in_features = model.fc.in_features
        model.fc = nn.Linear(in_features=_in_features, out_features=num_classes, bias=True)
    elif model_name == 'resnet34':
        model = resnet34(pretrained=pretrained)
        _in_features = model.fc.in_features
        model.fc = nn.Linear(in_features=_in_features, out_features=num_classes, bias=True)
    elif model_name == 'resnet50':
        model = resnet50(pretrained=pretrained)
        _in_features = model.fc.in_features
        model.fc = nn.Linear(in_features=_in_features, out_features=num_classes, bias=True)
    elif model_name == 'resnet101':
        model = resnet101(pretrained=pretrained)
        _in_features = model.fc.in_features
        model.fc = nn.Linear(in_features=_in_features, out_features=num_classes, bias=True)
    elif model_name == 'resnet152':
        model = resnet152(pretrained=pretrained)
        _in_features = model.fc.in_features
        model.fc = nn.Linear(in_features=_in_features, out_features=num_classes, bias=True)
    
    ### Multitask ABN V1
    elif model_name == 'mtabn_v1_resnet18':
        model = mtabn_v1_resnet18(pretrained=pretrained, num_classes=num_classes, residual_attention=residual_attention)
    elif model_name == 'mtabn_v1_resnet34':
        model = mtabn_v1_resnet34(pretrained=pretrained, num_classes=num_classes, residual_attention=residual_attention)
    elif model_name == 'mtabn_v1_resnet50':
        model = mtabn_v1_resnet50(pretrained=pretrained, num_classes=num_classes, residual_attention=residual_attention)
    elif model_name == 'mtabn_v1_resnet101':
        model = mtabn_v1_resnet101(pretrained=pretrained, num_classes=num_classes, residual_attention=residual_attention)
    elif model_name == 'mtabn_v1_resnet152':
        model = mtabn_v1_resnet152(pretrained=pretrained, num_classes=num_classes, residual_attention=residual_attention)
    
    ### Multitask ABN V2
    elif model_name == 'mtabn_v2_resnet18':
        model = mtabn_v2_resnet18(pretrained=pretrained, num_classes=num_classes, residual_attention=residual_attention)
    elif model_name == 'mtabn_v2_resnet34':
        model = mtabn_v2_resnet34(pretrained=pretrained, num_classes=num_classes, residual_attention=residual_attention)
    elif model_name == 'mtabn_v2_resnet50':
        model = mtabn_v2_resnet50(pretrained=pretrained, num_classes=num_classes, residual_attention=residual_attention)
    elif model_name == 'mtabn_v2_resnet101':
        model = mtabn_v2_resnet101(pretrained=pretrained, num_classes=num_classes, residual_attention=residual_attention)
    elif model_name == 'mtabn_v2_resnet152':
        model = mtabn_v2_resnet152(pretrained=pretrained, num_classes=num_classes, residual_attention=residual_attention)

    return model
