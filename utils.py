import os
import yaml
import random
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

import torch
from torch import nn
from torchvision.models import (vit_b_16, vit_l_16, vit_b_32, vit_l_32, \
    efficientnet_b0, efficientnet_b1, efficientnet_b2, \
    resnet18, resnet34, resnet50)

def create_experiment():
    os.makedirs('experiments', exist_ok=True)
    i = 0
    while True:
        if not os.path.exists(f'experiments/expr_{i}'):
            os.makedirs(f'experiments/expr_{i}',exist_ok=False)
            return f'experiments/expr_{i}'
        i += 1

def read_default_configs():
    with open('configs/default_configs.yaml','r') as f:
        configs = yaml.safe_load(f)
    return configs

def get_field_cfg(cfg, field):
    field_cfg = {}
    for key, value in cfg.items():
        if key.startswith(cfg[field]) and key!= cfg[field]:
            new_key = '_'.join(key.split('_')[1:])
            field_cfg[new_key] = value
    return field_cfg
class cfg:
    def __init__(self, d=None):
        if d is not None:
            for key, value in d.items():
                setattr(self, key, value)

def draw_log(history, expr_dir):
    plt.style.use("ggplot")
    plt.figure()
    plt.plot(history["train_loss"], label="train_loss")
    plt.plot(history["val_loss"], label="val_loss")
    plt.title("Training Loss on Dataset")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss")
    plt.legend(loc="lower left")
    plt.savefig(Path(expr_dir)/'loss.jpg')

    plt.figure()
    plt.plot(history["train_acc"], label="train_acc")
    plt.plot(history["val_acc"], label="val_acc")
    plt.title("Training Accuracy on Dataset")
    plt.xlabel("Epoch #")
    plt.ylabel("Accuracy")
    plt.legend(loc="lower left")
    plt.savefig(Path(expr_dir)/'acc.jpg')

def get_model(model_name:str,
              num_class:int,
              pretrained=True,
              freeze_backbone: bool = False):
    models = {
        'vit_b_16': vit_b_16,
        'vit_l_16': vit_l_16,
        'vit_b_32': vit_b_32,
        'vit_l_32': vit_l_32,
        'efficientnet_b0': efficientnet_b0,
        'efficientnet_b1': efficientnet_b1,
        'efficientnet_b2': efficientnet_b2,
        'resnet18': resnet18,
        'resnet34': resnet34,
        'resnet50': resnet50
    }
    if model_name not in models.keys():
        print('Invalid model name',model_name)
        exit()
    if pretrained:
        model = models[model_name](weights='DEFAULT')
    else:
        model = models[model_name]()
    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False
        
    if model_name.startswith('vit'):
        model.heads.head = nn.Linear(model.heads.head.in_features, num_class)
    elif model_name.startswith('efficientnet'):
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_class)
    else: # resnet
        model.fc = nn.Linear(model.fc.in_features, num_class)
    return model
    
def seed_torch(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True