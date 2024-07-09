import os
import yaml
import random
import logging
import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.models import (vit_b_16, vit_l_16, vit_b_32, vit_l_32, \
    efficientnet_b2)

from dataset import CatDataset

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

def get_model(model_name:str, num_class:int, pretrained=True):
    models = {
        'vit_b_16': vit_b_16,
        'vit_l_16': vit_l_16,
        'vit_b_32': vit_b_32,
        'vit_l_32': vit_l_32,
        'efficientnet_b2': efficientnet_b2,
    }
    if model_name not in models.keys():
        print('Invalid model name',model_name)
        exit()
    if pretrained:
        model = models[model_name](weights='DEFAULT')
    else:
        model = models[model_name]()
    if model_name.startswith('vit'):
        model.heads = nn.Linear(768, num_class)
    elif model_name.startswith('efficientnet'):
        model.classifier[1] = torch.nn.Linear(1408, num_class)
    return model
    
def seed_torch(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    args = parser.parse_args()
    # Experiment setup
    expr_dir = create_experiment()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(Path(expr_dir)/"experiment.log"),
            logging.StreamHandler()
        ]
    )
    # Configs
    args = vars(args)
    args = {k: v for k, v in args.items() if v is not None}
    default_configs = read_default_configs()
    default_configs.update(args)
    optim_cfg = get_field_cfg(default_configs, 'optimizer')
    configs = cfg(default_configs)
    # Seed and device
    seed_torch(configs.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Currently using {device}')
    # Data
    # train_dataloader, val_dataloader = data_loader(configs.data_dir, \
    #                                                 configs.batch_size,
    #                                                 configs.seed,
    #                                                 )
    img_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])
    train_dataset = CatDataset(configs.data_dir,
                               configs.train_class_file,
                               transform=img_transform)
    train_dataloader = DataLoader(train_dataset,
                                  batch_size=configs.batch_size,
                                  shuffle=True,
                                  num_workers=configs.num_workers)
    val_dataset = CatDataset(configs.data_dir,
                              configs.test_class_file,
                              transform=img_transform)
    val_dataloader = DataLoader(val_dataset,
                                 test=configs.batch_size,
                                 shuffle=False,
                                 num_workers=configs.num_workers)
    train_steps = len(train_dataloader.dataset) // configs.batch_size
    val_steps = len(val_dataloader.dataset) // configs.batch_size
    logging.info(f'Train step:{train_steps}, Val steps:{val_steps}')
    # Model
    model = get_model(configs.pretrained, device)
    logging.info(f'Use pretrained model: {configs.pretrained}')
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=configs.learning_rate,
        **optim_cfg,
        )
    
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
    }
    
    model = get_model('efficientnet_b2')
    print(model)