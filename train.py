import os
import sys
import yaml
import time
import random
import logging
import argparse
from tqdm import tqdm
from pathlib import Path
from distutils.util import strtobool

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchinfo import summary

from dataset import CatDataset
from utils import cfg, create_experiment, draw_log, get_field_cfg, get_model, read_default_configs, seed_torch

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir',type=str)
    parser.add_argument('--pretrained',type=lambda x: bool(strtobool(x)))
    parser.add_argument('--freeze_backbone',type=lambda x: bool(strtobool(x)))
    parser.add_argument('--optimizer',type=str)
    parser.add_argument('--batch_size',type=int)
    parser.add_argument('--seed',type=int)
    parser.add_argument('--epochs',type=int)
    parser.add_argument('--learning_rate',type=float)
    parser.add_argument('--summary',type=bool, default=False)
    args = parser.parse_args()
    # Experiment setup
    expr_dir = create_experiment()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(Path(expr_dir)/"experiment.log"),
            logging.StreamHandler(stream=sys.stdout)
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
    img_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),  # Rotate images by 10 degrees
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])
    train_dataset = CatDataset(configs.data_dir,
                               configs.train_annot,
                               configs.train_class_file,
                               transform=img_transform)
    train_dataloader = DataLoader(train_dataset,
                                  batch_size=configs.batch_size,
                                  shuffle=True,
                                  num_workers=configs.num_workers)
    val_dataset = CatDataset(configs.data_dir,
                             configs.test_annot,
                             configs.train_class_file,
                             transform=img_transform)
    val_dataloader = DataLoader(val_dataset,
                                 batch_size=configs.batch_size,
                                 shuffle=False,
                                 num_workers=configs.num_workers)
    train_steps = len(train_dataloader.dataset) // configs.batch_size
    val_steps = len(val_dataloader.dataset) // configs.batch_size
    logging.info(f'Train step: {train_steps}, Val steps: {val_steps}')
    logging.info(f'Train data summary {train_dataset.summary()}')
    logging.info(f'Val data summary {val_dataset.summary()}')
    # Model
    model = get_model(configs.model_name,
                      len(train_dataset.train_class),
                      configs.pretrained,
                      configs.freeze_backbone)
    total_params = sum(p.numel() for p in model.parameters())
    print(f'Total number of parameters: {total_params:,}')
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Total number of trainable parameters: {trainable_params:,}')
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    model = model.to(device)
    logging.info(f'Use pretrained model: {configs.pretrained}')
    criterion = nn.CrossEntropyLoss()
    trainable_params = [param for param in model.parameters() if param.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=configs.learning_rate,
        **optim_cfg,
        )
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
    }
    
    if configs.summary:
        logging.info(summary(model,input_size=(configs.batch_size,3,28,28)))
    start_time = time.time()
    for epoch in range(configs.epochs):
        model.train()
        total_train_loss = 0
        total_val_loss = 0

        train_correct = 0
        val_correct = 0

        for images, labels in tqdm(train_dataloader):
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_train_loss +=  loss.detach()
            train_correct += (outputs.argmax(1) == labels).type(torch.float).sum().item()

        model.eval()
        with torch.no_grad():
            correct, total = 0, 0
            for images, labels in tqdm(val_dataloader):
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                total_val_loss += criterion(outputs, labels)
                val_correct += (outputs.argmax(1) == labels).type(torch.float).sum().item()

        avg_train_loss = total_train_loss / train_steps
        avg_val_loss = total_val_loss / val_steps

        train_correct = train_correct / len(train_dataloader.dataset)
        val_correct = val_correct / len(val_dataloader.dataset)

        history['train_loss'].append(avg_train_loss.cpu().detach().numpy())
        history['train_acc'].append(train_correct)
        history['val_loss'].append(avg_val_loss.cpu().detach().numpy())
        history['val_acc'].append(val_correct)

        logging.info('[INFO] EPOCHS: {}/{}'.format(epoch + 1, configs.epochs))
        logging.info('Train loss {:.6f}, Train accuracy: {:.4f}'.format(avg_train_loss, train_correct))
        logging.info('Val loss: {:.6f}, Val accuracy: {:.4f}\n'.format(avg_val_loss, val_correct))
    end_time = time.time()
    logging.info('[INFO] total time taken to train the model: {:.2f}s'.format(end_time - start_time))
    
    draw_log(history, expr_dir)
    df = pd.DataFrame.from_dict(history)
    df.rename(columns={df.columns[0]:'ID'}).to_csv(Path(expr_dir)/'history.csv')
    torch.save(model.state_dict(), Path(expr_dir)/'model.pt')
    with open(Path(expr_dir)/'configs.yaml', 'w') as file:
        for key, val in vars(configs).items():
            file.write(f'{key}: {val}\n')