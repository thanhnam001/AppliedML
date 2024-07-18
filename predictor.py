import os
import tempfile
import requests

import gdown
import torch
import torch.nn.functional as F
from torch import nn
from torchvision import transforms
from tqdm import tqdm
from utils import get_model

def download_weights(uri, model_name=None, cached=None, md5=None, quiet=False):
    if uri.startswith('http'):
        return download(url=uri,model_name=model_name, quiet=quiet)
    return uri

def download(url, model_name=None, quiet=False):
    tmp_dir = tempfile.gettempdir()
    filename = url.split('/')[-1] 
    filename = filename if filename.endswith('pt') else model_name
    full_path = os.path.join(tmp_dir, filename)
    if os.path.exists(full_path):
        print('Model weight {} exists. Ignore download!'.format(full_path))
        return full_path
    print(full_path)
    gdown.download(url, full_path)
    # with requests.get(url, stream=True) as r:
    #     r.raise_for_status()
    #     with open(full_path, 'wb') as f:
    #         for chunk in tqdm(r.iter_content(chunk_size=8192)):
    #             # If you have chunk encoded response uncomment if
    #             # and set chunk_size parameter to None.
    #             #if chunk:
    #             f.write(chunk)
    return full_path

class Predictor:
    def __init__(self, configs):
        self.device = configs['device']
        
        weights = '/tmp/weights.pth'

        if configs['weights'].startswith('http'):
            weights = download_weights(configs['weights'], model_name=f"{configs['model_name']}.pt")
        else:
            weights = configs['weights']
        with open(configs['train_class_file'], 'r') as f:
            clss = f.readlines()
        self.train_class = [c.strip() for c in clss]
        self.class2idx = {clss: i for i, clss in enumerate(self.train_class)}
        self.idx2class = {i: clss for i, clss in enumerate(self.train_class)}
        self.model = get_model(configs['model_name'],
                                len(self.train_class),
                                False,
                                configs['freeze_backbone'])
        self.model = nn.DataParallel(self.model)
        print('Load', weights)
        weight = torch.load(weights, map_location=torch.device(self.device))
        self.model.eval()
        self.model.load_state_dict(weight, strict=True)
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ])
        
        
    def predict(self, image):
        with torch.no_grad():
            image = self.transform(image).unsqueeze(0)
            logits = self.model(image)
            output = F.softmax(logits, dim=1)
            idx = torch.argmax(output, dim=1).item()
            label = self.idx2class[idx]
        return label