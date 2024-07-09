import argparse
from tqdm import tqdm
from pathlib import Path
from collections import defaultdict
from PIL import Image

from sklearn.model_selection import train_test_split

import torch
from torch.utils.data import Dataset
from torch import nn

class CatDataset(Dataset):
    def __init__(self,
                 root_dir: str,
                 train_class_file: str,
                 transform):
        super().__init__()
        self.root_dir = root_dir
        with open(train_class_file, 'r') as f:
            train_class = f.readlines()
        self.train_class = [tc.strip() for tc in train_class]
        self.class2idx = {clss: i for i, clss in enumerate(train_class)}
        self.idx2class = {i: clss for i, clss in enumerate(train_class)}
        img_dirs = list(Path(root_dir).rglob('**/*.*g'))
        self.data = []
        print('Read data')
        for img_dir in tqdm(img_dirs):
            img_class = img_dir.parts[-2]
            if img_class in self.train_class:
                self.data.append({
                    'img_dir': img_dir,
                    'class': img_class
                })
        self.img_transform = transform
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        data = self.data[index]
        img_dir, lb = data['img_dir'], data['class']
        
        img = Image.open(img_dir).convert('RGB')
        img_ts = self.img_transform(img)
        
        lb = self.class2idx[lb]
        lb = torch.tensor(lb)
        return img_ts, lb
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
                    prog='Create ',
                    description='What the program does',
                    epilog='Text at the bottom of help')
    parser.add_argument('--img_dir',
                        type=str,
                        default='data/images')
    parser.add_argument('--num_class',
                        type=int,
                        default=10,
                        help='Number of most appeared classes use to train')
    args = parser.parse_args()
    save_dir = 'configs/train_class.txt'
    
    img_dirs = list(Path(args.img_dir).rglob('**/*.*g'))
    print('Total images count',len(img_dirs))
    class_count = defaultdict(int)
    for imd in tqdm(img_dirs):
        clss = imd.parts[-2]
        class_count[clss] += 1
    class_count = dict(class_count)
    sorted_class_count = sorted(class_count.items(), key=lambda x:x[1], reverse=True)
    num_class = args.num_class
    print('Num classes use to train', num_class)
    print('Save at', save_dir)
    with open(save_dir,'w') as f:
        f.write('\n'.join([x[0] for x in sorted_class_count[:num_class]]))
    