'''
Custom Dataset for CamVid Dataset
https://www.kaggle.com/datasets/carlolepelaars/camvid
'''

import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import torchvision.transforms.functional as transforms_F
import pandas as pd

class_df = pd.read_csv('/kaggle/input/camvid/CamVid/class_dict.csv')
# Create a dictionary that maps rgb value to 32 CamVid's class indices
RGB2label_dict = {
    (row['r'], row['g'], row['b']): idx
    for idx, row in class_df.iterrows()
}
label2RGB_dict = {
    v: k for k, v in RGB2label_dict.items()
}

class CamVidDataset(Dataset):
    def __init__(self, img_dir: str, label_dir: str, augmentation: bool=False):
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.augmentation = augmentation
        self.img_files = sorted(os.listdir(self.img_dir))
        self.label_files = sorted(os.listdir(self.label_dir))

        self.transform = transforms.Compose([
            transforms.Resize((384, 480)),
            transforms.ToTensor()
        ])
        
    def __len__(self):
        return len(self.img_files)

    def _augment(self, image: torch.Tensor, label: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # Horizontal flip with p=0.5
        if torch.randn(1) > 0.5:
            image = transforms_F.hflip(image)
            label = transforms_F.hflip(label)
        # Pad for cropping
        image = transforms_F.pad(image, (10, 10, 10, 10))
        label = transforms_F.pad(label, (10, 10, 10, 10))
        # RandomCrop
        i, j, h, w = transforms.RandomCrop.get_params(image, output_size=(384, 480))
        image = transforms_F.crop(image, i, j, h, w)
        label = transforms_F.crop(label, i, j, h, w)

        image = transforms.ColorJitter(brightness=0.1, contrast=0, saturation=0, hue=0.2)(image)
        return image, label
        
    def __getitem__(self, idx):
        img_file = self.img_files[idx]
        label_file = self.label_files[idx]

        img_path = os.path.join(self.img_dir, img_file)
        label_path = os.path.join(self.label_dir, label_file)

        image = Image.open(img_path)
        label = Image.open(label_path)

        # Transform and send to cpu/gpu
        image = self.transform(image).to(DEVICE)
        label = self.transform(label).to(DEVICE)

        # If augmentation is on, apply augmentation
        if self.augmentation:
            image, label = self._augment(image, label)

        # Masking label image pixel by pixel
        label = label.permute(1, 2, 0) # (C, H, W) -> (H, W, C)
        label = (label * 255).int() # Scale back to 0~255 as torch.ToTensor() scaled the image to 0~1
        masked_label = torch.zeros(label.size(0), label.size(1), dtype=torch.uint8, device=DEVICE)
        for rgb, idx in RGB2label_dict.items(): # Mask the pixels for every class type
            rgb_tensor = torch.tensor(rgb, device=DEVICE)
            masked_label[(label == rgb_tensor).all(axis=-1)] = idx

        return image, masked_label.long()
        