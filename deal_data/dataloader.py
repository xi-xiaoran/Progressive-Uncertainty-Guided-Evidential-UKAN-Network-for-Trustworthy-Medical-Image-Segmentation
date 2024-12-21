import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optin
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from sklearn.metrics import mean_squared_error
from PIL import Image
import matplotlib.pyplot as plt


class MedicalImageDataset(Dataset):
    def __init__(self, images_dir, masks_dir, transform=None, channels=1, resize=False):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.transform = transform
        self.images = sorted(os.listdir(images_dir))
        self.masks = sorted(os.listdir(masks_dir))
        self.channels = channels
        self.resize = resize

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.images_dir, self.images[idx])
        mask_path = os.path.join(self.masks_dir, self.masks[idx])
        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        if self.channels == 1:
            image = image.convert("l")
        if self.resize:
            image = image.resize((300,300), Image.Resampling.LANCZOS)
            mask = mask.resize((300,300), Image.Resampling.LANCZOS)

        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)
            mask = torch.where(mask > 0.5, 1.0, 0.0)



        return image, mask

data_transform = transforms.Compose([
    # transforms.CenterCrop((256,256)),
    transforms.Resize((256,256)),
    # transforms.Resize((128,128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0], std=[1])
])

def get_data_loader(images_root, masks_root, channels = 3):
    dataset = MedicalImageDataset(
        images_dir=images_root,
        masks_dir=masks_root,
        transform=data_transform,
        channels=channels,
        resize=False
    )
    # train_size = int(0.7 * len(dataset))
    # print(f'train_size={train_size}')
    # val_size = int(0.2 * len(dataset))
    # print(f'val_size={val_size}')
    # test_size = len(dataset) - train_size - val_size
    # print(f'test_size={test_size}')

    # train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, val_size, test_size])
    #
    # # Data loaders
    # train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    # val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
    # test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    # return train_loader, val_loader, test_loader
    return dataset

