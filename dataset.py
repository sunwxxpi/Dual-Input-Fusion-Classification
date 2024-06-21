import random
import os
import pandas as pd
import numpy as np
import cv2
import torchvision.transforms as transforms
from glob import glob
from PIL import Image
from torch.utils.data.dataset import Dataset
from torchvision.transforms.transforms import CenterCrop, Grayscale, RandomHorizontalFlip, RandomRotation


class AddGaussianNoise(object):
    def __init__(self, mean=0.0, variance=1.0, amplitude=1.0,p=1):

        self.mean = mean
        self.variance = variance
        self.amplitude = amplitude
        self.p=p

    def __call__(self, img):
        if random.uniform(0, 1) < self.p:
            img = np.array(img)
            h, w = img.shape
            N = self.amplitude * np.random.normal(loc=self.mean, scale=self.variance, size=(h, w))
            img = N + img
            img[img > 255] = 255                       
            img = Image.fromarray(img.astype('uint8')).convert('L')
            return img
        else:
            return img


class AddBlur(object):
    def __init__(self, kernel=3, p=1):
        self.kernel = kernel
        self.p = p

    def __call__(self, img):
        if random.uniform(0, 1) < self.p:
            img = np.array(img)
            img = cv2.blur(img, (self.kernel, self.kernel))
            img = Image.fromarray(img.astype('uint8')).convert('L')
            return img
        else:
            return img


class CustomDataset(Dataset):
    def __init__(self, root, transform, mask_transform, elastogram_transform):
        super().__init__()
        self.root = root
        self.csv = os.path.join(root, "label.csv")
        self.transform = transform
        self.mask_transform = mask_transform
        self.elastogram_transform = elastogram_transform
        self.info = pd.read_csv(self.csv)

    def __len__(self):
        return len(self.info)

    def __getitem__(self, index):
        patience_info = self.info.iloc[index]
        file_name = patience_info['name']
        file_path = glob(f"{self.root}/img/{file_name}")[0]
        mask_path = glob(f"{self.root}/mask/{file_name}")[0]
        elastogram_path = glob(f"{self.root}/elastogram/{file_name}")[0]
        label = patience_info['label']
        
        img = Image.open(file_path).convert('L')
        mask = Image.open(mask_path).convert('RGB')
        elastogram = Image.open(elastogram_path).convert('RGB')
        
        if self.transform is not None:
            img = self.transform(img)
            mask = self.mask_transform(mask)
            elastogram = self.elastogram_transform(elastogram)

        return {'imgs': img, 'masks': mask, 'elastograms': elastogram, 'labels': label, 'names': file_name}


def get_dataset(imgpath, img_size, mode='train'):
    train_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        AddGaussianNoise(amplitude=random.uniform(0, 1), p=0.5),
        AddBlur(kernel=3, p=0.5),
        transforms.RandomHorizontalFlip(p=0.5),
        # transforms.RandomVerticalFlip(p=0.5),
        # transforms.RandomRotation(20),
        transforms.ColorJitter(brightness=(0.5, 2), contrast=(0.5, 2)),
        transforms.ToTensor(),
        transforms.Normalize(mean=0.5, std=0.5)
    ])
    
    test_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=0.5, std=0.5)
    ])
    
    mask_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=0.5, std=0.5)
    ])

    elastogram_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=0.5, std=0.5)
    ])

    if mode =='train':
        transform = train_transform
    elif mode == 'test':
        transform = test_transform

    dataset = CustomDataset(imgpath, transform, mask_transform, elastogram_transform)

    return dataset