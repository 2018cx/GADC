from pathlib import Path
import random
from PIL import Image
from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder
import numpy as np
import torch
import os

class CustomDataset(Dataset):
    def __init__(self, root, image_size, transform=None):
        self.root = root
        self.image_size = image_size
        self.transform = transform
        self.image_folder = ImageFolder(root=root, transform=self.transform)
        self.class_to_idx = self.image_folder.class_to_idx
        self.num_classes = len(self.class_to_idx)
        self.class_names = list(self.class_to_idx.keys())
        self.class_images = []
        for i in range(1000):
            self.class_images.append(self.get_class_images(i))

    def __len__(self):
        return len(self.image_folder.samples)

    def __getitem__(self, index):
        img_path, class_idx = self.image_folder.samples[index]
        image = self.image_folder.loader(img_path)
        if self.transform:
            image = self.transform(image)
        return image, class_idx
    
    def get_class_images(self, class_idx):
        class_images = []
        for img_path, _class_idx in self.image_folder.samples:
            if _class_idx == class_idx:
                class_images.append((img_path, class_idx))
        return class_images
