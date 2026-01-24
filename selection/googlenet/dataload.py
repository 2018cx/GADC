import os
from typing import Tuple
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


def get_by_id(class_id, batch_size=64, num_workers=4):
    assert 0 <= class_id < 1000, 

    imagenet_root = '../ImageNet/train'

    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor()
    ])

    dataset = datasets.ImageFolder(imagenet_root, transform=transform)
    
    idx_to_class = {v: k for k, v in dataset.class_to_idx.items()}
    
    target_indices = [i for i, (_, label) in enumerate(dataset.samples) if label == class_id]
    
    subset = Subset(dataset, target_indices)
    loader = DataLoader(subset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return loader, idx_to_class[class_id]
