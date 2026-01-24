import torch
from torchvision.models import inception_v3, Inception_V3_Weights
import torch.nn.functional as F
from tqdm import tqdm
import torch.nn as nn
from torch.cuda.amp import autocast


def network(device="cuda"):
    weights = Inception_V3_Weights.DEFAULT
    model = inception_v3(weights=weights, transform_input=False)
    model.fc = nn.Identity()
    if hasattr(model, "dropout"):
        model.dropout = nn.Identity()
    model.eval().to(device)
    return model

def get_features(dataloader, device="cuda"):
    weights = Inception_V3_Weights.DEFAULT
    model = bnetwork(device=device)
    model.fc = torch.nn.Identity()          
    if hasattr(model, "dropout"):
        model.dropout = torch.nn.Identity() 
    model.eval().to(device)

    feats_all = []
    for images, _ in tqdm(dataloader, desc="Extract FID pool3"):
        images = images.to(device, non_blocking=True) * 2.0 - 1.0  
        with autocast():
            feats = model(images)
        feats = feats.float()
        feats_all.append(feats.detach().cpu())
    return torch.cat(feats_all, dim=0)  # [N, 2048]