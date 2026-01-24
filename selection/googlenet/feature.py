import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

# torchvision GoogLeNet
from torchvision.models import googlenet, GoogLeNet_Weights
from torch.cuda.amp import autocast

@torch.no_grad()
def build_googlenet(device="cuda"):
    """
    Build a GoogLeNet backbone that outputs 1024-d GAP features:
      - disable aux logits
      - replace the classifier head with Identity to expose the pooled features
    """
    weights = GoogLeNet_Weights.DEFAULT
    model = googlenet(weights=weights, aux_logits=True)
    model.fc = nn.Identity()    # return 1024-d features after GAP
    model.eval().to(device)
    return model, weights

@torch.no_grad()
def get_features(dataloader, device="cuda"):
    model, weights = build_googlenet(device=device)
    preprocess = weights.transforms()

    feats_all = []
    for images, _ in tqdm(dataloader, desc="Extract GoogLeNet GAP (1024)"):
        images = images.to(device, non_blocking=True)
        images = F.interpolate(images, size=(224, 224), mode='bilinear', align_corners=False)
        images = preprocess.normalize(preprocess.resize(images)) if hasattr(preprocess, "normalize") else images
        with autocast():
            feats = model(images)     # [B,1024]
        feats = feats.float().detach().cpu()
        feats_all.append(feats)
    return torch.cat(feats_all, dim=0)  # [N,1024]
