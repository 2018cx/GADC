import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

# torchvision ResNet
from torchvision.models import resnet50, ResNet50_Weights, resnet34, ResNet34_Weights
from torch.cuda.amp import autocast

def build_resnet(device="cuda"):
    weights = ResNet50_Weights.DEFAULT
    model = resnet50(weights=weights)
    model.fc = nn.Identity()    # return 2048-d features
    model.eval().to(device)
    return model

@torch.no_grad()
def get_features(dataloader, device="cuda"):
    weights = ResNet50_Weights.DEFAULT
    model = build_resnet(device=device)
    preprocess = weights.transforms()

    feats_all = []
    for images, _ in tqdm(dataloader, desc="Extract ResNet50 GAP (2048)"):
        images = images.to(device, non_blocking=True)
        images = F.interpolate(images, size=(224, 224), mode='bilinear', align_corners=False)
        # Normalize to ImageNet stats expected by pretrained weights
        images = preprocess.normalize(preprocess.resize(images)) if hasattr(preprocess, "normalize") else images
        with autocast():
            feats = model(images)     # [B,2048]
        feats = feats.float().detach().cpu()
        feats_all.append(feats)
    return torch.cat(feats_all, dim=0)  # [N,2048]
