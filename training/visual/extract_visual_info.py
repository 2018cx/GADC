import argparse
import copy
import logging
import os
import math
from pathlib import Path
from collections import OrderedDict

import json
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
import torch
from torch.utils.data import DataLoader
from torchvision.transforms import Normalize
from torchvision.utils import make_grid
from tqdm.auto import tqdm
from omegaconf import OmegaConf
from dataset import CustomDataset
from loss.losses import ReconstructionLoss_Single_Stage
from models.autoencoder import vae_models
from models.sit import SiT_models
from utils import load_encoders, count_trainable_params
from torchvision import transforms
from PIL import Image
import numpy as np

logger = get_logger(__name__)

CLIP_DEFAULT_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_DEFAULT_STD = (0.26862954, 0.26130258, 0.27577711)



def center_crop_arr(pil_image, image_size):
    """
    Center cropping implementation from ADM.
    https://github.com/openai/guided-diffusion/blob/8fb3ad9197f16bbc40620447b2742e13458d2831/guided_diffusion/image_datasets.py#L126
    """
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return Image.fromarray(arr[crop_y: crop_y + image_size, crop_x: crop_x + image_size])

def preprocess_raw_image(x, enc_type):
    resolution = x.shape[-1]
    if 'clip' in enc_type:
        x = x / 255.
        x = torch.nn.functional.interpolate(x, 224 * (resolution // 256), mode='bicubic')
        x = Normalize(CLIP_DEFAULT_MEAN, CLIP_DEFAULT_STD)(x)
    elif 'mocov3' in enc_type or 'mae' in enc_type:
        x = x / 255.
        x = Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)(x)
    elif 'dinov2' in enc_type:
        x = x / 255.
        x = Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)(x)
        x = torch.nn.functional.interpolate(x, 224 * (resolution // 256), mode='bicubic')
    elif 'dinov1' in enc_type:
        x = x / 255.
        x = Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)(x)
    elif 'jepa' in enc_type:
        x = x / 255.
        x = Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)(x)
        x = torch.nn.functional.interpolate(x, 224 * (resolution // 256), mode='bicubic')

    return x


def array2grid(x):
    nrow = round(math.sqrt(x.size(0)))
    x = make_grid(x.clamp(0, 1), nrow=nrow, value_range=(0, 1))
    x = x.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
    return x


def sample_posterior(moments, latents_scale=1., latents_bias=0.):
    mean, std = torch.chunk(moments, 2, dim=1)
    z = mean + std * torch.randn_like(mean)
    z = (z - latents_bias) * latents_scale # normalize
    return z 


@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    """
    Step the EMA model towards the current model.
    """
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())

    for name, param in model_params.items():
        name = name.replace("module.", "")
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)

    # Also perform EMA on BN buffers
    ema_buffers = OrderedDict(ema_model.named_buffers())
    model_buffers = OrderedDict(model.named_buffers())

    for name, buffer in model_buffers.items():
        name = name.replace("module.", "")
        if buffer.dtype in (torch.bfloat16, torch.float16, torch.float32, torch.float64):
            # Apply EMA only to float buffers
            ema_buffers[name].mul_(decay).add_(buffer.data, alpha=1 - decay)
        else:
            # Direct copy for non-float buffers
            ema_buffers[name].copy_(buffer)


def create_logger(logging_dir):
    """
    Create a logger that writes to a log file and stdout.
    """
    logging.basicConfig(
        level=logging.INFO,
        format='[\033[34m%(asctime)s\033[0m] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[logging.StreamHandler(), logging.FileHandler(f"{logging_dir}/log.txt")]
    )
    logger = logging.getLogger(__name__)
    return logger


def requires_grad(model, flag=True):
    """
    Set requires_grad flag for all parameters in a model.
    """
    for p in model.parameters():
        p.requires_grad = flag


#################################################################################
#                                  Training Loop                                #
#################################################################################

def main(args):    
    # set accelerator
    logging_dir = Path(args.output_dir, args.logging_dir)



    os.makedirs(args.output_dir, exist_ok=True)  # Make results folder (holds all experiment subfolders)
    save_dir = os.path.join(args.output_dir, args.exp_name)
    os.makedirs(save_dir, exist_ok=True)
    args_dict = vars(args)
    # Save to a JSON file
    json_dir = os.path.join(save_dir, "args.json")
    with open(json_dir, 'w') as f:
        json.dump(args_dict, f, indent=4)
    checkpoint_dir = f"{save_dir}/checkpoints"  # Stores saved model checkpoints
    os.makedirs(checkpoint_dir, exist_ok=True)
    logger = create_logger(save_dir)
    logger.info(f"Experiment directory created at {save_dir}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.seed is not None:
        set_seed(args.seed)

    # Create model:
    if args.vae == "f8d4":
        assert args.resolution % 8 == 0, "Image size must be divisible by 8 (for the VAE encoder)."
        latent_size = args.resolution // 8
        in_channels = 4
    elif args.vae == "f16d32":
        assert args.resolution % 16 == 0, "Image size must be divisible by 16 (for the VAE encoder)."
        latent_size = args.resolution // 16
        in_channels = 32
    else:
        raise NotImplementedError()

    if args.enc_type != None:
        encoders, encoder_types, architectures = load_encoders(
            args.enc_type, device, args.resolution
        )
    else:
        raise NotImplementedError()
    z_dims = [encoder.embed_dim for encoder in encoders] if args.enc_type != 'None' else [0]

    block_kwargs = {"fused_attn": args.fused_attn, "qk_norm": args.qk_norm}
    model = SiT_models[args.model](
        input_size=latent_size,
        in_channels=in_channels,
        num_classes=args.num_classes,
        class_dropout_prob=args.cfg_prob,
        z_dims=z_dims,
        encoder_depth=args.encoder_depth,
        bn_momentum=args.bn_momentum,
        **block_kwargs
    )

    # make a copy of the model for EMA
    model = model.to(device)
    # Load VAE and create a EMA of the VAE
    vae = vae_models[args.vae]().to(device)
    vae_ckpt = torch.load(args.vae_ckpt, map_location=device)
    vae.load_state_dict(vae_ckpt, strict=False)   # We may not have the projection layer in the VAE
    del vae_ckpt
    print("Total trainable params in VAE:", count_trainable_params(vae))

    # Load the VAE latents-stats for BN layer initialization
    latents_stats = torch.load(args.vae_ckpt.replace(".pt", "-latents-stats.pt"))
    latents_scale = latents_stats["latents_scale"].squeeze().to(device)
    latents_bias = latents_stats["latents_bias"].squeeze().to(device)

    model.init_bn(latents_bias=latents_bias, latents_scale=latents_scale)

    # Apply SyncBN if more than 1 GPU is used

    loss_cfg = OmegaConf.load(args.loss_cfg_path)
    vae_loss_fn = ReconstructionLoss_Single_Stage(loss_cfg).to(device)

    logger.info(f"SiT Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Setup optimizer (we used default Adam betas=(0.9, 0.999) and a constant learning rate of 1e-4 in our paper):
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True


    # Setup data

    with open("../training/baseline_dataset_list/merged.json", "r") as f:
        select_info = json.load(f) # list[dict, dict, ...]
    merged_dict = {}
    for d in select_info:
        merged_dict.update(d)
    
    transform = transforms.Compose([
        transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, 256)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
    ])
    
    train_dataset = CustomDataset(args.data_dir,image_size=256,transform=transform,
                            select_info=merged_dict, 
                            select_num=10000)
    
    local_batch_size = int(args.batch_size)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=local_batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False
    )
    logger.info(f"Dataset contains {len(train_dataset):,} images ({args.data_dir})")
    
    # Start with eval mode for all models
    model.eval()
    vae.eval()

    if args.disc_pretrained_ckpt is not None:
        # Load the discriminator from a pretrained checkpoint if provided
        disc_ckpt = torch.load(args.disc_pretrained_ckpt, map_location=device)
        vae_loss_fn.discriminator.load_state_dict(disc_ckpt)
        logger.info(f"Loaded discriminator from {args.disc_pretrained_ckpt}")

    # resume
    global_step = 0
    if args.resume_step > 0:
        ckpt_name = str(args.resume_step).zfill(7) +'.pt'
        ckpt_path = f'{args.cont_dir}/checkpoints/{ckpt_name}'

        # If the checkpoint exists, we load the checkpoint and resume the training
        ckpt = torch.load(ckpt_path, map_location='cpu')
        model.load_state_dict(ckpt['model'])
        vae.load_state_dict(ckpt['vae'])
        vae_loss_fn.discriminator.load_state_dict(ckpt['discriminator'])
        global_step = ckpt['steps']

    # Allow larger cache size for DYNAMo compilation
    torch._dynamo.config.cache_size_limit = 64
    torch._dynamo.config.accumulated_cache_size_limit = 512
    # Model compilation for better performance
    if args.compile:
        model = torch.compile(model, backend="inductor", mode="default")
        vae = torch.compile(vae, backend="inductor", mode="default")
        vae_loss_fn = torch.compile(vae_loss_fn, backend="inductor", mode="default")



    if not os.path.exists(args.svd_feat_dir):
        os.makedirs(args.svd_feat_dir, exist_ok=True)
    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=global_step,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=True,
    )

    # Labels to condition the model with (feel free to change):
    sample_batch_size = 64
    ys = torch.randint(1000, size=(sample_batch_size,), device=device)
    ys = ys.to(device)
    # Create sampling noise:
    n = ys.size(0)
    xT = torch.randn((n, in_channels, latent_size, latent_size), device=device)

    # main training loop
    count_idx = 0

    total_json = dict()
    for raw_image, y, s_index in train_dataloader:
        
        save_pt_list = []
        raw_image = raw_image.to(device)
        raw_image = ((raw_image + 1) / 2.) * 255 
        labels = y.to(device)
        z = None
        # extract the dinov2 features
        with torch.no_grad():
            zs = []
            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                for encoder, encoder_type, arch in zip(encoders, encoder_types, architectures):
                    raw_image_ = preprocess_raw_image(raw_image, encoder_type)
                    print(raw_image_.min(), raw_image_.max(), raw_image_.mean() ,"raw_image_")
                    z = encoder.forward_features(raw_image_)
                    if 'mocov3' in encoder_type: z = z = z[:, 1:] 
                    if 'dinov2' in encoder_type: z = z['x_norm_patchtokens']
                    print(z.shape)
                    zs.append(z)
        
        gt_zs = zs[0]
        for _index in range(gt_zs.shape[0]):
            _gt_zs = gt_zs[_index] 
            save_pt_list.append({
                "gt_zs": _gt_zs.to(torch.bfloat16),
            })

        
        for _index, save_pt in enumerate(save_pt_list):
            save_path = f"{args.svd_feat_dir}/s{count_idx:07d}.pt"
            torch.save(save_pt, save_path)
            total_json[train_dataset.image_folder.samples[s_index[_index]][0]] = save_path
            count_idx += 1

            if count_idx % 50 == 0:
                with open(f"{args.svd_feat_dir}/total_json.json", "w") as f:
                    json.dump(total_json, f, indent=4)

        with open(f"{args.svd_feat_dir}/total_json.json", "w") as f:
            json.dump(total_json, f, indent=4)

    model.eval()
    


def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Training")

    # logging params
    parser.add_argument("--output-dir", type=str, default="exps")
    parser.add_argument("--exp-name", type=str, required=True)
    parser.add_argument("--logging-dir", type=str, default="logs")
    parser.add_argument("--report-to", type=str, default="wandb")
    parser.add_argument("--sampling-steps", type=int, default=10000)
    parser.add_argument("--resume-step", type=int, default=0)
    parser.add_argument("--continue-train-exp-dir", type=str, default=None)
    parser.add_argument("--wandb-history-path", type=str, default=None)

    # SiT model params
    parser.add_argument("--model", type=str, default="SiT-XL/2", choices=SiT_models.keys(),
                        help="The model to train.")
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--encoder-depth", type=int, default=8)
    parser.add_argument("--qk-norm",  action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--fused-attn", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--bn-momentum", type=float, default=0.1)
    parser.add_argument("--compile", action=argparse.BooleanOptionalAction, default=True,
                        help="Whether to compile the model for faster training")

    # dataset params
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--resolution", type=int, choices=[256], default=256)
    parser.add_argument("--batch-size", type=int, default=256)

    # precision params
    parser.add_argument("--allow-tf32", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--mixed-precision", type=str, default="fp16", choices=["no", "fp16", "bf16"])

    # optimization params
    parser.add_argument("--epochs", type=int, default=1400)
    parser.add_argument("--max-train-steps", type=int, default=400000)
    parser.add_argument("--checkpointing-steps", type=int, default=50000)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--adam-beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam-beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam-weight-decay", type=float, default=0., help="Weight decay to use.")
    parser.add_argument("--adam-epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument("--max-grad-norm", default=1.0, type=float, help="Max gradient norm.")

    # seed params
    parser.add_argument("--seed", type=int, default=0)

    # cpu params
    parser.add_argument("--num-workers", type=int, default=4)

    # loss params
    parser.add_argument("--path-type", type=str, default="linear", choices=["linear", "cosine"])
    parser.add_argument("--prediction", type=str, default="v", choices=["v"],
                        help="currently we only support v-prediction")
    parser.add_argument("--cfg-prob", type=float, default=0.1)
    parser.add_argument("--enc-type", type=str, default='dinov2-vit-b')
    parser.add_argument("--proj-coeff", type=float, default=0.5)
    parser.add_argument("--weighting", default="uniform", type=str, choices=["uniform", "lognormal"],
                        help="Loss weihgting, uniform or lognormal")

    # vae params
    parser.add_argument("--vae", type=str, default="f8d4", choices=["f8d4", "f16d32"])
    parser.add_argument("--vae-ckpt", type=str, default="sdvae-f8d4.pt")

    # vae loss params
    parser.add_argument("--disc-pretrained-ckpt", type=str, default=None)
    parser.add_argument("--loss-cfg-path", type=str, default="configs/l1_lpips_kl_gan.yaml")
    parser.add_argument("--svd-feat-dir", type=str, default="./test")
    # vae training params
    parser.add_argument("--vae-learning-rate", type=float, default=1e-4)
    parser.add_argument("--disc-learning-rate", type=float, default=1e-4)
    parser.add_argument("--vae-align-proj-coeff", type=float, default=1.5)

    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)