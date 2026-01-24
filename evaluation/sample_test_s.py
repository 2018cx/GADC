# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Samples a large number of images from a pre-trained DiT model using DDP.
Subsequently saves a .npz file that can be used to compute FID and other
evaluation metrics via the ADM repo: https://github.com/openai/guided-diffusion/tree/main/evaluations

For a simple single-GPU/CPU sampling script, see sample.py.
"""
import torch
import torch.distributed as dist
from sit import SiT_models
from download import find_model
from diffusion import create_diffusion
from diffusion.samplers import euler_maruyama_sampler, euler_sampler
from diffusers.models import AutoencoderKL
from tqdm import tqdm
import os
from PIL import Image
import numpy as np
import math
import argparse
import torch.nn.functional as F
import random
import pandas as pd
import csv
from pathlib import Path

def setup_environment():
    if "RANK" not in os.environ:
        print("🚨 Warning: RANK is not set, assuming single GPU mode.")
        os.environ["RANK"] = "0"
        os.environ["WORLD_SIZE"] = "1"
        os.environ["MASTER_ADDR"] = "127.0.0.1"
        os.environ["MASTER_PORT"] = "29639"

    if int(os.environ["WORLD_SIZE"]) > 1:
        dist.init_process_group(backend="nccl")
        os.environ["MASTER_PORT"] = "29637"
        print(f"✅ Distributed initialized: rank {os.environ['RANK']} / {os.environ['WORLD_SIZE']}")
    else:
        print("✅ Running in single GPU mode. Distributed training is disabled.")

setup_environment()

def get_random_captions_from_indices(y, 
                                     data_path= "../training/text/caption_embeddings",
                                     class_index=None):
    n = len(y)  # batch_size
    embedding_list = []
    embedding_mask_list = []
    for i in range(n):
        cls_idx = y[i].item()  
        cls = class_index[cls_idx] 
        caption_file = os.path.join(data_path, cls + '.pt')
        embedding_mask_file = os.path.join(data_path, cls + '_mask.pt')
        embedding = torch.load(caption_file, map_location=torch.device('cpu')).squeeze() 
        embedding_mask = torch.load(embedding_mask_file, map_location=torch.device('cpu')).squeeze()
        embedding_list.append(embedding)
        embedding_mask_list.append(embedding_mask)
        
    return embedding_list, embedding_mask_list



def create_npz_from_sample_folder(sample_dir, num=50_000):
    """
    Builds a single .npz file from a folder of .png samples.
    """
    samples = []
    for i in tqdm(range(num), desc="Building .npz file from samples"):
        sample_pil = Image.open(f"{sample_dir}/{i:06d}.png")
        sample_np = np.asarray(sample_pil).astype(np.uint8)
        samples.append(sample_np)
    samples = np.stack(samples)
    assert samples.shape == (num, samples.shape[1], samples.shape[2], 3)
    npz_path = f"{sample_dir}.npz"
    np.savez(npz_path, arr_0=samples)
    print(f"Saved .npz file to {npz_path} [shape={samples.shape}].")
    return npz_path

@torch.no_grad()
def main(args):
    """
    Run sampling.
    """
    torch.backends.cuda.matmul.allow_tf32 = args.tf32  # True: fast but may lead to some small numerical differences
    assert torch.cuda.is_available(), "Sampling with DDP requires at least one GPU. sample.py supports CPU-only usage"
    torch.set_grad_enabled(False)

    class_ids = []
    with open("../training/text/extract_semantic_info/caption.txt", 'r') as f:
        for line in f:
            class_id = line.strip().split()[0]
            class_ids.append(class_id)
    

    # Setup DDP:
    if not dist.is_initialized():
        dist.init_process_group("nccl")

    rank = dist.get_rank()
    device = rank % torch.cuda.device_count()
    seed = args.global_seed * dist.get_world_size() + rank
    torch.manual_seed(seed)
    torch.cuda.set_device(device)
    print(f"Starting rank={rank}, seed={seed}, world_size={dist.get_world_size()}.")

    base = args.num_fid_samples // args.num_classes
    rem  = args.num_fid_samples % args.num_classes
    counts = [base + (1 if c < rem else 0) for c in range(args.num_classes)]
    balanced_classes = torch.cat([
        torch.full((counts[c],), c, dtype=torch.long) for c in range(args.num_classes)
    ]) if sum(counts) > 0 else torch.empty((0,), dtype=torch.long)

    g = torch.Generator().manual_seed(args.global_seed)
    perm = torch.randperm(balanced_classes.numel(), generator=g) if balanced_classes.numel() > 0 else torch.arange(0)
    balanced_classes = balanced_classes[perm]
    # finish test



    if args.ckpt is None:
        assert args.model == "DiT-XL/2", "Only DiT-XL/2 models are available for auto-download."
        assert args.image_size in [256, 512]
        assert args.num_classes == 1000

    # Load model:
    latent_size = args.image_size // 8
    block_kwargs = {"fused_attn": True, "qk_norm": False}
    model = SiT_models[args.model](
        input_size=latent_size,
        in_channels=4,
        num_classes=1000,
        class_dropout_prob=0.1,
        z_dims=[768],
        encoder_depth=8,
        bn_momentum=0.1,
        **block_kwargs
    ).to(device)
    # Auto-download a pre-trained model or load a custom DiT checkpoint from train.py:
    ckpt_path = args.ckpt or f"SiT-XL-2-{args.image_size}x{args.image_size}.pt"
    state_dict = find_model(ckpt_path)
    model.load_state_dict(state_dict)
    
    # latents_scale = model.state_dict()["bn.running_var"].clone().rsqrt().view(1, 4, 1, 1).to(device)
    # latents_bias = model.state_dict()["bn.running_mean"].clone().view(1, 4, 1, 1).to(device)
    model.eval()  # important!
    model = model
    model = torch.compile(model)
    vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{args.vae}").to(device)
    assert args.cfg_scale >= 1.0, "In almost all cases, cfg_scale be >= 1.0"
    using_cfg = args.cfg_scale > 1.0

    # Create folder to save samples:
    model_string_name = args.model.replace("/", "-")
    ckpt_string_name = os.path.basename(args.ckpt).replace(".pt", "") if args.ckpt else "pretrained"
    folder_name = f"{model_string_name}-{ckpt_string_name}-size-{args.image_size}-vae-{args.vae}-" \
                  f"cfg-{args.cfg_scale}-seed-{args.global_seed}"
    sample_folder_dir = f"{args.sample_dir}/{folder_name}"
    if rank == 0:
        os.makedirs(sample_folder_dir, exist_ok=True)
        print(f"Saving .png samples at {sample_folder_dir}")
    dist.barrier()

    # Figure out how many samples we need to generate on each GPU and how many iterations we need to run:
    n = args.per_proc_batch_size
    global_batch_size = n * dist.get_world_size()
    # To make things evenly-divisible, we'll sample a bit more than we need and then discard the extra samples:
    total_samples = int(math.ceil(args.num_fid_samples / global_batch_size) * global_batch_size)
    if rank == 0:
        print(f"Total number of images that will be sampled: {total_samples}")
    assert total_samples % dist.get_world_size() == 0, "total_samples must be divisible by world_size"
    samples_needed_this_gpu = int(total_samples // dist.get_world_size())
    assert samples_needed_this_gpu % n == 0, "samples_needed_this_gpu must be divisible by the per-GPU batch size"
    iterations = int(samples_needed_this_gpu // n)
    pbar = range(0, iterations, 1)
    pbar = tqdm(pbar) if rank == 0 else pbar
    total = 0
    for _ in pbar:
        # Sample inputs:
        z = torch.randn(n, model.in_channels, latent_size, latent_size, device=device)

        indices = (torch.arange(n, dtype=torch.long) * dist.get_world_size()) + rank + total
        valid_mask = indices < balanced_classes.numel()
        if valid_mask.any():
            selected = balanced_classes[indices[valid_mask].cpu()] 
            if selected.numel() < n:
                pad = selected[-1:].repeat(n - selected.numel())
                selected = torch.cat([selected, pad], dim=0)
            ccls = selected.to(device)
        else:
            ccls = torch.zeros(n, dtype=torch.long, device=device)

        y = ccls.clone()


        embedding_list, embedding_mask_list = get_random_captions_from_indices(y, class_index=class_ids)
        text_embedding = torch.stack(embedding_list, dim=0).to(device).to(torch.float32)
        text_embedding_mask = torch.stack(embedding_mask_list, dim=0).to(device).bool()

        assert not args.heun or args.mode == "ode", "Heun's method is only available for ODE sampling."

        # Sample images:
        sampling_kwargs = dict(
            model=model, 
            latents=z,
            y=ccls.to(device),
            num_steps=args.num_steps, 
            heun=args.heun,
            cfg_scale=args.cfg_scale,
            guidance_low=args.guidance_low,
            guidance_high=args.guidance_high,
            path_type=args.path_type,
            text_embedding=text_embedding,
            text_embedding_mask=text_embedding_mask
        )
        
        with torch.no_grad():
            if args.mode == "sde":
                samples = euler_maruyama_sampler(**sampling_kwargs).to(torch.float32)
            elif args.mode == "ode":
                samples = euler_sampler(**sampling_kwargs).to(torch.float32)
            else:
                raise NotImplementedError()
            # samples = (samples / latents_scale + latents_bias)
            # print(samples.mean(), samples.min(), samples.max(), samples.view(-1).std(dim=0), "test")
            torch.cuda.synchronize()
            if dist.is_initialized(): dist.barrier()

            if samples.shape[0] > 32:
                results = []
                for i in range(samples.shape[0] // 32):
                    samples_batch = samples[i*32:(i+1)*32]
                    samples_batch = vae.decode(samples_batch / 0.18215).sample
                    samples_batch = torch.clamp(127.5 * samples_batch + 128.0, 0, 255).permute(0, 2, 3, 1).to("cpu", dtype=torch.uint8).numpy()
                    results.append(samples_batch)
                samples = np.concatenate(results, axis=0)
            else:
                samples = vae.decode(samples / 0.18215).sample
                samples = torch.clamp(127.5 * samples + 128.0, 0, 255).permute(0, 2, 3, 1).to("cpu", dtype=torch.uint8).numpy()

            # Save samples to disk as individual .png files
            for i, sample in enumerate(samples):
                index = i * dist.get_world_size() + rank + total
                Image.fromarray(sample).save(f"{sample_folder_dir}/{index:06d}.png")
            torch.cuda.synchronize()
            if dist.is_initialized(): dist.barrier()
            total += global_batch_size

    # Make sure all processes have finished saving their samples before attempting to convert to .npz
    dist.barrier()
    if rank == 0:
        create_npz_from_sample_folder(sample_folder_dir, args.num_fid_samples)
        print("Done.")
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, choices=list(SiT_models.keys()), default="SiT-L/2")
    parser.add_argument("--vae",  type=str, choices=["ema", "mse"], default="ema")
    parser.add_argument("--sample-dir", type=str, default="1_10000")
    parser.add_argument("--per-proc-batch-size", type=int, default=64)
    parser.add_argument("--num-fid-samples", type=int, default=1_0000)
    parser.add_argument("--image-size", type=int, choices=[256, 512], default=256)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--cfg-scale",  type=float, default=1.5)
    parser.add_argument("--num-sampling-steps", type=int, default=250)
    parser.add_argument("--global-seed", type=int, default=6)
    parser.add_argument("--prob_cutmix", type=int, default=0.15)
    parser.add_argument("--tf32", default=True, action="store_true",
                        help="By default, use TF32 matmuls. This massively accelerates sampling on Ampere GPUs.")
    parser.add_argument("--ckpt", type=str, default="",
                        help="Optional path to a DiT checkpoint (default: auto-download a pre-trained DiT-XL/2 model).")
    parser.add_argument("--num-steps", type=int, default=250)
    parser.add_argument("--heun", default=False, action="store_true")
    parser.add_argument("--guidance-low", type=float, default=0.0)
    parser.add_argument("--guidance-high", type=float, default=1.0)
    parser.add_argument("--path-type", type=str, default="linear")
    parser.add_argument("--mode", type=str, default="ode")

    args = parser.parse_args()
    
    main(args)