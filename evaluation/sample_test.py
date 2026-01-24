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
from models_v3 import DiT_models
from download import find_model
from diffusion import create_diffusion
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
    model = DiT_models[args.model](
        input_size=latent_size,
        num_classes=args.num_classes
    ).to(device)
    # Auto-download a pre-trained model or load a custom DiT checkpoint from train.py:
    ckpt_path = args.ckpt or f"DiT-XL-2-{args.image_size}x{args.image_size}.pt"
    state_dict = find_model(ckpt_path)
    model.load_state_dict(state_dict)
    model.eval()  # important!
    model = model.to(torch.bfloat16)
    model = torch.compile(model)
    diffusion = create_diffusion(str(args.num_sampling_steps))
    vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{args.vae}").to(device)
    assert args.cfg_scale >= 1.0, "In almost all cases, cfg_scale be >= 1.0"
    using_cfg = args.cfg_scale > 1.0

    # Create folder to save samples:
    model_string_name = args.model.replace("/", "-")
    ckpt_string_name = os.path.basename(args.ckpt).replace(".pt", "") if args.ckpt else "pretrained"
    folder_name = f"{model_string_name}-{ckpt_string_name}-size-{args.image_size}-vae-{args.vae}-" \
                  f"cfg-{args.cfg_scale}-seed-{args.global_seed}-ours"
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
            selected = balanced_classes[indices[valid_mask].cpu()]  # 先在 CPU 上索引
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

        # Setup classifier-free guidance:
        if using_cfg:
            z = torch.cat([z, z], 0)
            y_null = torch.zeros_like(text_embedding).to(device)
            ccls = ccls.to(device)
            y = torch.cat([text_embedding, y_null], 0)
            text_embedding_mask = torch.cat([text_embedding_mask, torch.zeros_like(text_embedding_mask).bool()], 0)
            ccls_null = torch.tensor([1000] * n, device=device)
            ccls = torch.cat([ccls, ccls_null], 0)
            print(ccls.shape, text_embedding_mask.shape, y.shape)
            model_kwargs = dict(y=y.to(torch.bfloat16), ccls=ccls, attn_mask=text_embedding_mask, cfg_scale=args.cfg_scale)
            sample_fn = model.forward_with_cfg
        else:
            model_kwargs = dict(y=text_embedding.to(torch.bfloat16), ccls=ccls, attn_mask=text_embedding_mask)
            sample_fn = model.forward

        # Sample images:
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            samples = diffusion.p_sample_loop(
                sample_fn, z.shape, z.to(torch.bfloat16), clip_denoised=False, model_kwargs=model_kwargs, progress=False, device=device
            )
        if using_cfg:
            samples, _ = samples.chunk(2, dim=0)  # Remove null class samples

        results = []
        bs = 32
        num = samples.shape[0]
        for i in range(0, num, bs):
            samples_batch = samples[i:min(i + bs, num)]
            samples_batch = vae.decode(samples_batch / 0.18215).sample
            samples_batch = torch.clamp(127.5 * samples_batch + 128.0, 0, 255)\
                            .permute(0, 2, 3, 1).to("cpu", dtype=torch.uint8).numpy()
            results.append(samples_batch)

        samples = np.concatenate(results, axis=0) if len(results) > 1 else results[0]


        # Save samples to disk as individual .png files
        for i, sample in enumerate(samples):
            index = i * dist.get_world_size() + rank + total
            Image.fromarray(sample).save(f"{sample_folder_dir}/{index:06d}.png")
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
    parser.add_argument("--model", type=str, choices=list(DiT_models.keys()), default="DiT-L/2")
    parser.add_argument("--vae",  type=str, choices=["ema", "mse"], default="ema")
    parser.add_argument("--sample-dir", type=str, default="50_50_base")
    parser.add_argument("--per-proc-batch-size", type=int, default=250)
    parser.add_argument("--num-fid-samples", type=int, default=50_000)
    parser.add_argument("--image-size", type=int, choices=[256, 512], default=512)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--cfg-scale",  type=float, default=1.5)
    parser.add_argument("--num-sampling-steps", type=int, default=250)
    parser.add_argument("--global-seed", type=int, default=1)
    parser.add_argument("--prob_cutmix", type=int, default=0.15)
    parser.add_argument("--tf32", action=argparse.BooleanOptionalAction, default=True,
                        help="By default, use TF32 matmuls. This massively accelerates sampling on Ampere GPUs.")
    parser.add_argument("--ckpt", type=str, default="",
                        help="Optional path to a DiT checkpoint (default: auto-download a pre-trained DiT-XL/2 model).")

    parser.add_argument("--cutmix_noise_threshold_ratio", type=int, default=500)
    parser.add_argument("--do_cutmix_above_threshold_prob", type=float, default=0.5)

    args = parser.parse_args()
    
    args.sample_dir = args.sample_dir + "_" + args.ckpt.split("/")[-1].split(".")[0]
    main(args)