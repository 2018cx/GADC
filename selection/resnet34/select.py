# -*- coding: utf-8 -*-
import os
import json
import numpy as np
from typing import List, Tuple, Optional

import torch
from tqdm import tqdm
import torch.multiprocessing as mp
import torch.nn.functional as F

# torchvision ResNet
from torchvision.models import resnet50, ResNet50_Weights, resnet34, ResNet34_Weights

# project APIs
from dataload import get_by_id    # -> (loader, wnid)
from feature import get_features # -> torch.Tensor [N, D=2048]

# ---------------------------
# ResNet probs / CE / entropy
# ---------------------------
@torch.no_grad()
def get_resnet_probs_and_scores(dataloader, device: str):
    """
    Use pretrained ResNet-50 as the classifier head to produce 1000-d probabilities.
    Returns:
        probs_all:  [N, 1000]
        labels_all: [N]
        ce_all:     [N]    (per-sample cross-entropy vs GT)
        ent_all:    [N]    (per-sample predictive entropy)
        pmax_all:   [N]    (max class prob per sample)
    """
    weights = ResNet34_Weights.DEFAULT
    cls_model = resnet34(weights=weights).to(device).eval()
    preprocess = weights.transforms()

    probs_all, labels_all = [], []
    for images, labels in dataloader:
        images = images.to(device)
        images = F.interpolate(images, size=(224, 224), mode='bilinear', align_corners=False)
        images = preprocess.normalize(preprocess.resize(images)) if hasattr(preprocess, "normalize") else images

        logits = cls_model(images)             # [B,1000]
        probs  = torch.softmax(logits, dim=1)  # [B,1000]

        probs_all.append(probs.cpu())
        labels_all.append(labels)

    probs_all  = torch.cat(probs_all,  dim=0).to(device)   # [N,1000]
    labels_all = torch.cat(labels_all, dim=0).to(device)   # [N]
    ce_all  = -torch.log(torch.gather(probs_all, 1, labels_all.view(-1,1)).clamp_min(1e-12)).squeeze(1)
    ent_all = -(probs_all.clamp_min(1e-12) * probs_all.clamp_min(1e-12).log()).sum(dim=1)
    pmax_all = probs_all.max(dim=1).values
    return probs_all, labels_all, ce_all, ent_all, pmax_all

# ============================================================
# ============ Single-sided Partial OT (batched) =============
# ============================================================
def partial_single_sided_ot_cost_batched(
    x: torch.Tensor,                # [B, m, D]  (distilled subset; must fully ship -> equality)
    y: torch.Tensor,                # [B, n, D]  (real set; capacity -> inequality)
    epsilon: float = 10.0,
    n_iter: int = 100,
    capacity_scale: float = 1.2,    # >= 1.0
    uniform_marginals: bool = True,
    mu: Optional[torch.Tensor] = None,      # [B, m]  (sum=1)
    nu_cap: Optional[torch.Tensor] = None,  # [B, n] (sum = capacity_scale)

    # --- dummy row positive cost ---
    dummy_cost: Optional[float] = None,     # None -> median(C_real) * dummy_cost_factor
    dummy_cost_factor: float = 0.05,
) -> torch.Tensor:                          # -> [B] returns ONLY transport term <C,π_real>
    """
    Single-sided partial OT via dummy-source trick, with positive dummy cost δ.

    Enforce:
        sum_i π_{ij} <= (nu_cap)_j
        sum_j π_{ij}  =  mu_i

    Add one dummy source row with supply s = sum(nu_cap) - 1 (>=0) and positive cost δ to discourage idling.
    """
    assert capacity_scale >= 1.0, "capacity_scale must be >= 1."

    B, m, D = x.shape
    n = y.shape[1]
    device = x.device
    dtype  = x.dtype
    EPS = 1e-300

    C_real = torch.cdist(x, y, p=2).pow(2)  # [B, m, n]

    if dummy_cost is None:
        med = C_real.detach().median().item()
        delta = max(1e-12, float(med) * float(dummy_cost_factor))
    else:
        delta = float(dummy_cost)

    C_dummy = torch.full((B, 1, n), delta, device=device, dtype=dtype)  # [B,1,n]
    C_aug   = torch.cat([C_real, C_dummy], dim=1)                        # [B,m+1,n]

    K = torch.exp(-C_aug / epsilon).clamp_min(EPS)                       # [B,m+1,n]

    if uniform_marginals or (mu is None):
        mu = torch.full((B, m), 1.0 / m, device=device, dtype=dtype)
    else:
        mu = mu.to(device=device, dtype=dtype)

    if uniform_marginals or (nu_cap is None):
        nu_cap = torch.full((B, n), capacity_scale / n, device=device, dtype=dtype)
    else:
        nu_cap = nu_cap.to(device=device, dtype=dtype)
        totals = nu_cap.sum(dim=1)
        assert torch.all(totals >= 1.0 - 1e-8), "sum(nu_cap) must be >= 1."

    s = (nu_cap.sum(dim=1, keepdim=True) - 1.0).clamp_min(0.0)  # [B,1]
    mu_aug = torch.cat([mu, s], dim=1)                           # [B,m+1]
    nu = nu_cap                                                  # [B,n]

    u = torch.ones((B, m + 1), device=device, dtype=dtype)
    v = torch.ones((B, n), device=device, dtype=dtype)

    for _ in range(n_iter):
        Kv  = torch.bmm(K, v.unsqueeze(-1)).squeeze(-1).clamp_min(EPS)
        u   = mu_aug / Kv
        KTu = torch.bmm(K.transpose(1, 2), u.unsqueeze(-1)).squeeze(-1).clamp_min(EPS)
        v   = nu / KTu

    T_aug = u.unsqueeze(2) * K * v.unsqueeze(1)                  # [B,m+1,n]
    T_real = T_aug[:, :m, :]                                     # drop dummy row
    cost = torch.sum(T_real * C_real, dim=(1, 2))
    return cost


@torch.no_grad()
def batch_mean_var(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    B, K, D = x.shape
    mu = x.mean(dim=1)                           # [B, D]
    var = x.var(dim=1, unbiased=True) if K > 1 else torch.zeros_like(mu)
    return mu, var

@torch.no_grad()
def moment_diag_to_target_batched(
    mu_b: torch.Tensor,          # [B, D]
    var_b: torch.Tensor,         # [B, D]
    mu_t: torch.Tensor,          # [D]
    var_t: torch.Tensor,         # [D]
) -> torch.Tensor:               # [B]
    dm = (mu_b - mu_t.unsqueeze(0))**2
    term_mean = dm.sum(dim=1)                            # [B]
    vb = var_b.clamp_min(0.0)
    vt = var_t.clamp_min(0.0)
    term_cov = (vb + vt - 2.0 * torch.sqrt(vb * vt + 1e-20)).sum(dim=1)  # [B]
    return 1.25 * term_mean + term_cov

# ---------------------------
# === KL soft-target regularizer ===
# ---------------------------
@torch.no_grad()
def get_class_soft_target(probs_all: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    if temperature != 1.0:
        p = (probs_all.clamp_min(1e-12)) ** (1.0 / temperature)
        p = p / p.sum(dim=1, keepdim=True)
    else:
        p = probs_all
    p_bar = p.mean(dim=0)                         # [1000]
    p_bar = (p_bar / p_bar.sum()).clamp_min(1e-12)
    return p_bar

def kl_soft_target_batch(
    probs_candidates: torch.Tensor,   # [N_cand, 1000]
    sel_chunk: torch.LongTensor,      # [b, K]
    target_soft: torch.Tensor,        # [1000]
    temperature: float = 1.0,
) -> torch.Tensor:                    # [b]
    if temperature != 1.0:
        p = (probs_candidates.clamp_min(1e-12)) ** (1.0 / temperature)
        p = p / p.sum(dim=1, keepdim=True)
    else:
        p = probs_candidates

    p_sel = p[sel_chunk]                              # [b,K,1000]
    p_mean = p_sel.mean(dim=1).clamp_min(1e-12)       # [b,1000]
    p_mean = p_mean / p_mean.sum(dim=1, keepdim=True)

    t = target_soft.view(1, -1).expand_as(p_mean)     # [b,1000]
    kl = (t * (t.log() - p_mean.log())).sum(dim=1)    # KL(t || \hat p_S)
    return kl

# ---------------------------
# === On-the-fly target stats ===
# ---------------------------
@torch.no_grad()
def compute_target_stats_on_the_fly(
    features_full: torch.Tensor,  # [N, D], all real features of current class
    prefer_fp32: bool = True
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute μ, σ² from current class features on the fly (no disk I/O).
    """
    mu = features_full.mean(dim=0)                            # [D]
    var = (features_full.var(dim=0, unbiased=True)
           if features_full.size(0) > 1 else torch.zeros_like(mu))
    if prefer_fp32:
        mu = mu.float(); var = var.float()
    return mu, var

def compute_costs_in_batches(
    features: torch.Tensor,             # [N, D]
    all_sel: torch.LongTensor,          # [M, K]
    epsilon: float,
    sinkhorn_iter: int,
    batch_size: int,
    capacity_scale: float,
    moment_weight: float,
    mu_t: Optional[torch.Tensor],
    var_t: Optional[torch.Tensor],
    scale_match: Optional[float] = None,
    ce_all: Optional[torch.Tensor] = None,     # [N]
    ent_all: Optional[torch.Tensor] = None,    # [N]
    ce_weight: float = 0.0,
    ent_weight: float = 0.0,
    probs_all_cand: Optional[torch.Tensor] = None,  # [N_cand,1000]
    target_soft: Optional[torch.Tensor] = None,     # [1000]
    kl_weight: float = 0.0,
    kl_temperature: float = 1.0,
) -> torch.Tensor:                             # [M]
    M, K = all_sel.shape
    device = features.device
    costs = torch.empty(M, device=device)

    use_moment = (mu_t is not None) and (var_t is not None) and (moment_weight > 0.0)
    use_ce  = (ce_all is not None)  and (ce_weight  > 0.0)
    use_ent = (ent_all is not None) and (ent_weight > 0.0)
    use_kl  = (probs_all_cand is not None) and (target_soft is not None) and (kl_weight > 0.0)

    for start in range(0, M, batch_size):
        end = min(start + batch_size, M)
        sel_chunk = all_sel[start:end]              # [b, K]
        b = sel_chunk.size(0)
        x = features[sel_chunk]                     # [b, K, D]
        y = features.unsqueeze(0).expand(b, -1, -1) # [b, N, D]

        total = partial_single_sided_ot_cost_batched(
            x, y, epsilon=epsilon, n_iter=sinkhorn_iter, capacity_scale=capacity_scale
        )  # [b]

        if use_moment:
            mu_b, var_b = batch_mean_var(x)
            moment_diag = moment_diag_to_target_batched(mu_b, var_b, mu_t, var_t)
            if scale_match is not None:
                moment_diag = moment_diag * scale_match
            total = total + moment_weight * moment_diag

        if use_kl:
            kl_reg = kl_soft_target_batch(
                probs_all_cand, sel_chunk, target_soft, temperature=kl_temperature
            )  # [b]
            total = total + kl_weight * kl_reg

        if use_ce:
            ce_reg = ce_all[sel_chunk].mean(dim=1)          # [b]
            total  = total + ce_weight * ce_reg

        if use_ent:
            ent_reg = ent_all[sel_chunk].mean(dim=1)        # [b]
            total   = total + ent_weight * ent_reg

        costs[start:end] = total

    return costs

# ---------------------------
# Local swap refinement
# ---------------------------
def global_refine_with_chunks(
    features: torch.Tensor,                # [N, D]
    selected: List[int],
    epsilon: float = 10.0,
    sinkhorn_iter: int = 20,
    max_iter: int = 10,
    batch_size: int = 512,
    capacity_scale: float = 1.2,
    moment_weight: float = 0.0,
    mu_t: Optional[torch.Tensor] = None,
    var_t: Optional[torch.Tensor] = None,
    scale_match: Optional[torch.Tensor] = None,
    ce_all: Optional[torch.Tensor] = None,
    ent_all: Optional[torch.Tensor] = None,
    ce_weight: float = 0.0,
    ent_weight: float = 0.0,
    probs_all_cand: Optional[torch.Tensor] = None,  # [N_cand,1000]
    target_soft: Optional[torch.Tensor] = None,     # [1000]
    kl_weight: float = 0.0,
    kl_temperature: float = 1.0,
) -> List[int]:
    device = features.device
    N = features.size(0)
    sel = torch.tensor(selected, dtype=torch.long, device=device)
    K = sel.size(0)

    def objective_for_sel(sel_idx: torch.Tensor) -> float:
        x = features[sel_idx].unsqueeze(0)                    # [1,K,D]
        y = features.unsqueeze(0)                             # [1,N,D]
        total = partial_single_sided_ot_cost_batched(
            x, y, epsilon, sinkhorn_iter, capacity_scale
        )[0]

        if (mu_t is not None) and (var_t is not None) and (moment_weight > 0.0):
            mu_b, var_b = batch_mean_var(x)                   # [1,D],[1,D]
            moment_diag = moment_diag_to_target_batched(mu_b, var_b, mu_t, var_t)[0]
            if scale_match is not None:
                moment_diag = moment_diag * scale_match
            total = total + moment_weight * moment_diag

        if (probs_all_cand is not None) and (target_soft is not None) and (kl_weight > 0.0):
            kl_reg = kl_soft_target_batch(
                probs_all_cand, sel_idx.unsqueeze(0), target_soft, temperature=kl_temperature
            )[0]
            total = total + kl_weight * kl_reg

        if (ce_all is not None) and (ce_weight > 0.0):
            total = total + ce_weight * ce_all[sel_idx].mean()

        if (ent_all is not None) and (ent_weight > 0.0):
            total = total + ent_weight * ent_all[sel_idx].mean()

        return total.item()

    curr_cost = objective_for_sel(sel)

    for _ in range(max_iter):
        mask = torch.ones(N, dtype=torch.bool, device=device)
        mask[sel] = False
        unselected = torch.nonzero(mask, as_tuple=False).view(-1)

        rows = []
        for i in range(K):
            row = sel.clone().unsqueeze(0).repeat(unselected.numel(), 1)
            row[:, i] = unselected
            rows.append(row)
        all_sel = torch.cat(rows, dim=0)  # [M, K]

        costs = compute_costs_in_batches(
            features, all_sel, epsilon, sinkhorn_iter, batch_size, capacity_scale,
            moment_weight, mu_t, var_t, scale_match,
            ce_all=ce_all, ent_all=ent_all, ce_weight=ce_weight, ent_weight=ent_weight,
            probs_all_cand=probs_all_cand,
            target_soft=target_soft,
            kl_weight=kl_weight, kl_temperature=kl_temperature
        )
        min_idx = torch.argmin(costs).item()
        best_cost = costs[min_idx].item()

        if best_cost + 1e-8 < curr_cost:
            sel = all_sel[min_idx]
            curr_cost = best_cost
            print(f"[Refine] cost ↓ {best_cost:.6f}")
        else:
            break

    return sel.tolist()

# ---------------------------
# Greedy init
# ---------------------------
def greedy_sinkhorn_selection_batch(
    features: torch.Tensor,          # [N, D]
    n_select: int,
    epsilon: float = 10.0,
    sinkhorn_iter: int = 50,
    batch_size: int = 512,
    capacity_scale: float = 1.2,
    moment_weight: float = 0.0,
    mu_t: Optional[torch.Tensor] = None,
    var_t: Optional[torch.Tensor] = None,
    scale_match: Optional[float] = None,
    ce_all: Optional[torch.Tensor] = None,
    ent_all: Optional[torch.Tensor] = None,
    ce_weight: float = 0.0,
    ent_weight: float = 0.0,
    probs_all_cand: Optional[torch.Tensor] = None,  # [N_cand,1000]
    target_soft: Optional[torch.Tensor] = None,     # [1000]
    kl_weight: float = 0.0,
    kl_temperature: float = 1.0,
) -> List[int]:
    device = features.device
    N, D = features.shape
    y_full_1 = features.unsqueeze(0)  # [1, N, D]

    def single_cost(i: int) -> float:
        x = features[i:i+1].unsqueeze(0)  # [1,1,D]
        total = partial_single_sided_ot_cost_batched(
            x, y_full_1, epsilon, sinkhorn_iter, capacity_scale
        )[0]

        if (mu_t is not None) and (var_t is not None) and (moment_weight > 0.0):
            mu_b, var_b = batch_mean_var(x)  # [1,D],[1,D]
            moment_diag = moment_diag_to_target_batched(mu_b, var_b, mu_t, var_t)[0]
            if scale_match is not None:
                moment_diag = moment_diag * scale_match
            total = total + moment_weight * moment_diag

        if (probs_all_cand is not None) and (target_soft is not None) and (kl_weight > 0.0):
            kl_reg = kl_soft_target_batch(
                probs_all_cand, torch.tensor([[i]], device=device, dtype=torch.long),
                target_soft, temperature=kl_temperature
            )[0]
            total = total + kl_weight * kl_reg

        if (ce_all is not None) and (ce_weight > 0.0):
            total = total + ce_weight * ce_all[i]

        if (ent_all is not None) and (ent_weight > 0.0):
            total = total + ent_weight * ent_all[i]

        return total.item()

    # best single start
    best_cost = float("inf")
    best_idx = 0
    for i in range(N):
        c = single_cost(i)
        if c < best_cost:
            best_cost = c
            best_idx = i
    selected = [best_idx]

    # iteratively add
    for _ in tqdm(range(1, n_select), desc="Greedy Sinkhorn Select"):
        mask = torch.ones(N, dtype=torch.bool, device=device)
        mask[selected] = False
        candidates = torch.nonzero(mask, as_tuple=False).view(-1)  # [B]
        B = candidates.size(0)

        best_cand = None
        best_cost = float("inf")
        sel_feat = features[selected]  # [k, D]

        for start in range(0, B, batch_size):
            end = min(start + batch_size, B)
            cand_chunk = candidates[start:end]              # [b]
            b = cand_chunk.size(0)

            x_base = sel_feat.unsqueeze(0).expand(b, -1, -1)   # [b, k, D]
            x_new  = features[cand_chunk].unsqueeze(1)         # [b, 1, D]
            x_batch = torch.cat([x_base, x_new], dim=1)        # [b, k+1, D]
            y_batch = features.unsqueeze(0).expand(b, -1, -1)  # [b, N, D]

            ot_chunk = partial_single_sided_ot_cost_batched(
                x_batch, y_batch, epsilon, sinkhorn_iter, capacity_scale
            )  # [b]

            total = ot_chunk.clone()

            if (mu_t is not None) and (var_t is not None) and (moment_weight > 0.0):
                mu_b, var_b = batch_mean_var(x_batch)           # [b,D],[b,D]
                moment_diag = moment_diag_to_target_batched(mu_b, var_b, mu_t, var_t)  # [b]
                if scale_match is not None:
                    moment_diag = moment_diag * scale_match
                total = total + moment_weight * moment_diag

            if (probs_all_cand is not None) and (target_soft is not None) and (kl_weight > 0.0):
                sel_prefix = torch.tensor(selected, device=device).unsqueeze(0).expand(b, -1)
                sel_mat = torch.cat([sel_prefix, cand_chunk.unsqueeze(1)], dim=1)  # [b,k+1]
                kl_reg = kl_soft_target_batch(
                    probs_all_cand, sel_mat, target_soft, temperature=kl_temperature
                )
                total = total + kl_weight * kl_reg

            if (ce_all is not None) and (ce_weight > 0.0):
                total = total + ce_weight * ce_all[cand_chunk]

            if (ent_all is not None) and (ent_weight > 0.0):
                total = total + ent_weight * ent_all[cand_chunk]

            min_val, idx_local = total.min(dim=0)
            if min_val.item() < best_cost:
                best_cost = min_val.item()
                best_cand = cand_chunk[idx_local].item()

        selected.append(best_cand)

    return selected

# ---------------------------
# Save json paths
# ---------------------------
def save_selected_images_as_json(all_selected_paths: List[str], save_path: str):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w") as f:
        json.dump(all_selected_paths, f, indent=4)
    print(f"✅ Saved {len(all_selected_paths)} paths to {save_path}")

# ---------------------------
# Worker (per process/GPU)
# ---------------------------
def spawn_entry(rank: int, class_shards, shard_outs, cfg):
    worker_run(
        rank=rank,
        world_size=len(class_shards),
        class_ids=class_shards[rank],
        json_shard_out=shard_outs[rank],
        cfg=cfg,
    )

def worker_run(
    rank: int,
    world_size: int,
    class_ids: List[int],
    json_shard_out: str,
    cfg: dict,
):
    torch.set_num_threads(1)
    if torch.cuda.is_available():
        torch.cuda.set_device(rank)
        device = f"cuda:{rank}"
    else:
        device = "cpu"

    use_fp16      = cfg["use_fp16"]
    epsilon       = cfg["epsilon"]
    sinkhorn_iter = cfg["sinkhorn_iter"]
    capacity_scale= cfg["capacity_scale"]
    n_select      = cfg["n_select_per_class"]
    batch_size    = cfg["batch_size"]
    refine_iters  = cfg["refine_iters"]

    moment_weight      = cfg.get("moment_weight", 0.0)
    moment_scale_match = cfg.get("moment_scale_match", None)

    ce_weight   = cfg.get("ce_weight", 0.0)
    ent_weight  = cfg.get("ent_weight", 0.0)

    kl_weight        = cfg.get("kl_weight", 0.0)
    kl_temperature   = cfg.get("kl_temperature", 1.0)

    min_pmax_for_candidates = cfg.get("min_pmax_for_candidates", 0.0)

    shard_paths: List[str] = []

    for class_id in class_ids:
        print(f".[rank {rank}] class {class_id}")
        loader, wnid = get_by_id(class_id=class_id)

        subset_indices = loader.dataset.indices
        image_paths_full = [loader.dataset.dataset.samples[i][0] for i in subset_indices]

        probs_all, labels_all, ce_all, ent_all, pmax_all = get_resnet_probs_and_scores(loader, device)

        feats = get_features(loader).to(device)  # [N, D]
        if use_fp16:
            feats = feats.half()
        if feats.dim() > 2:
            feats = feats.flatten(start_dim=1)
        elif feats.dim() == 1:
            feats = feats.unsqueeze(0)
        feats = feats.contiguous()

        Np, Nf = probs_all.size(0), feats.size(0)
        Nmin = min(Np, Nf, len(image_paths_full))
        if (Np != Nmin) or (Nf != Nmin):
            print(f"[rank {rank}] class {class_id} length mismatch probs={Np}, feats={Nf}, paths={len(image_paths_full)} -> truncate {Nmin}")
            probs_all = probs_all[:Nmin]; labels_all = labels_all[:Nmin]
            ce_all    = ce_all[:Nmin];   ent_all    = ent_all[:Nmin]; pmax_all = pmax_all[:Nmin]
            feats     = feats[:Nmin]
            image_paths_full = image_paths_full[:Nmin]

        candidate_mask = (pmax_all >= min_pmax_for_candidates)
        if candidate_mask.sum().item() < n_select:
            candidate_mask = torch.ones_like(candidate_mask, dtype=torch.bool, device=device)

        cand_idx = torch.nonzero(candidate_mask, as_tuple=False).view(-1)
        feats_c  = feats[cand_idx]
        ce_c     = ce_all[cand_idx] if ce_weight > 0 else None
        ent_c    = ent_all[cand_idx] if ent_weight > 0 else None
        probs_c  = probs_all[cand_idx]                                  # [N_cand, 1000]
        paths_c  = [image_paths_full[i] for i in cand_idx.tolist()]

        target_soft = get_class_soft_target(probs_all, temperature=kl_temperature) if (kl_weight > 0) else None

        mu_t, var_t = (None, None)
        if moment_weight > 0.0:
            mu_t, var_t = compute_target_stats_on_the_fly(features_full=feats, prefer_fp32=True)

        if feats_c.size(0) < n_select:
            print(f"[rank {rank}] class {class_id}: not enough candidates ({feats_c.size(0)}<{n_select}), select all.")
            refined_idx_local = list(range(min(feats_c.size(0), n_select)))
        else:
            init_idx_local = greedy_sinkhorn_selection_batch(
                feats_c, n_select=n_select,
                epsilon=epsilon, sinkhorn_iter=sinkhorn_iter,
                batch_size=batch_size, capacity_scale=capacity_scale,
                moment_weight=moment_weight, mu_t=mu_t, var_t=var_t, scale_match=moment_scale_match,
                ce_all=ce_c, ent_all=ent_c, ce_weight=ce_weight, ent_weight=ent_weight,
                probs_all_cand=probs_c, target_soft=target_soft,
                kl_weight=kl_weight, kl_temperature=kl_temperature
            )
            refined_idx_local = init_idx_local
            refined_idx_local = global_refine_with_chunks(
                feats_c, init_idx_local, epsilon=epsilon, sinkhorn_iter=sinkhorn_iter,
                max_iter=refine_iters, batch_size=batch_size, capacity_scale=capacity_scale,
                moment_weight=moment_weight, mu_t=mu_t, var_t=var_t, scale_match=moment_scale_match,
                ce_all=ce_c, ent_all=ent_c, ce_weight=ce_weight, ent_weight=ent_weight,
                probs_all_cand=probs_c, target_soft=target_soft,
                kl_weight=kl_weight, kl_temperature=kl_temperature
            )

        shard_paths.extend([paths_c[idx] for idx in refined_idx_local])

    save_selected_images_as_json(shard_paths, json_shard_out)

# ---------------------------
# Main (spawn)
# ---------------------------
def shard_list(xs: List[int], k: int) -> List[List[int]]:
    return [xs[i::k] for i in range(k)]

def main():
    n_gpus = torch.cuda.device_count()
    world_size = max(1, n_gpus)
    print(f"[Main] world_size = {world_size}")

    cfg = dict(
        use_fp16=False,
        # --- OT params (single-sided partial) ---
        epsilon=20.0,
        sinkhorn_iter=20,
        capacity_scale=1.05,
        n_select_per_class=10,
        batch_size=256,
        refine_iters=30,
        moment_weight=1.0,
        moment_scale_match=6.0,  
        ce_weight=10,
        ent_weight=0.0,
        kl_weight=0.0,
        kl_temperature=1.0,
        min_pmax_for_candidates=0.0,
        json_out="test.json",
    )

    all_classes = list(range(1000))
    class_shards = shard_list(all_classes, world_size)

    shard_outs = [f"{cfg['json_out']}.rank{r}.json" for r in range(world_size)]
    os.makedirs(os.path.dirname(cfg["json_out"]), exist_ok=True)

    if world_size == 1:
        worker_run(0, 1, class_shards[0], shard_outs[0], cfg)
    else:
        mp.spawn(
            spawn_entry,
            args=(class_shards, shard_outs, cfg),
            nprocs=world_size,
            daemon=False,
            join=True,
        )

    all_paths: List[str] = []
    for p in shard_outs:
        if os.path.exists(p):
            with open(p, "r") as f:
                all_paths.extend(json.load(f))
    save_selected_images_as_json(all_paths, cfg["json_out"])

    for p in shard_outs:
        try:
            os.remove(p)
        except Exception:
            pass

if __name__ == "__main__":
    main()
