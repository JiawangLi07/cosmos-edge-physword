# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Future-only frozen visual targets and valid-masked, FP32 PW supervision."""

import torch
from torch.nn import functional as F


@torch.no_grad()
def encode_future_targets(network, processor, future_rgb, valid):
    """Encode original clip frames 8/12/16/20 independently; never mutate a condition."""
    vision = network.language_model.visual
    vision.eval().requires_grad_(False)
    device = next(vision.parameters()).device
    targets = torch.zeros((1, 4, network.hidden_size), device=device, dtype=torch.float32)
    for block, frame in enumerate((3, 7, 11, 15)):  # future container begins at original frame 5
        if not bool(valid[0, block]):
            continue
        data = processor.image_processor(images=[future_rgb[frame].cpu()], return_tensors="pt")
        spatial = data["spatial_shapes"].to(device)
        grid = torch.cat((torch.ones((1, 1), dtype=spatial.dtype, device=device), spatial), dim=1)
        features = vision.get_image_features(data["pixel_values"].to(device), grid)
        targets[0, block] = features.float().mean(0)
    return targets.detach()


def word_ce(logits, labels, valid):
    if (
        logits.ndim != labels.ndim + 1
        or logits.shape[:-1] != labels.shape
        or valid.shape != labels.shape
        or valid.dtype != torch.bool
    ):
        raise ValueError("PW CE requires [B,K,V] logits and matching label/mask tensors")
    labels, valid = labels.to(logits.device), valid.to(logits.device)
    if bool(((labels[valid] < 0) | (labels[valid] >= logits.shape[-1])).any()):
        raise ValueError("Valid PW labels must be configured class IDs")
    selected = logits.float()[valid]
    if not bool(torch.isfinite(selected).all()):
        raise FloatingPointError("Nonfinite valid word logits")
    count = int(valid.sum())
    return (F.cross_entropy(selected, labels[valid]) if count else selected.sum()), count


def latent_cosine(hidden, targets, valid, eps=1e-8):
    if hidden.shape != targets.shape or valid.shape != hidden.shape[:-1] or valid.dtype != torch.bool:
        raise ValueError("Cosine requires matching [B,4,D] states/targets and [B,4] mask")
    targets = targets.detach().to(device=hidden.device, dtype=torch.float32)
    mask = valid.to(hidden.device) & torch.isfinite(targets).all(-1) & (targets.norm(dim=-1) > eps)
    z, g = hidden.float()[mask], targets[mask]
    if not bool(torch.isfinite(z).all()):
        raise FloatingPointError("Nonfinite valid PW hidden states")
    count = int(mask.sum())
    # Select BEFORE computing cosine: multiplying NaN targets by zero would still leak NaNs.
    loss = (1 - F.cosine_similarity(z, g, dim=-1, eps=eps)).mean() if count else z.sum()
    return loss, count
