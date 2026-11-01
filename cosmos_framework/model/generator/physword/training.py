# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Opt-in single-device A/B/C training; reuses the M2 prefix, AR and native joint forward."""

import math
from contextlib import contextmanager
from dataclasses import dataclass

import torch

from cosmos_framework.inference.physword import generate_physical_words, repack_physical_words
from cosmos_framework.model.generator.algorithm.loss.flow_matching import compute_flow_matching_loss
from cosmos_framework.model.generator.physword.losses import encode_future_targets, latent_cosine, word_ce
from cosmos_framework.model.generator.physword.multimodal import (
    RGBHistory,
    build_reasoner_prefix,
    embed_reasoner_condition,
)
from cosmos_framework.model.generator.utils.data_and_condition import GenerationDataClean


@dataclass(frozen=True)
class LossConfig:
    lambda_word: float = 1.0
    lambda_latent: float = 1.0
    video_loss_scale: float = 10.0  # Existing Edge RF scale, applied exactly ONCE below.
    normalize_loss_by_active: bool = False

    def __post_init__(self):
        if any(not math.isfinite(x) or x < 0 for x in (self.lambda_word, self.lambda_latent, self.video_loss_scale)):
            raise ValueError("Loss coefficients must be finite and nonnegative")


@dataclass(frozen=True)
class PhysWordBatch:
    history: RGBHistory
    instruction: str
    future_rgb: torch.Tensor  # [16,3,H,W] uint8; TARGET ONLY
    word_labels: torch.Tensor  # [1,4] int64; invalid entries may be -100
    valid_word: torch.Tensor  # [1,4] bool
    valid_latent: torch.Tensor  # [1,4] bool

    def __post_init__(self):
        if self.future_rgb.shape != (16, *self.history.frames.shape[1:]) or self.future_rgb.dtype != torch.uint8:
            raise ValueError("Training target must contain exactly sixteen future RGB frames")
        if self.word_labels.shape != (1, 4) or self.word_labels.dtype != torch.long:
            raise ValueError("M3 supports B=1, K=4 int64 word labels")
        for mask in (self.valid_word, self.valid_latent):
            if mask.shape != (1, 4) or mask.dtype != torch.bool:
                raise ValueError("Validity masks must be [1,4] bool")
        valid_labels = self.word_labels[self.valid_word.to(self.word_labels.device)]
        if bool(((valid_labels < 0) | (valid_labels >= 16)).any()):
            raise ValueError("Valid word labels must be in [0,16)")


def forward_teacher_words(network, prefix, labels, valid_word):
    """A: start,GT(w1),GT(w2),GT(w3); current target never occupies its prediction row."""
    device = network.language_model.model.embed_tokens.weight.device
    base = network.physword.config.token_id_base
    previous = labels[0, :3].to(device)
    valid_previous = valid_word[0, :3].to(device)
    # No fabricated GT for missing labels: reuse neutral start embedding as missing feedback.
    previous_ids = torch.where(valid_previous, base + 1 + previous, base).long()
    base_embed = network.language_model.model.embed_tokens
    inputs = torch.cat(
        (embed_reasoner_condition(network, prefix), network.physword(previous_ids, base_embed).unsqueeze(0)), 1
    )
    positions = torch.cat(
        (prefix.position_ids, torch.arange(prefix.next_position, prefix.next_position + 3).expand(3, -1)), 1
    )
    hidden = network.language_model.model.reasoner_forward(
        input_ids=None, inputs_embeds=inputs, cache=None, position_ids=positions.unsqueeze(1).to(device)
    )
    z = hidden[:, prefix.prediction_positions.to(device)]
    return {
        "hidden": z,
        "logits": network.physword.word_head(z),
        "input_ids": torch.cat((prefix.input_ids, previous_ids.cpu())),
        "position_ids": positions,
    }


@contextmanager
def rollout_eval(network):
    # Restore every submodule's previous flag; frozen vision must stay eval after rollout.
    states = [(module, module.training) for module in network.modules()]
    try:
        network.eval()
        yield
    finally:
        for module, mode in states:
            module.training = mode


class PhysWordTrainingStep:
    """A reusable differentiable step, separate from the default OmniMoT trainer/recipes.

    Construct heads on net, select exact phase groups, then call this step. An external
    caller owns backward/optimizer. Only the tiny probe executes optimizer updates here.
    encode_video adapts the already loaded VAE with its checkpoint normalization.
    """

    def __init__(
        self, network, processor, vae, encode_video, rectified_flow, special_tokens, phase, losses=LossConfig()
    ):
        if phase not in ("A", "B") or network.physword is None or network.parallel_dims is not None:
            raise ValueError("M3 requires Phase A/B, enabled PW and an unsharded network")
        if getattr(getattr(network, "_physword_phase_config", None), "phase", None) != phase:
            raise ValueError("Build matching exact phase optimizer groups before the training step")
        self.phase_config = network._physword_phase_config
        self.net, self.processor = network, processor
        self.vae = vae.eval().requires_grad_(False)
        self.encode_video, self.flow = encode_video, rectified_flow
        self.special_tokens, self.phase, self.config = special_tokens, phase, losses

    def __call__(self, batch: PhysWordBatch, *, timesteps=None, noise=None):
        net, cfg = self.net, self.config
        if net._physword_phase_config != self.phase_config:
            raise ValueError("Phase changed; rebuild the training step and optimizer")
        net.train()
        net.language_model.visual.eval().requires_grad_(False)
        self.vae.eval()
        device = net.language_model.model.embed_tokens.weight.device
        if device.type != "cuda":
            raise ValueError("Native M3 MoT training requires CUDA")
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            prefix = build_reasoner_prefix(net, self.processor, batch.history, batch.instruction)
            teacher = forward_teacher_words(net, prefix, batch.word_labels, batch.valid_word)
            targets = encode_future_targets(net, self.processor, batch.future_rgb, batch.valid_latent)
            loss_word, count_word = word_ce(teacher["logits"], batch.word_labels, batch.valid_word)
            loss_latent, count_latent = latent_cosine(teacher["hidden"], targets, batch.valid_latent)
            prediction = packed = velocity = None
            loss_video = teacher["hidden"][:, :0].float().sum()
            raw_video = loss_video
            count_video = 0
            if self.phase == "B":
                # B accepts the ORIGINAL observed prefix only; teacher feedback is never reused.
                with rollout_eval(net):
                    prediction = generate_physical_words(net, prefix)
                with torch.no_grad():
                    observed = batch.history.frames.permute(1, 0, 2, 3).unsqueeze(0).float() / 127.5 - 1
                    future = batch.future_rgb.permute(1, 0, 2, 3).unsqueeze(0).float() / 127.5 - 1
                    history_latents = self.encode_video(observed).detach().float()
                    full = self.encode_video(torch.cat((observed, future), 2)).detach().float()
                    if history_latents.shape[:3] != (1, 48, 2) or full.shape[:3] != (1, 48, 6):
                        raise ValueError("Expected native Wan 5/21-frame -> 2/6-latent encoding")
                    # Even though VAE is causal, supply condition from HISTORY-ONLY encoding.
                    clean = torch.cat((history_latents, full[:, :, 2:]), 2)
                    if timesteps is None:
                        u = self.flow.sample_train_time(1)
                        timesteps = self.flow.get_discrete_timestamp(u, {"device": device, "dtype": torch.float32})
                    timesteps = timesteps.to(device=device, dtype=torch.float32).reshape(1)
                    sigma = self.flow.get_sigmas(timesteps, {"device": device, "dtype": torch.float32})
                    noise = torch.randn_like(clean) if noise is None else noise.to(clean)
                    if noise.shape != clean.shape or not bool(torch.isfinite(noise).all()):
                        raise ValueError("Noise must be finite and match full clean latent shape")
                    xt, vt = self.flow.get_interpolation([noise], [clean], [sigma.reshape(1, 1, 1, 1, 1)])
                    xt = torch.cat((history_latents, xt[0][:, :, 2:]), 2)
                    gen_data = GenerationDataClean(
                        batch_size=1, is_image_batch=False, x0_tokens_vision=[xt], fps_vision=torch.tensor([10.0])
                    )
                # C: native M2 repacking and native two-way attention, without Oracle PW.
                packed = repack_physical_words(
                    net, prediction.condition, gen_data, timesteps.cpu(), self.special_tokens
                )
                packed.to_cuda()
                out = net(packed)
                velocity = out["preds_vision"]
                raw_video, _ = compute_flow_matching_loss(
                    pred=[v.float() for v in velocity],
                    target=vt,
                    condition_mask=packed.vision.condition_mask,
                    timesteps=timesteps.reshape(1, 1),
                    has_valid_tokens=True,
                    rectified_flow=self.flow,
                    tensor_kwargs_fp32={"device": device, "dtype": torch.float32},
                    normalize_by_active=cfg.normalize_loss_by_active,
                )
                loss_video = raw_video * cfg.video_loss_scale  # Native RF helper returns UNscaled loss.
                count_video = 4 * clean.shape[1] * clean.shape[3] * clean.shape[4]
            total = loss_video + cfg.lambda_word * loss_word + cfg.lambda_latent * loss_latent
        return {
            "losses": {
                "total": total,
                "video": loss_video,
                "video_raw": raw_video,
                "word": loss_word,
                "latent": loss_latent,
            },
            "counts": {
                "word": count_word,
                "latent": count_latent,
                "word_samples": int(count_word > 0),
                "latent_samples": int(count_latent > 0),
                "video_elements": count_video,
                "video_samples": int(self.phase == "B"),
            },
            "teacher": teacher,
            "targets": targets,
            "prediction": prediction,
            "packed": packed,
            "velocity": velocity,
            "prefix": prefix,
        }
