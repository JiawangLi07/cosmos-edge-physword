# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""B=1, four-step PW decoding and fresh joint MoT repacking. No training or KV handoff."""

from dataclasses import dataclass

import torch

from cosmos_framework.data.generator.sequence_packing.packers import pack_input_sequence
from cosmos_framework.data.generator.sequence_packing.sequence import SequencePlan
from cosmos_framework.model.generator.mot.unified_mot import ReasonerKVCache
from cosmos_framework.model.generator.physword.multimodal import (
    ReasonerCondition,
    RGBHistory,
    build_reasoner_prefix,
    embed_reasoner_condition,
    prepare_history_video_data,
)


@dataclass
class PhysWordPrediction:
    token_ids: torch.Tensor  # [4], excludes start
    class_ids: torch.Tensor  # [4]
    prediction_positions: torch.Tensor  # [4], UND-local
    token_positions: torch.Tensor  # [4], UND-local
    prediction_hidden_states: torch.Tensor  # [1,4,D]
    condition: ReasonerCondition  # context + start + four predicted tokens, no EOS/BOV yet
    cache_length: int  # includes the final sampled token (explicit flush)
    debug_cache: ReasonerKVCache | None = None  # diagnostic only, NEVER passed to GEN


@torch.no_grad()
def generate_physical_words(network, prefix: ReasonerCondition, *, capture_cache: bool = False) -> PhysWordPrediction:
    if network.training or network.language_model.training:
        raise ValueError("PW greedy inference requires eval mode")
    if network.physword is None or prefix.word_class_ids or prefix.boundaries:
        raise ValueError("Expected an enabled PW model and a fresh shared context + start prefix")
    if network.parallel_dims is not None:
        raise ValueError("M2 PW decoding supports unsharded B=1 only")
    model = network.language_model.model
    device = model.embed_tokens.weight.device
    cache = ReasonerKVCache.empty(len(model.layers))
    hidden = model.reasoner_forward(
        input_ids=None,
        inputs_embeds=embed_reasoner_condition(network, prefix),
        cache=cache,
        position_ids=prefix.position_ids.unsqueeze(1).to(device),
    )
    classes = []
    states = []
    for step in range(4):
        z = hidden[:, -1:, :]
        states.append(z)
        cls = int(network.physword.word_head(z[:, 0]).argmax(-1).item())
        classes.append(cls)
        token_id = network.physword.config.token_id_base + 1 + cls
        embeds = network.physword(torch.tensor([[token_id]], device=device), model.embed_tokens)
        positions = torch.full((3, 1, 1), prefix.next_position + step, device=device, dtype=torch.long)
        # Also consume w4, so AR/cache covers exactly the token-repacked PW prefix.
        # Its output is not used to predict a fifth word.
        hidden = model.reasoner_forward(input_ids=None, inputs_embeds=embeds, cache=cache, position_ids=positions)
    condition = prefix.with_words(tuple(classes), network.physword.config.token_id_base)
    return PhysWordPrediction(
        token_ids=condition.input_ids[condition.token_positions],
        class_ids=torch.tensor(classes),
        prediction_positions=condition.prediction_positions,
        token_positions=condition.token_positions,
        prediction_hidden_states=torch.cat(states, dim=1),
        condition=condition,
        cache_length=cache.seq_len,
        debug_cache=cache if capture_cache else None,
    )


def repack_physical_words(
    network,
    condition: ReasonerCondition,
    gen_data,
    timesteps: torch.Tensor,
    special_tokens: dict[str, int],
    *,
    rgb_prefix_only: bool = False,
):
    if network.physword is None or network.parallel_dims is not None:
        raise ValueError("M2 requires enabled PW and an unsharded B=1 model")
    base = network.physword.config.token_id_base
    if condition.vocabulary_hash and condition.vocabulary_hash != network.physword.config.vocabulary_hash:
        raise ValueError("Repack vocabulary fingerprint differs from the model")
    if condition.rgb_prefix_only != rgb_prefix_only:
        raise ValueError("PW and matched RGB-prefix modes must be explicit and consistent")
    if not rgb_prefix_only and int(condition.input_ids[condition.context_length]) != base:
        raise ValueError("Prefix and network must share the M1 PW namespace")
    expected = torch.tensor([base + 1 + c for c in condition.word_class_ids])
    if not rgb_prefix_only and not torch.equal(condition.input_ids[condition.context_length + 1 :], expected):
        raise ValueError("Repack requires exactly the four predicted private PW IDs, without existing boundaries")
    if gen_data.batch_size != 1 or gen_data.x0_tokens_vision is None or len(gen_data.x0_tokens_vision) != 1:
        raise ValueError("M2 repacking requires B=1 video latents")
    if timesteps.shape != (1,) or not bool(torch.isfinite(timesteps).all()):
        raise ValueError("M2 repacking requires one finite diffusion timestep")
    if gen_data.fps_vision is None or gen_data.fps_vision.numel() != 1 or float(gen_data.fps_vision[0]) != 10.0:
        raise ValueError("GEN and PW must use the same 10Hz model clock")
    if any(
        getattr(gen_data, name) is not None
        for name in ("x0_tokens_action", "x0_tokens_sound", "x0_tokens_lidar", "x0_tokens_radar")
    ):
        raise ValueError("M2 does not accept action, audio, tactile or other sensor conditioning")
    latent = gen_data.x0_tokens_vision[0]
    if latent.ndim != 5 or latent.shape[:3] != (1, 48, 6):
        raise ValueError("5 observed + 16 future frames must give six causal VAE latents")
    cfg = network.config
    return pack_input_sequence(
        sequence_plans=[SequencePlan(has_text=True, has_vision=True, condition_frame_indexes_vision=[0, 1])],
        input_text_indexes=[[]],
        gen_data_clean=gen_data,
        input_timesteps=timesteps.cpu(),
        special_tokens=special_tokens,
        latent_patch_size=cfg.latent_patch_size,
        physword=cfg.physword,
        reasoner_condition=condition.with_boundaries(special_tokens),
        unified_3d_mrope_temporal_modality_margin=15000,
        enable_fps_modulation=True,
        base_fps=24.0,
        temporal_compression_factor=4,
    )


@torch.no_grad()
def predict_physical_word_video(
    network,
    processor,
    history: RGBHistory,
    instruction: str,
    gen_data,
    timesteps: torch.Tensor,
    special_tokens: dict[str, int],
):
    """One-step video velocity entry: history→native RGB prefix→PW AR→joint forward."""
    prefix = build_reasoner_prefix(network, processor, history, instruction)
    prediction = generate_physical_words(network, prefix)
    packed = repack_physical_words(network, prediction.condition, gen_data, timesteps, special_tokens)
    packed.to_cuda()
    return prediction, packed, network(packed)


@torch.no_grad()
def predict_physical_word_video_from_rgb(
    network,
    processor,
    history: RGBHistory,
    instruction: str,
    encode_history,
    future_noise: torch.Tensor,
    timesteps: torch.Tensor,
    special_tokens: dict[str, int],
):
    data = prepare_history_video_data(history, encode_history, future_noise)
    return predict_physical_word_video(network, processor, history, instruction, data, timesteps, special_tokens)
