# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Independent word table; packing and two-way MoT remain native."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch import nn

from cosmos_framework.model.generator.physword.config import PhysWordConfig

if TYPE_CHECKING:
    from cosmos_framework.data.generator.sequence_packing.sequence import PackedSequence, SequencePlan

ATOMIC_ACTIONS = (
    "approach",
    "move",
    "pick",
    "place",
    "flip",
    "rotate",
    "toggle",
    "open",
    "close",
    "push",
    "pull",
    "insert",
    "press",
    "click",
    "strike",
    "handover",
)
PW_TOKENS = ("<pw_start>",) + tuple(f"<pw_{word}_v>" for word in ATOMIC_ACTIONS)


def manual_word_ids(words: list[str], config: PhysWordConfig) -> list[int]:
    if not words:
        raise ValueError("M1 requires a nonempty manual Physical Word sequence")
    try:
        return [config.token_id_base] + [config.token_id_base + 1 + ATOMIC_ACTIONS.index(word) for word in words]
    except ValueError as exc:
        raise ValueError(f"Expected atomic action names from {ATOMIC_ACTIONS}; got {words}") from exc


def append_manual_words(
    text_ids: list[list[int]],
    plans: list[SequencePlan],
    words: list[list[str]] | None,
    config: PhysWordConfig,
) -> list[list[int]]:
    """Append before native EOS/BOV, before offsets and positions are finalized."""
    if not config.enabled:
        return text_ids
    if words is None or len(words) != len(plans) or len(text_ids) != len(plans):
        raise ValueError("Enabled M1 needs one caption and one manual PW sequence per sample")
    for plan in plans:
        if (
            not plan.has_text
            or not plan.has_vision
            or plan.text_view_ids is not None
            or plan.has_action
            or plan.has_sound
            or plan.has_lidar
            or plan.has_radar
        ):
            raise ValueError("M1 supports only RGB + one caption per sample, without action or other modalities")
    if any(token < 0 or token >= config.token_id_base for sample in text_ids for token in sample):
        raise ValueError("Base text IDs collide with the private Physical Word namespace")
    return [list(ids) + manual_word_ids(ws, config) for ids, ws in zip(text_ids, words, strict=True)]


class PhysicalWordEmbedding(nn.Module):
    """Seventeen rows; no LM-head resizing, extra Transformer or final-z injection."""

    def __init__(self, hidden_size: int, base_vocab_size: int, config: PhysWordConfig):
        super().__init__()
        if config.token_id_base < base_vocab_size:
            raise ValueError("Physical Word IDs must be outside the base embedding vocabulary")
        self.config = config
        with torch.random.fork_rng(devices=[]):
            self.embedding = nn.Embedding(len(PW_TOKENS), hidden_size)
            self.word_head = nn.Linear(hidden_size, len(ATOMIC_ACTIONS), bias=False)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        if self.embedding.weight.is_meta:
            return
        generator = torch.Generator(device=self.embedding.weight.device).manual_seed(self.config.embedding_seed)
        with torch.no_grad():
            self.embedding.weight.normal_(mean=0.0, std=0.02, generator=generator)
            self.word_head.weight.copy_(self.embedding.weight[1:])

    def forward(self, ids: torch.Tensor, base_embedding: nn.Embedding) -> torch.Tensor:
        pw_indexes = ids - self.config.token_id_base
        is_pw = (pw_indexes >= 0) & (pw_indexes < len(PW_TOKENS))
        embeddings = base_embedding(ids.masked_fill(is_pw, 0))
        pw_embeddings = self.embedding(pw_indexes.clamp(0, len(PW_TOKENS) - 1))
        result = torch.where(is_pw.unsqueeze(-1), pw_embeddings, embeddings)
        # FP32 optimizer masters + BF16 compute: native RoPE derives its dtype from
        # embeddings. Keep Q/K/V dtypes aligned without changing attention math.
        if result.is_cuda and torch.is_autocast_enabled("cuda"):
            result = result.to(torch.get_autocast_dtype("cuda"))
        return result


def physical_word_trace(packed: PackedSequence, config: PhysWordConfig) -> list[dict]:
    """Global buffer indexes and sample boundaries, for host-side diagnostics."""
    result = []
    ids = packed.text_ids.detach().cpu().tolist()
    positions = packed.text_indexes.detach().cpu().tolist()
    offsets = [0]
    for length in packed.sample_lens:
        offsets.append(offsets[-1] + length)
    for sample, (start, end) in enumerate(zip(offsets[:-1], offsets[1:], strict=True)):
        tokens = []
        for token_id, global_index in zip(ids, positions, strict=True):
            pw_index = token_id - config.token_id_base
            if start <= global_index < end and 0 <= pw_index < len(PW_TOKENS):
                tokens.append(
                    {
                        "token": PW_TOKENS[pw_index],
                        "token_id": token_id,
                        "global_index": global_index,
                        "sample_local_index": global_index - start,
                    }
                )
        result.append({"sample_id": sample, "sample_start": start, "sample_end": end, "tokens": tokens})
    return result
