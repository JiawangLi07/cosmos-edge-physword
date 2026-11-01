# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Shared RGB-conditioned UND inputs. Future supervision never belongs in these inputs."""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, replace

import torch

from cosmos_framework.model.generator.reasoner.nemotron_3_dense_vl.reasoner_multimodal_utils import (
    prepare_multimodal_reasoner_inputs,
)
from cosmos_framework.utils.generator.source_video_timing import build_source_video_timing


@dataclass(frozen=True)
class PWTiming:
    source_fps: float = 10.0
    model_fps: float = 10.0
    history_frames: int = 5
    future_frames: int = 16
    stride: int = 4
    blocks: int = 4
    start_seconds: float = 0.0

    def __post_init__(self):
        if (self.history_frames, self.future_frames, self.stride, self.blocks) != (5, 16, 4, 4):
            raise ValueError("M2 supports history=5, future=16, stride=4, K=4 only")
        if self.model_fps != 10 or not math.isfinite(self.source_fps) or self.source_fps <= 0:
            raise ValueError("M2 requires model_fps=10 and positive finite source_fps")
        if not math.isfinite(self.start_seconds):
            raise ValueError("start_seconds must be finite")

    @property
    def target_frames(self) -> tuple[int, ...]:
        return tuple(self.history_frames - 1 + self.stride * k for k in range(1, self.blocks + 1))

    @property
    def target_seconds(self) -> tuple[float, ...]:
        return tuple(self.start_seconds + frame / self.model_fps for frame in self.target_frames)

    @property
    def resolved_frames(self) -> int:
        return self.history_frames + self.future_frames


@dataclass(frozen=True)
class RGBHistory:
    frames: torch.Tensor  # [5,3,H,W] uint8; ONLY observed, already resampled frames
    timing: PWTiming = PWTiming()
    source_frame_indexes: tuple[int, ...] = (0, 1, 2, 3, 4)
    source_pts_seconds: tuple[float, ...] = (0.0, 0.1, 0.2, 0.3, 0.4)

    def __post_init__(self):
        if self.frames.ndim != 4 or self.frames.shape[:2] != (5, 3) or self.frames.dtype != torch.uint8:
            raise ValueError("RGBHistory accepts exactly [5,3,H,W] uint8 observed frames")
        if min(self.frames.shape[-2:]) <= 0:
            raise ValueError("History RGB dimensions must be positive")
        if len(self.source_frame_indexes) != 5 or len(self.source_pts_seconds) != 5:
            raise ValueError("One source frame index and PTS is required per observed frame")
        expected = [self.timing.start_seconds + k / self.timing.model_fps for k in range(5)]
        if any(not math.isfinite(t) or abs(t - e) > 1e-6 for t, e in zip(self.source_pts_seconds, expected)):
            raise ValueError("History must be resampled to the declared 10Hz model clock")
        if any(b < a for a, b in zip(self.source_frame_indexes, self.source_frame_indexes[1:])):
            raise ValueError("Source frame order must be nondecreasing")
        if any(not isinstance(i, int) or i < 0 for i in self.source_frame_indexes):
            raise ValueError("Source frame indexes must be nonnegative integers")


@dataclass(frozen=True)
class FutureVisualTargets:
    """Supervision-only boundary; no condition builder or AR API accepts this type."""

    frames: torch.Tensor  # [16,3,H,W]
    word_class_ids: torch.Tensor  # [4]

    def __post_init__(self):
        if self.frames.ndim != 4 or self.frames.shape[:2] != (16, 3) or self.word_class_ids.shape != (4,):
            raise ValueError("Future targets require 16 RGB frames and four word labels")
        if self.frames.dtype != torch.uint8 or self.word_class_ids.dtype != torch.long:
            raise ValueError("Future RGB/labels must be uint8/int64")
        if bool(((self.word_class_ids < 0) | (self.word_class_ids >= 16)).any()):
            raise ValueError("Future word targets must be in [0,16)")


@dataclass(frozen=True)
class ReasonerCondition:
    input_ids: torch.Tensor  # [U], CPU, private PW IDs allowed
    position_ids: torch.Tensor  # [3,U], real multimodal mRoPE, CPU
    visual_indexes: torch.Tensor  # [R], UND-local indexes
    visual_features: torch.Tensor  # [R,D], frozen SigLIP2 + projector
    context_length: int  # excludes pw_start
    next_position: int  # multimodal position cursor, NOT physical sequence length
    timing: PWTiming
    rgb_sha256: str
    word_class_ids: tuple[int, ...] = ()
    boundaries: bool = False

    def __post_init__(self):
        size = self.input_ids.numel()
        if self.input_ids.ndim != 1 or self.position_ids.shape != (3, size):
            raise ValueError("Reasoner condition IDs/positions must be [U] / [3,U]")
        if self.input_ids.device.type != "cpu" or self.position_ids.device.type != "cpu":
            raise ValueError("Packing metadata must remain on CPU")
        if self.input_ids.dtype != torch.long or self.position_ids.dtype != torch.long:
            raise ValueError("Native UND IDs and mRoPE positions must be int64")
        expected_size = self.context_length + 1 + len(self.word_class_ids) + 2 * self.boundaries
        if self.context_length <= 0 or size != expected_size or self.next_position != int(self.position_ids.max()) + 1:
            raise ValueError("Condition length, PW suffix or mRoPE cursor is inconsistent")
        if (
            self.visual_indexes.ndim != 1
            or self.visual_indexes.device.type != "cpu"
            or self.visual_indexes.dtype != torch.long
        ):
            raise ValueError("Visual indexes must be UND-local CPU int64 metadata")
        if not self.visual_indexes.numel() or not bool(
            ((self.visual_indexes >= 0) & (self.visual_indexes < self.context_length)).all()
        ):
            raise ValueError("Visual features must belong exclusively to the historical context")
        if not bool((self.visual_indexes[1:] > self.visual_indexes[:-1]).all()):
            raise ValueError("Visual placeholder indexes must be ordered and unique")
        if self.visual_features.ndim != 2 or self.visual_features.shape[0] != self.visual_indexes.numel():
            raise ValueError("Visual features and UND placeholder indexes disagree")
        if self.visual_features.requires_grad:
            raise ValueError("SigLIP2 and projector outputs must be frozen")
        if self.boundaries and len(self.word_class_ids) != 4:
            raise ValueError("Finalized conditions require four predicted words")
        if not 0 <= len(self.word_class_ids) <= 4 or any(not 0 <= c < 16 for c in self.word_class_ids):
            raise ValueError("Expected up to four visual Physical Word classes")

    @property
    def prediction_positions(self) -> torch.Tensor:
        return torch.arange(self.context_length, self.context_length + 4)

    @property
    def token_positions(self) -> torch.Tensor:
        return self.prediction_positions + 1

    def with_words(self, class_ids: tuple[int, ...], token_id_base: int) -> ReasonerCondition:
        if self.boundaries:
            raise ValueError("Cannot extend a finalized condition")
        # Always rebuild from the ORIGINAL context + start, never from a partially appended suffix.
        size = self.context_length + 1
        suffix = torch.tensor([token_id_base + 1 + c for c in class_ids], dtype=torch.long)
        start = int(self.position_ids[:, size - 1].max()) + 1
        positions = torch.arange(start, start + len(class_ids)).expand(3, -1).clone()
        return replace(
            self,
            input_ids=torch.cat((self.input_ids[:size], suffix)),
            position_ids=torch.cat((self.position_ids[:, :size], positions), dim=1),
            word_class_ids=class_ids,
            next_position=start + len(class_ids),
        )

    def with_boundaries(self, special_tokens: dict[str, int]) -> ReasonerCondition:
        if self.boundaries or len(self.word_class_ids) != 4:
            raise ValueError("Repacking requires exactly four predicted words and no existing EOS/BOV")
        ids = torch.tensor([special_tokens["eos_token_id"], special_tokens["start_of_generation"]])
        positions = torch.arange(self.next_position, self.next_position + 2).expand(3, -1).clone()
        return replace(
            self,
            input_ids=torch.cat((self.input_ids, ids)),
            position_ids=torch.cat((self.position_ids, positions), dim=1),
            next_position=self.next_position + 2,
            boundaries=True,
        )


@torch.no_grad()
def build_reasoner_prefix(network, processor, history: RGBHistory, instruction: str) -> ReasonerCondition:
    """Apply native chat/vision preprocessing ONCE; share the resulting frozen features."""
    from cosmos_framework.data.generator.processors.cosmos3_edge_processing import NemotronNanoV3BridgeProcessor

    if not isinstance(processor, NemotronNanoV3BridgeProcessor):
        raise TypeError("M2 requires the native Cosmos3-Edge multimodal processor")
    if not isinstance(history, RGBHistory) or not isinstance(instruction, str) or not instruction.strip():
        raise ValueError("Expected an observed RGBHistory and nonempty language instruction")
    if network.physword is None:
        raise ValueError("Enable physword before building an RGB reasoning prefix")
    lm = network.language_model
    if getattr(lm, "visual", None) is None:
        raise RuntimeError("Load Edge SigLIP2/projector locally first; M2 never downloads weights implicitly")
    lm.visual.eval().requires_grad_(False)
    device = lm.model.embed_tokens.weight.device
    rgb = history.frames.detach().cpu().contiguous()
    digest = hashlib.sha256(rgb.numpy().tobytes()).hexdigest()
    timing = build_source_video_timing(
        list(history.source_frame_indexes),
        torch.tensor(history.source_pts_seconds, dtype=torch.float64),
        torch.full((5,), 1 / history.timing.source_fps, dtype=torch.float64),
        digest,
    )
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "video", "video": list(rgb), "fps": history.timing.model_fps, "source_video_timing": timing},
                {"type": "text", "text": instruction},
            ],
        }
    ]
    data = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
        do_sample_frames=False,
    )
    ids = torch.as_tensor(data["input_ids"], dtype=torch.long, device=device).reshape(1, -1)
    mask = torch.as_tensor(data["attention_mask"], device=device).reshape(1, -1)
    if not bool(mask.all()) or int(data["video_grid_thw"][:, 0].sum()) != 5:
        raise ValueError("M2 requires unpadded B=1 input and exactly five retained video frames")
    if bool((ids >= network.physword.config.token_id_base).any()):
        raise ValueError("Instruction cannot provide private PW IDs")
    embeds, visual_mask, _, positions, _ = prepare_multimodal_reasoner_inputs(
        lm,
        input_ids=ids,
        pixel_values_videos=data["pixel_values_videos"].to(device),
        video_grid_thw=data["video_grid_thw"].to(device),
        attention_mask=mask,
    )
    next_position = int(positions.max()) + 1
    input_ids = torch.cat((ids[0].cpu(), torch.tensor([network.physword.config.token_id_base])))
    position_ids = torch.cat((positions[:, 0].cpu(), torch.full((3, 1), next_position)), dim=1)
    return ReasonerCondition(
        input_ids=input_ids,
        position_ids=position_ids,
        visual_indexes=visual_mask[0].nonzero().flatten().cpu(),
        visual_features=embeds[visual_mask].detach(),
        context_length=ids.shape[1],
        next_position=next_position + 1,
        timing=history.timing,
        rgb_sha256=digest,
    )


def embed_reasoner_condition(network, condition: ReasonerCondition) -> torch.Tensor:
    """Shared AR/replay embedding path; Joint MoT scatters the same frozen payload."""
    base = network.language_model.model.embed_tokens
    ids = condition.input_ids.to(base.weight.device)
    embeds = network.physword(ids, base)
    return embeds.index_copy(
        0,
        condition.visual_indexes.to(embeds.device),
        condition.visual_features.to(device=embeds.device, dtype=embeds.dtype),
    ).unsqueeze(0)


def replay_physical_words(network, condition: ReasonerCondition) -> torch.Tensor:
    """Differentiable replay primitive, without any loss or optimizer."""
    return network.language_model.model.reasoner_forward(
        input_ids=None,
        cache=None,
        inputs_embeds=embed_reasoner_condition(network, condition),
        position_ids=condition.position_ids.unsqueeze(1).to(network.language_model.model.embed_tokens.weight.device),
    )


@torch.no_grad()
def prepare_history_video_data(history: RGBHistory, encode_history, future_noise: torch.Tensor):
    """Inference video condition: encode FIVE observed frames only; future input is noise."""
    from cosmos_framework.model.generator.utils.data_and_condition import GenerationDataClean

    pixels = history.frames.permute(1, 0, 2, 3).unsqueeze(0).float() / 127.5 - 1
    clean = encode_history(pixels).detach()
    if clean.ndim != 5 or clean.shape[:3] != (1, 48, 2):
        raise ValueError("History encoder must yield [1,48,2,H_lat,W_lat]")
    if future_noise.shape != (1, 48, 4, *clean.shape[-2:]):
        raise ValueError("Exactly four future noise latents are required")
    return GenerationDataClean(
        batch_size=1,
        is_image_batch=False,
        x0_tokens_vision=[torch.cat((clean, future_noise.to(clean)), dim=2)],
        fps_vision=torch.tensor([history.timing.model_fps]),
    )
