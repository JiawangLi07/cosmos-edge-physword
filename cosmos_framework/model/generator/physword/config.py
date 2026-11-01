# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

from dataclasses import dataclass


@dataclass
class PhysWordConfig:
    """Opt-in Physical Words; private model IDs, never HF tokenizer IDs."""

    enabled: bool = False
    token_id_base: int = 1 << 30
    embedding_seed: int = 1729

    def __post_init__(self) -> None:
        if not isinstance(self.enabled, bool):
            raise ValueError("physword.enabled must be a boolean")
        if isinstance(self.token_id_base, bool) or not isinstance(self.token_id_base, int) or self.token_id_base < 0:
            raise ValueError("physword.token_id_base must be a nonnegative integer")


def resolve_physword_config(value: object = None) -> PhysWordConfig:
    if value is None:
        return PhysWordConfig()
    if isinstance(value, PhysWordConfig):
        return value
    if hasattr(value, "items"):
        return PhysWordConfig(**dict(value.items()))
    return PhysWordConfig(
        enabled=value.enabled,
        token_id_base=value.token_id_base,
        embedding_seed=value.embedding_seed,
    )
