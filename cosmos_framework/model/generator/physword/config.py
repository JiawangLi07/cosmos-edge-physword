# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

import hashlib
import json
import re
from dataclasses import dataclass

LEGACY_WORDS = (
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


@dataclass
class PhysWordConfig:
    """Opt-in Physical Words; private model IDs, never HF tokenizer IDs."""

    enabled: bool = False
    token_id_base: int = 1 << 30
    embedding_seed: int = 1729
    vocabulary: tuple[str, ...] = LEGACY_WORDS
    vocabulary_id: str = "physword_atomic_v1"
    definitions: tuple[str, ...] = ()
    effector: str = "scene"

    def __post_init__(self) -> None:
        if not isinstance(self.enabled, bool):
            raise ValueError("physword.enabled must be a boolean")
        if isinstance(self.token_id_base, bool) or not isinstance(self.token_id_base, int) or self.token_id_base < 0:
            raise ValueError("physword.token_id_base must be a nonnegative integer")
        self.vocabulary = tuple(self.vocabulary)
        self.definitions = tuple(self.definitions) or self.vocabulary
        if (
            not self.vocabulary
            or len(set(self.vocabulary)) != len(self.vocabulary)
            or any(not re.fullmatch(r"[a-z]+", w) or w in {"unknown", "idle"} for w in self.vocabulary)
        ):
            raise ValueError("PW vocabulary must contain unique, defined lowercase words, without unknown/idle")
        if len(self.definitions) != len(self.vocabulary) or any(not d.strip() for d in self.definitions):
            raise ValueError("One nonempty definition per PW class is required")
        if self.effector not in {"scene", "left", "right"}:
            raise ValueError("PW effector must be explicit: scene, left or right")

    @property
    def tokens(self):
        return ("<pw_start>",) + tuple(f"<pw_{word}_v>" for word in self.vocabulary)

    @property
    def vocabulary_hash(self):
        content = dict(
            id=self.vocabulary_id, words=self.vocabulary, definitions=self.definitions, effector=self.effector
        )
        return hashlib.sha256(json.dumps(content, sort_keys=True).encode()).hexdigest()


def resolve_physword_config(value: object = None) -> PhysWordConfig:
    if value is None:
        return PhysWordConfig()
    if isinstance(value, PhysWordConfig):
        return value
    if hasattr(value, "items"):
        return PhysWordConfig(**dict(value.items()))
    return PhysWordConfig(
        **{name: getattr(value, name) for name in PhysWordConfig.__dataclass_fields__ if hasattr(value, name)}
    )
