"""Versioned configuration and deterministic, atomic artifact IO."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

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
VERSION = "physword-data-v1"


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def digest(data):
    return hashlib.sha256(json.dumps(data, sort_keys=True, allow_nan=False).encode()).hexdigest()


def file_sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


@dataclass(frozen=True)
class PipelineConfig:
    sampling_fps: float = 10.0
    train_ratio: float = 0.9
    test_ratio: float = 0.1
    validation_ratio: float = 0.0  # fraction of TRAINING groups, never of test
    seed: int = 42
    history_frames: int = 5
    future_frames: int = 16
    pw_num_blocks: int = 4
    pw_stride: int = 4
    window_stride: int = 1
    tail_policy: str = "drop"
    min_confidence: float = 0.6
    boundary_tolerance_s: float = 0.05
    max_frame_age_s: float = 0.15
    action_conditioning: bool = False
    proprioception_conditioning: bool = False
    tactile_conditioning: bool = False
    modality: str = "RGB"

    def __post_init__(self):
        if any((self.action_conditioning, self.proprioception_conditioning, self.tactile_conditioning)):
            raise ValueError("PhysWord data conditions are RGB + instruction only")
        if self.modality != "RGB" or self.tail_policy != "drop":
            raise ValueError("Only RGB and explicit tail_policy=drop are supported in v1")
        for key in ("history_frames", "future_frames", "pw_num_blocks", "pw_stride", "window_stride"):
            if type(getattr(self, key)) is not int or getattr(self, key) < 1:
                raise ValueError(f"{key} must be a positive integer")
        for key in ("sampling_fps", "max_frame_age_s"):
            if not math.isfinite(getattr(self, key)) or getattr(self, key) <= 0:
                raise ValueError(f"{key} must be positive and finite")
        if not (
            0 < self.train_ratio < 1 and 0 < self.test_ratio < 1 and abs(self.train_ratio + self.test_ratio - 1) < 1e-9
        ):
            raise ValueError("train/test ratios must be positive and sum to one")
        if not 0 <= self.validation_ratio < 1 or not 0 <= self.min_confidence <= 1:
            raise ValueError("Invalid validation ratio or confidence")
        if not math.isfinite(self.boundary_tolerance_s) or self.boundary_tolerance_s < 0:
            raise ValueError("Invalid boundary tolerance")
        if (self.history_frames - 1) % 4 or (self.history_frames + self.future_frames - 1) % 4:
            raise ValueError("Wan causal history and full clips must each have 1+4m frames")
        if self.pw_num_blocks * self.pw_stride > self.future_frames:
            raise ValueError("PW target exceeds future horizon")

    @property
    def total_frames(self):
        return self.history_frames + self.future_frames

    @property
    def target_offsets(self):
        return [self.history_frames - 1 + k * self.pw_stride for k in range(1, self.pw_num_blocks + 1)]

    def to_dict(self):
        return asdict(self)
