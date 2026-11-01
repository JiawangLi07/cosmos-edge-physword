"""Lazy RGB dataset and optional M3 bridge; future supervision is never a condition."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

import numpy as np
from PIL import Image

from cosmos_framework.data.generator.physword.vocabulary import (
    require_legacy_model_vocabulary,
    vocabulary_for,
    vocabulary_hash,
)


def read_rgb(paths):
    images = []
    for path in paths:
        with Image.open(path) as im:
            im.load()
            if im.mode != "RGB":
                raise ValueError(f"Expected RGB image: {path}")
            images.append(np.asarray(im).copy().transpose(2, 0, 1))
    return np.stack(images)  # [T,3,H,W] uint8, never silently resize


class TrainingWindowDataset:
    def __init__(self, manifest, split="train", require_valid_pw=False, require_valid_arm_pw=False):
        self.manifest = Path(manifest).resolve()
        records = [json.loads(line) for line in self.manifest.read_text().splitlines() if line.strip()]
        legacy = vocabulary_for({})
        for record in records:
            record.setdefault("pw_vocabulary_id", legacy["id"])
            record.setdefault("pw_vocabulary", legacy["words"])
            record.setdefault("pw_vocabulary_hash", vocabulary_hash(legacy))
            actual_hash = vocabulary_hash({"id": record["pw_vocabulary_id"], "words": record["pw_vocabulary"]})
            if actual_hash != record["pw_vocabulary_hash"]:
                raise ValueError("Vocabulary ID mapping fingerprint mismatch")
        if len({record["pw_vocabulary_hash"] for record in records}) > 1:
            raise ValueError("Cannot mix vocabularies in a training manifest")
        self.records = [r for r in records if r["split"] == split and (not require_valid_pw or any(r["pw_valid_mask"]))]
        if require_valid_arm_pw:
            self.records = [r for r in self.records if any(any(arm) for arm in r["pw_arm_valid_mask"])]
        self.base = self.manifest.parent

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        r = self.records[index]
        # Explicit allowlist: no raw action/state/tactile fields are forwarded.
        result = {
            key: r[key]
            for key in (
                "instruction",
                "episode_id",
                "original_episode_id",
                "window_id",
                "model_fps",
                "source_fps",
                "history_frame_indices",
                "future_frame_indices",
                "pw_target_frame_indices",
                "pw_target_source_frame_indices",
                "history_source_frame_indices",
                "future_source_frame_indices",
                "history_source_pts",
                "future_source_pts",
                "history_source_timestamps",
                "future_source_timestamps",
                "history_timestamps",
                "future_timestamps",
                "pw_target_resampled_timestamps",
                "pw_target_source_timestamps",
                "pw_mask_reasons",
                "pw_effectors",
                "pw_effector_coordinate",
                "pw_arm_mask_reasons",
                "metadata_path",
                "video_path",
                "feature_cache_key",
                "annotation_status",
                "pw_vocabulary_id",
                "pw_vocabulary",
                "pw_vocabulary_hash",
            )
        }
        result["history_rgb"] = read_rgb([self.base / p for p in r["history_frame_paths"]])
        result["future_rgb"] = read_rgb([self.base / p for p in r["future_frame_paths"]])
        result["pw_class_targets"] = np.array(r["pw_class_targets"], dtype=np.int64)
        result["pw_valid_mask"] = np.array(r["pw_valid_mask"], dtype=np.bool_)
        result["pw_arm_class_targets"] = np.array(r["pw_arm_class_targets"], dtype=np.int64)
        result["pw_arm_valid_mask"] = np.array(r["pw_arm_valid_mask"], dtype=np.bool_)
        result["pw_target_timestamps"] = np.array(r["pw_target_timestamps"], dtype=np.float64)
        for key in ("task_vocabulary_hash", "selected_effector"):
            if key in r:
                result[key] = r[key]
        if "pw_latent_valid_mask" in r:
            result["pw_latent_valid_mask"] = np.array(r["pw_latent_valid_mask"], dtype=np.bool_)
        return result


def collate_windows(samples):
    """Torch imports stay optional for preprocessing; metadata stays a per-sample list."""
    import torch

    tensors = {
        "history_rgb",
        "future_rgb",
        "pw_class_targets",
        "pw_valid_mask",
        "pw_target_timestamps",
        "pw_arm_class_targets",
        "pw_arm_valid_mask",
    }
    if any(sample["pw_vocabulary_hash"] != samples[0]["pw_vocabulary_hash"] for sample in samples):
        raise ValueError("Batch vocabulary mappings disagree")
    if any(sample["pw_effectors"] != samples[0]["pw_effectors"] for sample in samples):
        raise ValueError("Batch effectors/order disagree; use separate loaders for different embodiments")
    return {
        key: torch.from_numpy(np.stack([s[key] for s in samples])) if key in tensors else [s[key] for s in samples]
        for key in samples[0]
    }


def make_dataloader(
    manifest, split="train", batch_size=1, seed=42, num_workers=0, require_valid_pw=False, require_valid_arm_pw=False
):
    import torch
    from torch.utils.data import DataLoader

    dataset = TrainingWindowDataset(
        manifest, split=split, require_valid_pw=require_valid_pw, require_valid_arm_pw=require_valid_arm_pw
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=split == "train" and len(dataset) > 0,
        num_workers=num_workers,
        collate_fn=collate_windows,
        generator=torch.Generator().manual_seed(seed),
    )


def condition_inputs(sample):
    """Formal inference input API. Changing GT labels/future RGB cannot affect it."""
    return {
        "history_rgb": sample["history_rgb"],
        "instruction": sample["instruction"],
        "model_fps": sample["model_fps"],
        "history_timestamps": sample["history_timestamps"],
    }


def to_rgb_history(sample):
    """M2/M3 compatibility: legacy RGBHistory's PTS field means MODEL clock.

    Actual capture times remain in the dataset; do not overwrite them to satisfy
    the M2 validator. Only this legacy condition payload uses resampled timestamps.
    M2 currently restricts 5/16/4/4; other config shapes fail explicitly here.
    """
    import torch

    from cosmos_framework.model.generator.physword.multimodal import PWTiming, RGBHistory

    history = sample["history_rgb"]
    future = sample["future_rgb"]
    timing = PWTiming(
        source_fps=sample["source_fps"] or sample["model_fps"],
        model_fps=sample["model_fps"],
        history_frames=len(history),
        future_frames=len(future),
        blocks=len(sample["pw_class_targets"]),
        stride=sample["pw_target_frame_indices"][0] - sample["history_frame_indices"][-1],
        start_seconds=sample["history_timestamps"][0],
    )
    return RGBHistory(
        torch.as_tensor(history),
        timing=timing,
        source_frame_indexes=tuple(sample["history_source_frame_indices"]),
        source_pts_seconds=tuple(sample["history_timestamps"]),
    )


def build_condition(sample, network, processor):
    from cosmos_framework.model.generator.physword.multimodal import build_reasoner_prefix

    return build_reasoner_prefix(network, processor, to_rgb_history(sample), sample["instruction"])


def m3_target_payload(sample):
    """Supervision-only: compatible field contract for M3 PhysWordTargets factory.

    Instantiate model-side types via an explicit factory to avoid importing M3's
    work-in-progress training modules into the standalone CPU pipeline.
    """
    require_legacy_model_vocabulary(sample)
    import torch

    future = torch.as_tensor(sample["future_rgb"])
    first_future = sample["future_frame_indices"][0]
    offsets = [i - first_future for i in sample["pw_target_frame_indices"]]
    return {
        "future_rgb": future,
        "pw_class_targets": torch.as_tensor(sample["pw_class_targets"], dtype=torch.long),
        "pw_valid_mask": torch.as_tensor(sample["pw_valid_mask"], dtype=torch.bool),
        "pw_target_timestamps": torch.as_tensor(sample["pw_target_timestamps"], dtype=torch.float64),
        "target_rgb": future[offsets],
        "pw_target_frame_indices": sample["pw_target_frame_indices"],
        "episode_id": sample["episode_id"],
    }


def to_m3_targets(sample, factory: Callable):
    """factory(**payload) can translate names to the exact installed M3 target type."""
    return factory(**m3_target_payload(sample))


def feature_cache_identity(sample, teacher_sha256, processor_config, pooling="spatial_mean"):
    """M3 owns SigLIP2 target computation; invalidate cache for teacher/processor changes."""
    from cosmos_framework.data.generator.physword.common import digest

    return digest(
        {
            "frames": sample["feature_cache_key"],
            "teacher_sha256": teacher_sha256,
            "processor_config": processor_config,
            "pooling": pooling,
        }
    )


def to_m3_batch(sample, batch_type=None, *, pw_config=None):
    """Exact installed M3 PhysWordBatch contract (B=1); only targets carry future GT.

    M3 currently names this type PhysWordBatch, rather than PhysWordTargets. The
    optional factory allows integration before the independent M3 branch is merged.
    """
    if pw_config is None:
        require_legacy_model_vocabulary(sample)
    elif (
        sample.get("task_vocabulary_hash") != pw_config.vocabulary_hash
        or tuple(sample["pw_vocabulary"]) != pw_config.vocabulary
    ):
        raise ValueError("Task sample vocabulary/definitions/effector differs from the model")
    import torch

    if batch_type is None:
        from cosmos_framework.model.generator.physword.training import PhysWordBatch

        batch_type = PhysWordBatch
    mask = torch.as_tensor(sample["pw_valid_mask"], dtype=torch.bool).unsqueeze(0)
    return batch_type(
        history=to_rgb_history(sample),
        instruction=sample["instruction"],
        future_rgb=torch.as_tensor(sample["future_rgb"]),
        word_labels=torch.as_tensor(sample["pw_class_targets"], dtype=torch.long).unsqueeze(0),
        valid_word=mask,
        valid_latent=torch.as_tensor(
            sample.get("pw_latent_valid_mask", sample["pw_valid_mask"]), dtype=torch.bool
        ).unsqueeze(0),
    )
