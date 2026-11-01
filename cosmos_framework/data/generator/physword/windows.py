"""Construct RGB history/future windows after an immutable episode split."""

from __future__ import annotations

from pathlib import Path

from cosmos_framework.data.generator.physword.annotation import label_at, validate_annotation
from cosmos_framework.data.generator.physword.common import VERSION, digest


def generate_windows(metadata, annotation, split_entry, episode_dir, config):
    validate_annotation(annotation, metadata)
    ep = metadata["episode"]
    if ep["episode_id"] != split_entry["episode_id"]:
        raise ValueError("Episode/split mismatch")
    if not ep.get("instruction") or not ep["instruction"].strip():
        return [], {"excluded_missing_instruction": True, "windows": 0}
    if config.sampling_fps != metadata["model_fps"]:
        raise ValueError("Window and processed frame clocks disagree")
    episode_dir = Path(episode_dir).resolve()
    frames = metadata["frame_mapping"]
    records = []
    dropped_invalid = 0
    for start in range(0, max(0, len(frames) - config.total_frames + 1), config.window_stride):
        history = list(range(start, start + config.history_frames))
        future = list(range(start + config.history_frames, start + config.total_frames))
        if not all(frames[i]["valid"] for i in history + future):
            dropped_invalid += 1
            continue
        targets = [start + offset for offset in config.target_offsets]
        labels, masks, reasons = [], [], []
        for i in targets:
            label, valid, reason = label_at(annotation, frames[i]["source_elapsed_s"], config)
            labels.append(label)
            masks.append(valid)
            reasons.append(reason)
        effectors = annotation.get("effectors") or sorted({s["effector"] for s in annotation["stages"]})
        arm_targets = [
            [label_at(annotation, frames[i]["source_elapsed_s"], config, effector=arm) for i in targets]
            for arm in effectors
        ]
        record = {
            "version": VERSION,
            "episode_id": ep["episode_id"],
            "original_episode_id": ep["original_episode_id"],
            "lineage_id": ep["lineage_id"],
            "group_id": split_entry["group_id"],
            "split": split_entry["split"],
            "task_id": ep["task_id"],
            "dataset_source": ep["dataset_source"],
            "instruction": ep["instruction"],
            "source_episode_ref": ep["source_ref"],
            "camera": ep["camera"],
            "model_fps": metadata["model_fps"],
            "source_fps": metadata["source_fps"],
            "video_path": str(episode_dir / metadata["video_path"]),
            "metadata_path": str(episode_dir / "metadata.json"),
            "annotation_path": str(episode_dir / "annotation.json"),
            "media_fingerprint": metadata["fingerprint"],
            "annotation_hash": digest(annotation),
            "history_frame_indices": history,
            "future_frame_indices": future,
            "pw_target_frame_indices": targets,
            "pw_class_targets": labels,
            "pw_valid_mask": masks,
            "pw_mask_reasons": reasons,
            "pw_effectors": effectors,
            "pw_effector_coordinate": annotation.get("effector_coordinate", "unspecified"),
            "pw_arm_class_targets": [[target[0] for target in arm] for arm in arm_targets],
            "pw_arm_valid_mask": [[target[1] for target in arm] for arm in arm_targets],
            "pw_arm_mask_reasons": [[target[2] for target in arm] for arm in arm_targets],
            "pw_target_timestamps": [frames[i]["source_elapsed_s"] for i in targets],
            "pw_target_resampled_timestamps": [frames[i]["resampled_timestamp"] for i in targets],
            "pw_target_source_timestamps": [frames[i]["source_timestamp"] for i in targets],
            "pw_target_source_frame_indices": [frames[i]["source_frame_index"] for i in targets],
            "source_time_origin": metadata["source_time_origin"],
            "history_frame_paths": [str(episode_dir / frames[i]["path"]) for i in history],
            "future_frame_paths": [str(episode_dir / frames[i]["path"]) for i in future],
            "history_source_frame_indices": [frames[i]["source_frame_index"] for i in history],
            "future_source_frame_indices": [frames[i]["source_frame_index"] for i in future],
            "history_source_timestamps": [frames[i]["source_timestamp"] for i in history],
            "future_source_timestamps": [frames[i]["source_timestamp"] for i in future],
            "history_timestamps": [frames[i]["resampled_timestamp"] for i in history],
            "future_timestamps": [frames[i]["resampled_timestamp"] for i in future],
            "history_source_pts": [frames[i]["source_pts_s"] for i in history],
            "future_source_pts": [frames[i]["source_pts_s"] for i in future],
            "feature_cache_key": digest({"media": metadata["fingerprint"], "targets": targets}),
            "annotation_status": annotation["provenance"],
        }
        record["window_id"] = f"{ep['episode_id']}__{start:06d}"
        records.append(record)
    return records, {
        "excluded_missing_instruction": False,
        "windows": len(records),
        "dropped_invalid_frame_windows": dropped_invalid,
        "too_short": len(frames) < config.total_frames,
        "tail_policy": "drop",
        "tail_candidate_starts_dropped": min(len(frames), config.total_frames - 1),
    }
