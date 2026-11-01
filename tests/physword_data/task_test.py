"""Task-specific vocab and immutable episode holdout, using independent synthetic manifests."""

import json

import pytest

from cosmos_framework.data.generator.physword.common import PipelineConfig, file_sha256
from cosmos_framework.data.generator.physword.task import config_from_manifest, prepare_task


def source_manifest(tmp_path):
    records = []
    for ep, labels in [("episode_0001", [0, 1, 1, 1]), ("episode_0017", [0, 2, 1, 1])]:
        records.append(
            {
                "task_id": "robot",
                "episode_id": ep,
                "original_episode_id": ep,
                "lineage_id": ep,
                "group_id": ep,
                "media_fingerprint": ep,
                "split": "train",
                "pw_effectors": ["left", "right"],
                "pw_arm_class_targets": [labels, [1] * 4],
                "pw_arm_valid_mask": [[True, False, True, True], [True] * 4],
                "pw_vocabulary": ["approach", "hold", "rotate"],
                "model_fps": 10,
                "history_frame_indices": list(range(5)),
                "future_frame_indices": list(range(5, 21)),
                "pw_target_frame_indices": [8, 12, 16, 20],
                "history_frame_paths": [f"{ep}/h.png"] * 5,
                "future_frame_paths": [f"{ep}/f.png"] * 16,
            }
        )
    source = tmp_path / "source.jsonl"
    source.write_text("".join(json.dumps(r) + "\n" for r in records))
    vocab = tmp_path / "vocabulary.json"
    vocab.write_text(json.dumps({"definitions": {w: w for w in ["approach", "hold", "rotate"]}}))
    return source, vocab


def test_frozen_train_only_task_vocabulary_unknown_mask_and_episode_split(tmp_path):
    source, vocab = source_manifest(tmp_path)
    before = file_sha256(source)
    manifest, protocol = prepare_task(
        source, vocab, tmp_path / "task", task_id="robot", effector="left", test_episodes=["episode_0017"]
    )
    assert file_sha256(source) == before
    assert manifest["words"] == ["approach", "hold"]
    assert protocol["window_counts"] == {"train": 1, "test": 1}
    assert config_from_manifest(manifest).effector == "left"
    records = [json.loads(s) for s in (tmp_path / "task/windows.jsonl").read_text().splitlines()]
    assert records[1]["pw_class_targets"][1] == -100 and not records[1]["pw_valid_mask"][1]
    assert all(records[1]["pw_latent_valid_mask"])
    assert records[1]["source_split"] == "train" and records[1]["split"] == "test"
    for mode in ("native", "matched_rgb", "full_pw"):
        spec = json.loads((tmp_path / f"task/{mode}.json").read_text())
        assert spec["rf"]["distribution"] == "waver" and spec["max_train_steps"] == 0


def test_task_rejects_semantic_or_order_drift_and_capture_leakage(tmp_path):
    source, vocab = source_manifest(tmp_path)
    manifest, _ = prepare_task(
        source, vocab, tmp_path / "task", task_id="robot", effector="left", test_episodes=["episode_0017"]
    )
    manifest["definitions"]["hold"] = "Changed semantics"
    with pytest.raises(ValueError, match="fingerprint"):
        config_from_manifest(manifest)
    records = [json.loads(s) for s in source.read_text().splitlines()]
    records[1]["lineage_id"] = records[0]["lineage_id"]
    source.write_text("".join(json.dumps(r) + "\n" for r in records))
    with pytest.raises(ValueError, match="leakage"):
        prepare_task(source, vocab, tmp_path / "leak", task_id="robot", effector="left", test_episodes=["episode_0017"])


def test_horizon_must_equal_future_and_no_implicit_empty_test(tmp_path):
    with pytest.raises(ValueError, match="equal"):
        PipelineConfig(future_frames=20)
    source, vocab = source_manifest(tmp_path)
    with pytest.raises(ValueError, match="disjoint"):
        prepare_task(source, vocab, tmp_path / "task", task_id="robot", effector="left")
