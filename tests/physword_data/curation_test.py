"""Removal must preserve payloads, split membership, normalization population and recovery."""

import json
from pathlib import Path

import numpy as np
import pytest

from cosmos_framework.data.generator.physword.adapters import discover_real
from cosmos_framework.data.generator.physword.common import read_json, write_json
from tools.physword_data.curation import Quarantine, capture_identity, stage_lerobot_removal


def test_quarantine_backup_does_not_mutate_shared_hardlinks(tmp_path):
    original = tmp_path / "active/stats.json"
    original.parent.mkdir()
    original.write_text("old")
    alias = tmp_path / "alias.json"
    alias.hardlink_to(original)
    quarantine = Quarantine(tmp_path / "quarantine")
    quarantine.replace_text(original, "new")
    assert original.read_text() == "new" and alias.read_text() == "old"
    assert Path(quarantine.entries[0]["quarantine"]).read_text() == "old"
    quarantine.move(original)
    assert not original.exists() and all(entry["complete"] for entry in read_json(quarantine.journal_path))


def test_persistent_capture_exclusion_survives_directory_renumbering(tmp_path):
    manifest = {
        "episode": 48,
        "start_wall": 123.0,
        "stop_wall": 126.0,
        "start_mono": 10.0,
        "cameras": {"main": {"frames": 3}},
    }
    directory = tmp_path / "episode_0999"
    (directory / "main").mkdir(parents=True)
    (directory / "main/color_take001.mp4").touch()
    write_json(directory / "manifest.json", manifest)
    write_json(
        tmp_path / "_episode_exclusions.json",
        {"original_episode_ids": [capture_identity(manifest)], "source_sha256": []},
    )
    episodes, rejected = discover_real(tmp_path, "fixture")
    assert not episodes and rejected[0]["reason"] == "excluded_original_episode"


def test_lerobot_removal_dense_indexes_and_train_only_statistics(tmp_path):
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    source, staging = tmp_path / "source", tmp_path / "staging"
    info = {
        "total_episodes": 3,
        "total_frames": 12,
        "splits": {"train": "0:2", "val": "2:3"},
        "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
        "video_path": "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",
        "features": {"action": {"shape": [2]}, "observation.state": {"shape": [2]}},
    }
    write_json(source / "meta/info.json", info)
    rows, all_action, all_state = [], [], []
    for i in range(3):
        x = np.arange(8).reshape(4, 2).astype(float) + 100 * i
        table = pa.table(
            {
                "episode_index": [i] * 4,
                "index": list(range(i * 4, i * 4 + 4)),
                "frame_index": [0, 1, 2, 3],
                "action": x.tolist(),
                "observation.state": x.tolist(),
                "annotation.time_valid": [True] * 4,
                "annotation.action_valid": [True, True, True, False],
            }
        )
        path = source / info["data_path"].format(chunk_index=0, file_index=i)
        path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(table, path)
        rows.append(
            {
                "episode_index": i,
                "source_name": f"episode_{i + 1:04d}",
                "length": 4,
                "dataset_from_index": 4 * i,
                "dataset_to_index": 4 * i + 4,
                "data/chunk_index": 0,
                "data/file_index": i,
                "stats/action/count": [3],
                "stats/observation.state/count": [4],
            }
        )
        if i < 2:
            all_action.append(x[:3])
            all_state.append(x)
    meta = source / "meta/episodes/chunk-000/file-000.parquet"
    meta.parent.mkdir(parents=True)
    pq.write_table(pa.Table.from_pylist(rows), meta)
    write_json(source / "meta/stats.json", {"action": {"count": [6]}, "observation.state": {"count": [8]}})
    original = pq.read_table(source / "data/chunk-000/file-002.parquet")
    result = stage_lerobot_removal(source, staging, "episode_0002")
    assert result["remaining_episodes"] == 2 and result["remaining_rows"] == 8
    assert result["statistics_population"] == "train"
    assert result["splits"] == {"train": "0:1", "val": "1:2"}
    updated = pq.read_table(staging / "data/chunk-000/file-001.parquet")
    assert updated["episode_index"].to_pylist() == [1] * 4
    assert updated["index"].to_pylist() == [4, 5, 6, 7]
    assert updated["action"].equals(original["action"])
    assert pq.read_table(source / "data/chunk-000/file-002.parquet").equals(original)
    stats = read_json(staging / "meta/stats.json")
    assert stats["action"]["count"] == [3] and stats["action"]["max"] == [4.0, 5.0]
    assert [
        r["source_name"] for r in pq.read_table(staging / "meta/episodes/chunk-000/file-000.parquet").to_pylist()
    ] == ["episode_0001", "episode_0003"]


def test_curated_split_inherits_surviving_memberships():
    from cosmos_framework.data.generator.physword.split import inherit_split

    def entry(name, split):
        return {
            "episode_id": name,
            "original_episode_id": name,
            "lineage_id": name,
            "content_sha256": name,
            "group_id": name,
            "split": split,
        }

    parent = {"seed": 42, "episodes": [entry("a", "train"), entry("b", "test"), entry("removed", "test")]}
    fresh = {"seed": 42, "episodes": [entry("a", "test"), entry("b", "train")]}
    result = inherit_split(fresh, parent)
    assert [r["split"] for r in result["episodes"]] == ["train", "test"]
    with pytest.raises(ValueError, match="no assignment"):
        inherit_split({"seed": 42, "episodes": [entry("new", "train")]}, parent)


def test_short_episode_screen_uses_capture_clock_not_integer_frame_rate(tmp_path):
    from tools.physword_data.curation import scan_short_episodes

    for name, capture_span in [("episode_0001", 4.9), ("episode_0002", 5.1)]:
        folder = tmp_path / name
        (folder / "main").mkdir(parents=True)
        manifest = {
            "episode": int(name[-4:]),
            "start_wall": 100.0,
            "stop_wall": 105.5,
            "start_mono": 10.0,
            "stop_mono": 15.5,
            "status": "closed",
            "recording_errors": [],
            "cameras": {"main": {"frames": 3}},
        }
        write_json(folder / "manifest.json", manifest)
        (folder / "main/timestamps_take001.jsonl").write_text(
            "\n".join(
                json.dumps({"i": index + 1, "mono": 10.0 + delta, "wall": 100.0 + delta})
                for index, delta in enumerate([0.0, capture_span / 2, capture_span])
            )
        )
    report = scan_short_episodes(tmp_path)
    assert report["episodes_scanned"] == 2
    assert [row["source_name"] for row in report["short_episodes"]] == ["episode_0001"]
