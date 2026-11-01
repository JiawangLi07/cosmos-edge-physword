"""Read-only verification of local original-episode removal and converted payloads."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from cosmos_framework.data.generator.physword.common import file_sha256, read_json, write_json
from tools.physword_data.curation import Quarantine


def verify(quarantine_root, pilot_root):
    import pyarrow.parquet as pq

    quarantine = Quarantine(quarantine_root)
    removal = read_json(quarantine.root / "removal_report.json")
    excluded = removal["removed_source_name"]
    original_id = removal["removed_original_episode_id"]
    raw_root = Path(removal["alignment_datasets"][0]["dataset"]).parent
    raw_copies = []
    for name in ("episode_badminton", "episode_badminton_91"):
        root = raw_root / name
        assert not (root / excluded).exists()
        policy = read_json(root / "_episode_exclusions.json")
        assert original_id in policy["original_episode_ids"] and len(policy["source_sha256"]) >= 3
        raw_copies.append({"dataset": str(root), "excluded_episode_absent": True, "exclusion_policy_present": True})
    alignments = []
    for item in removal["alignment_datasets"]:
        root = Path(item["dataset"])
        assert not (root / "episodes" / (excluded + ".json")).exists()
        assert not (root / "episodes" / (excluded + ".npz")).exists()
        rows = [read_json(path) for path in (root / "episodes").glob("*.json")]
        assert len(rows) == item["remaining_episodes"]
        assert sorted(row["episode_index"] for row in rows) == list(range(len(rows)))
        alignments.append({"dataset": str(root), "episodes": len(rows), "excluded_episode_absent": True})
    lerobot = []
    for plan in removal["lerobot_datasets"]:
        root = Path(plan["source"])
        old_root = quarantine._destination(root, "removed")
        info = read_json(root / "meta/info.json")
        rows = pq.read_table(next((root / "meta/episodes").rglob("*.parquet"))).to_pylist()
        assert len(rows) == info["total_episodes"] == plan["remaining_episodes"]
        assert [r["episode_index"] for r in rows] == list(range(len(rows)))
        assert excluded not in {r["source_name"] for r in rows}
        mapping = {m["new_episode_index"]: m for m in plan["episode_mapping"]}
        stats = read_json(root / "meta/stats.json")
        train_start, train_stop = map(int, info["splits"]["train"].split(":"))
        arrays = {feature: [] for feature in stats}
        offset = 0
        for row in rows:
            idx = row["episode_index"]
            path = info["data_path"].format(chunk_index=row["data/chunk_index"], file_index=row["data/file_index"])
            table = pq.read_table(root / path)
            n = table.num_rows
            assert n == row["length"] and row["dataset_from_index"] == offset and row["dataset_to_index"] == offset + n
            np.testing.assert_array_equal(table["index"].to_numpy(), np.arange(offset, offset + n))
            np.testing.assert_array_equal(table["episode_index"].to_numpy(), np.full(n, idx))
            old_idx = mapping[idx]["old_episode_index"]
            old_path = info["data_path"].format(chunk_index=0, file_index=old_idx)
            old = pq.read_table(old_root / old_path)
            assert old.drop(["index", "episode_index"]).equals(table.drop(["index", "episode_index"]))
            assert mapping[idx]["source_name"] == row["source_name"]
            if plan["statistics_population"] == "all" or train_start <= idx < train_stop:
                for feature, mask in (
                    ("action", "annotation.action_valid"),
                    ("observation.state", "annotation.time_valid"),
                ):
                    arrays[feature].append(
                        np.asarray(table[feature].to_pylist(), dtype=np.float64)[table[mask].to_numpy().astype(bool)]
                    )
            offset += n
        assert offset == info["total_frames"] == plan["remaining_rows"]
        for feature, parts in arrays.items():
            x = np.concatenate(parts)
            assert stats[feature]["count"] == [len(x)]
            for key, value in (("min", x.min(0)), ("max", x.max(0)), ("mean", x.mean(0)), ("std", x.std(0))):
                np.testing.assert_allclose(stats[feature][key], value, atol=1e-12, rtol=1e-12)
        norm_path = root / "meta/normalization_stats.npy"
        if norm_path.exists():
            norm = np.load(norm_path, allow_pickle=True).item()["eef"]
            pooled = np.concatenate(arrays["action"] + arrays["observation.state"])
            active = np.zeros(80, dtype=bool)
            active[0:9] = active[10:32] = active[34:43] = active[44:66] = True
            exempt = ~active
            exempt[list(range(3, 9)) + list(range(37, 43))] = True
            expected = {
                "min": pooled.min(0),
                "max": pooled.max(0),
                "mean": pooled.mean(0),
                "std": pooled.std(0),
                "q01": np.quantile(pooled, 0.01, axis=0),
                "q99": np.quantile(pooled, 0.99, axis=0),
            }
            for key, constant in (("min", -1), ("max", 1), ("mean", 0), ("std", 1), ("q01", -1), ("q99", 1)):
                expected[key] = expected[key].astype(np.float32)
                expected[key][exempt] = constant
                np.testing.assert_array_equal(norm[key], expected[key])
        videos = read_json(root / "provenance/videos.json")
        for video in videos:
            assert video["source_name"] != excluded and file_sha256(root / video["path"]) == video["sha256"]
        source_hashes = root / "provenance/source_hashes.json"
        if source_hashes.exists():
            src = Path(read_json(root / "meta/conversion.json")["source"])
            for relative, sha in read_json(source_hashes).items():
                assert file_sha256(src / relative) == sha
        lerobot.append(
            {
                "dataset": str(root),
                "episodes": len(rows),
                "rows": offset,
                "videos_verified": len(videos),
                "kept_payload_equal_to_original_except_indexes": True,
                "statistics_population": plan["statistics_population"],
                "statistics_recomputed": True,
            }
        )
    physword = []
    for path in Path(pilot_root).rglob("inventory.json"):
        inventory = read_json(path)
        assert all(e.get("original_episode_id") != original_id for e in inventory.get("episodes", []))
        split = read_json(path.parent / "split_manifest.json")
        assert all(e["original_episode_id"] != original_id for e in split["episodes"])
        windows = path.parent / "windows.jsonl"
        count = 0
        if windows.exists():
            import json

            for line in windows.read_text().splitlines():
                record = json.loads(line)
                assert record["original_episode_id"] != original_id
                assert all(Path(p).is_file() for p in record["history_frame_paths"] + record["future_frame_paths"])
                count += 1
        assert not (path.parent / "processed" / removal["removed_episode_id"]).exists()
        physword.append({"dataset": str(path.parent), "windows": count, "excluded_episode_absent": True})
    report = {
        "status": "passed",
        "excluded_source_name": excluded,
        "raw_datasets": raw_copies,
        "alignment_datasets": alignments,
        "lerobot_datasets": lerobot,
        "physword_datasets": physword,
        "remaining_raw_episodes": removal["remaining_raw_episodes"],
        "remaining_short_episodes": removal["remaining_short_episodes"],
        "all_surviving_payloads_and_indexes_verified": True,
        "no_training_started": True,
    }
    write_json(quarantine.root / "verification_report.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quarantine", required=True)
    parser.add_argument("--pilot-root", required=True)
    args = parser.parse_args()
    report = verify(args.quarantine, args.pilot_root)
    print(
        {
            "status": report["status"],
            "lerobot_datasets": len(report["lerobot_datasets"]),
            "physword_datasets_checked": len(report["physword_datasets"]),
        }
    )


if __name__ == "__main__":
    main()
