"""Source-aware, recoverable removal utilities for inspected episode dataset layouts.

PyArrow is optional and needed only for LeRobot data curation. No training is run.
"""

from __future__ import annotations

import os
import shutil
from collections import Counter
from pathlib import Path

import numpy as np

from cosmos_framework.data.generator.physword.common import digest, file_sha256, read_json, write_json
from cosmos_framework.data.generator.physword.media import read_capture_timestamps


def capture_identity(manifest):
    return "capture:" + digest({key: manifest.get(key) for key in ("episode", "start_wall", "start_mono", "stop_wall")})


def scan_short_episodes(raw_root, threshold_seconds=5.0):
    if threshold_seconds <= 0:
        raise ValueError("Duration threshold must be positive")
    results = []
    for path in sorted(Path(raw_root).glob("*/manifest.json")):
        manifest = read_json(path)
        clocks = {}
        for camera, spec in manifest["cameras"].items():
            stamps, _ = read_capture_timestamps(path.parent / camera / "timestamps_take001.jsonl", spec["frames"])
            clocks[camera] = {
                "frames": len(stamps),
                "first_mono": float(stamps[0]),
                "last_mono": float(stamps[-1]),
                "duration_s": float(stamps[-1] - stamps[0]),
            }
        recording_duration = manifest["stop_mono"] - manifest["start_mono"]
        results.append(
            {
                "source_name": path.parent.name,
                "source_path": str(path.parent),
                "original_episode_id": capture_identity(manifest),
                "capture_number": manifest["episode"],
                "recording_duration_s": recording_duration,
                "cameras": clocks,
                "common_camera_duration_s": min(c["last_mono"] for c in clocks.values())
                - max(c["first_mono"] for c in clocks.values()),
                "status": manifest["status"],
                "recording_errors": manifest.get("recording_errors", []),
                "short": recording_duration < threshold_seconds or clocks["main"]["duration_s"] < threshold_seconds,
            }
        )
    return {
        "threshold_seconds": threshold_seconds,
        "episodes_scanned": len(results),
        "short_episodes": [r for r in results if r["short"]],
        "episodes": results,
    }


class Quarantine:
    """Every mutation gets a durable recovery record; active files are never truncated in place."""

    def __init__(self, root):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.journal_path = self.root / "journal.json"
        self.entries = read_json(self.journal_path) if self.journal_path.exists() else []

    def _destination(self, source, category):
        source = Path(source).absolute()
        return self.root / category / source.relative_to(source.anchor)

    def record(self, operation, source, destination):
        entry = {"operation": operation, "original": str(source), "quarantine": str(destination), "complete": False}
        self.entries.append(entry)
        write_json(self.journal_path, self.entries)
        return entry

    def backup(self, source):
        source = Path(source)
        if not source.exists() and not source.is_symlink():
            return
        destination = self._destination(source, "before")
        if destination.exists() or destination.is_symlink():
            return
        destination.parent.mkdir(parents=True, exist_ok=True)
        entry = self.record("backup", source, destination)
        if source.is_symlink():
            destination.symlink_to(os.readlink(source))
        elif source.is_file():
            os.link(source, destination)
        else:
            raise ValueError("Back up individual files; whole directories use move")
        entry["complete"] = True
        write_json(self.journal_path, self.entries)

    def move(self, source):
        source = Path(source)
        if not source.exists() and not source.is_symlink():
            return
        destination = self._destination(source, "removed")
        if destination.exists() or destination.is_symlink():
            raise FileExistsError(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        entry = self.record("move", source, destination)
        source.rename(destination)
        entry["complete"] = True
        write_json(self.journal_path, self.entries)

    def replace_text(self, path, text):
        path = Path(path)
        self.backup(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(path.name + ".curation-tmp")
        temporary.write_text(text)
        temporary.replace(path)  # breaks old hard links without mutating their contents

    def replace_json(self, path, data):
        import json

        self.replace_text(path, json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def _fit_stats(arrays):
    x = np.concatenate(arrays).astype(np.float64)
    return {
        "min": x.min(0).tolist(),
        "max": x.max(0).tolist(),
        "mean": x.mean(0).tolist(),
        "std": x.std(0).tolist(),
        "count": [len(x)],
    }


def _format(info, kind, chunk, file, **kwargs):
    return info[kind].format(chunk_index=chunk, file_index=file, **kwargs)


def stage_lerobot_removal(source, staging, source_name, alignment_root=None):
    """Build a dense, self-consistent filtered LeRobot tree before swapping any active root.

    Original training/validation membership and stats population are retained. Media
    is hard-linked by resolved inode, so renumbering never follows shifted symlinks.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    source, staging = Path(source).resolve(), Path(staging).resolve()
    if staging.exists():
        raise FileExistsError(staging)
    info = read_json(source / "meta/info.json")
    meta_paths = sorted((source / "meta/episodes").rglob("*.parquet"))
    if len(meta_paths) != 1:
        raise ValueError("Audited adapter expects one episode metadata shard")
    metadata = pq.read_table(meta_paths[0])
    rows = metadata.to_pylist()
    if len(rows) != info["total_episodes"] or [r["episode_index"] for r in rows] != list(range(len(rows))):
        raise ValueError("Source episode indexes must be dense and sorted")
    provenance_path = source / "provenance/episodes.json"
    provenance = read_json(provenance_path) if provenance_path.exists() else []
    provenance_by_id = {r["episode_index"]: r for r in provenance}
    alignments = {}
    if alignment_root:
        for path in sorted((Path(alignment_root) / "episodes").glob("*.json")):
            item = read_json(path)
            alignments[item["episode_index"]] = item
    for row in rows:
        old = row["episode_index"]
        name = (
            row.get("source_name")
            or provenance_by_id.get(old, {}).get("source_name")
            or alignments.get(old, {}).get("source_name")
        )
        if not name:
            raise ValueError("Cannot infer source lineage from converted file numbers")
        row["source_name"] = name
    removed = [r for r in rows if r["source_name"] == source_name]
    if not removed:
        raise ValueError("Requested source episode is absent")
    keep = [r for r in rows if r["source_name"] != source_name]
    split_membership = {}
    for split, interval in info["splits"].items():
        a, b = map(int, interval.split(":"))
        split_membership[split] = set(range(a, b))
    existing_stats = read_json(source / "meta/stats.json")
    features = ("action", "observation.state")
    all_counts = {f: sum(r[f"stats/{f}/count"][0] for r in rows) for f in features}
    train_counts = {
        f: sum(r[f"stats/{f}/count"][0] for r in rows if r["episode_index"] in split_membership["train"])
        for f in features
    }
    population = "all" if all(existing_stats[f]["count"][0] == all_counts[f] for f in features) else "train"
    if population == "train" and any(existing_stats[f]["count"][0] != train_counts[f] for f in features):
        raise ValueError("Unknown source statistics population; refusing to change normalization semantics")
    staging.mkdir(parents=True)
    for path in source.iterdir():
        if path.name in {"data", "videos", "meta", "provenance", "conversion_report.json", "validation_report.json"}:
            continue
        if path.is_file():
            shutil.copy2(path, staging / path.name)
    (staging / "provenance").mkdir()
    if (source / "provenance").exists():
        for path in (source / "provenance").iterdir():
            if path.is_file() and path.name not in {"episodes.json", "videos.json", "source_hashes.json"}:
                shutil.copy2(path, staging / "provenance" / path.name)
    (staging / "meta").mkdir()
    for path in (source / "meta").iterdir():
        if path.is_file() and path.name not in {
            "info.json",
            "stats.json",
            "normalization_stats.npy",
            "stats_provenance.json",
        }:
            shutil.copy2(path, staging / "meta" / path.name)
    aggregate = {f: [] for f in features}
    new_rows, new_provenance, videos = [], [], []
    new_split_ids = {split: [] for split in split_membership}
    offset = 0
    window_counts = Counter()
    invalid_rows = 0
    mapping = []
    used_data_files = set()
    for new_id, old_row in enumerate(keep):
        old_id = old_row["episode_index"]
        row = dict(old_row)
        old_rel = _format(info, "data_path", row["data/chunk_index"], row["data/file_index"])
        if old_rel in used_data_files:
            raise ValueError("Multiple episodes share one source shard; explicit row filtering adapter required")
        used_data_files.add(old_rel)
        table = pq.read_table(source / old_rel)
        n = table.num_rows
        if n != row["length"] or not np.all(table["episode_index"].to_numpy() == old_id):
            raise ValueError("Source episode table layout mismatch")
        original = table
        for key, values in (("episode_index", np.full(n, new_id)), ("index", np.arange(offset, offset + n))):
            table = table.set_column(
                table.column_names.index(key), key, pa.array(values, type=table.schema.field(key).type)
            )
        assert table.drop(["episode_index", "index"]).equals(original.drop(["episode_index", "index"]))
        row.update(episode_index=new_id, dataset_from_index=offset, dataset_to_index=offset + n)
        row["data/chunk_index"], row["data/file_index"] = 0, new_id
        new_rel = _format(info, "data_path", 0, new_id)
        dest = staging / new_rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(table, dest, compression="zstd")
        member_splits = [split for split, ids in split_membership.items() if old_id in ids]
        if len(member_splits) != 1:
            raise ValueError("Every source episode must belong to exactly one split")
        new_split_ids[member_splits[0]].append(new_id)
        window_counts[member_splits[0]] += int(row.get("valid_windows_33", 0))
        invalid_rows += int((~table["annotation.time_valid"].to_numpy()).sum())
        if population == "all" or old_id in split_membership["train"]:
            for feature, mask in (
                ("action", "annotation.action_valid"),
                ("observation.state", "annotation.time_valid"),
            ):
                aggregate[feature].append(
                    np.asarray(table[feature].to_pylist(), dtype=np.float64)[table[mask].to_numpy().astype(bool)]
                )
        for key in sorted(info["features"]):
            if not key.startswith("observation.images."):
                continue
            prefix = f"videos/{key}/"
            old_video = source / _format(
                info, "video_path", old_row[prefix + "chunk_index"], old_row[prefix + "file_index"], video_key=key
            )
            if old_row[prefix + "from_timestamp"] != 0:
                raise ValueError("Shared video containers require a separate slicing adapter")
            new_video_rel = _format(info, "video_path", 0, new_id, video_key=key)
            dst = staging / new_video_rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            os.link(old_video.resolve(strict=True), dst)
            assert os.path.samefile(dst, old_video)
            row[prefix + "chunk_index"], row[prefix + "file_index"] = 0, new_id
            videos.append(
                {
                    "path": new_video_rel,
                    "source": str(old_video),
                    "sha256": file_sha256(dst),
                    "source_name": row["source_name"],
                }
            )
        prov = dict(provenance_by_id.get(old_id, {}))
        prov.update(
            episode_index=new_id,
            original_converted_episode_index=old_id,
            source_name=row["source_name"],
            data_path=new_rel,
            rows=n,
            valid_windows_33=int(row.get("valid_windows_33", 0)),
        )
        if old_id in alignments:
            prov.setdefault("raw_robot_path", str(Path(alignments[old_id]["source_directory"]) / "robot_states.h5"))
            prov.setdefault("raw_robot_sha256", alignments[old_id]["source_robot_sha256"])
        new_provenance.append(prov)
        new_rows.append(row)
        mapping.append({"source_name": row["source_name"], "old_episode_index": old_id, "new_episode_index": new_id})
        offset += n
    new_splits = {}
    for split, ids in new_split_ids.items():
        if ids:
            assert ids == list(range(ids[0], ids[-1] + 1))
            new_splits[split] = f"{ids[0]}:{ids[-1] + 1}"
    info.update(total_episodes=len(new_rows), total_frames=offset, splits=new_splits)
    # The extra metadata source_name explicitly survives dense renumbering.
    schema = metadata.schema
    if "source_name" not in schema.names:
        schema = schema.append(pa.field("source_name", pa.string()))
    meta_dest = staging / meta_paths[0].relative_to(source)
    meta_dest.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(new_rows, schema=schema), meta_dest, compression="zstd")
    stats = {f: _fit_stats(aggregate[f]) for f in features}
    write_json(staging / "meta/stats.json", stats)
    write_json(staging / "meta/info.json", info)
    if (source / "meta/normalization_stats.npy").exists():
        original_norm = np.load(source / "meta/normalization_stats.npy", allow_pickle=True).item()
        if set(original_norm) != {"eef"} or len(original_norm["eef"]["min"]) != 80:
            raise ValueError("Unsupported normalization schema")
        pooled = np.concatenate(aggregate["action"] + aggregate["observation.state"])
        norm = {
            k: v.astype(np.float32)
            for k, v in {
                "min": pooled.min(0),
                "max": pooled.max(0),
                "mean": pooled.mean(0),
                "std": pooled.std(0),
                "q01": np.quantile(pooled, 0.01, axis=0),
                "q99": np.quantile(pooled, 0.99, axis=0),
            }.items()
        }
        active = np.zeros(80, dtype=bool)
        active[0:9] = active[10:32] = active[34:43] = active[44:66] = True
        exempt = ~active
        exempt[list(range(3, 9)) + list(range(37, 43))] = True
        for key, value in (("min", -1), ("max", 1), ("mean", 0), ("std", 1), ("q01", -1), ("q99", 1)):
            norm[key][exempt] = value
        np.save(staging / "meta/normalization_stats.npy", {"eef": norm})
    write_json(staging / "provenance/episodes.json", new_provenance)
    write_json(staging / "provenance/videos.json", videos)
    result = {
        "source": str(source),
        "removed_source_name": source_name,
        "removed_converted_episode_indexes": [r["episode_index"] for r in removed],
        "removed_rows": sum(r["length"] for r in removed),
        "remaining_episodes": len(new_rows),
        "remaining_rows": offset,
        "statistics_population": population,
        "statistics_counts": {f: stats[f]["count"][0] for f in features},
        "splits": new_splits,
        "valid_windows_33": dict(window_counts),
        "videos": len(videos),
        "episode_mapping": mapping,
    }
    write_json(staging / "provenance/episode_removal.json", result)
    write_json(
        staging / "meta/stats_provenance.json",
        {"population": population, "excluded_source_names": [source_name], "fit_after_episode_removal": True},
    )
    for filename in ("conversion_report.json", "meta/conversion.json"):
        if not (source / filename).exists():
            continue
        report = read_json(source / filename)
        for key in ("total_episodes", "episodes"):
            if key in report:
                report[key] = len(new_rows)
        for key in ("total_rows", "rows"):
            if key in report:
                report[key] = offset
        for key, value in (
            ("videos", len(videos)),
            ("train_episodes", len(new_split_ids.get("train", []))),
            ("val_episodes", len(new_split_ids.get("val", []))),
        ):
            if key in report:
                report[key] = value
        if "valid_windows_33" in report:
            report["valid_windows_33"] = (
                dict(window_counts) if isinstance(report["valid_windows_33"], dict) else sum(window_counts.values())
            )
        if "valid_windows" in report:
            report["valid_windows"] = dict(window_counts)
        report["episode_removal"] = {
            k: result[k] for k in ("removed_source_name", "removed_converted_episode_indexes", "removed_rows")
        }
        if "normalization_population" in report:
            report["normalization_population"] = (
                f"{population} retained episodes only; exclusion applied; original identity slot policy preserved"
            )
        if "invalid_time_rows_excluded" in report:
            report["invalid_time_rows_excluded"] = invalid_rows
        for key, feature in (("train_action_rows", "action"), ("train_state_rows", "observation.state")):
            if key in report:
                report[key] = stats[feature]["count"][0]
        write_json(staging / filename, report)
    write_json(
        staging / "validation_report.json",
        {
            "passed": True,
            "scope": "post-removal metadata/index/kept payload/media-inode/statistics checks; no retraining or FK recomputation",
            **{k: v for k, v in result.items() if k != "episode_mapping"},
        },
    )
    return result
