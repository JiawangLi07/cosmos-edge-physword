"""Apply an audited original-episode exclusion across inspected local dataset copies."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from cosmos_framework.data.generator.physword.cli import build, qa, visualize
from cosmos_framework.data.generator.physword.common import PipelineConfig, digest, file_sha256, read_json, write_json
from tools.physword_data.curation import Quarantine, capture_identity, scan_short_episodes
from tools.physword_data.relabel_scene import publish_review
from tools.physword_data.source_audit import audit_inventory


def _matches(row, episode_id, original_id):
    return isinstance(row, dict) and (
        row.get("episode_id") == episode_id
        or row.get("original_episode_id") == original_id
        or row.get("lineage_id") == original_id
    )


def _filter_records(value, episode_id, original_id):
    if isinstance(value, list):
        return [
            _filter_records(item, episode_id, original_id)
            for item in value
            if not _matches(item, episode_id, original_id) and item != episode_id
        ]
    if isinstance(value, dict):
        return {key: _filter_records(item, episode_id, original_id) for key, item in value.items()}
    return value


def filter_physword(root, quarantine, episode_id, original_id):
    root = Path(root)
    inventory = read_json(root / "inventory.json")
    removed = [e for e in inventory["episodes"] if _matches(e, episode_id, original_id)]
    previous_exclusion = any(_matches(e, episode_id, original_id) for e in inventory.get("exclusions", []))
    if not removed and not previous_exclusion:
        return None
    split = read_json(root / "split_manifest.json")
    parent_split_hash = file_sha256(root / "split_manifest.json")
    split["episodes"] = [e for e in split["episodes"] if not _matches(e, episode_id, original_id)]
    split["original_group_count"] = len({e["group_id"] for e in split["episodes"]})
    split.setdefault("curation_parent_sha256", parent_split_hash)
    note = "One excluded original episode removed; surviving train/test assignments retained without reshuffling."
    if note not in split.setdefault("notes", []):
        split["notes"].append(note)
    inventory["episodes"] = [e for e in inventory["episodes"] if not _matches(e, episode_id, original_id)]
    inventory["counts"] = dict(Counter(e["dataset_source"] for e in inventory["episodes"]))
    inventory["missing_instruction"] = sum(not e.get("instruction") for e in inventory["episodes"])
    inventory["profiles"] = [p for p in inventory["profiles"] if not _matches(p, episode_id, original_id)]
    if not previous_exclusion:
        inventory.setdefault("exclusions", []).append(
            {"episode_id": episode_id, "original_episode_id": original_id, "reason": "user_confirmed_truncated_episode"}
        )
    entries = {e["episode_id"]: e for e in split["episodes"]}
    split["inventory_hash"] = digest(
        [
            dict(e, content_sha256=entries[e["episode_id"]]["content_sha256"])
            for e in sorted(inventory["episodes"], key=lambda e: e["episode_id"])
        ]
    )
    quarantine.replace_json(root / "inventory.json", inventory)
    quarantine.replace_json(root / "split_manifest.json", split)
    for filename in ("prepare_status.json", "annotation_status.json", "window_status.json"):
        if (root / filename).exists():
            quarantine.replace_json(
                root / filename, _filter_records(read_json(root / filename), episode_id, original_id)
            )
    quarantine.move(root / "processed" / episode_id)
    cache = root / "annotation_cache"
    bad_keys = set()
    for path in cache.glob("*.json"):
        data = read_json(path)
        if _matches(data, episode_id, original_id):
            bad_keys.add(path.name.split(".")[0])
    for key in sorted(bad_keys):
        for path in cache.glob(key + ".*"):
            quarantine.move(path)
    for filename in (
        "qa_stdout.json",
        "visualize_stdout.json",
        "browser_resource_check.json",
        "browser_script_check.json",
    ):
        quarantine.move(root / filename)
    for name in ("windows.jsonl", "window_status.json", "quality_report.json", "index.html"):
        quarantine.backup(root / name)
    for meta_path in (root / "processed").glob("*/metadata.json"):
        meta = read_json(meta_path)
        missing_hashes = any("sha256" not in frame for frame in meta["frame_mapping"]) or "video_sha256" not in meta
        if missing_hashes:
            for frame in meta["frame_mapping"]:
                frame.setdefault("sha256", file_sha256(meta_path.parent / frame["path"]))
            meta.setdefault("video_sha256", file_sha256(meta_path.parent / meta["video_path"]))
            meta["checksum_provenance"] = (
                "Legacy fixture hashes first recorded during episode cleanup; not a retrospective source-pixel identity claim"
            )
            quarantine.replace_json(meta_path, meta)
    config = PipelineConfig(**read_json(root / "config_resolved.json")["pipeline"])
    # Never create new windows in a source tree that had only media preparation.
    if (root / "windows.jsonl").exists():
        build(root, config)
        report = qa(root)
        visualize(root, config)
        counts = {
            key: report[key]
            for key in (
                "episodes_processed",
                "frames_processed",
                "windows",
                "split_episode_counts",
                "split_window_counts",
            )
        }
    else:
        counts = {"episodes_processed": len(list((root / "processed").glob("*/metadata.json")))}
    if (root / "source_audit.json").exists():
        quarantine.replace_json(root / "source_audit.json", audit_inventory(inventory))
    return {"dataset": str(root), "removed_inventory_records": max(len(removed), int(previous_exclusion)), **counts}


def _filter_download_manifest(path, source_name, quarantine):
    data = read_json(path)
    verified = data.get("verified_files") if isinstance(data, dict) else None
    if not isinstance(verified, list):
        return
    kept, removed = [], []
    for row in verified:
        if isinstance(row, dict) and source_name in Path(row.get("path", "")).parts:
            removed.append(row)
        else:
            kept.append(row)
    if not removed:
        return
    data["verified_files"] = kept
    for key in ("expected_files", "total_files"):
        if key in data:
            data[key] = len(kept)
    size = sum(row.get("bytes", 0) for row in kept)
    for key in ("expected_bytes", "total_bytes", "verified_bytes"):
        if key in data:
            data[key] = size
    if "episodes" in data:
        data["episodes"] -= 1
    if "downloaded_files" in data and all("mode" in row for row in kept):
        downloaded = [r for r in kept if r["mode"].startswith("downloaded")]
        reused = [r for r in kept if r["mode"].startswith("reused")]
        data["downloaded_files"] = len(downloaded)
        data["downloaded_bytes"] = sum(r.get("bytes", 0) for r in downloaded)
        if "reused_files" in data:
            data["reused_files"] = len(reused)
    data["local_curation_excluded_source_names"] = sorted(
        set(data.get("local_curation_excluded_source_names", []) + [source_name])
    )
    data["origin_revision_retained_but_local_membership_filtered"] = True
    quarantine.replace_json(path, data)


def apply_removal(raw_root, pilot_root, quarantine_root, source_name="episode_0038"):
    raw_root, pilot_root = Path(raw_root).resolve(), Path(pilot_root).resolve()
    quarantine = Quarantine(quarantine_root)
    target = raw_root / "episode_badminton" / source_name
    source_target = target if (target / "manifest.json").exists() else quarantine._destination(target, "removed")
    manifest = read_json(source_target / "manifest.json")
    original_id = capture_identity(manifest)
    episode_id = "badminton__" + source_name
    hashes = [file_sha256(source_target / name / "color_take001.mp4") for name in manifest["cameras"]]
    plans = read_json(quarantine.root / "lerobot_filter_plan.json")
    for plan in plans:
        if plan["removed_source_name"] != source_name:
            raise ValueError("Staged removal plan is mismatched")
        if not Path(plan["staging"]).is_dir():
            completed = Path(plan["source"]) / "provenance/episode_removal.json"
            if not completed.exists() or read_json(completed)["removed_source_name"] != source_name:
                raise ValueError("Neither staged nor already-committed filtered dataset exists")
    affected_roots = []
    for path in pilot_root.rglob("inventory.json"):
        data = read_json(path)
        if any(_matches(e, episode_id, original_id) for e in data.get("episodes", []) + data.get("exclusions", [])):
            affected_roots.append(path.parent)
    # All media and per-row arrays are already staged/checked before these root swaps.
    for plan in plans:
        source, stage = Path(plan["source"]), Path(plan["staging"])
        if stage.exists():
            quarantine.move(source)
            stage.rename(source)
    # Recompute source hashes against the now-filtered source tree, never shifted old IDs.
    for plan in plans:
        root = Path(plan["source"])
        if not (root / "meta/conversion.json").exists():
            continue
        contract = read_json(root / "meta/conversion.json")
        src = Path(contract["source"])
        table_paths = (
            sorted((src / "meta").rglob("*.json"))
            + sorted((src / "meta").rglob("*.parquet"))
            + sorted((src / "data").rglob("*.parquet"))
        )
        write_json(
            root / "provenance/source_hashes.json", {str(p.relative_to(src)): file_sha256(p) for p in table_paths}
        )
        video_records = read_json(root / "provenance/videos.json")
        for row in video_records:
            row["source"] = str(src / row["path"])
            assert file_sha256(src / row["path"]) == row["sha256"]
        write_json(root / "provenance/videos.json", video_records)
    # Original source numbers stay stable even though converted indexes become dense.
    alignment_reports = []
    for name in ("aligned_native_v2", "aligned_native_v2_146"):
        root = raw_root / name
        target_json = root / "episodes" / (source_name + ".json")
        old_record = read_json(target_json if target_json.exists() else quarantine._destination(target_json, "removed"))
        removed_index = old_record["episode_index"]
        quarantine.move(target_json)
        quarantine.move(root / "episodes" / (source_name + ".npz"))
        records = []
        for path in sorted((root / "episodes").glob("*.json")):
            record = read_json(path)
            if record["episode_index"] > removed_index and "original_converted_episode_index" not in record:
                record["original_converted_episode_index"] = record["episode_index"]
                record["episode_index"] -= 1
                quarantine.replace_json(path, record)
            records.append(record)
        report = read_json(root / "audit_report.json")
        if "episodes" in report:
            report["episodes"] = records
        for key in ("found_episodes", "prepared_episodes", "total_episodes"):
            if key in report:
                report[key] = len(records)
        values = {
            "total_robot_rows": sum(r.get("robot_rows", 0) for r in records),
            "total_hand_rows": sum(r.get("hand_rows", 0) for r in records),
            "total_aligned_rows": sum(r["aligned_rows"] for r in records),
            "total_rows": sum(r["aligned_rows"] for r in records),
            "total_invalid_rows": sum(r["invalid_rows"] for r in records),
            "total_valid_windows_33": sum(r["valid_windows_33"] for r in records),
            "valid_windows_33": sum(r["valid_windows_33"] for r in records),
        }
        for key, value in values.items():
            if key in report:
                report[key] = value
        report["excluded_source_names"] = [source_name]
        quarantine.replace_json(root / "audit_report.json", report)
        alignment_reports.append({"dataset": str(root), "remaining_episodes": len(records)})
    # Clean PhysWord versions deepest-first before relocating their shared source media.
    physword_reports = []
    for root in sorted(affected_roots, key=lambda p: len(p.parts), reverse=True):
        physword_reports.append(filter_physword(root, quarantine, episode_id, original_id))
    # Source copies have distinct paths/hard links, both must leave discovery roots.
    for name in ("episode_badminton", "episode_badminton_91"):
        root = raw_root / name
        quarantine.move(root / source_name)
        policy_path = root / "_episode_exclusions.json"
        policy = read_json(policy_path) if policy_path.exists() else {}
        policy["original_episode_ids"] = sorted(set(policy.get("original_episode_ids", []) + [original_id]))
        policy["source_sha256"] = sorted(set(policy.get("source_sha256", []) + hashes))
        policy["reason"] = "Explicit user exclusion: truncated recording; wrist2 capture timeout"
        quarantine.replace_json(policy_path, policy)
    # Stale cache images are not part of any surviving review.
    for directory in (pilot_root / "badminton/multiview_review", pilot_root / "badminton_inspection"):
        for path in directory.glob("*" + source_name + "*"):
            quarantine.move(path)
    for path in raw_root.rglob("*.json"):
        if "manifest" in path.name and path.name not in {"manifest.json"}:
            _filter_download_manifest(path, source_name, quarantine)
    active = pilot_root / "badminton"
    if (active / "scene_v2/quality_report.json").exists():
        for name in ("index.html", "index_v1.html"):
            quarantine.backup(active / name)
        publish_review(active, active / "scene_v2")
    audit = scan_short_episodes(raw_root / "episode_badminton")
    write_json(quarantine.root / "duration_audit_after.json", audit)
    result = {
        "removed_source_name": source_name,
        "removed_episode_id": episode_id,
        "removed_original_episode_id": original_id,
        "source_duration_s": manifest["stop_mono"] - manifest["start_mono"],
        "source_recording_status": manifest["status"],
        "source_recording_errors": manifest["recording_errors"],
        "lerobot_datasets": plans,
        "alignment_datasets": alignment_reports,
        "physword_datasets": physword_reports,
        "remaining_short_episodes": audit["short_episodes"],
        "remaining_raw_episodes": audit["episodes_scanned"],
        "quarantine_root": str(quarantine.root),
        "remote_mutations": False,
        "training_started": False,
    }
    write_json(quarantine.root / "removal_report.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", required=True)
    parser.add_argument("--pilot-root", required=True)
    parser.add_argument("--quarantine", required=True)
    parser.add_argument("--source-name", default="episode_0038")
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    if not args.apply:
        parser.error("Inspect duration_audit_before and staged LeRobot plan, then use --apply")
    result = apply_removal(args.raw_root, args.pilot_root, args.quarantine, args.source_name)
    print(
        json.dumps(
            {key: result[key] for key in ("removed_source_name", "remaining_raw_episodes", "remaining_short_episodes")},
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
