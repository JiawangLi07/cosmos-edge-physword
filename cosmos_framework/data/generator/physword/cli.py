"""CPU-only entry point: python -m cosmos_framework.data.generator.physword.cli."""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path

import h5py
import numpy as np
import yaml
from PIL import Image, ImageDraw

from cosmos_framework.data.generator.physword.adapters import discover_libero, discover_real, manifest_adapter
from cosmos_framework.data.generator.physword.annotation import (
    AnnotationAdapter,
    pending_annotation,
    validate_annotation,
)
from cosmos_framework.data.generator.physword.common import PipelineConfig, digest, read_json, write_json
from cosmos_framework.data.generator.physword.media import inspect_episode, process_episode
from cosmos_framework.data.generator.physword.quality import quality_report, visualize_episode
from cosmos_framework.data.generator.physword.split import build_split, check_leakage, source_content_hash
from cosmos_framework.data.generator.physword.windows import generate_windows


def load_config(path):
    path = Path(path).resolve()
    raw = yaml.safe_load(path.read_text())
    allowed = {"pipeline", "sources", "annotation"}
    if set(raw) - allowed:
        raise ValueError(f"Unknown top-level keys: {set(raw) - allowed}")
    for source in raw.get("sources", []):
        for key in ("root", "manifest"):
            if source.get(key):
                source[key] = str((path.parent / source[key]).resolve())
    adapter = raw.get("annotation", {})
    if adapter.get("directory"):
        adapter["directory"] = str((path.parent / adapter["directory"]).resolve())
    return PipelineConfig(**raw.get("pipeline", {})), raw


def audit(raw, config, output, probe_limit=3):
    output = Path(output)
    if (output / "inventory.json").exists():
        raise ValueError("Audit output exists; use a fresh root to preserve the split manifest")
    episodes, rejected = [], []
    for source in raw.get("sources", []):
        kind = source["adapter"]
        kwargs = {k: v for k, v in source.items() if k != "adapter"}
        if kind == "real_robot":
            found, bad = discover_real(**kwargs)
        elif kind == "libero_hdf5":
            found, bad = discover_libero(**kwargs)
        elif kind == "manifest":
            found, bad = manifest_adapter(kwargs["manifest"])
        else:
            raise ValueError(f"Unknown source adapter: {kind}")
        episodes.extend(found)
        rejected.extend(bad)
    probes = []
    # Representative probes, not a claim of full corpus decode verification.
    for ep in sorted(episodes, key=lambda e: e["episode_id"])[:probe_limit]:
        try:
            probes.append({"episode_id": ep["episode_id"], "audit": inspect_episode(ep)})
        except Exception as exc:
            probes.append({"episode_id": ep["episode_id"], "error": type(exc).__name__, "detail": str(exc)})
    split = build_split(episodes, config)
    check_leakage(split)
    write_json(output / "split_manifest.json", split)
    write_json(
        output / "config_resolved.json",
        {"pipeline": config.to_dict(), "sources": raw["sources"], "annotation": raw.get("annotation", {})},
    )
    inventory = {
        "episodes": episodes,
        "rejected": rejected,
        "profiles": probes,
        "counts": dict(Counter(e["dataset_source"] for e in episodes)),
        "rejection_reasons": dict(Counter(e["reason"] for e in rejected)),
        "missing_instruction": sum(not e.get("instruction") for e in episodes),
        "probe_limit": probe_limit,
        "config_hash": digest(config.to_dict()),
    }
    write_json(output / "inventory.json", inventory)
    return {
        "episodes": len(episodes),
        "rejected": len(rejected),
        "missing_instruction": inventory["missing_instruction"],
    }


def selected_episodes(inventory, names, limit):
    episodes = inventory["episodes"]
    if names:
        names = set(names)
        available = {ep["episode_id"] for ep in episodes}
        if names - available:
            raise ValueError(f"Unknown episode IDs: {names - available}")
        episodes = [ep for ep in episodes if ep["episode_id"] in names]
    if limit is not None:
        episodes = episodes[:limit]
    return episodes


def prepare(output, config, names=None, limit=3):
    output = Path(output)
    inventory = read_json(output / "inventory.json")
    if inventory["config_hash"] != digest(config.to_dict()):
        raise ValueError("Config differs from the audited split; use a new output root")
    successes, failures = [], []
    split_entries = {entry["episode_id"]: entry for entry in read_json(output / "split_manifest.json")["episodes"]}
    for ep in selected_episodes(inventory, names, limit):
        start = time.perf_counter()
        try:
            if source_content_hash(ep) != split_entries[ep["episode_id"]]["content_sha256"]:
                raise ValueError("Source RGB changed after episode split; create a fresh audited output")
            meta = process_episode(ep, output / "processed", config)
            path = output / "processed" / ep["episode_id"] / "annotation.json"
            if not path.exists():
                write_json(path, pending_annotation(meta))
            successes.append(
                {"episode_id": ep["episode_id"], "frames": meta["frame_count"], "seconds": time.perf_counter() - start}
            )
        except Exception as exc:
            failures.append({"episode_id": ep["episode_id"], "error": type(exc).__name__, "detail": str(exc)})
    result = {"successes": successes, "failures": failures}
    status_path = output / "prepare_status.json"
    prior = read_json(status_path) if status_path.exists() else {"successes": [], "failures": []}
    succeeded = {row["episode_id"]: row for row in prior["successes"]}
    failed = {row["episode_id"]: row for row in prior["failures"]}
    for row in successes:
        succeeded[row["episode_id"]] = row
        failed.pop(row["episode_id"], None)
    for row in failures:
        failed[row["episode_id"]] = row
        succeeded.pop(row["episode_id"], None)
    write_json(
        status_path, {"successes": list(succeeded.values()), "failures": list(failed.values()), "last_run": result}
    )
    return result


def annotate(output, raw, names=None, limit=3, allow_network=False, approved_pilot=None):
    output = Path(output)
    metadata = sorted((output / "processed").glob("*/metadata.json"))
    if names:
        metadata = [p for p in metadata if p.parent.name in names]
    if limit is not None:
        metadata = metadata[:limit]
    if len(metadata) > 3:
        if not approved_pilot:
            raise PermissionError(
                "Batch annotation requires --approved-pilot reviewed.json after pilot quality assessment"
            )
        review = read_json(approved_pilot)
        if review.get("approved_for_batch") is not True or not review.get("reviewer") or not review.get("pilot_report"):
            raise PermissionError("Pilot review must name reviewer, pilot_report and approved_for_batch=true")
    adapter = AnnotationAdapter(raw.get("annotation", {"kind": "pending"}), output / "annotation_cache")
    completed, failures = [], []
    for path in metadata:
        meta = read_json(path)
        try:
            annotation = adapter.annotate(meta, path.parent, allow_network=allow_network)
            write_json(path.parent / "annotation.json", annotation)
            completed.append(meta["episode"]["episode_id"])
        except Exception as exc:
            failures.append({"episode_id": meta["episode"]["episode_id"], "error": type(exc).__name__})
    result = {"completed": completed, "failures": failures, "adapter": raw.get("annotation", {}).get("kind", "pending")}
    write_json(output / "annotation_status.json", result)
    return result


def build(output, config):
    output = Path(output).resolve()
    split = read_json(output / "split_manifest.json")
    check_leakage(split)
    entries = {e["episode_id"]: e for e in split["episodes"]}
    records, statuses = [], []
    for path in sorted((output / "processed").glob("*/metadata.json")):
        meta = read_json(path)
        if meta["config"] != config.to_dict():
            raise ValueError("Build config differs from processed media")
        annotation = read_json(path.parent / "annotation.json")
        windows, status = generate_windows(
            meta, annotation, entries[meta["episode"]["episode_id"]], path.parent, config
        )
        records.extend(windows)
        statuses.append({"episode_id": meta["episode"]["episode_id"], **status})
    dest = output / "windows.jsonl"
    tmp = dest.with_suffix(".tmp")
    tmp.write_text("".join(json.dumps(r, ensure_ascii=False, allow_nan=False) + "\n" for r in records))
    tmp.replace(dest)
    write_json(output / "window_status.json", statuses)
    return {"windows": len(records), "episodes": len(statuses)}


def qa(output):
    output = Path(output).resolve()
    split = read_json(output / "split_manifest.json")
    records = [json.loads(line) for line in (output / "windows.jsonl").read_text().splitlines() if line]
    metadata = sorted((output / "processed").glob("*/metadata.json"))
    status_path = output / "prepare_status.json"
    failures = read_json(status_path)["failures"] if status_path.exists() else []
    return quality_report(metadata, records, split, output / "quality_report.json", failures)


def visualize(output, config):
    output = Path(output).resolve()
    records = [json.loads(line) for line in (output / "windows.jsonl").read_text().splitlines() if line]
    reports = []
    for path in sorted((output / "processed").glob("*/metadata.json")):
        meta = read_json(path)
        annotation = validate_annotation(read_json(path.parent / "annotation.json"), meta)
        selected = [r for r in records if r["episode_id"] == meta["episode"]["episode_id"]]
        reports.append(visualize_episode(meta, annotation, path.parent, selected, seed=config.seed))
    (output / "index.html").write_text(
        "<!doctype html><meta charset='utf-8'><h1>PhysWord pilot review</h1>"
        + "".join(f"<p><a href='{Path(p).relative_to(output)}'>{Path(p).parent.parent.name}</a></p>" for p in reports)
    )
    return {"reports": reports}


def fixture(output):
    """Construct synthetic RGB-only episodes; never represent them as real robot GT."""
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    episodes = []
    for i in range(10):
        name = f"synthetic_{i:04d}"
        path = output / f"{name}.hdf5"
        # Irregular timestamps exercise a non-integer sampling ratio and gaps.
        ts = np.r_[0, np.cumsum(np.tile([0.032, 0.048, 0.041, 0.039], 20))]
        frames = []
        for j, t in enumerate(ts):
            im = Image.new("RGB", (128, 96), (30 + i * 5, 35, 50))
            d = ImageDraw.Draw(im)
            x = int(10 + 25 * t)
            d.rectangle((x, 30, x + 15, 45), fill=(210, 80 + i * 3, 40))
            d.text((2, 76), f"SYNTH {i} {j}", fill="white")
            frames.append(np.asarray(im))
        with h5py.File(path, "w") as f:
            f.create_dataset("rgb", data=np.stack(frames))
            f.create_dataset("timestamps", data=ts)
            # Poison fields verify no adapter forwards non-RGB data.
            f.create_dataset("actions", data=np.ones((len(frames), 7)) * 999)
        episodes.append(
            {
                "episode_id": name,
                "original_episode_id": name,
                "lineage_id": name,
                "dataset_source": "synthetic_fixture",
                "collection": "synthetic_fixture",
                "task_id": f"synthetic_motion_{i % 2}",
                "instruction": "Observe the synthetic moving square",
                "source_kind": "hdf5",
                "source_ref": str(path),
                "group": "/",
                "image_key": "rgb",
                "timestamp_key": "timestamps",
                "source_fps": None,
                "camera": "rgb",
                "cameras": {},
            }
        )
    write_json(output / "episodes.json", episodes)
    cfg = {
        "pipeline": PipelineConfig().to_dict(),
        "sources": [{"adapter": "manifest", "manifest": "episodes.json"}],
        "annotation": {"kind": "pending"},
    }
    write_json(output / "config.json", cfg)
    return {"episodes": len(episodes), "config": str(output / "config.json")}


def fixture_labels(output):
    for path in sorted((Path(output) / "processed").glob("*/metadata.json")):
        meta = read_json(path)
        if meta["episode"]["dataset_source"] != "synthetic_fixture":
            raise ValueError("fixture-labels is prohibited on real data")
        annotation = pending_annotation(meta)
        end = annotation["stages"][0]["end_time"]
        base = {
            "confidence": 1.0,
            "effector": "single",
            "ambiguous": False,
            "candidates": [],
            "evidence": "Synthetic fixture moving square; constructed motion label, not a robot event.",
        }
        annotation["stages"] = [
            dict(base, start_time=0.0, end_time=1.6, action="move"),
            dict(base, start_time=1.6, end_time=end, action="move"),
        ]
        annotation["provenance"] = {"provider": "synthetic_fixture", "model": "constructed_motion", "reviewed": False}
        write_json(path.parent / "annotation.json", validate_annotation(annotation, meta))
    return {"status": "synthetic_only_labels_written"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command", choices=["audit", "prepare", "annotate", "build", "qa", "visualize", "fixture", "fixture-labels"]
    )
    parser.add_argument("--config")
    parser.add_argument("--output", required=True)
    parser.add_argument("--episode", action="append")
    parser.add_argument(
        "--limit", type=int, default=3, help="Pilot cap; 0 selects all, annotate >3 requires reviewed pilot"
    )
    parser.add_argument(
        "--probe-limit", type=int, default=3, help="Number of full multi-camera source probes during audit; 0=all"
    )
    parser.add_argument("--allow-network", action="store_true")
    parser.add_argument("--approved-pilot")
    args = parser.parse_args()
    start = time.perf_counter()
    if args.command == "fixture":
        result = fixture(args.output)
    elif args.command == "fixture-labels":
        result = fixture_labels(args.output)
    elif args.command == "qa":
        result = qa(args.output)
    else:
        if not args.config:
            parser.error("--config is required for this command")
        config, raw = load_config(args.config)
        limit = args.limit or None
        if args.limit < 0:
            parser.error("--limit must be nonnegative")
        if args.command == "audit":
            result = audit(raw, config, args.output, probe_limit=args.probe_limit or None)
        elif args.command == "prepare":
            result = prepare(args.output, config, args.episode, limit)
        elif args.command == "annotate":
            result = annotate(args.output, raw, args.episode, limit, args.allow_network, args.approved_pilot)
        elif args.command == "build":
            result = build(args.output, config)
        elif args.command == "visualize":
            result = visualize(args.output, config)
    print(
        json.dumps(
            {"command": args.command, "elapsed_seconds": time.perf_counter() - start, "result": result},
            ensure_ascii=False,
        )
    )
    if result.get("failures"):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
