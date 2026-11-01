"""Adapters based on inspected source layouts; all original references survive export.

A JSON manifest adapter supports arbitrary video datasets without guessing layout.
LIBERO RGB HDF5 needs explicit timestamp key or user-declared FPS (never assumed).
"""

from __future__ import annotations

import json
from pathlib import Path

import h5py

from cosmos_framework.data.generator.physword.common import digest, read_json


def discover_real(root, collection, instruction=None, instruction_map=None, camera="main"):
    root = Path(root).resolve()
    if not root.is_dir():
        raise FileNotFoundError(root)
    episodes, rejected = [], []
    for path in sorted(root.glob("*/manifest.json")):
        m = read_json(path)
        identity = {k: m.get(k) for k in ("episode", "start_wall", "start_mono", "stop_wall")}
        # Original capture identity remains stable across renumbering/re-encoding.
        original = "capture:" + digest(identity)
        ep_id = f"{collection}__{path.parent.name}"
        text = (instruction_map or {}).get(ep_id, m.get("instruction", instruction))
        instruction_evidence = None
        inferred_source = None
        if isinstance(text, dict):
            instruction_evidence = text.get("evidence")
            inferred_source = text.get("source", "visual_inference")
            text = text["text"]
        cameras = {}
        for name in m.get("cameras", {}):
            video = path.parent / name / "color_take001.mp4"
            timestamps = path.parent / name / "timestamps_take001.jsonl"
            cameras[name] = {
                "video_path": str(video),
                "timestamp_path": str(timestamps),
                "manifest_frames": m["cameras"][name]["frames"],
            }
        if camera not in cameras or not Path(cameras[camera]["video_path"]).exists():
            rejected.append({"episode_id": ep_id, "reason": "missing_selected_rgb", "path": str(path)})
            continue
        episodes.append(
            {
                "episode_id": ep_id,
                "original_episode_id": original,
                "lineage_id": original,
                "dataset_source": "real_robot",
                "collection": collection,
                "task_id": collection,
                "instruction": text,
                "instruction_evidence": instruction_evidence,
                "instruction_source": inferred_source
                or (
                    "override"
                    if instruction_map and ep_id in instruction_map
                    else "manifest"
                    if m.get("instruction")
                    else "collection_metadata"
                    if instruction
                    else "missing"
                ),
                "source_kind": "video",
                "camera": camera,
                "cameras": cameras,
                "source_ref": str(path),
                "manifest_episode_number": m.get("episode"),
                "manifest_start_mono": m.get("start_mono"),
                "warnings": [] if text else ["missing_instruction: excluded from training windows"],
            }
        )
    return episodes, rejected


def discover_libero(root, image_key="obs/agentview_rgb", timestamp_key=None, source_fps=None, instructions=None):
    root = Path(root).resolve()
    if not root.is_dir():
        raise FileNotFoundError(root)
    episodes, rejected = [], []
    for path in sorted(root.glob("*.hdf5")):
        with h5py.File(path, "r") as f:
            for name, group in f.get("data", {}).items():
                if not isinstance(group, h5py.Group):
                    continue
                ep_id = f"libero__{path.stem}__{name}"
                if image_key not in group:
                    rejected.append(
                        {
                            "episode_id": ep_id,
                            "reason": "missing_rgb",
                            "path": str(path),
                            "available_keys": list(group.keys()),
                        }
                    )
                    continue
                image = group[image_key]
                if image.ndim != 4 or image.shape[-1] != 3 or image.dtype.name != "uint8":
                    rejected.append(
                        {"episode_id": ep_id, "reason": "unsupported_rgb_layout", "shape": list(image.shape)}
                    )
                    continue
                if timestamp_key is None and source_fps is None:
                    rejected.append({"episode_id": ep_id, "reason": "missing_timestamps_and_declared_fps"})
                    continue
                # Instruction comes from stored language or explicit mapping, never filename guessing.
                attrs = dict(f["data"].attrs)
                problem = attrs.get("problem_info", "{}")
                if isinstance(problem, bytes):
                    problem = problem.decode()
                problem = json.loads(problem) if isinstance(problem, str) else {}
                instruction = (instructions or {}).get(path.stem, problem.get("language_instruction"))
                original = "libero:" + digest({"task": path.stem, "demo": name})
                episodes.append(
                    {
                        "episode_id": ep_id,
                        "original_episode_id": original,
                        "lineage_id": original,
                        "dataset_source": "LIBERO",
                        "collection": "LIBERO",
                        "task_id": path.stem,
                        "instruction": instruction,
                        "instruction_source": "hdf5_or_override",
                        "source_kind": "hdf5",
                        "source_ref": str(path),
                        "group": f"data/{name}",
                        "image_key": image_key,
                        "timestamp_key": timestamp_key,
                        "source_fps": source_fps,
                        "camera": image_key,
                        "cameras": {},
                        "warnings": [] if instruction else ["missing_instruction"],
                    }
                )
    return episodes, rejected


def manifest_adapter(path):
    """Explicit episode JSON list. Paths resolve relative to this manifest, not cwd.

    Video records require lineage_id/original_episode_id even for transformed copies.
    Optional per-camera timestamp_values + timestamp_origin support external clocks.
    """
    path = Path(path).resolve()
    episodes = read_json(path)
    ids = set()
    for ep in episodes:
        for key in (
            "episode_id",
            "original_episode_id",
            "lineage_id",
            "dataset_source",
            "task_id",
            "source_kind",
            "camera",
        ):
            if not ep.get(key):
                raise ValueError(f"Manifest episode missing {key}")
        if ep["episode_id"] in ids:
            raise ValueError("Duplicate episode ID")
        ids.add(ep["episode_id"])
        if Path(ep["episode_id"]).name != ep["episode_id"] or ep["episode_id"] in {".", ".."}:
            raise ValueError("Unsafe episode_id")
        ep.setdefault("warnings", [])
        ep.setdefault("collection", ep["dataset_source"])
        for cam in ep.get("cameras", {}).values():
            for key in ("video_path", "timestamp_path"):
                if cam.get(key):
                    cam[key] = str((path.parent / cam[key]).resolve())
        if ep.get("source_ref"):
            ep["source_ref"] = str((path.parent / ep["source_ref"]).resolve())
    return episodes, []
