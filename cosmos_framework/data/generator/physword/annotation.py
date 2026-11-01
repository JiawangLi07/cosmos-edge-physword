"""Stage annotation contract, strict validation, and configurable offline/VLM adapters."""

from __future__ import annotations

import json
import math
import os
import subprocess
import time
import urllib.request
from pathlib import Path

import jsonschema

from cosmos_framework.data.generator.physword.common import ATOMIC_ACTIONS, digest, read_json, write_json

SCHEMA_PATH = Path(__file__).with_name("schemas") / "atomic_stages.schema.json"
PROMPT_PATH = Path(__file__).with_name("annotation_prompt.txt")


def validate_annotation(annotation, metadata):
    jsonschema.Draft202012Validator(read_json(SCHEMA_PATH)).validate(annotation)
    if (
        annotation["episode_id"] != metadata["episode"]["episode_id"]
        or annotation["media_fingerprint"] != metadata["fingerprint"]
    ):
        raise ValueError("Annotation is for different episode/media/preprocessing")
    for stage in annotation["stages"]:
        if not all(math.isfinite(stage[key]) for key in ("start_time", "end_time", "confidence")):
            raise ValueError("Nonfinite stage value")
    end = metadata["frame_mapping"][-1]["source_elapsed_s"] + 1 / metadata["model_fps"]
    previous_start = -1.0
    prior = []
    for stage in annotation["stages"]:
        start, stop = stage["start_time"], stage["end_time"]
        if not (0 <= start < stop <= end + 1e-7) or start < previous_start:
            raise ValueError("Stages must be sorted, positive-duration, and within episode bounds")
        for other in prior:
            if start < other["end_time"]:
                # Per-effector parallel stages are preserved. Every multi-label target is masked.
                if (
                    stage["effector"] not in {"left", "right"}
                    or other["effector"] not in {"left", "right"}
                    or stage["effector"] == other["effector"]
                ):
                    raise ValueError("Overlapping stages require distinct explicit left/right effectors")
        previous_start = start
        prior.append(stage)
    if annotation.get("effectors"):
        for effector in annotation["effectors"]:
            arm_stages = [stage for stage in annotation["stages"] if stage["effector"] == effector]
            if (
                not arm_stages
                or abs(arm_stages[0]["start_time"]) > 1e-7
                or abs(arm_stages[-1]["end_time"] - end) > 1e-7
            ):
                raise ValueError("Declared effectors need complete episode stage coverage, including idle/unknown")
            if any(abs(a["end_time"] - b["start_time"]) > 1e-7 for a, b in zip(arm_stages, arm_stages[1:])):
                raise ValueError("Per-arm stage gaps/overlaps are prohibited for declared effectors")
        if any(stage["effector"] not in annotation["effectors"] for stage in annotation["stages"]):
            raise ValueError("Stage does not belong to a declared effector")
    return annotation


def label_at(annotation, timestamp, config, effector=None):
    """Half-open stages on ACTUAL selected observation time, not rounded frame index."""
    stages = [s for s in annotation["stages"] if effector is None or s["effector"] == effector]
    matches = [s for s in stages if s["start_time"] <= timestamp < s["end_time"]]
    if not matches:
        return 0, False, "unannotated_gap"
    if effector is None:
        active = [s for s in matches if s["action"] != "idle"]
        if not active:
            return 0, False, "idle"
        matches = active
    if len(matches) > 1:
        return 0, False, "parallel_or_multilabel"
    stage = matches[0]
    if stage["ambiguous"] or stage["candidates"]:
        return 0, False, "ambiguous"
    if stage["action"] not in ATOMIC_ACTIONS:
        return 0, False, stage["action"]
    if stage["confidence"] < config.min_confidence:
        return 0, False, "low_confidence"
    # Ignore outer episode edges; uncertain internal transitions are masked.
    boundaries = {s["start_time"] for s in stages if s["start_time"] > 0}
    boundaries.update(s["end_time"] for s in stages[:-1])
    if any(abs(timestamp - b) <= config.boundary_tolerance_s + 1e-9 for b in boundaries):
        return 0, False, "stage_boundary_uncertainty"
    return ATOMIC_ACTIONS.index(stage["action"]), True, "valid"


def pending_annotation(metadata):
    return {
        "episode_id": metadata["episode"]["episode_id"],
        "media_fingerprint": metadata["fingerprint"],
        "time_coordinate": metadata["time_coordinate"],
        "stages": [
            {
                "start_time": 0.0,
                "end_time": metadata["frame_mapping"][-1]["source_elapsed_s"] + 1 / metadata["model_fps"],
                "action": "unknown",
                "confidence": 0.0,
                "effector": "unspecified",
                "ambiguous": False,
                "candidates": [],
                "evidence": "Pending visual annotation; no semantic GT asserted.",
            }
        ],
        "provenance": {"provider": "pending", "model": "none", "reviewed": False},
    }


class AnnotationAdapter:
    """VLM transport: explicit command or JSON HTTP contract; no implicit paid calls.

    Adapter input contains the COMPLETE episode video, all frames and true timestamps.
    Custom transport maps this to a provider's video API; output is the schema object.
    `json_file` imports previously produced annotations without invoking a model.
    """

    def __init__(self, config, cache_dir):
        self.config = config
        self.cache_dir = Path(cache_dir)

    def annotate(self, metadata, episode_dir, allow_network=False):
        episode_dir = Path(episode_dir).resolve()
        request = {
            "model": self.config.get("model"),
            "effectors": self.config.get("effectors", ["left", "right"]),
            "effector_coordinate": self.config.get("effector_coordinate", "selected_camera_image_sides"),
            "parameters": self.config.get("parameters", {}),
            "prompt": PROMPT_PATH.read_text(),
            "schema": read_json(SCHEMA_PATH),
            "episode_id": metadata["episode"]["episode_id"],
            "instruction": metadata["episode"].get("instruction"),
            "media_fingerprint": metadata["fingerprint"],
            "time_coordinate": metadata["time_coordinate"],
            "video_path": str(episode_dir / metadata["video_path"]),
            "frames": [{**r, "path": str(episode_dir / r["path"])} for r in metadata["frame_mapping"]],
        }
        public_config = {k: v for k, v in self.config.items() if k not in {"api_key"}}
        if "api_key" in self.config:
            raise ValueError("Use api_key_env, never store literal API keys in config")
        if self.config.get("kind") == "json_file":
            from cosmos_framework.data.generator.physword.common import file_sha256

            source_annotation = Path(self.config["directory"]) / f"{request['episode_id']}.json"
            public_config["source_annotation_sha256"] = file_sha256(source_annotation)
        key = digest({"request": request, "adapter": public_config})
        cached = self.cache_dir / f"{key}.json"
        state = self.cache_dir / f"{key}.state.json"
        if cached.exists():
            return validate_annotation(read_json(cached), metadata)
        kind = self.config.get("kind", "pending")
        if kind == "pending":
            return pending_annotation(metadata)
        if kind == "http_json" and not allow_network:
            raise PermissionError("HTTP annotation requires explicit --allow-network; review the pilot request first")
        attempts = int(self.config.get("max_attempts", 3))
        if not 1 <= attempts <= 10:
            raise ValueError("max_attempts must be 1..10")
        timeout = float(self.config.get("timeout_seconds", 120))
        if not 0 < timeout <= 3600:
            raise ValueError("timeout_seconds must be in (0,3600]")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        write_json(self.cache_dir / f"{key}.request.json", request)
        for attempt in range(1, attempts + 1):
            write_json(state, {"status": "running", "attempt": attempt, "episode_id": request["episode_id"]})
            try:
                if kind == "json_file":
                    annotation = read_json(Path(self.config["directory"]) / f"{request['episode_id']}.json")
                elif kind == "command":
                    # Explicit argv, no shell. Local command is responsible for its own provider/cost policy.
                    cmd = self.config["command"]
                    if not isinstance(cmd, list) or not cmd or not all(isinstance(s, str) for s in cmd):
                        raise ValueError("command must be a nonempty argv list")
                    response = subprocess.run(
                        cmd, input=json.dumps(request).encode(), capture_output=True, check=True, timeout=timeout
                    )
                    annotation = json.loads(response.stdout)
                elif kind == "http_json":
                    endpoint = self.config["endpoint"]
                    if not endpoint.startswith("https://"):
                        raise ValueError("Annotation HTTP transport requires HTTPS")
                    headers = {"Content-Type": "application/json"}
                    env = self.config.get("api_key_env")
                    if env:
                        headers["Authorization"] = "Bearer " + os.environ[env]
                    req = urllib.request.Request(endpoint, data=json.dumps(request).encode(), headers=headers)
                    with urllib.request.urlopen(req, timeout=timeout) as response:
                        annotation = json.load(response)
                else:
                    raise ValueError(f"Unknown annotation adapter: {kind}")
                validate_annotation(annotation, metadata)
                write_json(cached, annotation)
                write_json(state, {"status": "complete", "attempt": attempt, "cache_key": key})
                return annotation
            except Exception as exc:
                # Do not store response bodies, credentials or provider exception strings.
                write_json(state, {"status": "failed", "attempt": attempt, "error_type": type(exc).__name__})
                if attempt == attempts:
                    raise RuntimeError(f"Annotation failed ({type(exc).__name__}); see {state}") from None
                time.sleep(min(float(self.config.get("retry_backoff_seconds", 1)) * 2 ** (attempt - 1), 10))
        raise AssertionError("unreachable")
