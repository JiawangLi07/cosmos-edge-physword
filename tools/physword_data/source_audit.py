"""Read-only full-corpus manifest/clock audit; video decode coverage stays explicit."""

from __future__ import annotations

import argparse
import json
from collections import Counter

import numpy as np

from cosmos_framework.data.generator.physword.common import read_json, write_json
from cosmos_framework.data.generator.physword.media import read_capture_timestamps


def audit_inventory(inventory):
    rows, failures = [], []
    for episode in inventory["episodes"]:
        if episode["source_kind"] != "video":
            continue
        manifest = read_json(episode["source_ref"])
        for name, camera in episode["cameras"].items():
            if not camera.get("timestamp_path"):
                continue
            try:
                ts, wall = read_capture_timestamps(camera["timestamp_path"], camera["manifest_frames"])
                delta = np.diff(ts)
                wall_delta = np.asarray(wall, dtype=float) - ts
                details = manifest["cameras"][name]
                writer = details.get("diagnostics", {}).get("writer", {})
                rows.append(
                    {
                        "episode_id": episode["episode_id"],
                        "original_episode_id": episode["original_episode_id"],
                        "camera": name,
                        "video_path": camera["video_path"],
                        "timestamp_path": camera["timestamp_path"],
                        "manifest_frames": len(ts),
                        "timestamp_rows": len(ts),
                        "timestamp_span_s": float(ts[-1] - ts[0]),
                        "capture_first_mono": float(ts[0]),
                        "capture_last_mono": float(ts[-1]),
                        "writer_fps_declared": writer.get("fps"),
                        "max_capture_gap_s": float(delta.max()) if len(delta) else 0,
                        "measured_capture_fps": float((len(ts) - 1) / (ts[-1] - ts[0])) if len(ts) > 1 else None,
                        "wall_minus_mono_range_s": float(np.ptp(wall_delta)),
                        "frame_count_status": "manifest_vs_timestamp_only; ffprobe/decode verified only for pilot/profiles",
                    }
                )
            except Exception as exc:
                failures.append({"episode_id": episode["episode_id"], "camera": name, "error": type(exc).__name__})
    return {
        "episodes": len(inventory["episodes"]),
        "camera_timestamp_logs_checked": len(rows),
        "failures": failures,
        "source_frame_rows": sum(r["timestamp_rows"] for r in rows),
        "declared_writer_fps_counts": dict(Counter(str(r["writer_fps_declared"]) for r in rows)),
        "camera_counts": dict(Counter(r["camera"] for r in rows)),
        "largest_capture_gap_s": max((r["max_capture_gap_s"] for r in rows), default=None),
        "profiles_fully_probed": [p["episode_id"] for p in inventory["profiles"] if "audit" in p],
        "rows": rows,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = audit_inventory(read_json(args.inventory))
    write_json(args.output, result)
    print(json.dumps({k: v for k, v in result.items() if k != "rows"}))


if __name__ == "__main__":
    main()
