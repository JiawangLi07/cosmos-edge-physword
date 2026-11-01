"""PTS-aware, causal sample-and-hold onto a uniform model clock, no interpolation."""

from __future__ import annotations

import json
import math
import shutil
import subprocess
from fractions import Fraction
from pathlib import Path

import h5py
import numpy as np
from PIL import Image

from cosmos_framework.data.generator.physword.common import VERSION, digest, file_sha256, write_json


def run(args):
    return subprocess.run([str(a) for a in args], check=True, capture_output=True).stdout


def probe_video(path):
    raw = json.loads(
        run(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_streams",
                "-show_frames",
                "-show_entries",
                "stream=width,height,avg_frame_rate,r_frame_rate,nb_frames,duration:frame=best_effort_timestamp_time,pkt_duration_time",
                "-of",
                "json",
                path,
            ]
        )
    )
    stream = raw["streams"][0]
    pts = [float(f["best_effort_timestamp_time"]) for f in raw["frames"]]
    validate_timestamps(pts)
    return {
        "width": stream["width"],
        "height": stream["height"],
        "frame_count": len(pts),
        "source_fps": float(Fraction(stream["avg_frame_rate"])),
        "nominal_fps": stream["r_frame_rate"],
        "pts_seconds": pts,
        "container_duration": float(stream.get("duration", pts[-1] - pts[0])),
    }


def validate_timestamps(timestamps):
    t = np.asarray(timestamps, dtype=np.float64)
    if t.ndim != 1 or not len(t) or not np.isfinite(t).all() or (np.diff(t) <= 0).any():
        raise ValueError("Timestamps must be nonempty, finite, strictly increasing")
    return t


def read_capture_timestamps(path, expected):
    rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    frames = [r for r in rows if "i" in r]
    if len(frames) != expected or [r["i"] for r in frames] != list(range(1, expected + 1)):
        raise ValueError("Capture timestamps must map 1:1 in decode order (i=1..N)")
    return validate_timestamps([r["mono"] for r in frames]), [r.get("wall") for r in frames]


def time_grid(timestamps, fps):
    t = validate_timestamps(timestamps)
    if not math.isfinite(fps) or fps <= 0:
        raise ValueError("Invalid FPS")
    elapsed = t - t[0]
    grid = np.arange(int(np.floor(elapsed[-1] * fps + 1e-8)) + 1) / fps
    # Last actual observation AT OR BEFORE each grid point: no future RGB in history.
    indices = np.searchsorted(elapsed, grid + 1e-9, side="right") - 1
    return grid, indices, elapsed[indices]


def inspect_episode(ep):
    cameras = {}
    if ep["source_kind"] == "hdf5":
        with h5py.File(ep["source_ref"], "r") as f:
            g = f[ep["group"]]
            im = g[ep["image_key"]]
            if ep.get("timestamp_key"):
                ts = validate_timestamps(g[ep["timestamp_key"]][:])
                clock = "hdf5_timestamps"
            else:
                fps = ep.get("source_fps")
                if fps is None or not math.isfinite(fps) or fps <= 0:
                    raise ValueError("HDF5 without timestamps needs explicit positive source_fps")
                ts = np.arange(len(im)) / fps
                clock = "declared_fps_reconstructed_NOT_measured"
            if len(ts) != len(im):
                raise ValueError("HDF5 timestamps/image length mismatch")
            info = {
                "width": im.shape[2],
                "height": im.shape[1],
                "frame_count": len(im),
                "source_fps": ep.get("source_fps"),
                "pts_seconds": None,
                "timestamps": ts.tolist(),
                "clock": clock,
            }
            cameras[ep["camera"]] = info
    elif ep["source_kind"] == "video":
        for name, cam in ep["cameras"].items():
            if not Path(cam["video_path"]).is_file():
                cameras[name] = {"error": "missing_camera_video"}
                continue
            info = probe_video(cam["video_path"])
            if cam.get("timestamp_path"):
                ts, wall = read_capture_timestamps(cam["timestamp_path"], info["frame_count"])
                info.update(timestamps=ts.tolist(), wall_timestamps=wall, clock="capture_monotonic")
            elif cam.get("timestamp_values") is not None:
                ts = validate_timestamps(cam["timestamp_values"])
                if len(ts) != info["frame_count"]:
                    raise ValueError("Timestamp/frame count mismatch")
                info.update(timestamps=ts.tolist(), clock=cam.get("clock", "external"))
            else:
                info.update(timestamps=info["pts_seconds"], clock="container_pts")
            ts = np.array(info["timestamps"])
            pts = np.array(info["pts_seconds"])
            info["pts_capture_drift_s"] = float(np.max(np.abs((ts - ts[0]) - (pts - pts[0]))))
            info["max_source_gap_s"] = float(np.diff(ts).max()) if len(ts) > 1 else 0
            info["measured_fps"] = float((len(ts) - 1) / (ts[-1] - ts[0])) if len(ts) > 1 else None
            if cam.get("manifest_frames") is not None and cam["manifest_frames"] != info["frame_count"]:
                raise ValueError("Manifest frame count != decoded PTS count")
            cameras[name] = info
    else:
        raise ValueError("Unsupported source_kind")
    selected = cameras[ep["camera"]]
    if "error" in selected:
        raise ValueError(selected["error"])
    sync = {}
    for name, info in cameras.items():
        if name == ep["camera"] or "error" in info:
            continue
        if info["clock"] == selected["clock"] == "capture_monotonic":
            base = np.array(selected["timestamps"])
            other = np.array(info["timestamps"])
            pos = np.searchsorted(other, base).clip(0, len(other) - 1)
            prev = (pos - 1).clip(0, len(other) - 1)
            error = np.minimum(np.abs(other[pos] - base), np.abs(other[prev] - base))
            sync[name] = {
                "status": "shared_monotonic_clock",
                "start_offset_s": float(other[0] - base[0]),
                "end_offset_s": float(other[-1] - base[-1]),
                "nearest_p95_s": float(np.quantile(error, 0.95)),
                "nearest_max_s": float(error.max()),
            }
        else:
            sync[name] = {"status": "unverified_no_shared_clock"}
    return {"cameras": cameras, "sync": sync}


def process_episode(ep, output, config):
    output = Path(output).resolve() / ep["episode_id"]
    audit = inspect_episode(ep)
    info = audit["cameras"][ep["camera"]]
    timestamps = info["timestamps"]
    grid, indices, source_elapsed = time_grid(timestamps, config.sampling_fps)
    inputs = {"episode": ep, "config": config.to_dict(), "audit": audit, "version": VERSION}
    video = ep.get("cameras", {}).get(ep["camera"], {}).get("video_path", ep["source_ref"])
    inputs["source_sha256"] = file_sha256(video)
    fingerprint = digest(inputs)
    if (output / "metadata.json").exists():
        from cosmos_framework.data.generator.physword.common import read_json

        old = read_json(output / "metadata.json")
        if old["fingerprint"] != fingerprint:
            raise ValueError(f"Output exists with different inputs/config: {output}; use a new output root")
        if (
            all(
                (output / r["path"]).is_file() and r.get("sha256") == file_sha256(output / r["path"])
                for r in old["frame_mapping"]
            )
            and (output / old["video_path"]).is_file()
            and old.get("video_sha256") == file_sha256(output / old["video_path"])
        ):
            return old
    output.mkdir(parents=True, exist_ok=True)
    staging = output / ".staging"
    if staging.exists():
        shutil.rmtree(staging)
    (staging / "frames").mkdir(parents=True)
    mapping = []
    video_name = f"rgb_{config.sampling_fps:g}hz.mp4"
    if ep["source_kind"] == "video":
        # Decode sequentially with passthrough timestamps; no ffmpeg CFR insertion.
        args = [
            "ffmpeg",
            "-v",
            "error",
            "-threads",
            "1",
            "-i",
            video,
            "-map",
            "0:v:0",
            "-vsync",
            "0",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-threads",
            "1",
            "pipe:1",
        ]
        decoder = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        stream = decoder.stdout
        h5 = None
    else:
        h5 = h5py.File(ep["source_ref"], "r")
        decoder = None
        stream = None
    try:
        cursor = 0
        for source_index in range(info["frame_count"]):
            if stream:
                size = info["width"] * info["height"] * 3
                raw = stream.read(size)
                if len(raw) != size:
                    raise ValueError(f"Truncated/invalid decoded frame {source_index}")
                image = Image.frombytes("RGB", (info["width"], info["height"]), raw)
            else:
                image = Image.fromarray(h5[ep["group"]][ep["image_key"]][source_index])
            while cursor < len(indices) and indices[cursor] == source_index:
                name = f"frames/{cursor:06d}.png"
                image.save(staging / name)
                age = float(grid[cursor] - source_elapsed[cursor])
                mapping.append(
                    {
                        "sha256": file_sha256(staging / name),
                        "frame_index": cursor,
                        "source_frame_index": source_index,
                        "source_timestamp": timestamps[source_index],
                        "source_elapsed_s": float(source_elapsed[cursor]),
                        "source_pts_s": info["pts_seconds"][source_index] if info["pts_seconds"] else None,
                        "resampled_timestamp": float(grid[cursor]),
                        "frame_age_s": age,
                        "valid": age <= config.max_frame_age_s,
                        "image_std": float(np.asarray(image).std()),
                        "path": name,
                    }
                )
                cursor += 1
        if decoder and decoder.wait() != 0:
            raise ValueError("Video decode failed")
    finally:
        if stream:
            stream.close()
        if decoder and decoder.poll() is None:
            decoder.kill()
            decoder.wait()
        if h5:
            h5.close()
    if len(mapping) != len(grid):
        raise ValueError("Incomplete resampling")
    run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-threads",
            "1",
            "-framerate",
            config.sampling_fps,
            "-i",
            staging / "frames/%06d.png",
            "-c:v",
            "libx264",
            "-threads",
            "1",
            "-pix_fmt",
            "yuv420p",
            "-vf",
            "pad=ceil(iw/2)*2:ceil(ih/2)*2",
            staging / video_name,
        ]
    )
    if (output / "frames").exists():
        shutil.rmtree(output / "frames")
    (staging / "frames").replace(output / "frames")
    (staging / video_name).replace(output / video_name)
    staging.rmdir()
    meta = {
        "version": VERSION,
        "episode": ep,
        "fingerprint": fingerprint,
        "source_sha256": inputs["source_sha256"],
        "model_fps": config.sampling_fps,
        "sampling_method": "previous_observation_no_interpolation",
        "source_fps": info["source_fps"],
        "source_time_origin": timestamps[0],
        "time_coordinate": "seconds_since_first_selected_camera_observation",
        "frame_count": len(mapping),
        "width": info["width"],
        "height": info["height"],
        "video_path": video_name,
        "video_sha256": file_sha256(output / video_name),
        "frame_paths": [r["path"] for r in mapping],
        "frame_mapping": mapping,
        "audit": audit,
        "config": config.to_dict(),
    }
    write_json(output / "metadata.json", meta)
    return meta
