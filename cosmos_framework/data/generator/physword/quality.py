"""Executable data invariants, label statistics and standalone visual inspection."""

from __future__ import annotations

import html
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from cosmos_framework.data.generator.physword.annotation import label_at, validate_annotation
from cosmos_framework.data.generator.physword.common import (
    PipelineConfig,
    digest,
    file_sha256,
    read_json,
    write_json,
)
from cosmos_framework.data.generator.physword.media import probe_video, time_grid
from cosmos_framework.data.generator.physword.split import check_leakage
from cosmos_framework.data.generator.physword.vocabulary import vocabulary_for, vocabulary_hash


def validate_record(record, metadata, annotation, config):
    vocabulary = vocabulary_for(annotation)
    words = vocabulary["words"]
    assert record["pw_vocabulary_id"] == vocabulary["id"]
    assert record["pw_vocabulary"] == words and record["pw_vocabulary_hash"] == vocabulary_hash(vocabulary)
    h, f, p = (record[k] for k in ("history_frame_indices", "future_frame_indices", "pw_target_frame_indices"))
    assert len(h) == config.history_frames and len(f) == config.future_frames, "history/future length"
    assert h + f == list(range(h[0], h[0] + config.total_frames)), "history/future boundary"
    assert not set(h) & set(f) and max(h) < min(f), "history/future overlap"
    assert p == [h[0] + o for o in config.target_offsets] and set(p) <= set(f), "PW offsets"
    assert record["media_fingerprint"] == metadata["fingerprint"], "changed media"
    assert record["annotation_hash"] == digest(annotation), "changed annotation"
    frames = metadata["frame_mapping"]
    for prefix, indexes in (("history", h), ("future", f)):
        assert record[f"{prefix}_source_timestamps"] == [frames[i]["source_timestamp"] for i in indexes]
        assert record[f"{prefix}_timestamps"] == [frames[i]["resampled_timestamp"] for i in indexes]
        assert record[f"{prefix}_source_frame_indices"] == [frames[i]["source_frame_index"] for i in indexes]
    assert (
        len(record["pw_class_targets"]) == len(record["pw_valid_mask"]) == len(record["pw_target_timestamps"]) == len(p)
    )
    for j, i in enumerate(p):
        assert 0 <= record["pw_class_targets"][j] < len(words), "invalid head class"
        assert record["pw_target_timestamps"][j] == frames[i]["source_elapsed_s"], "target time misalignment"
        assert record["pw_target_source_timestamps"][j] == frames[i]["source_timestamp"]
        assert record["pw_target_source_frame_indices"][j] == frames[i]["source_frame_index"]
        expected = label_at(annotation, frames[i]["source_elapsed_s"], config)
        assert (record["pw_class_targets"][j], record["pw_valid_mask"][j], record["pw_mask_reasons"][j]) == expected, (
            "mask/label alignment"
        )
    expected_effectors = annotation.get("effectors") or sorted({s["effector"] for s in annotation["stages"]})
    assert record["pw_effectors"] == expected_effectors, "arm identity/order"
    assert (
        len(record["pw_arm_class_targets"])
        == len(record["pw_arm_valid_mask"])
        == len(record["pw_arm_mask_reasons"])
        == len(expected_effectors)
    )
    for arm_index, arm in enumerate(expected_effectors):
        assert len(record["pw_arm_class_targets"][arm_index]) == len(record["pw_arm_valid_mask"][arm_index]) == len(p)
        for j, index in enumerate(p):
            actual = (
                record["pw_arm_class_targets"][arm_index][j],
                record["pw_arm_valid_mask"][arm_index][j],
                record["pw_arm_mask_reasons"][arm_index][j],
            )
            assert actual == label_at(annotation, frames[index]["source_elapsed_s"], config, effector=arm), (
                "per-arm label/mask alignment"
            )
    assert all(frames[i]["valid"] for i in h + f), "invalid frames in training window"
    assert all(frames[i]["source_elapsed_s"] <= frames[i]["resampled_timestamp"] + 1e-8 for i in h), (
        "future observation in history"
    )
    forbidden = {"action", "actions", "proprioception", "tactile", "state", "condition"}
    assert not forbidden & set(record), "unexpected model condition fields"


def _timeline(annotation, width, duration):
    result = []
    for stage in annotation["stages"]:
        x = width * stage["start_time"] / duration
        w = width * (stage["end_time"] - stage["start_time"]) / duration
        color = "#d5d9dd" if stage["action"] in {"unknown", "idle"} else "#a1d6c8"
        y = 45 if stage["effector"] == "right" else 5
        title = html.escape(
            f"{stage['effector']} {stage['action']} {stage['start_time']:.3f}..{stage['end_time']:.3f}: {stage['evidence']}"
        )
        result.append(
            f'<g><title>{title}</title><rect x="{x}" y="{y}" width="{w}" height="35" fill="{color}" stroke="white"/><text x="{x + 2}" y="{y + 20}" font-size="10">{html.escape(stage["action"])}</text></g>'
        )
    return f'<svg viewBox="0 0 {width} 85" role="img" aria-label="Stage timeline">' + "".join(result) + "</svg>"


def visualize_episode(metadata, annotation, episode_dir, records, seed=42, full_contact_sheets=True):
    words = vocabulary_for(annotation)["words"]
    episode_dir = Path(episode_dir)
    report_dir = episode_dir / "review"
    report_dir.mkdir(exist_ok=True)
    frames = metadata["frame_mapping"]
    sheets = []
    if full_contact_sheets:
        # Every processed observation is visible, in time order, rather than highlights only.
        for first in range(0, len(frames), 60):
            subset = frames[first : first + 60]
            canvas = Image.new("RGB", (6 * 192, ((len(subset) + 5) // 6) * 166), "white")
            draw = ImageDraw.Draw(canvas)
            for offset, frame in enumerate(subset):
                with Image.open(episode_dir / frame["path"]) as im:
                    im.thumbnail((192, 142))
                    x, y = offset % 6 * 192, offset // 6 * 166
                    canvas.paste(im, (x, y))
                    draw.text(
                        (x + 2, y + 144), f"#{frame['frame_index']} src={frame['source_elapsed_s']:.3f}s", fill="black"
                    )
            name = f"contact_{first:06d}.jpg"
            canvas.save(report_dir / name)
            sheets.append(name)
    rng = random.Random(seed)
    samples = rng.sample(records, min(3, len(records)))
    gallery = []
    for record in samples:
        indexes = record["history_frame_indices"] + record["pw_target_frame_indices"]
        canvas = Image.new("RGB", (192 * len(indexes), 224), "white")
        draw = ImageDraw.Draw(canvas)
        for j, i in enumerate(indexes):
            with Image.open(episode_dir / frames[i]["path"]) as im:
                im.thumbnail((192, 142))
                canvas.paste(im, (j * 192, 0))
            text = f"H #{i}" if j < len(record["history_frame_indices"]) else f"F #{i}"
            if j >= len(record["history_frame_indices"]):
                k = j - len(record["history_frame_indices"])
                if "vocabulary" not in annotation:
                    text += " " + (words[record["pw_class_targets"][k]] if record["pw_valid_mask"][k] else "MASK")
            draw.text((j * 192 + 2, 145), text, fill="black")
            draw.text((j * 192 + 2, 162), f"src={frames[i]['source_elapsed_s']:.3f}s", fill="black")
            if j >= len(record["history_frame_indices"]):
                k = j - len(record["history_frame_indices"])
                for arm_index, arm in enumerate(record["pw_effectors"]):
                    valid = record["pw_arm_valid_mask"][arm_index][k]
                    word = (
                        words[record["pw_arm_class_targets"][arm_index][k]]
                        if valid or "vocabulary" in annotation
                        else "MASK"
                    )
                    if not valid and "vocabulary" in annotation:
                        word += " *"
                    draw.text((j * 192 + 2, 180 + 16 * arm_index), f"{arm}: {word}", fill="black")
        name = f"{record['window_id']}.jpg"
        canvas.save(report_dir / name)
        gallery.append(name)

    def vtt_time(seconds):
        milliseconds = round(seconds * 1000)
        return f"{milliseconds // 3600000:02d}:{milliseconds // 60000 % 60:02d}:{milliseconds // 1000 % 60:02d}.{milliseconds % 1000:03d}"

    cues = ["WEBVTT", ""]
    for i, frame in enumerate(frames):
        matches = [s for s in annotation["stages"] if s["start_time"] <= frame["source_elapsed_s"] < s["end_time"]]
        text = (
            " | ".join(f"{s['effector']}: {s['action']}" + (" [ambiguous]" if s["ambiguous"] else "") for s in matches)
            or "unknown"
        )
        if "vocabulary" not in annotation and len([stage for stage in matches if stage["action"] != "idle"]) > 1:
            text += " [legacy single-sequence MASK]"
        start = frame["resampled_timestamp"]
        cues.extend(
            [
                f"{vtt_time(start)} --> {vtt_time(start + 1 / metadata['model_fps'])}",
                text + f"; source={frame['source_elapsed_s']:.3f}s",
                "",
            ]
        )
    (report_dir / "stages.vtt").write_text("\n".join(cues))
    duration = frames[-1]["source_elapsed_s"] + 1 / metadata["model_fps"]
    body = [
        "<!doctype html><meta charset='utf-8'><title>PhysWord episode review</title>",
        "<style>body{font-family:sans-serif;max-width:1250px;margin:auto}img,svg{max-width:100%}video{max-width:640px}</style>",
        f"<h1>{html.escape(metadata['episode']['episode_id'])}</h1>",
        f"<p>Instruction: {html.escape(metadata['episode'].get('instruction') or 'MISSING — excluded from training')}</p>",
        f"<p>Annotation provenance: {html.escape(str(annotation['provenance']))}. Full sequence contact sheets use actual source times.</p>",
        f"<video id='episode-video' controls src='../{html.escape(metadata['video_path'])}'><track kind='captions' src='stages.vtt' srclang='en' label='Atomic stages' default></video>",
        f"<p>Vocabulary: {html.escape(vocabulary_for(annotation)['id'])}; independent left/right labels. * marks a masked supervision boundary, not an absent action label.</p>",
        "<p id='current-stages' style='font-size:1.3em;font-weight:bold'></p>",
        "<p>Timeline lanes: upper=left, lower=right. Click a stage start time in the table to seek the video.</p>",
        _timeline(annotation, 1200, duration),
        "<h2>Independent arm stages</h2><table border='1' cellpadding='5'><tr><th>Arm</th><th>Source start</th><th>Source end</th><th>Word</th><th>Confidence</th></tr>",
        *[
            f"<tr><td>{html.escape(stage['effector'])}</td><td><button class='seek-stage' data-source='{stage['start_time']}'>{stage['start_time']:.3f}</button></td><td>{stage['end_time']:.3f}</td><td>{html.escape(stage['action'])}</td><td>{stage['confidence']:.2f}</td></tr>"
            for stage in annotation["stages"]
        ],
        "</table>",
        "<h2>History + future PW keyframes</h2>",
    ]
    body.extend(f"<img src='{html.escape(name)}' alt='History and future keyframes'>" for name in gallery)
    body.append("<h2>Complete episode at model observation rate</h2>")
    body.extend(f"<img src='{name}' alt='Consecutive observations'>" for name in sheets)
    # Only encode trusted scalar timing/action fields, escaping '<' for script embedding.
    payload = {
        "frames": [{"source": frame["source_elapsed_s"], "playback": frame["resampled_timestamp"]} for frame in frames],
        "stages": [
            {key: stage[key] for key in ("start_time", "end_time", "action", "effector")}
            for stage in annotation["stages"]
        ],
    }
    serialized = json.dumps(payload).replace("<", "\\u003c")
    body.append(
        "<script>const review="
        + serialized
        + ";"
        + """
const video=document.getElementById('episode-video');
function showStages(){let frame=review.frames[0];for(const item of review.frames){if(item.playback>video.currentTime+1e-7)break;frame=item;}
const active=review.stages.filter(s=>s.start_time<=frame.source && frame.source<s.end_time);
document.getElementById('current-stages').textContent=active.map(s=>s.effector+': '+s.action).join(' | ')+' ; source='+frame.source.toFixed(3)+'s';}
video.addEventListener('timeupdate',showStages);showStages();
document.querySelectorAll('.seek-stage').forEach(button=>button.addEventListener('click',()=>{const source=Number(button.dataset.source);const frame=review.frames.find(f=>f.source>=source)||review.frames[review.frames.length-1];video.currentTime=frame.playback;showStages();}));
</script>"""
    )
    (report_dir / "index.html").write_text("\n".join(body))
    return str(report_dir / "index.html")


def quality_report(metadata_list, records, split, output, failures=None):
    check_leakage(split)
    by_ep = defaultdict(list)
    entries = {e["episode_id"]: e for e in split["episodes"]}
    processed_ids = {read_json(path)["episode"]["episode_id"] for path in metadata_list}
    for record in records:
        assert record["episode_id"] in processed_ids, "window references unprocessed episode"
        entry = entries[record["episode_id"]]
        for key in ("split", "group_id", "original_episode_id", "lineage_id"):
            assert record[key] == entry[key], f"window/split manifest disagreement: {key}"
        by_ep[record["episode_id"]].append(record)
    vocabularies = [vocabulary_for(read_json(Path(path).parent / "annotation.json")) for path in metadata_list]
    if len({vocabulary_hash(v) for v in vocabularies}) > 1:
        raise ValueError("Quality report cannot merge distinct vocabularies")
    vocabulary = vocabularies[0] if vocabularies else vocabulary_for({})
    words = vocabulary["words"]
    tasks = defaultdict(
        lambda: {
            "episodes": 0,
            "windows": 0,
            "windows_with_valid_pw": 0,
            "targets": 0,
            "valid_targets": 0,
            "words": Counter(),
            "mask_reasons": Counter(),
            "unique_frame_words": Counter(),
            "unique_frame_mask_reasons": Counter(),
            "stage_words": Counter(),
            "stage_durations_s": Counter(),
        }
    )
    episodes = []
    for meta_path in metadata_list:
        meta = read_json(meta_path)
        directory = Path(meta_path).parent
        config = PipelineConfig(**meta["config"])
        annotation = validate_annotation(read_json(directory / "annotation.json"), meta)
        frames = meta["frame_mapping"]
        assert meta["frame_count"] == len(frames) > 0
        grid = np.array([r["resampled_timestamp"] for r in frames])
        assert np.allclose(grid, np.arange(len(grid)) / meta["model_fps"], atol=1e-8), "FPS grid"
        assert [f["frame_index"] for f in frames] == list(range(len(frames)))
        assert all(f["source_timestamp"] == meta["source_time_origin"] + f["source_elapsed_s"] for f in frames)
        assert all(f["source_elapsed_s"] <= f["resampled_timestamp"] + 1e-8 for f in frames)
        source_info = meta["audit"]["cameras"][meta["episode"]["camera"]]
        expected_grid, expected_indices, expected_elapsed = time_grid(source_info["timestamps"], meta["model_fps"])
        assert len(expected_grid) == len(frames)
        for index, frame in enumerate(frames):
            source_index = int(expected_indices[index])
            assert frame["source_frame_index"] == source_index, "source frame mapping"
            assert frame["source_timestamp"] == source_info["timestamps"][source_index], "source timestamp mapping"
            assert abs(frame["source_elapsed_s"] - expected_elapsed[index]) < 1e-9
            age = frame["resampled_timestamp"] - frame["source_elapsed_s"]
            assert abs(frame["frame_age_s"] - age) < 1e-9
            assert frame["valid"] == (age <= config.max_frame_age_s)
            if source_info["pts_seconds"]:
                assert frame["source_pts_s"] == source_info["pts_seconds"][source_index]
            assert file_sha256(directory / frame["path"]) == frame["sha256"], "frame checksum changed"
            with Image.open(directory / frame["path"]) as im:
                im.load()
                assert im.mode == "RGB" and im.size == (meta["width"], meta["height"]), "image decode/resolution"
        assert file_sha256(directory / meta["video_path"]) == meta["video_sha256"], "video checksum changed"
        video = probe_video(directory / meta["video_path"])
        assert video["frame_count"] == len(frames) and abs(video["source_fps"] - meta["model_fps"]) < 1e-6
        assert np.allclose(video["pts_seconds"], expected_grid, atol=1e-6), "exported video PTS grid"
        assert (video["width"], video["height"]) == (
            meta["width"] + meta["width"] % 2,
            meta["height"] + meta["height"] % 2,
        )
        ep_records = by_ep[meta["episode"]["episode_id"]]
        task = tasks[meta["episode"]["task_id"]]
        task["episodes"] += 1
        for name in ("unique_frame_words", "unique_frame_mask_reasons", "stage_words", "stage_durations_s"):
            task.setdefault(name, Counter())
        for stage in annotation["stages"]:
            task["stage_words"][stage["action"]] += 1
            task["stage_durations_s"][stage["action"]] += stage["end_time"] - stage["start_time"]
        for frame in frames:
            label, valid, reason = label_at(annotation, frame["source_elapsed_s"], config)
            task["unique_frame_words"][words[label]] += int(valid)
            if not valid:
                task["unique_frame_mask_reasons"][reason] += 1
        task.setdefault("arms", {})
        for arm in annotation.get("effectors") or sorted({s["effector"] for s in annotation["stages"]}):
            arm_stats = task["arms"].setdefault(
                arm, {"targets": 0, "valid_targets": 0, "words": Counter(), "mask_reasons": Counter()}
            )
            arm_stats.setdefault("unique_frame_words", Counter())
            arm_stats.setdefault("unique_frame_mask_reasons", Counter())
            for frame in frames:
                label, valid, reason = label_at(annotation, frame["source_elapsed_s"], config, effector=arm)
                if valid:
                    arm_stats["unique_frame_words"][words[label]] += 1
                else:
                    arm_stats["unique_frame_mask_reasons"][reason] += 1
            for record in ep_records:
                arm_index = record["pw_effectors"].index(arm)
                for label, valid, reason in zip(
                    record["pw_arm_class_targets"][arm_index],
                    record["pw_arm_valid_mask"][arm_index],
                    record["pw_arm_mask_reasons"][arm_index],
                ):
                    arm_stats["targets"] += 1
                    arm_stats["valid_targets"] += int(valid)
                    if valid:
                        arm_stats["words"][words[label]] += 1
                    else:
                        arm_stats["mask_reasons"][reason] += 1
        for record in ep_records:
            validate_record(record, meta, annotation, config)
            task["windows"] += 1
            task["windows_with_valid_pw"] += int(any(record["pw_valid_mask"]))
            task["targets"] += len(record["pw_valid_mask"])
            for label, valid, reason in zip(
                record["pw_class_targets"], record["pw_valid_mask"], record["pw_mask_reasons"]
            ):
                task["valid_targets"] += int(valid)
                if valid:
                    task["words"][words[label]] += 1
                else:
                    task["mask_reasons"][reason] += 1
        episodes.append(
            {
                "episode_id": meta["episode"]["episode_id"],
                "frames": len(frames),
                "windows": len(ep_records),
                "stage_count": len(annotation["stages"]),
                "invalid_stale_frames": sum(not f["valid"] for f in frames),
                "near_uniform_frames": sum(f["image_std"] < 2 for f in frames),
                "source_clock": meta["audit"]["cameras"][meta["episode"]["camera"]]["clock"],
                "sync": meta["audit"]["sync"],
                "annotation_provenance": annotation["provenance"],
            }
        )
    for task in tasks.values():
        for stats in task.get("arms", {}).values():
            stats["missing_rate"] = 1 - stats["valid_targets"] / stats["targets"] if stats["targets"] else None
            stats["words"] = {word: stats["words"][word] for word in words}
            stats["unique_frame_words"] = {word: stats["unique_frame_words"][word] for word in words}
        task["missing_rate"] = 1 - task["valid_targets"] / task["targets"] if task["targets"] else None
        task["words"] = {word: task["words"][word] for word in words}
        task["unique_frame_words"] = {word: task["unique_frame_words"][word] for word in words}
    report = {
        "status": "passed_integrity_checks" if not failures else "partial_with_failures",
        "seed": split["seed"],
        "vocabulary": vocabulary,
        "vocabulary_hash": vocabulary_hash(vocabulary),
        "episodes_processed": len(episodes),
        "frames_processed": sum(e["frames"] for e in episodes),
        "windows": len(records),
        "windows_with_valid_pw": sum(any(r["pw_valid_mask"]) for r in records),
        "windows_with_valid_arm_pw": sum(any(any(arm) for arm in r["pw_arm_valid_mask"]) for r in records),
        "windows_with_all_arm_pw": sum(all(all(arm) for arm in r["pw_arm_valid_mask"]) for r in records),
        "split_episode_counts": dict(Counter(e["split"] for e in split["episodes"])),
        "split_window_counts": dict(Counter(r["split"] for r in records)),
        "tasks": dict(tasks),
        "episodes": episodes,
        "failures": failures or [],
        "checks": [
            "fps_grid",
            "timestamps",
            "image_decode",
            "resolution_and_video_length",
            "episode_leakage",
            "atomic_vocabulary",
            "stage_intervals",
            "target_time_alignment",
            "ambiguity_mask",
            "history_future_boundary",
        ],
        "semantic_quality": "See annotation provenance and manual pilot review; integrity checks do not establish semantic accuracy.",
    }
    write_json(output, report)
    return report
