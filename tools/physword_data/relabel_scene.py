"""Rebuild the three badminton pilot annotations with a versioned scene vocabulary.

Reuses immutable 10Hz frames/video through links and preserves original episode split.
No VLM API, model training or writes to the original annotations are performed.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

from cosmos_framework.data.generator.physword.annotation import AnnotationAdapter, validate_annotation
from cosmos_framework.data.generator.physword.cli import build, qa, visualize
from cosmos_framework.data.generator.physword.common import PipelineConfig, read_json, write_json
from cosmos_framework.data.generator.physword.vocabulary import vocabulary_for


def relabel_scene(source, output, profile=None):
    source, output = Path(source).resolve(), Path(output).resolve()
    profile = Path(profile).resolve() if profile else Path(__file__).with_name("scene_v2").resolve()
    metadata_paths = sorted((source / "processed").glob("*/metadata.json"))
    if not 1 <= len(metadata_paths) <= 3:
        raise ValueError("This relabel command is scoped to the existing 1..3 badminton pilot episodes")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Choose a fresh version output directory")
    vocabulary = read_json(profile / "vocabulary.json")
    for path in metadata_paths:
        meta = read_json(path)
        if meta["episode"].get("collection") != "badminton":
            raise ValueError("Only badminton is in scope")
        annotation = read_json(profile / "annotations" / f"{path.parent.name}.json")
        validate_annotation(annotation, meta)
        if vocabulary_for(annotation) != vocabulary:
            raise ValueError("Pilot annotation vocabulary differs from profile")
    output.mkdir(parents=True, exist_ok=True)
    for filename in (
        "inventory.json",
        "split_manifest.json",
        "config_resolved.json",
        "prepare_status.json",
        "source_audit.json",
    ):
        if (source / filename).exists():
            shutil.copyfile(source / filename, output / filename)
    config = read_json(output / "config_resolved.json")
    adapter_config = {
        "kind": "json_file",
        "directory": str(profile / "annotations"),
        "vocabulary_file": str(profile / "vocabulary.json"),
        "prompt_file": str(profile / "prompt.txt"),
    }
    config["annotation"] = adapter_config
    write_json(output / "config_resolved.json", config)
    write_json(output / "vocabulary.json", vocabulary)
    adapter = AnnotationAdapter(adapter_config, output / "annotation_cache")
    for path in metadata_paths:
        meta = read_json(path)
        episode_dir = output / "processed" / path.parent.name
        episode_dir.mkdir(parents=True)
        (episode_dir / "frames").symlink_to(path.parent / "frames", target_is_directory=True)
        (episode_dir / meta["video_path"]).symlink_to(path.parent / meta["video_path"])
        write_json(episode_dir / "metadata.json", meta)
        annotation = adapter.annotate(meta, episode_dir)
        write_json(episode_dir / "annotation.json", annotation)
    pipeline_config = PipelineConfig(**config["pipeline"])
    build(output, pipeline_config)
    report = qa(output)
    visualize(output, pipeline_config)
    write_json(
        output / "relabel_provenance.json",
        {
            "source": str(source),
            "profile": str(profile),
            "vocabulary_id": vocabulary["id"],
            "network_calls": 0,
            "training_started": False,
            "source_media_reused": True,
            "original_split_preserved": True,
        },
    )
    return report


def publish_review(source, output):
    """Update existing browser entry points, preserving the old review assets."""
    source, output = Path(source).resolve(), Path(output).resolve()
    if not output.is_relative_to(source):
        raise ValueError("Published version must live inside the served source directory")
    if read_json(output / "quality_report.json")["status"] != "passed_integrity_checks":
        raise ValueError("Publish only after integrity checks pass")
    versions, archives = [], []
    for path in sorted((output / "processed").glob("*/metadata.json")):
        name = path.parent.name
        old_review = source / "processed" / name / "review"
        archive = old_review.with_name("review_v1")
        if old_review.exists() and not archive.exists():
            old_review.rename(archive)
        old_review.mkdir(parents=True, exist_ok=True)
        target = os.path.relpath(path.parent / "review/index.html", old_review)
        (old_review / "index.html").write_text(
            "<!doctype html><meta charset='utf-8'><meta http-equiv='refresh' content='0;url="
            + target
            + "'><a href='"
            + target
            + "'>Open current independent-arm scene annotations</a>"
        )
        versions.append(f"<p><a href='{(path.parent / 'review/index.html').relative_to(source)}'>{name}</a></p>")
        archives.append(f"<p><a href='{(archive / 'index.html').relative_to(source)}'>{name} v1</a></p>")
    (source / "index_v1.html").write_text(
        "<!doctype html><meta charset='utf-8'><h1>Archived sixteen-class pilot</h1>" + "".join(archives)
    )
    (source / "index.html").write_text(
        "<!doctype html><meta charset='utf-8'><title>Badminton scene labels v2</title>"
        "<h1>Badminton — 场景词表 v2</h1><p>左右臂独立标注；单词原语；无 idle / unknown 标签。"
        "持物为 hold，空手保持原位为 rest。</p>"
        + "".join(versions)
        + "<p><a href='index_v1.html'>历史 v1 抽检（旧词表，仅供对照）</a></p>"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--profile")
    parser.add_argument(
        "--publish", action="store_true", help="Update existing HTML entry points after successful relabel"
    )
    args = parser.parse_args()
    report = relabel_scene(args.source, args.output, args.profile)
    if args.publish:
        publish_review(args.source, args.output)
    print(
        json.dumps(
            {
                key: report[key]
                for key in (
                    "status",
                    "episodes_processed",
                    "frames_processed",
                    "windows",
                    "windows_with_valid_arm_pw",
                    "vocabulary_hash",
                )
            }
        )
    )


if __name__ == "__main__":
    main()
