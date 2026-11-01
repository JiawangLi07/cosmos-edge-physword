"""Offline data contract tests, including corruption, VFR and leakage regressions."""

from __future__ import annotations

import copy
import json
import subprocess
import sys
from dataclasses import replace

import h5py
import numpy as np
import pytest
from PIL import Image

from cosmos_framework.data.generator.physword.adapters import discover_libero, discover_real, manifest_adapter
from cosmos_framework.data.generator.physword.annotation import (
    AnnotationAdapter,
    label_at,
    pending_annotation,
    validate_annotation,
)
from cosmos_framework.data.generator.physword.cli import audit, build, fixture, fixture_labels, load_config, prepare, qa
from cosmos_framework.data.generator.physword.common import (
    ATOMIC_ACTIONS,
    PipelineConfig,
    read_json,
    write_json,
)
from cosmos_framework.data.generator.physword.dataset import TrainingWindowDataset, condition_inputs
from cosmos_framework.data.generator.physword.media import (
    probe_video,
    process_episode,
    read_capture_timestamps,
    time_grid,
    validate_timestamps,
)
from cosmos_framework.data.generator.physword.quality import validate_record
from cosmos_framework.data.generator.physword.split import build_split, check_leakage
from cosmos_framework.data.generator.physword.windows import generate_windows


@pytest.fixture(scope="module")
def prepared(tmp_path_factory):
    root = tmp_path_factory.mktemp("physword")
    fixture(root / "source")
    config, raw = load_config(root / "source/config.json")
    output = root / "output"
    audit(raw, config, output, probe_limit=1)
    result = prepare(output, config, limit=None)
    assert not result["failures"]
    fixture_labels(output)
    build(output, config)
    return output, config


@pytest.fixture
def example(prepared):
    output, config = prepared
    meta = read_json(output / "processed/synthetic_0000/metadata.json")
    annotation = read_json(output / "processed/synthetic_0000/annotation.json")
    entries = read_json(output / "split_manifest.json")["episodes"]
    entry = next(e for e in entries if e["episode_id"] == "synthetic_0000")
    return output, config, meta, annotation, entry


def test_complete_synthetic_pipeline(prepared):
    output, _ = prepared
    report = qa(output)
    assert report["episodes_processed"] == 10
    assert report["frames_processed"] == 330
    assert report["windows"] == 130
    assert report["split_episode_counts"] == {"train": 9, "test": 1}
    assert report["windows_with_valid_pw"] > 0
    sample = TrainingWindowDataset(output / "windows.jsonl")[0]
    assert sample["history_rgb"].shape == (5, 3, 96, 128)
    assert sample["future_rgb"].shape == (16, 3, 96, 128)
    assert sample["pw_class_targets"].dtype == np.int64
    assert sample["pw_valid_mask"].dtype == np.bool_
    assert not {"actions", "tactile", "state", "proprioception"} & set(sample)


def test_configurable_horizon_and_rejections():
    assert replace(PipelineConfig(), future_frames=32, pw_num_blocks=8).target_offsets == list(range(8, 37, 4))
    for kwargs in (
        {"history_frames": 4},
        {"future_frames": 15},
        {"pw_num_blocks": 5},
        {"sampling_fps": float("nan")},
        {"action_conditioning": True},
        {"proprioception_conditioning": True},
        {"tactile_conditioning": True},
        {"tail_policy": "silent_pad"},
    ):
        with pytest.raises(ValueError):
            PipelineConfig(**kwargs)


@pytest.mark.parametrize("bad", [[0, 0], [0, -1], [0, float("nan")], [], [[0, 1]]])
def test_invalid_timestamp_rejection(bad):
    with pytest.raises(ValueError):
        validate_timestamps(bad)


def test_causal_grid_vfr_no_interpolation():
    source = np.array([11.0, 11.04, 11.09, 11.17, 11.26, 11.41])
    grid, indexes, actual = time_grid(source, 10)
    assert indexes.tolist() == [0, 2, 3, 4, 4]
    assert np.all(actual <= grid + 1e-8)
    assert np.allclose(actual, source[indexes] - source[0])
    assert grid.tolist() == [0, 0.1, 0.2, 0.3, 0.4]


def test_real_vfr_container_pts(tmp_path):
    for i in range(6):
        Image.new("RGB", (64, 48), (i * 30, 60, 80)).save(tmp_path / f"{i}.png")
    concat = tmp_path / "input.txt"
    concat.write_text(
        "".join(f"file '{i}.png'\nduration {d}\n" for i, d in enumerate([0.04, 0.12, 0.04, 0.20, 0.08, 0.12]))
    )
    video = tmp_path / "vfr.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(concat),
            "-vsync",
            "vfr",
            "-c:v",
            "libx264",
            "-threads",
            "1",
            "-pix_fmt",
            "yuv420p",
            str(video),
        ],
        check=True,
    )
    info = probe_video(video)
    assert len(set(np.round(np.diff(info["pts_seconds"]), 3))) > 1
    ep = {
        "episode_id": "vfr",
        "original_episode_id": "vfr",
        "lineage_id": "vfr",
        "instruction": "Observe colors",
        "dataset_source": "synthetic_fixture",
        "task_id": "vfr",
        "source_kind": "video",
        "source_ref": str(video),
        "camera": "main",
        "cameras": {"main": {"video_path": str(video)}},
    }
    meta = process_episode(ep, tmp_path / "processed", PipelineConfig())
    _, indices, _ = time_grid(info["pts_seconds"], 10)
    assert [r["source_frame_index"] for r in meta["frame_mapping"]] == indices.tolist()
    assert all(r["source_elapsed_s"] <= r["resampled_timestamp"] + 1e-8 for r in meta["frame_mapping"])


def test_frame_timestamp_count_mismatch(tmp_path):
    path = tmp_path / "timestamps.jsonl"
    path.write_text('{"event":"start"}\n{"i":1,"mono":10}\n')
    with pytest.raises(ValueError, match="1:1"):
        read_capture_timestamps(path, 2)


def test_original_split_lineage_and_content_aliases(prepared):
    output, config = prepared
    episodes = read_json(output / "inventory.json")["episodes"]
    # Mirror/crop aliases share lineage even if their source bytes differ.
    alias = dict(episodes[0], episode_id="mirror", source_ref=episodes[1]["source_ref"])
    manifest = build_split(episodes + [alias], config)
    check_leakage(manifest)
    entries = {e["episode_id"]: e for e in manifest["episodes"]}
    assert entries[episodes[0]["episode_id"]]["split"] == entries["mirror"]["split"]
    assert entries[episodes[1]["episode_id"]]["split"] == entries["mirror"]["split"]
    assert build_split(list(reversed(episodes)), config) == build_split(episodes, config)
    corrupted = copy.deepcopy(manifest)
    corrupted["episodes"].append(
        dict(entries["mirror"], split="test" if entries["mirror"]["split"] == "train" else "train")
    )
    with pytest.raises(ValueError, match="leakage"):
        check_leakage(corrupted)


def test_split_validation_from_train_only(prepared):
    output, config = prepared
    episodes = read_json(output / "inventory.json")["episodes"]
    first = build_split(episodes, config)
    second = build_split(episodes, replace(config, validation_ratio=0.25))
    assert {e["episode_id"] for e in first["episodes"] if e["split"] == "test"} == {
        e["episode_id"] for e in second["episodes"] if e["split"] == "test"
    }
    check_leakage(second)


def test_window_offsets_and_real_time_mapping(example):
    output, config, meta, annotation, entry = example
    records, _ = generate_windows(meta, annotation, entry, output / "processed/synthetic_0000", config)
    assert records[0]["pw_target_frame_indices"] == [8, 12, 16, 20]
    assert records[1]["pw_target_frame_indices"] == [9, 13, 17, 21]
    for record in records:
        validate_record(record, meta, annotation, config)
    assert records[0]["pw_target_timestamps"] != records[0]["pw_target_resampled_timestamps"]
    record = copy.deepcopy(records[0])
    record["future_frame_indices"][0] = 4
    with pytest.raises(AssertionError):
        validate_record(record, meta, annotation, config)


def test_short_and_missing_instruction_explicit_policy(example):
    output, config, meta, annotation, entry = example
    meta["frame_mapping"] = meta["frame_mapping"][:10]
    annotation = pending_annotation(meta)
    windows, status = generate_windows(meta, annotation, entry, output, config)
    assert windows == [] and status["too_short"]
    meta["episode"]["instruction"] = None
    _, status = generate_windows(meta, annotation, entry, output, config)
    assert status["excluded_missing_instruction"]


def test_stale_source_frames_drop_entire_window(example):
    output, config, meta, annotation, entry = example
    meta["frame_mapping"][15]["valid"] = False
    records, status = generate_windows(meta, annotation, entry, output, config)
    assert records == [] and status["dropped_invalid_frame_windows"] == 13


@pytest.mark.parametrize(
    "mutate",
    [
        lambda a: a["stages"][0].update(action="grasp"),
        lambda a: a["stages"][0].update(end_time=-1),
        lambda a: a["stages"][0].update(confidence=float("nan")),
        lambda a: a["stages"][0].update(unexpected=True),
        lambda a: a["stages"][0].update(end_time=10000),
        lambda a: a.update(media_fingerprint="a" * 64),
        lambda a: a["stages"].reverse(),
        lambda a: a["stages"][0].update(end_time=2.0),
    ],
)
def test_strict_annotation_schema_and_intervals(example, mutate):
    _, _, meta, annotation, _ = example
    mutate(annotation)
    with pytest.raises(Exception):
        validate_annotation(annotation, meta)


def test_parallel_actions_ambiguity_unknown_and_boundaries(example):
    _, config, meta, annotation, _ = example
    stage = annotation["stages"][0]
    annotation["stages"] = [
        dict(stage, start_time=0.0, end_time=2.0, action="approach", effector="left"),
        dict(stage, start_time=0.0, end_time=2.0, action="pick", effector="right"),
    ]
    validate_annotation(annotation, meta)
    assert label_at(annotation, 1.0, config) == (0, False, "parallel_or_multilabel")
    annotation["stages"] = [dict(stage, start_time=0.0, end_time=2.0, ambiguous=True, candidates=["pick", "move"])]
    assert label_at(annotation, 1.0, config)[2] == "ambiguous"
    for word in ("unknown", "idle"):
        annotation["stages"][0].update(action=word, ambiguous=False, candidates=[])
        assert label_at(annotation, 1.0, config) == (0, False, word)
    annotation["stages"][0].update(action="move", confidence=0.3)
    assert label_at(annotation, 1.0, config)[2] == "low_confidence"
    annotation["stages"] = [
        dict(stage, start_time=0.0, end_time=1.0, action="approach"),
        dict(stage, start_time=1.0, end_time=2.0, action="move"),
    ]
    assert label_at(annotation, 1.0, config)[2] == "stage_boundary_uncertainty"
    assert label_at(annotation, 1.2, config) == (1, True, "valid")
    assert label_at(annotation, 2.0, config)[2] == "unannotated_gap"  # half-open intervals
    assert "unknown" not in ATOMIC_ACTIONS and len(ATOMIC_ACTIONS) == 16


def test_future_and_gt_cannot_enter_condition(prepared):
    output, _ = prepared
    sample = TrainingWindowDataset(output / "windows.jsonl")[0]
    before = condition_inputs(sample)
    sample["future_rgb"][:] = 255
    sample["pw_class_targets"][:] = 15
    sample["actions"] = np.ones(7) * 999
    after = condition_inputs(sample)
    assert np.array_equal(before["history_rgb"], after["history_rgb"])
    assert before["instruction"] == after["instruction"]
    assert set(after) == {"history_rgb", "instruction", "model_fps", "history_timestamps"}


def test_libero_missing_rgb_and_fps_never_guessed(tmp_path):
    path = tmp_path / "task.hdf5"
    with h5py.File(path, "w") as f:
        f.create_dataset("data/demo_0/actions", data=np.zeros((32, 7)))
    episodes, rejected = discover_libero(tmp_path)
    assert not episodes and rejected[0]["reason"] == "missing_rgb"
    with h5py.File(path, "a") as f:
        f.create_dataset("data/demo_0/obs/agentview_rgb", data=np.zeros((32, 16, 16, 3), dtype=np.uint8))
    episodes, rejected = discover_libero(tmp_path)
    assert not episodes and rejected[0]["reason"] == "missing_timestamps_and_declared_fps"
    episodes, rejected = discover_libero(tmp_path, source_fps=20, instructions={"task": "Observe fixture"})
    assert len(episodes) == 1 and not rejected
    assert episodes[0]["source_fps"] == 20


def test_reencoded_real_identity_not_directory_number(tmp_path):
    m = {
        "episode": 407,
        "start_wall": 123.0,
        "start_mono": 12.0,
        "stop_wall": 126.0,
        "cameras": {"main": {"frames": 3}},
    }
    for name in ("episode_0001", "episode_0999"):
        folder = tmp_path / name
        (folder / "main").mkdir(parents=True)
        (folder / "main/color_take001.mp4").touch()
        write_json(folder / "manifest.json", m)
    episodes, _ = discover_real(tmp_path, "fixture")
    assert episodes[0]["original_episode_id"] == episodes[1]["original_episode_id"]
    assert not episodes[0]["instruction"]


def test_annotation_cache_and_failed_response_never_cached(example, tmp_path):
    _, _, meta, annotation, _ = example
    path = tmp_path / "labels"
    write_json(path / f"{meta['episode']['episode_id']}.json", annotation)
    adapter = AnnotationAdapter({"kind": "json_file", "directory": str(path)}, tmp_path / "cache")
    assert adapter.annotate(meta, tmp_path) == annotation
    assert adapter.annotate(meta, tmp_path) == annotation
    annotation["provenance"]["reviewed"] = True
    write_json(path / f"{meta['episode']['episode_id']}.json", annotation)
    assert adapter.annotate(meta, tmp_path)["provenance"]["reviewed"]
    broken = AnnotationAdapter(
        {
            "kind": "command",
            "command": [sys.executable, "-c", "print('{}')"],
            "max_attempts": 2,
            "retry_backoff_seconds": 0,
        },
        tmp_path / "broken",
    )
    with pytest.raises(RuntimeError):
        broken.annotate(meta, tmp_path)
    states = list((tmp_path / "broken").glob("*.state.json"))
    assert len(states) == 1 and read_json(states[0])["status"] == "failed"
    assert read_json(states[0])["attempt"] == 2
    assert not [
        p for p in (tmp_path / "broken").glob("*.json") if not p.name.endswith((".request.json", ".state.json"))
    ]


def test_network_needs_explicit_opt_in(example, tmp_path):
    _, _, meta, _, _ = example
    adapter = AnnotationAdapter(
        {"kind": "http_json", "model": "unspecified", "endpoint": "https://example.invalid"}, tmp_path
    )
    with pytest.raises(PermissionError):
        adapter.annotate(meta, tmp_path)


def test_interrupted_preprocessing_recovers_and_changed_config_rejected(example):
    output, config, meta, _, _ = example
    recovered = process_episode(meta["episode"], output / "processed", config)
    assert recovered == meta
    with pytest.raises(ValueError, match="different inputs"):
        process_episode(meta["episode"], output / "processed", replace(config, min_confidence=0.8))


def test_quality_detects_corrupt_png(example, tmp_path):
    _, _, meta, _, _ = example
    # A failed image decode is a hard error, not a silent alternate episode selection.
    path = tmp_path / "bad.png"
    path.write_bytes(b"not a png")
    from cosmos_framework.data.generator.physword.dataset import read_rgb

    with pytest.raises(Exception):
        read_rgb([path])


def test_manifest_paths_and_explicit_lineage_required(tmp_path):
    path = tmp_path / "episodes.json"
    write_json(path, [{"episode_id": "missing_lineage"}])
    with pytest.raises(ValueError):
        manifest_adapter(path)


def test_quality_rejects_window_split_reassignment(prepared, tmp_path):
    from cosmos_framework.data.generator.physword.quality import quality_report

    output, _ = prepared
    records = [json.loads(line) for line in (output / "windows.jsonl").read_text().splitlines()]
    records[0]["split"] = "train" if records[0]["split"] == "test" else "test"
    with pytest.raises(AssertionError, match="manifest disagreement"):
        quality_report(
            list((output / "processed").glob("*/metadata.json")),
            records,
            read_json(output / "split_manifest.json"),
            tmp_path / "bad.json",
        )


def test_corrupted_cached_frames_are_regenerated(example):
    output, config, meta, _, _ = example
    path = output / "processed/synthetic_0000/frames/000000.png"
    expected = path.read_bytes()
    path.write_bytes(b"corrupted partial artifact")
    recovered = process_episode(meta["episode"], output / "processed", config)
    assert path.read_bytes() == expected
    assert recovered["frame_mapping"][0]["sha256"] == meta["frame_mapping"][0]["sha256"]


def test_libero_rgb_timestamp_adapter_end_to_end(tmp_path):
    path = tmp_path / "libero_rgb_fixture.hdf5"
    with h5py.File(path, "w") as f:
        group = f.create_group("data")
        group.attrs["problem_info"] = json.dumps({"language_instruction": "Observe fixture colors"})
        frames = np.zeros((61, 32, 32, 3), dtype=np.uint8)
        frames[:, :, :, 0] = np.arange(61)[:, None, None]
        f.create_dataset("data/demo_0/obs/agentview_rgb", data=frames)
        f.create_dataset("data/demo_0/timestamps", data=np.arange(61) / 20)
    episodes, rejected = discover_libero(tmp_path, timestamp_key="timestamps")
    assert len(episodes) == 1 and not rejected
    assert episodes[0]["instruction"] == "Observe fixture colors"
    meta = process_episode(episodes[0], tmp_path / "processed", PipelineConfig())
    assert meta["frame_count"] == 31
    assert meta["audit"]["cameras"]["obs/agentview_rgb"]["clock"] == "hdf5_timestamps"


def test_dualarm_targets_preserve_simultaneous_words_and_arm_isolation(example):
    output, config, meta, annotation, entry = example
    end = meta["frame_mapping"][-1]["source_elapsed_s"] + 1 / meta["model_fps"]
    template = annotation["stages"][0]
    annotation.update(effectors=["left", "right"], effector_coordinate="selected_camera_image_sides")
    annotation["stages"] = [
        dict(template, start_time=0.0, end_time=end, action="move", effector="left"),
        dict(template, start_time=0.0, end_time=end, action="pick", effector="right"),
    ]
    validate_annotation(annotation, meta)
    records, _ = generate_windows(meta, annotation, entry, output, config)
    first = records[0]
    assert first["pw_effectors"] == ["left", "right"]
    assert first["pw_arm_class_targets"] == [[1] * 4, [2] * 4]
    assert first["pw_arm_valid_mask"] == [[True] * 4, [True] * 4]
    assert first["pw_valid_mask"] == [False] * 4
    validate_record(first, meta, annotation, config)
    before = label_at(annotation, 0.9, config, effector="left")
    annotation["stages"][1]["action"] = "insert"
    assert label_at(annotation, 0.9, config, effector="left") == before
    assert label_at(annotation, 0.9, config, effector="right")[0] == 11
    annotation["stages"][1]["action"] = "move"
    assert label_at(annotation, 0.9, config)[2] == "parallel_or_multilabel"  # even identical words retain both arms
    annotation["stages"][1]["action"] = "idle"
    assert label_at(annotation, 0.9, config) == (1, True, "valid")
    assert label_at(annotation, 0.9, config, effector="right") == (0, False, "idle")


def test_declared_arm_tracks_require_full_coverage(example):
    _, _, meta, annotation, _ = example
    end = meta["frame_mapping"][-1]["source_elapsed_s"] + 1 / meta["model_fps"]
    template = annotation["stages"][0]
    annotation.update(effectors=["left", "right"], effector_coordinate="selected_camera_image_sides")
    annotation["stages"] = [
        dict(template, start_time=0.0, end_time=end, effector="left"),
        dict(template, start_time=0.2, end_time=end, effector="right"),
    ]
    with pytest.raises(ValueError, match="complete episode"):
        validate_annotation(annotation, meta)


def test_prepare_refuses_source_change_since_split_and_keeps_failure_state(prepared, tmp_path):
    output, config = prepared
    inventory = read_json(output / "inventory.json")
    inventory["episodes"][0]["source_ref"] = inventory["episodes"][1]["source_ref"]
    write_json(tmp_path / "inventory.json", inventory)
    write_json(tmp_path / "split_manifest.json", read_json(output / "split_manifest.json"))
    result = prepare(tmp_path, config, limit=1)
    assert not result["successes"] and len(result["failures"]) == 1
    assert "Source RGB changed" in result["failures"][0]["detail"]
    result = prepare(tmp_path, config, names=[inventory["episodes"][2]["episode_id"]], limit=1)
    assert result["successes"] and not result["failures"]
    assert len(read_json(tmp_path / "prepare_status.json")["failures"]) == 1


@pytest.fixture
def scene_annotation(example):
    from pathlib import Path

    _, _, meta, annotation, _ = example
    annotation = copy.deepcopy(annotation)
    annotation["vocabulary"] = read_json(
        Path(__file__).resolve().parents[2] / "tools/physword_data/scene_v2/vocabulary.json"
    )
    end = meta["frame_mapping"][-1]["source_elapsed_s"] + 1 / meta["model_fps"]
    template = annotation["stages"][0]
    annotation.update(effectors=["left", "right"], effector_coordinate="selected_camera_image_sides")
    annotation["stages"] = [
        dict(template, start_time=0.0, end_time=end, action="hold", effector="left"),
        dict(template, start_time=0.0, end_time=end, action="rest", effector="right"),
    ]
    return annotation


def test_scene_vocabulary_has_valid_independent_hold_and_rest_targets(example, scene_annotation):
    output, config, meta, _, entry = example
    validate_annotation(scene_annotation, meta)
    records, _ = generate_windows(meta, scene_annotation, entry, output, config)
    record = records[0]
    assert record["pw_vocabulary_id"] == "badminton_scene_v2"
    assert record["pw_arm_class_targets"] == [[17] * 4, [16] * 4]
    assert record["pw_arm_valid_mask"] == [[True] * 4, [True] * 4]
    assert record["pw_valid_mask"] == [False] * 4  # Never collapse the two arms to one target.
    validate_record(record, meta, scene_annotation, config)
    scene_annotation["stages"][0]["confidence"] = 0.2
    assert label_at(scene_annotation, 0.8, config, effector="left") == (17, False, "low_confidence")


@pytest.mark.parametrize("invalid", ["idle", "unknown", "two words", "reach_for", "Hold"])
def test_scene_profile_rejects_forbidden_or_multiple_words(example, scene_annotation, invalid):
    _, _, meta, _, _ = example
    scene_annotation["stages"][0]["action"] = invalid
    with pytest.raises(Exception):
        validate_annotation(scene_annotation, meta)


def test_scene_vocabulary_definitions_and_identity_are_strict(scene_annotation):
    from cosmos_framework.data.generator.physword.vocabulary import validate_vocabulary, vocabulary_for

    vocabulary = scene_annotation["vocabulary"]
    vocabulary["words"].append("hold")
    with pytest.raises(ValueError):
        validate_vocabulary(vocabulary)
    vocabulary["words"].pop()
    vocabulary["definitions"].pop("hold")
    with pytest.raises(ValueError):
        vocabulary_for(scene_annotation)


def test_scene_vocabulary_cannot_reach_legacy_word_head(scene_annotation):
    from cosmos_framework.data.generator.physword.vocabulary import require_legacy_model_vocabulary

    sample = {"pw_vocabulary_id": "badminton_scene_v2", "pw_vocabulary": scene_annotation["vocabulary"]["words"]}
    with pytest.raises(ValueError, match="legacy 16-class"):
        require_legacy_model_vocabulary(sample)


def test_scene_boundaries_keep_action_names_without_forcing_validity(example, scene_annotation):
    _, config, meta, _, _ = example
    first = scene_annotation["stages"][0]
    end = first["end_time"]
    scene_annotation["stages"] = [
        dict(first, end_time=1.0),
        scene_annotation["stages"][1],
        dict(first, start_time=1.0, end_time=end, action="adjust"),
    ]
    scene_annotation["stages"].sort(key=lambda s: (s["start_time"], s["end_time"]))
    validate_annotation(scene_annotation, meta)
    assert label_at(scene_annotation, 1.0, config, effector="left") == (21, False, "stage_boundary_uncertainty")
    assert label_at(scene_annotation, 1.2, config, effector="left") == (21, True, "valid")


def test_scene_adapter_import_is_bound_to_explicit_vocabulary(example, scene_annotation, tmp_path):
    _, _, meta, _, _ = example
    write_json(tmp_path / "vocabulary.json", scene_annotation["vocabulary"])
    write_json(tmp_path / f"{meta['episode']['episode_id']}.json", scene_annotation)
    adapter = AnnotationAdapter(
        {"kind": "json_file", "directory": str(tmp_path), "vocabulary_file": str(tmp_path / "vocabulary.json")},
        tmp_path / "cache",
    )
    assert adapter.annotate(meta, tmp_path) == scene_annotation
    assert adapter.annotate(meta, tmp_path) == scene_annotation
    unconfigured = AnnotationAdapter(
        {"kind": "json_file", "directory": str(tmp_path), "max_attempts": 1}, tmp_path / "other_cache"
    )
    with pytest.raises(RuntimeError):
        unconfigured.annotate(meta, tmp_path)


def test_dataset_rejects_mixed_class_mappings(example, scene_annotation, tmp_path):
    output, config, meta, original, entry = example
    a, _ = generate_windows(meta, original, entry, output, config)
    b, _ = generate_windows(meta, scene_annotation, entry, output, config)
    path = tmp_path / "mixed.jsonl"
    path.write_text(json.dumps(a[0]) + "\n" + json.dumps(b[0]) + "\n")
    with pytest.raises(ValueError, match="mix vocabularies"):
        TrainingWindowDataset(path)


def test_scene_bridges_reject_before_loading_torch_or_model(scene_annotation):
    from cosmos_framework.data.generator.physword.dataset import m3_target_payload, to_m3_batch

    sample = {"pw_vocabulary_id": "badminton_scene_v2", "pw_vocabulary": scene_annotation["vocabulary"]["words"]}
    for bridge in (m3_target_payload, to_m3_batch):
        with pytest.raises(ValueError, match="legacy 16-class"):
            bridge(sample)
