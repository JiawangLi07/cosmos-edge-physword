"""Frozen single-task/effector vocabulary and a non-destructive pilot split overlay."""

import argparse
import json
from collections import Counter
from pathlib import Path

from cosmos_framework.data.generator.physword.common import file_sha256
from cosmos_framework.data.generator.physword.vocabulary import vocabulary_hash
from cosmos_framework.model.generator.physword.config import PhysWordConfig


def config_from_manifest(manifest):
    config = PhysWordConfig(
        enabled=True,
        vocabulary=tuple(manifest["words"]),
        vocabulary_id=manifest["id"],
        definitions=tuple(manifest["definitions"][w] for w in manifest["words"]),
        effector=manifest["effector"],
    )
    if config.vocabulary_hash != manifest["model_vocabulary_hash"]:
        raise ValueError("Task vocabulary content/order/definitions fingerprint mismatch")
    return config


def prepare_task(source, vocabulary_source, output, *, task_id, effector, test_episodes=()):
    """Use train-only labels for vocab; explicit episode holdout never edits the source.

    A pilot holdout from globally Train episodes is recorded as a separate protocol.
    It must be frozen before any actual-data model fitting; global splits stay intact.
    """
    source, output = Path(source).resolve(), Path(output).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Choose a fresh task artifact directory")
    records = [json.loads(line) for line in source.read_text().splitlines() if line.strip()]
    records = [r for r in records if r["task_id"] == task_id]
    if not records:
        raise ValueError("No windows for the requested task")
    if effector not in {"left", "right"}:
        raise ValueError("Select exactly one explicit effector")
    episodes = {r["episode_id"] for r in records}
    if set(test_episodes) - episodes:
        raise ValueError("Unknown pilot test episode")
    split_by_episode = {r["episode_id"]: ("test" if r["episode_id"] in test_episodes else r["split"]) for r in records}
    if len(set(split_by_episode.values())) < 2 or "train" not in split_by_episode.values():
        raise ValueError("Task pilot needs disjoint train and test episodes")
    # Copies/aliases of the same original capture must never straddle the overlay.
    for key in ("group_id", "original_episode_id", "lineage_id", "media_fingerprint"):
        groups = {}
        for r in records:
            value = json.dumps(r[key], sort_keys=True)
            groups.setdefault(value, set()).add(split_by_episode[r["episode_id"]])
        if any(len(x) > 1 for x in groups.values()):
            raise ValueError(f"Capture leakage across task splits: {key}")
    counts, totals, masked = Counter(), Counter(), Counter()
    for r in records:
        arm = r["pw_effectors"].index(effector)
        split = split_by_episode[r["episode_id"]]
        for cls, valid in zip(r["pw_arm_class_targets"][arm], r["pw_arm_valid_mask"][arm], strict=True):
            totals[split] += 1
            if not valid:
                masked[split] += 1
            elif not 0 <= cls < len(r["pw_vocabulary"]):
                raise ValueError("Valid source annotation class is outside its vocabulary")
            elif split == "train":
                counts[r["pw_vocabulary"][cls]] += 1
    words = sorted(counts)
    if not words or {"unknown", "idle"}.intersection(words):
        raise ValueError("Task has no valid training vocabulary or contains unknown labels")
    definitions = json.loads(Path(vocabulary_source).read_text())["definitions"]
    manifest = {
        "schema": 1,
        "id": f"{task_id}_{effector}_v1",
        "task_id": task_id,
        "effector": effector,
        "words": words,
        "definitions": {w: definitions[w] for w in words},
        "order": "lexicographic; valid TRAIN labels only",
        "source_windows_sha256": file_sha256(source),
        "source_vocabulary_sha256": file_sha256(vocabulary_source),
        "train_label_counts": {w: counts[w] for w in words},
        "train_label_frequencies": {w: counts[w] / sum(counts.values()) for w in words},
        "mask_by_split": {s: {"total": n, "masked": masked[s], "ratio": masked[s] / n} for s, n in totals.items()},
        "missing_feedback": "neutral pw_start; never fabricate a class label",
        "unknown_eval_label_policy": "mask CE only; visual target validity remains independent",
    }
    config = PhysWordConfig(
        enabled=True,
        vocabulary=tuple(words),
        vocabulary_id=manifest["id"],
        definitions=tuple(definitions[w] for w in words),
        effector=effector,
    )
    manifest["model_vocabulary_hash"] = config.vocabulary_hash
    mapped, distribution = [], {s: Counter() for s in totals}
    unknown, fully_valid = Counter(), Counter()
    for original in records:
        r = dict(original)
        source_words = list(r["pw_vocabulary"])
        if r["model_fps"] != 10 or len(r["future_frame_indices"]) != 16:
            raise ValueError("Current model requires 10Hz / 16 future frames")
        history_end = r["history_frame_indices"][-1]
        if r["pw_target_frame_indices"] != [history_end + 4 * k for k in range(1, 5)]:
            raise ValueError("PW targets must cover exactly the video future horizon")
        if r["pw_target_frame_indices"][-1] != r["future_frame_indices"][-1]:
            raise ValueError("PW horizon differs from video future horizon")
        arm = r["pw_effectors"].index(effector)
        labels, valid = [], []
        for cls, ok in zip(r["pw_arm_class_targets"][arm], r["pw_arm_valid_mask"][arm], strict=True):
            word = r["pw_vocabulary"][cls] if 0 <= cls < len(r["pw_vocabulary"]) else None
            known = word in words
            if ok and not known:
                unknown[split_by_episode[r["episode_id"]]] += 1
            labels.append(words.index(word) if known else -100)
            valid.append(bool(ok and known))
            if ok and known:
                distribution[split_by_episode[r["episode_id"]]][word] += 1
        # Preserve provenance and keep the optional per-arm fields in the SAME
        # ID space as the overlay vocabulary. Raw 25-class annotations stay intact
        # in the source manifest and in these explicitly named source fields.
        r["source_pw_vocabulary"] = source_words
        r["source_pw_arm_class_targets"] = r["pw_arm_class_targets"]
        r["source_pw_arm_valid_mask"] = r["pw_arm_valid_mask"]
        mapped_arms, arm_masks, arm_reasons = [], [], []
        original_reasons = r.get("pw_arm_mask_reasons", [[None] * 4 for _ in r["pw_effectors"]])
        for original_labels, original_mask, old_reasons in zip(
            r["pw_arm_class_targets"], r["pw_arm_valid_mask"], original_reasons, strict=True
        ):
            arm_words = [source_words[c] if 0 <= c < len(source_words) else None for c in original_labels]
            mapped_arms.append([words.index(w) if w in words else -100 for w in arm_words])
            arm_masks.append([bool(m and w in words) for m, w in zip(original_mask, arm_words, strict=True)])
            arm_reasons.append(
                [
                    None if m else (reason or "unknown_task_word")
                    for m, reason in zip(arm_masks[-1], old_reasons, strict=True)
                ]
            )
        reasons = list(r.get("pw_arm_mask_reasons", [[None] * 4] * 2)[arm])
        for block, ok in enumerate(valid):
            reasons[block] = None if ok else (reasons[block] or "unknown_task_word")
        r.update(
            source_split=r["split"],
            split=split_by_episode[r["episode_id"]],
            pw_class_targets=labels,
            pw_valid_mask=valid,
            pw_latent_valid_mask=[True] * 4,
            pw_vocabulary_id=manifest["id"],
            pw_vocabulary=words,
            pw_vocabulary_hash=vocabulary_hash({"id": manifest["id"], "words": words}),
            task_vocabulary_hash=config.vocabulary_hash,
            selected_effector=effector,
            pw_mask_reasons=reasons,
            pw_arm_class_targets=mapped_arms,
            pw_arm_valid_mask=arm_masks,
            pw_arm_mask_reasons=arm_reasons,
        )
        fully_valid[r["split"]] += int(all(valid))
        # Resolve paths without changing any original metadata or images.
        for key in ("history_frame_paths", "future_frame_paths"):
            r[key] = [str((source.parent / p).resolve()) for p in r[key]]
        mapped.append(r)
    output.mkdir(parents=True)
    (output / "vocabulary.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    (output / "windows.jsonl").write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in mapped))
    protocol = {
        "schema": 1,
        "source_windows": str(source),
        "source_windows_sha256": file_sha256(source),
        "task_id": task_id,
        "effector": effector,
        "split_by_episode": split_by_episode,
        "pilot_holdout_from_global_train": sorted(test_episodes),
        "global_split_modified": False,
        "freeze_before_actual_data_training": True,
        "validation_split": None,
        "window_counts": dict(Counter(r["split"] for r in mapped)),
        "fully_word_valid_windows": dict(fully_valid),
        "label_counts": {s: dict(v) for s, v in distribution.items()},
        "unknown_eval_labels_masked": dict(unknown),
        "model_vocabulary_hash": config.vocabulary_hash,
        "windows_sha256": file_sha256(output / "windows.jsonl"),
        "timing": {"fps": 10, "history": 5, "future": 16, "pw_k": 4, "pw_stride": 4, "future_seconds": 1.6},
    }
    (output / "protocol.json").write_text(json.dumps(protocol, indent=2, sort_keys=True) + "\n")
    common = {
        "schema": 1,
        "task_protocol": "protocol.json",
        "vocabulary": "vocabulary.json",
        "windows": "windows.jsonl",
        "checkpoint": "nvidia/Cosmos3-Edge",
        "revision": "344d602b128d1bbdacb43b08d0a3626f46343e29",
        "seed": 20261101,
        "batch_size": 1,
        "num_workers": 0,
        "timing": protocol["timing"],
        "rf": {
            "distribution": "waver",
            "resolution": "480",
            "shift": 5,
            "num_train_timesteps": 1000,
            "continuous": True,
            "weight": "uniform",
            "loss_scale": 10,
            "normalize_by_active": False,
        },
        "optimizer": {"lr_gen": 1e-5, "lr_und": 1e-6, "lr_pw": 1e-4, "reasoner_layers": [24, 25, 26, 27]},
        "evaluation": {
            "split": "test",
            "seeds": [20261101, 20261102, 20261103],
            "sigmas": [0.12345, 0.3746, 0.82731],
            "metric": "single-step future velocity MSE",
            "cfg": False,
            "joint_cache": False,
            "ema": False,
        },
        "formal_training_started": False,
        "max_train_steps": 0,
    }
    for mode in ("native", "matched_rgb", "full_pw"):
        spec = {
            **common,
            "mode": mode,
            "und_rgb_history": mode != "native",
            "predicted_pw": mode == "full_pw",
            "lambda_word": int(mode == "full_pw"),
            "lambda_latent": int(mode == "full_pw"),
        }
        (output / f"{mode}.json").write_text(json.dumps(spec, indent=2, sort_keys=True) + "\n")
    return manifest, protocol


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--windows", required=True)
    parser.add_argument("--vocabulary", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--effector", choices=["left", "right"], required=True)
    parser.add_argument("--test-episode", action="append", default=[])
    args = parser.parse_args()
    _, protocol = prepare_task(
        args.windows,
        args.vocabulary,
        args.output,
        task_id=args.task,
        effector=args.effector,
        test_episodes=args.test_episode,
    )
    print(json.dumps(protocol, indent=2))


if __name__ == "__main__":
    main()
