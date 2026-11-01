"""Split ORIGINAL capture groups before windowing, joining all known aliases/copies."""

from __future__ import annotations

import hashlib
import random
from collections import defaultdict

import h5py

from cosmos_framework.data.generator.physword.common import VERSION, digest, file_sha256


def source_content_hash(ep):
    if ep["source_kind"] == "video":
        return file_sha256(ep["cameras"][ep["camera"]]["video_path"])
    with h5py.File(ep["source_ref"], "r") as f:
        data = f[ep["group"]][ep["image_key"]]
        h = hashlib.sha256()
        h.update(str((data.shape, str(data.dtype))).encode())
        for i in range(len(data)):
            h.update(data[i].tobytes())
        return h.hexdigest()


def _allocate(groups, ratio, rng):
    """Global rounded quota with task stratification; singleton strata stay train."""
    strata = defaultdict(list)
    for key, eps in sorted(groups.items()):
        strata[tuple(sorted({e["task_id"] for e in eps}))].append(key)
    quotas = {}
    for task, keys in strata.items():
        rng.shuffle(keys)
        quotas[task] = min(int(len(keys) * ratio), len(keys) - 1)
    target = min(round(len(groups) * ratio), max(0, len(groups) - 1))
    while sum(quotas.values()) < target:
        candidates = [t for t, keys in strata.items() if quotas[t] < len(keys) - 1]
        if not candidates:
            break
        task = max(candidates, key=lambda t: (len(strata[t]) * ratio - quotas[t], t))
        quotas[task] += 1
    selected = {key for t, keys in strata.items() for key in keys[: quotas[t]]}
    return selected


def build_split(episodes, config):
    ids = [ep["episode_id"] for ep in episodes]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate episode ID")
    parent = {i: i for i in ids}

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    seen = {}
    enriched = []
    for ep in sorted(episodes, key=lambda e: e["episode_id"]):
        ep = dict(ep, content_sha256=source_content_hash(ep))
        enriched.append(ep)
        for field in ("original_episode_id", "lineage_id", "content_sha256"):
            key = field, ep[field]
            if key in seen:
                a, b = find(ep["episode_id"]), find(seen[key])
                parent[max(a, b)] = min(a, b)
            seen[key] = ep["episode_id"]
    groups = defaultdict(list)
    for ep in enriched:
        groups[find(ep["episode_id"])].append(ep)
    rng = random.Random(config.seed)
    test = _allocate(groups, config.test_ratio, rng)
    training = {key: eps for key, eps in groups.items() if key not in test}
    validation = _allocate(training, config.validation_ratio, rng) if config.validation_ratio else set()
    entries = []
    for group, eps in sorted(groups.items()):
        split = "test" if group in test else "validation" if group in validation else "train"
        for ep in eps:
            entries.append(
                {
                    "episode_id": ep["episode_id"],
                    "original_episode_id": ep["original_episode_id"],
                    "lineage_id": ep["lineage_id"],
                    "content_sha256": ep["content_sha256"],
                    "group_id": group,
                    "task_id": ep["task_id"],
                    "source_ref": ep["source_ref"],
                    "split": split,
                }
            )
    return {
        "version": VERSION,
        "seed": config.seed,
        "requested_train_ratio": config.train_ratio,
        "requested_test_ratio": config.test_ratio,
        "validation_fraction_of_train": config.validation_ratio,
        "split_unit": "original_episode_group",
        "episodes": entries,
        "original_group_count": len(groups),
        "notes": [
            "Task strata too small for a held-out group remain train; quotas use nearest feasible integer.",
            "Explicit lineage is REQUIRED for unknown transformed copies; content hashes detect byte-identical RGB sources.",
            "No visual near-duplicate detector; provenance-less crops/re-encodes must supply lineage before inclusion.",
        ],
        "inventory_hash": digest(enriched),
    }


def check_leakage(manifest):
    for field in ("episode_id", "original_episode_id", "lineage_id", "content_sha256", "group_id"):
        seen = {}
        for entry in manifest["episodes"]:
            value = entry[field]
            if value in seen and seen[value] != entry["split"]:
                raise ValueError(f"Episode leakage via {field}: {value}")
            seen[value] = entry["split"]
    return True


def inherit_split(manifest, parent):
    """Preserve surviving memberships after curation instead of reshuffling test episodes."""
    if manifest["seed"] != parent["seed"]:
        raise ValueError("Inherited split seed differs from requested experiment seed")
    assignments = {entry["original_episode_id"]: entry["split"] for entry in parent["episodes"]}
    for entry in manifest["episodes"]:
        if entry["original_episode_id"] not in assignments:
            raise ValueError("New episode has no assignment in the canonical curated split")
        entry["split"] = assignments[entry["original_episode_id"]]
    check_leakage(manifest)
    manifest.setdefault("notes", []).append(
        "Surviving memberships inherited from canonical curated split; no reshuffle after exclusion."
    )
    manifest["parent_split_hash"] = digest(parent)
    return manifest
