# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Strict single-device PW resume bundle. Separate from native DCP/FSDP/export."""

import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import torch

from cosmos_framework.data.generator.processors.cosmos3_edge_processing import build_cosmos3_edge_processor
from cosmos_framework.model.generator.physword.conditioning import ATOMIC_ACTIONS, PW_TOKENS


def pw_mapping(network):
    base = network.physword.config.token_id_base
    return {token: base + index for index, token in enumerate(PW_TOKENS)}


def _sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_base_state(network, state):
    """Warm start native-remapped base weights; only the two new PW keys may be absent."""
    expected = network.state_dict()
    missing = expected.keys() - state.keys()
    if missing - {"physword.embedding.weight", "physword.word_head.weight"} or state.keys() - expected.keys():
        raise ValueError("Base checkpoint has missing/unexpected non-PW parameters")
    if any(state[name].shape != expected[name].shape for name in state):
        raise ValueError("Base checkpoint tensor shape mismatch")
    return network.load_state_dict(state, strict=False)


def save_checkpoint(path, network, processor, optimizer, phase_config, loss_config, *, step, revisions):
    path = Path(path)
    if path.exists() and any(path.iterdir()):
        raise FileExistsError("Choose an empty checkpoint directory to preserve previous checkpoints")
    if not revisions or step < 0:
        raise ValueError("Checkpoint needs explicit base/teacher provenance and a nonnegative step")
    path.mkdir(parents=True, exist_ok=True)
    processor.tokenizer.save_pretrained(path)
    processor.image_processor.to_json_file(path / "preprocessor_config.json")
    processor.video_processor.to_json_file(path / "video_preprocessor_config.json")
    (path / "chat_template.jinja").write_text(processor.chat_template)
    manifest = {
        "schema": 1,
        "step": step,
        "mapping": pw_mapping(network),
        "classes": list(ATOMIC_ACTIONS),
        "physword": asdict(network.physword.config),
        "phase": asdict(phase_config),
        "losses": asdict(loss_config),
        "revisions": revisions,
        "optimizer_names": [g["parameter_names"] for g in optimizer.param_groups],
        "processor_hashes": {p.name: _sha(p) for p in path.iterdir() if p.is_file()},
    }
    # No pickled modules: tensors + optimizer/RNG primitives only; strict load below.
    torch.save(
        {
            "network": network.state_dict(),
            "optimizer": optimizer.state_dict(),
            "rng_cpu": torch.get_rng_state(),
            "rng_cuda": torch.cuda.get_rng_state(next(network.parameters()).device),
        },
        path / "state.pt",
    )
    manifest["state_sha256"] = _sha(path / "state.pt")
    (path / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")


def load_checkpoint(path, network, optimizer, phase_config, loss_config, *, expected_revisions):
    path = Path(path)
    manifest = json.loads((path / "manifest.json").read_text())

    def canonical(x):
        return json.loads(json.dumps(x))

    checks = {
        "schema": 1,
        "mapping": pw_mapping(network),
        "classes": list(ATOMIC_ACTIONS),
        "physword": asdict(network.physword.config),
        "phase": asdict(phase_config),
        "losses": asdict(loss_config),
        "revisions": expected_revisions,
        "optimizer_names": [g["parameter_names"] for g in optimizer.param_groups],
    }
    for name, expected in checks.items():
        if manifest[name] != canonical(expected):
            raise ValueError(f"Checkpoint {name} differs; refusing a silent phase/mapping/teacher change")
    for name, digest in manifest["processor_hashes"].items():
        if Path(name).name != name or _sha(path / name) != digest:
            raise ValueError(f"Checkpoint processor integrity failed: {name}")
    if _sha(path / "state.pt") != manifest["state_sha256"]:
        raise ValueError("Checkpoint state integrity failed")
    state = torch.load(path / "state.pt", map_location="cpu", weights_only=True)
    current = network.state_dict()
    if current.keys() != state["network"].keys() or any(current[n].shape != state["network"][n].shape for n in current):
        raise ValueError("Resume requires all network keys and exact shapes, including PW and vision")
    network.load_state_dict(state["network"], strict=True)
    optimizer.load_state_dict(state["optimizer"])
    processor = build_cosmos3_edge_processor(str(path))  # Native local loader, no model downloads.
    torch.set_rng_state(state["rng_cpu"])
    torch.cuda.set_rng_state(state["rng_cuda"], device=next(network.parameters()).device)
    return processor, manifest
