# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Strict single-device PW resume bundle. Separate from native DCP/FSDP/export."""

import hashlib
import inspect
import json
import random
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from cosmos_framework.data.generator.processors.cosmos3_edge_processing import build_cosmos3_edge_processor


def pw_mapping(network):
    base = network.physword.config.token_id_base
    return {token: base + index for index, token in enumerate(network.physword.config.tokens)}


def _canonical(value):
    return json.loads(json.dumps(value, default=str, sort_keys=True, allow_nan=False))


def training_protocol(training_step, data_protocol, *, baseline="full_pw"):
    """Behavior-affecting state outside network.state_dict, checked strictly on resume."""
    flow = training_step.flow
    return _canonical(
        {
            "rf": {
                "distribution": flow.train_time_sampler.distribution,
                "weight": flow.train_time_weight.weight,
                # Diffusers constructs this internal list from a set. Its order
                # changes across processes and has no scheduler behavior.
                "scheduler": {k: v for k, v in flow.noise_scheduler.config.items() if k != "_use_default_values"},
                "scheduler_class": f"{type(flow.noise_scheduler).__module__}.{type(flow.noise_scheduler).__qualname__}",
                "resolution": training_step.resolution,
                "continuous_train_shift": training_step.rf_shift,
            },
            "data": data_protocol,
            "baseline": baseline,
            "timing": {"fps": 10, "history": 5, "future": 16, "k": 4, "stride": 4},
            "missing_feedback": "neutral pw_start; never fabricate a class label",
            "preprocessing": training_step.preprocessing.contract(),
            "visual_processor": {
                "image": training_step.processor.image_processor.to_dict(),
                "video": training_step.processor.video_processor.to_dict(),
            },
        }
    )


def _component_contract(component):
    if component is None:
        return None
    result = {"class": f"{type(component).__module__}.{type(component).__qualname__}"}
    if isinstance(component, torch.optim.lr_scheduler.LRScheduler):
        mutable = {"last_epoch", "_last_lr", "_step_count", "_get_lr_called_within_step"}
        result["configuration"] = {k: v for k, v in component.state_dict().items() if k not in mutable}
    if isinstance(component, torch.optim.lr_scheduler.LambdaLR):
        result["lr_functions"] = [_function_contract(fn) for fn in component.lr_lambdas]
    if isinstance(component, torch.optim.swa_utils.AveragedModel):
        result["use_buffers"] = component.use_buffers
        result["avg_fn"] = _function_contract(component.avg_fn)
        result["multi_avg_fn"] = _function_contract(component.multi_avg_fn)
    if hasattr(component, "is_enabled"):
        result["enabled"] = component.is_enabled()
    return _canonical(result)


def _function_contract(function):
    """Capture common pure schedule/EMA function choices, including closure decay."""
    if function is None:
        return None
    function = inspect.unwrap(function)
    result = {"module": function.__module__, "name": function.__qualname__}
    if inspect.isfunction(function):
        result["bytecode"] = hashlib.sha256(function.__code__.co_code).hexdigest()
        primitive = (str, int, float, bool, type(None))
        result["constants"] = [x for x in function.__code__.co_consts if isinstance(x, primitive)]
        closure = inspect.getclosurevars(function)
        result["values"] = {
            k: v for k, v in {**closure.nonlocals, **closure.globals}.items() if isinstance(v, primitive)
        }
        result["defaults"] = function.__defaults__
    else:
        result["attributes"] = vars(function)
    return _canonical(result)


def _rng_state(device):
    numpy = np.random.get_state()
    return {
        "python": random.getstate(),
        "numpy": {
            "kind": numpy[0],
            "keys": torch.from_numpy(numpy[1].copy()),
            "position": numpy[2],
            "has_gauss": numpy[3],
            "cached_gaussian": numpy[4],
        },
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state(device),
    }


def _restore_rng(state, device):
    random.setstate(state["python"])
    numpy = state["numpy"]
    np.random.set_state(
        (numpy["kind"], numpy["keys"].numpy(), numpy["position"], numpy["has_gauss"], numpy["cached_gaussian"])
    )
    torch.set_rng_state(state["torch_cpu"])
    torch.cuda.set_rng_state(state["torch_cuda"], device=device)


def _sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_base_state(network, state, *, tokenizer=None, source=None):
    """Warm start native-remapped base weights; only the two new PW keys may be absent."""
    expected = network.state_dict()
    missing = expected.keys() - state.keys()
    if missing - {"physword.embedding.weight", "physword.word_head.weight"} or state.keys() - expected.keys():
        raise ValueError("Base checkpoint has missing/unexpected non-PW parameters")
    if any(state[name].shape != expected[name].shape for name in state):
        raise ValueError("Base checkpoint tensor shape mismatch")
    if network.physword.initialization is not None:
        raise ValueError("Warm-start cannot overwrite a semantically initialized/resumed model")
    result = network.load_state_dict(state, strict=False)
    if tokenizer is not None:
        from cosmos_framework.model.generator.physword.initialization import initialize_physical_words

        initialize_physical_words(network, tokenizer, source=source)
    return result


def save_checkpoint(
    path,
    network,
    processor,
    optimizer,
    phase_config,
    loss_config,
    *,
    step,
    revisions,
    scheduler=None,
    ema=None,
    scaler=None,
    experiment=None,
    training_state=None,
    vae=None,
):
    path = Path(path)
    if path.exists() and any(path.iterdir()):
        raise FileExistsError("Choose an empty checkpoint directory to preserve previous checkpoints")
    if not revisions or step < 0:
        raise ValueError("Checkpoint needs explicit base/teacher provenance and a nonnegative step")
    if training_state is not None and (
        training_state.get("global_step") != step or training_state.get("accumulation_step", -1) < 0
    ):
        raise ValueError("Global step and accumulation state must agree with the checkpoint boundary")
    if (network.physword.initialization or {}).get("source", {}).get("verified_pretrained"):
        if not experiment or not {"rf", "data", "timing", "baseline", "preprocessing", "visual_processor"}.issubset(
            experiment
        ):
            raise ValueError("Real pretrained runs require the complete RF/data/experiment protocol")
        if training_state is None or not {"data_cursor", "sampler"}.intersection(training_state):
            raise ValueError("Real pretrained runs require an explicit dataset cursor or sampler state")
    path.mkdir(parents=True, exist_ok=True)
    processor.tokenizer.save_pretrained(path)
    processor.image_processor.to_json_file(path / "preprocessor_config.json")
    processor.video_processor.to_json_file(path / "video_preprocessor_config.json")
    (path / "chat_template.jinja").write_text(processor.chat_template)
    manifest = {
        "schema": 3,
        "step": step,
        "mapping": pw_mapping(network),
        "classes": list(network.physword.config.vocabulary),
        "vocabulary_hash": network.physword.config.vocabulary_hash,
        "initialization": network.physword.initialization,
        "network_config": _canonical(network.config.to_dict()),
        "physword": asdict(network.physword.config),
        "phase": asdict(phase_config),
        "losses": asdict(loss_config),
        "revisions": revisions,
        "experiment": _canonical(experiment or {}),
        "components": {
            k: _component_contract(v)
            for k, v in {"scheduler": scheduler, "ema": ema, "scaler": scaler, "vae": vae}.items()
        },
        "optimizer_names": [g["parameter_names"] for g in optimizer.param_groups],
        "processor_hashes": {p.name: _sha(p) for p in path.iterdir() if p.is_file()},
    }
    # No pickled modules: tensors + optimizer/RNG primitives only; strict load below.
    torch.save(
        {
            "network": network.state_dict(),
            "optimizer": optimizer.state_dict(),
            "rng": _rng_state(next(network.parameters()).device),
            "training_state": training_state or {"global_step": step, "accumulation_step": 0, "data_cursor": None},
            "components": {
                k: v.state_dict()
                for k, v in {"scheduler": scheduler, "ema": ema, "scaler": scaler, "vae": vae}.items()
                if v is not None
            },
            "gradients": {n: p.grad.detach().cpu() for n, p in network.named_parameters() if p.grad is not None},
        },
        path / "state.pt",
    )
    manifest["state_sha256"] = _sha(path / "state.pt")
    (path / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")


def load_checkpoint(
    path,
    network,
    optimizer,
    phase_config,
    loss_config,
    *,
    expected_revisions,
    scheduler=None,
    ema=None,
    scaler=None,
    experiment=None,
    vae=None,
):
    path = Path(path)
    manifest = json.loads((path / "manifest.json").read_text())

    def canonical(x):
        return json.loads(json.dumps(x))

    checks = {
        "schema": 3,
        "mapping": pw_mapping(network),
        "classes": list(network.physword.config.vocabulary),
        "vocabulary_hash": network.physword.config.vocabulary_hash,
        "network_config": _canonical(network.config.to_dict()),
        "physword": asdict(network.physword.config),
        "phase": asdict(phase_config),
        "losses": asdict(loss_config),
        "revisions": expected_revisions,
        "experiment": _canonical(experiment or {}),
        "components": {
            k: _component_contract(v)
            for k, v in {"scheduler": scheduler, "ema": ema, "scaler": scaler, "vae": vae}.items()
        },
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
    network.physword.initialization = manifest["initialization"]
    optimizer.load_state_dict(state["optimizer"])
    for name, component in {"scheduler": scheduler, "ema": ema, "scaler": scaler, "vae": vae}.items():
        if component is not None:
            component.load_state_dict(state["components"][name])
    for name, parameter in network.named_parameters():
        gradient = state["gradients"].get(name)
        parameter.grad = None if gradient is None else gradient.to(parameter)
    processor = build_cosmos3_edge_processor(str(path))  # Native local loader, no model downloads.
    manifest["training_state"] = state["training_state"]
    _restore_rng(state["rng"], next(network.parameters()).device)
    return processor, manifest
