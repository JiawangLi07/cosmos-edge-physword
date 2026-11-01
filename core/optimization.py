# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Phase selection by exact module paths and full parameter names, never substring matching."""

import math
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class PhaseConfig:
    phase: str = "B"
    reasoner_layers: tuple[int, ...] = ()
    train_pw_embeddings: bool = True
    train_word_head: bool = True
    train_final_und_norm: bool = True
    lr_pw: float = 1e-4
    lr_und: float = 1e-6
    lr_gen: float = 1e-5
    weight_decay: float = 0.01

    def __post_init__(self):
        if any(
            type(flag) is not bool
            for flag in (self.train_pw_embeddings, self.train_word_head, self.train_final_und_norm)
        ):
            raise ValueError("Phase training switches must be booleans")
        if self.phase not in ("A", "B") or len(set(self.reasoner_layers)) != len(self.reasoner_layers):
            raise ValueError("Expected Phase A/B and unique explicit UND layer indexes")
        if self.phase == "A" and not (self.train_pw_embeddings and self.train_word_head):
            raise ValueError("Phase A trains PW embeddings and Word Head")
        if any(not math.isfinite(x) or x <= 0 for x in (self.lr_pw, self.lr_und, self.lr_gen)):
            raise ValueError("Group learning rates must be finite and positive")
        if not math.isfinite(self.weight_decay) or self.weight_decay < 0:
            raise ValueError("Weight decay must be nonnegative and finite")


def build_phase_optimizer(network, config: PhaseConfig):
    if network.physword is None or network.parallel_dims is not None:
        raise ValueError("PW phase optimizer requires registered PW modules and an unsharded network")
    named = dict(network.named_parameters())
    groups = []
    claimed = set()

    def collect(label, paths, lr):
        names = []
        for path in paths:
            module = network.get_submodule(path)  # exact path; a missing module is an error
            names.extend(f"{path}.{key}" for key, _ in module.named_parameters())
        names = sorted(names)
        if not names:
            return
        if len(set(names)) != len(names) or claimed.intersection(names) or any(n not in named for n in names):
            raise ValueError(f"Invalid/overlapping exact parameter group: {label}")
        claimed.update(names)
        groups.append(dict(name=label, params=[named[n] for n in names], parameter_names=names, lr=lr))

    if config.train_pw_embeddings:
        collect("pw_embedding", ["physword.embedding"], config.lr_pw)
    if config.train_word_head:
        collect("word_head", ["physword.word_head"], config.lr_pw)
    und_paths = []
    for index in config.reasoner_layers:
        if (
            not isinstance(index, int)
            or isinstance(index, bool)
            or not 0 <= index < len(network.language_model.model.layers)
        ):
            raise ValueError(f"Unknown UND layer {index}")
        root = f"language_model.model.layers.{index}"
        und_paths += [
            f"{root}.{part}"
            for part in (
                "self_attn.q_proj",
                "self_attn.k_proj",
                "self_attn.v_proj",
                "self_attn.o_proj",
                "mlp",
                "input_layernorm",
                "post_attention_layernorm",
            )
        ]
        und_paths += [
            f"{root}.{part}" for part in ("self_attn.q_norm", "self_attn.k_norm") if _exists(network, f"{root}.{part}")
        ]
    if config.reasoner_layers and config.train_final_und_norm:
        und_paths.append("language_model.model.norm")
    collect("und", und_paths, config.lr_und)
    if config.phase == "B":
        gen_paths = ["time_embedder", "vae2llm", "llm2vae", "language_model.model.norm_moe_gen"]
        norm_paths = []
        for index in range(len(network.language_model.model.layers)):
            root = f"language_model.model.layers.{index}"
            gen_paths += [
                f"{root}.{part}"
                for part in (
                    "self_attn.q_proj_moe_gen",
                    "self_attn.k_proj_moe_gen",
                    "self_attn.v_proj_moe_gen",
                    "self_attn.o_proj_moe_gen",
                    "self_attn.q_norm_moe_gen",
                    "self_attn.k_norm_moe_gen",
                    "mlp_moe_gen",
                    "input_layernorm_moe_gen",
                    "post_attention_layernorm_moe_gen",
                )
            ]
            norm_path = f"{root}.self_attn.k_norm_und_for_gen"
            if _exists(network, norm_path):
                norm_paths.append(norm_path)
        collect("gen", gen_paths, config.lr_gen)
        collect("gen_und_k_norm", norm_paths, config.lr_gen)
    # Resolve the entire allowlist before changing requires_grad or constructing optimizer/FSDP.
    if not claimed:
        raise ValueError("Phase selected no trainable parameters")
    if any(named[name].dtype != torch.float32 for name in claimed):
        raise ValueError("Use FP32 master parameters with BF16 autocast for single-device PW training")
    for name, parameter in named.items():
        parameter.requires_grad_(name in claimed)
        parameter.grad = None
    network._physword_phase_config = config
    return torch.optim.AdamW(groups, weight_decay=config.weight_decay)


def _exists(network, path):
    try:
        network.get_submodule(path)
        return True
    except AttributeError:
        return False


def snapshot_parameters(network):
    return {name: value.detach().cpu().clone() for name, value in network.named_parameters()}


def optimizer_audit(network, optimizer, before=None):
    named = dict(network.named_parameters())
    expected = {n for n, p in named.items() if p.requires_grad}
    seen = []
    report = []
    for group in optimizer.param_groups:
        names = group["parameter_names"]
        if [id(named[n]) for n in names] != [id(p) for p in group["params"]]:
            raise AssertionError("Optimizer parameter identity does not match the exact name manifest")
        seen.extend(names)
        entries = []
        for name in names:
            p = named[name]
            entry = {
                "name": name,
                "numel": p.numel(),
                "grad_norm": 0.0 if p.grad is None else p.grad.float().norm().item(),
            }
            if before is not None:
                delta = p.detach().cpu().float() - before[name].float()
                entry.update(update_norm=delta.norm().item(), changed=not torch.equal(p.detach().cpu(), before[name]))
            entries.append(entry)
        report.append(
            {
                "name": group["name"],
                "lr": group["lr"],
                "numel": sum(e["numel"] for e in entries),
                "grad_norm": math.sqrt(sum(e["grad_norm"] ** 2 for e in entries)),
                "parameters": entries,
            }
        )
    if len(seen) != len(set(seen)) or set(seen) != expected:
        raise AssertionError("Every trainable parameter must occur in exactly one optimizer group")
    if before is not None:
        for name, p in named.items():
            if name not in expected and not torch.equal(p.detach().cpu(), before[name]):
                raise AssertionError(f"Frozen parameter changed: {name}")
    return report
