# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""ONE synthetic tiny optimizer step only. No actual Edge training or dataset fitting."""

import argparse
import json
from pathlib import Path

import torch

from cosmos_framework.model.generator.diffusion.rectified_flow import RectifiedFlow
from cosmos_framework.model.generator.physword.checkpoint import save_checkpoint
from cosmos_framework.model.generator.physword.optimization import (
    PhaseConfig,
    build_phase_optimizer,
    optimizer_audit,
    snapshot_parameters,
)
from cosmos_framework.model.generator.physword.preprocessing import PreprocessingConfig
from cosmos_framework.model.generator.physword.training import LossConfig, PhysWordBatch, PhysWordTrainingStep
from cosmos_framework.scripts.physword_m2_probe import tiny_multimodal_fixture, tiny_vae
from cosmos_framework.scripts.physword_probe import SPECIAL_TOKENS

SYNTHETIC_REVISIONS = {
    "base": "random tiny seed20261101",
    "vision": "random tiny seed20261103",
    "vae": "random tiny seed20261105 identity normalization",
    "data": "SYNTHETIC uint8 seed20261108",
}


def synthetic_training(phase="B", **overrides):
    net, processor, history = tiny_multimodal_fixture()
    net.float()  # FP32 optimizer master parameters; forward uses BF16 autocast.
    vae = tiny_vae()
    cfg = PhaseConfig(phase=phase, reasoner_layers=(0, 1), **overrides)
    optimizer = build_phase_optimizer(net, cfg)
    future = torch.randint(
        0, 256, (16, 3, 64, 64), dtype=torch.uint8, generator=torch.Generator().manual_seed(20261108)
    )
    batch = PhysWordBatch(
        history,
        "Pick the object and move it to the target.",
        future,
        torch.tensor([[2, 1, 1, 3]]),
        torch.ones(1, 4, dtype=torch.bool),
        torch.ones(1, 4, dtype=torch.bool),
    )
    flow = RectifiedFlow(None, shift=3, device="cuda")

    def encode(pixels):
        return vae.encode(pixels.cuda(), (torch.zeros(48, device="cuda"), torch.ones(48, device="cuda")))

    step = PhysWordTrainingStep(
        net,
        processor,
        vae,
        encode,
        flow,
        SPECIAL_TOKENS,
        phase,
        LossConfig(),
        preprocessing=PreprocessingConfig((64, 64)),
    )
    kwargs = {
        "timesteps": flow.noise_scheduler.timesteps[500:501],
        "noise": torch.randn((1, 48, 6, 4, 4), generator=torch.Generator().manual_seed(12)),
    }
    return net, processor, vae, cfg, optimizer, batch, step, kwargs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=["A", "B"], default="B")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError("Choose a fresh output directory")
    net, processor, vae, cfg, optimizer, batch, step, kwargs = synthetic_training(args.phase)
    before = snapshot_parameters(net)
    optimizer.zero_grad(set_to_none=True)
    result = step(batch, **kwargs)
    result["losses"]["total"].backward()
    optimizer.step()
    groups = optimizer_audit(net, optimizer, before)
    args.output.mkdir(parents=True, exist_ok=True)
    report = {
        "data": "SYNTHETIC; random tiny weights; one optimizer update only",
        "phase": args.phase,
        "losses": {k: float(v.detach()) for k, v in result["losses"].items()},
        "counts": result["counts"],
        "optimizer_groups": groups,
        "frozen_unchanged": True,
    }
    (args.output / "step.json").write_text(json.dumps(report, indent=2) + "\n")
    save_checkpoint(
        args.output / "checkpoint", net, processor, optimizer, cfg, step.config, step=1, revisions=SYNTHETIC_REVISIONS
    )
    print(json.dumps({k: v for k, v in report.items() if k != "optimizer_groups"}, indent=2))


if __name__ == "__main__":
    main()
