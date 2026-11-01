"""One explicitly Tiny optimizer step using real task annotations and RGB.

This is an integration check, not task fitting. Geometry is explicitly resized
to 64x64 for the random native Tiny model; the original files are never written.
"""

import argparse
import json
from dataclasses import asdict, replace
from pathlib import Path

import torch
from torch.nn import functional as F

from cosmos_framework.data.generator.physword.dataset import TrainingWindowDataset, to_m3_batch
from cosmos_framework.data.generator.physword.task import config_from_manifest
from cosmos_framework.model.generator.diffusion.rectified_flow import RectifiedFlow
from cosmos_framework.model.generator.physword.checkpoint import save_checkpoint, training_protocol
from cosmos_framework.model.generator.physword.initialization import initialize_physical_words
from cosmos_framework.model.generator.physword.optimization import (
    PhaseConfig,
    build_phase_optimizer,
    optimizer_audit,
    snapshot_parameters,
)
from cosmos_framework.model.generator.physword.training import PhysWordTrainingStep
from cosmos_framework.scripts.physword_m2_probe import tiny_multimodal_fixture, tiny_vae
from cosmos_framework.scripts.physword_probe import SPECIAL_TOKENS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError("Choose a fresh probe output directory")
    args.output.mkdir(parents=True)
    manifest = json.loads((args.task_dir / "vocabulary.json").read_text())
    config = config_from_manifest(manifest)
    data_protocol = json.loads((args.task_dir / "protocol.json").read_text())
    dataset = TrainingWindowDataset(args.task_dir / "windows.jsonl")
    index = next(i for i, r in enumerate(dataset.records) if not all(r["pw_valid_mask"]))
    sample = dataset[index]
    batch = to_m3_batch(sample, pw_config=config)
    observed = (
        F.interpolate(batch.history.frames.float(), size=(64, 64), mode="bilinear", align_corners=False).round().byte()
    )
    future = F.interpolate(batch.future_rgb.float(), size=(64, 64), mode="bilinear", align_corners=False).round().byte()
    batch = replace(batch, history=replace(batch.history, frames=observed), future_rgb=future)
    reports = {}
    for phase in ("A", "B"):
        net, processor, _ = tiny_multimodal_fixture(pw_config=asdict(config))
        net.float()
        initialization = initialize_physical_words(
            net, processor.tokenizer, source={"kind": "SYNTHETIC random native Tiny UND", "verified_pretrained": False}
        )
        cfg = PhaseConfig(phase=phase, reasoner_layers=(0, 1))
        optimizer = build_phase_optimizer(net, cfg)
        vae = tiny_vae() if phase == "B" else None

        def encode(pixels, frozen_vae=vae):
            return frozen_vae.encode(pixels.cuda(), (torch.zeros(48, device="cuda"), torch.ones(48, device="cuda")))

        flow = RectifiedFlow(None, train_time_distribution="waver", shift=5, device="cuda")
        step = PhysWordTrainingStep(
            net, processor, vae, encode if vae is not None else None, flow, SPECIAL_TOKENS, phase
        )
        before = snapshot_parameters(net)
        optimizer.zero_grad(set_to_none=True)
        result = step(batch, timesteps=torch.tensor([374.6]))
        result["losses"]["total"].backward()
        optimizer.step()
        audit = optimizer_audit(net, optimizer, before)
        protocol = training_protocol(step, data_protocol)
        output = args.output / phase
        output.mkdir()
        save_checkpoint(
            output / "checkpoint",
            net,
            processor,
            optimizer,
            cfg,
            step.config,
            step=1,
            revisions={"base": "SYNTHETIC Tiny", "data": data_protocol["windows_sha256"]},
            experiment=protocol,
            vae=vae,
            training_state={
                "global_step": 1,
                "accumulation_step": 0,
                "data_window_id": sample["window_id"],
                "data_cursor": index + 1,
            },
        )
        reports[phase] = {
            "weights": "SYNTHETIC random native Tiny; NOT real Edge weights",
            "data": "real badminton left-arm annotations and decoded real RGB, explicitly resized to 64x64",
            "source_rgb_shape": list(sample["history_rgb"].shape),
            "window_id": sample["window_id"],
            "vocabulary": list(config.vocabulary),
            "initialization": initialization,
            "losses": {k: v.detach().item() for k, v in result["losses"].items()},
            "counts": result["counts"],
            "optimizer_groups": audit,
            "frozen_parameters_unchanged": True,
        }
        (output / "result.json").write_text(json.dumps(reports[phase], indent=2) + "\n")
        del result, net, optimizer, step, vae
        torch.cuda.empty_cache()
    (args.output / "summary.json").write_text(
        json.dumps(
            {
                p: {k: v for k, v in r.items() if k not in {"optimizer_groups", "initialization"}}
                for p, r in reports.items()
            },
            indent=2,
        )
        + "\n"
    )
    print((args.output / "summary.json").read_text())


if __name__ == "__main__":
    main()
