"""Separate-process Tiny ABC resume acceptance; no pretrained resources or fitting."""

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch

from cosmos_framework.data.generator.physword.sampling import StatefulWindowSampler
from cosmos_framework.model.generator.physword.checkpoint import load_checkpoint, save_checkpoint, training_protocol
from cosmos_framework.scripts.physword_m3_probe import SYNTHETIC_REVISIONS, synthetic_training


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("save", "resume"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    net, processor, vae, cfg, optimizer, batch, step, _ = synthetic_training()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=8)
    scaler = torch.amp.GradScaler("cuda", init_scale=128)
    ema = torch.optim.swa_utils.AveragedModel(
        net, device="cpu", multi_avg_fn=torch.optim.swa_utils.get_ema_multi_avg_fn(0.9)
    )
    sampler = StatefulWindowSampler(3, "frozen synthetic split", seed=7)
    random.seed(3)
    np.random.seed(4)
    protocol = training_protocol(step, {"split": "frozen SYNTHETIC"})

    def update():
        index = sampler.next_index()
        draws = (random.random(), float(np.random.random()), torch.rand(3), torch.rand(3, device="cuda").cpu())
        optimizer.zero_grad(set_to_none=True)
        out = step(batch)  # Fresh continuous RF sample and noise, all three branches.
        scaler.scale(out["losses"]["total"]).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        ema.update_parameters(net)
        return {
            "losses": {k: v.detach().cpu() for k, v in out["losses"].items()},
            "draws": draws,
            "sample_index": index,
            "model": {k: v.detach().cpu().clone() for k, v in net.state_dict().items()},
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "ema": ema.state_dict(),
            "scaler": scaler.state_dict(),
            "sampler": sampler.state_dict(),
        }

    if args.mode == "save":
        args.output.mkdir(parents=True, exist_ok=False)
        update()
        save_checkpoint(
            args.output / "checkpoint",
            net,
            processor,
            optimizer,
            cfg,
            step.config,
            step=1,
            revisions=SYNTHETIC_REVISIONS,
            scheduler=scheduler,
            scaler=scaler,
            ema=ema,
            experiment=protocol,
            vae=vae,
            training_state={"global_step": 1, "accumulation_step": 0, "sampler": sampler.state_dict()},
        )
        torch.save(update(), args.output / "uninterrupted.pt")
        print("Saved after step 1 and recorded uninterrupted step 2", flush=True)
        return

    processor, manifest = load_checkpoint(
        args.output / "checkpoint",
        net,
        optimizer,
        cfg,
        step.config,
        expected_revisions=SYNTHETIC_REVISIONS,
        scheduler=scheduler,
        scaler=scaler,
        ema=ema,
        experiment=protocol,
        vae=vae,
    )
    step.processor = processor
    sampler.load_state_dict(manifest["training_state"]["sampler"])
    actual = update()
    expected = torch.load(args.output / "uninterrupted.pt", map_location="cpu", weights_only=True)

    def exact(a, b):
        if isinstance(a, torch.Tensor):
            torch.testing.assert_close(a.cpu(), b.cpu(), atol=0, rtol=0)
        elif isinstance(a, dict):
            assert a.keys() == b.keys()
            for key in a:
                exact(a[key], b[key])
        elif isinstance(a, (tuple, list)):
            assert len(a) == len(b)
            for x, y in zip(a, b, strict=True):
                exact(x, y)
        else:
            assert a == b, (a, b)

    exact(actual, expected)
    result = {
        "status": "PASS",
        "weights": "SYNTHETIC native Tiny",
        "phase": "full A+B+C",
        "next_losses": {k: float(v) for k, v in actual["losses"].items()},
        "next_loss_max_abs": 0,
        "all_model_parameters_max_abs": 0,
        "optimizer_scheduler_ema_scaler_equal": True,
        "all_rng_draws_equal": True,
        "sampler_equal": True,
        "process": "independent Python interpreter",
    }
    (args.output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
