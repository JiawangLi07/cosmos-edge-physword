"""Real pinned Edge initialization and RGB-only single-step acceptance, never fitting."""

import argparse
import gc
import json
from dataclasses import replace
from pathlib import Path

import torch
from diffusers import AutoencoderKLWan
from torch.nn import functional as F

from cosmos_framework.data.generator.physword.dataset import TrainingWindowDataset, to_m3_batch
from cosmos_framework.data.generator.physword.task import config_from_manifest
from cosmos_framework.inference.physword import generate_physical_words, repack_physical_words
from cosmos_framework.inference.physword_baselines import pack_video_baseline
from cosmos_framework.inference.physword_diagnostics import mask_physical_word_suffix
from cosmos_framework.model.generator.diffusion.rectified_flow import RectifiedFlow
from cosmos_framework.model.generator.physword.checkpoint import load_checkpoint, save_checkpoint, training_protocol
from cosmos_framework.model.generator.physword.loading import load_local_edge
from cosmos_framework.model.generator.physword.multimodal import build_reasoner_prefix
from cosmos_framework.model.generator.physword.optimization import (
    PhaseConfig,
    build_phase_optimizer,
    optimizer_audit,
    snapshot_parameters,
)
from cosmos_framework.model.generator.physword.training import PhysWordTrainingStep, rollout_eval
from cosmos_framework.model.generator.utils.data_and_condition import GenerationDataClean


def real_checkpoint_roundtrip(
    net, processor, opt, scheduler, cfg, step, batch, next_batch, data, special, args, config, record
):
    """Compare uninterrupted vs resumed real A update, plus logits/AR/video restoration."""
    protocol = training_protocol(step, json.loads((args.task_dir / "protocol.json").read_text()))
    protocol["inference_scheduler"] = json.loads((args.checkpoint / "scheduler/scheduler_config.json").read_text())
    revisions = {"base": "344d602b128d1bbdacb43b08d0a3626f46343e29", "data": protocol["data"]["windows_sha256"]}

    def inference(network, proc, train_step):
        with torch.no_grad():
            logits = train_step(batch)["teacher"]["logits"].cpu().clone()
            with rollout_eval(network), torch.autocast("cuda", dtype=torch.bfloat16):
                prefix = build_reasoner_prefix(network, proc, batch.history, batch.instruction)
                prediction = generate_physical_words(network, prefix)
                p = repack_physical_words(network, prediction.condition, data, torch.tensor([374.6]), special)
                p.to_cuda()
                velocity = network(p)["preds_vision"][0].cpu().clone()
        return logits, prediction.class_ids.cpu().clone(), velocity

    expected_logits, expected_ids, expected_velocity = inference(net, processor, step)
    opt.zero_grad(set_to_none=True)
    save_checkpoint(
        args.output / "checkpoint",
        net,
        processor,
        opt,
        cfg,
        step.config,
        step=1,
        revisions=revisions,
        scheduler=scheduler,
        experiment=protocol,
        training_state={
            "global_step": 1,
            "accumulation_step": 0,
            "data_cursor": 1,
            "next_window_id": "badminton__episode_0001__000001",
        },
    )
    expected_rng = (torch.rand(3), torch.rand(3, device="cuda").cpu())
    expected = step(next_batch)
    expected_losses = {k: v.detach().cpu().clone() for k, v in expected["losses"].items()}
    expected["losses"]["total"].backward()
    opt.step()
    scheduler.step()
    expected_parameters = {n: p.detach().cpu().clone() for n, p in net.named_parameters() if p.requires_grad}
    expected_lr = scheduler.state_dict()
    del expected
    opt.zero_grad(set_to_none=True)
    net.cpu()
    gc.collect()
    torch.cuda.empty_cache()
    print("Reloading fresh real Edge without semantic reinitialization", flush=True)
    restored, _, _ = load_local_edge(args.checkpoint, config, initialize_pw=False)
    restored.physword.float()
    restored.cuda()
    opt2 = build_phase_optimizer(restored, cfg)
    scheduler2 = torch.optim.lr_scheduler.CosineAnnealingLR(opt2, T_max=8)
    proc2, manifest = load_checkpoint(
        args.output / "checkpoint",
        restored,
        opt2,
        cfg,
        step.config,
        expected_revisions=revisions,
        scheduler=scheduler2,
        experiment=protocol,
    )
    actual_rng = (torch.rand(3), torch.rand(3, device="cuda").cpu())
    rng_equal = all(torch.equal(a, b) for a, b in zip(expected_rng, actual_rng, strict=True))
    flow = RectifiedFlow(None, train_time_distribution="waver", shift=5, device="cuda")
    step2 = PhysWordTrainingStep(restored, proc2, None, None, flow, special, "A")
    actual_logits, actual_ids, actual_velocity = inference(restored, proc2, step2)
    logits_diff = (actual_logits - expected_logits).abs().max().item()
    velocity_diff = (actual_velocity - expected_velocity).abs().max().item()
    actual = step2(next_batch)
    loss_diff = max((actual["losses"][k].detach().cpu() - v).abs().item() for k, v in expected_losses.items())
    actual["losses"]["total"].backward()
    opt2.step()
    scheduler2.step()
    update_diff = max(
        (dict(restored.named_parameters())[n].detach().cpu() - v).abs().max().item()
        for n, v in expected_parameters.items()
    )
    success = (
        logits_diff == velocity_diff == loss_diff == update_diff == 0
        and rng_equal
        and torch.equal(actual_ids, expected_ids)
        and scheduler2.state_dict() == expected_lr
    )
    record(
        "real_checkpoint_roundtrip",
        {
            "status": "PASS" if success else "FAIL",
            "word_logits_max_abs": logits_diff,
            "predicted_ids_equal": torch.equal(actual_ids, expected_ids),
            "velocity_max_abs": velocity_diff,
            "next_loss_max_abs": loss_diff,
            "next_parameter_update_max_abs": update_diff,
            "rng_equal": rng_equal,
            "scheduler_equal": scheduler2.state_dict() == expected_lr,
            "data_cursor": manifest["training_state"]["data_cursor"],
            "semantic_reinitialized_on_resume": False,
            "checkpoint_bytes": (args.output / "checkpoint/state.pt").stat().st_size,
            "trainable": "real PW embeddings and Word Head; UND/GEN frozen",
        },
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--task-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--phase-a-step", action="store_true", help="One PW/head-only diagnostic update; no fitting")
    parser.add_argument("--checkpoint-roundtrip", action="store_true")
    args = parser.parse_args()
    if args.checkpoint_roundtrip and not args.phase_a_step:
        raise ValueError("Real interrupted resume validation requires --phase-a-step")
    args.output.mkdir(parents=True, exist_ok=True)
    result = {"weights": "REAL pretrained Edge", "formal_training": False}

    def record(stage, data):
        result[stage] = data
        (args.output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
        print(stage, data.get("status", "PASS"), flush=True)

    torch.manual_seed(20261101)
    config = config_from_manifest(json.loads((args.task_dir / "vocabulary.json").read_text()))
    print("Loading pinned real UND/GEN/vision/projector on CPU", flush=True)
    net, processor, loading = load_local_edge(args.checkpoint, config)
    record("loading", {"status": "PASS", **loading})
    sample = TrainingWindowDataset(args.task_dir / "windows.jsonl")[0]
    batch = to_m3_batch(sample, pw_config=config)

    # Small diagnostic spatial size is explicit; timestamps and GT labels are unchanged.
    def resize(x):
        return F.interpolate(x.float(), size=(64, 64), mode="bilinear", align_corners=False).round().byte()

    batch = replace(
        batch, history=replace(batch.history, frames=resize(batch.history.frames)), future_rgb=resize(batch.future_rgb)
    )
    record(
        "input",
        {
            "status": "PASS",
            "data": "REAL RGB and left-arm annotations; explicitly resized 480x640 -> 64x64",
            "window_id": sample["window_id"],
            "history": 5,
            "future": 16,
            "fps": 10,
        },
    )
    print("Loading real published Wan VAE; encode before moving the large MoT to GPU", flush=True)
    vae, info = AutoencoderKLWan.from_pretrained(
        str(args.checkpoint / "vae"),
        local_files_only=True,
        torch_dtype=torch.bfloat16,
        output_loading_info=True,
        low_cpu_mem_usage=False,
    )
    if info.get("missing_keys") or info.get("unexpected_keys") or info.get("mismatched_keys"):
        raise ValueError(f"Wan VAE loading mismatch: {info}")
    vae = vae.eval().requires_grad_(False).cuda()
    vcfg = vae.config
    mean = torch.tensor(vcfg.latents_mean, device="cuda", dtype=torch.float32).view(1, -1, 1, 1, 1)
    std = torch.tensor(vcfg.latents_std, device="cuda", dtype=torch.float32).view(1, -1, 1, 1, 1)
    with torch.no_grad():
        observed = batch.history.frames.permute(1, 0, 2, 3).unsqueeze(0).cuda().float() / 127.5 - 1
        future = batch.future_rgb.permute(1, 0, 2, 3).unsqueeze(0).cuda().float() / 127.5 - 1
        h = vae.encode(observed.bfloat16()).latent_dist.mode().float()
        full = vae.encode(torch.cat((observed, future), 2).bfloat16()).latent_dist.mode().float()
        difference = (h - full[:, :, :2]).abs().max().item()
        clean = torch.cat(((h - mean) / std, (full[:, :, 2:] - mean) / std), 2)
    record(
        "vae",
        {
            "status": "PASS" if difference <= 1e-3 else "FAIL",
            "missing": [],
            "unexpected": [],
            "history_shape": list(h.shape),
            "full_shape": list(full.shape),
            "history_prefix_max_abs": difference,
            "history_prefix_atol": 1e-3,
            "normalization": "published latents_mean / latents_std",
        },
    )
    vae.cpu()
    del h, full, observed, future
    gc.collect()
    torch.cuda.empty_cache()
    net.cuda()
    special = {
        "eos_token_id": processor.tokenizer.eos_token_id,
        "start_of_generation": processor.tokenizer.convert_tokens_to_ids("<|vision_start|>"),
        "end_of_generation": processor.tokenizer.convert_tokens_to_ids("<|vision_end|>"),
    }
    noise = torch.randn(clean.shape, generator=torch.Generator().manual_seed(12)).cuda()
    sigma = torch.tensor([0.3746], device="cuda")
    t = sigma * 1000
    flow = RectifiedFlow(None, train_time_distribution="waver", shift=5, device="cuda")
    mask = torch.tensor([1, 1, 0, 0, 0, 0], device="cuda").float().reshape(6, 1, 1)
    xt, _ = flow.noise_vision([noise], [clean], sigma.reshape(1, 1), [mask])
    data = GenerationDataClean(batch_size=1, is_image_batch=False, x0_tokens_vision=xt, fps_vision=torch.tensor([10.0]))
    baseline = {}
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for mode in ("native", "matched_rgb", "full_pw"):
            packed = pack_video_baseline(
                net, processor, batch.history, batch.instruction, data, t.cpu(), special, mode=mode
            )
            packed.to_cuda()
            velocity = net(packed)["preds_vision"][0]
            if not bool(torch.isfinite(velocity).all()):
                raise FloatingPointError("Nonfinite real Edge velocity")
            baseline[mode] = {
                "shape": list(velocity.shape),
                "future_velocity_rms": velocity[:, :, 2:].float().square().mean().sqrt().item(),
            }
            if mode == "native":
                original_pw = net.physword
                net.physword = None
                try:
                    off = net(packed)["preds_vision"][0]
                finally:
                    net.physword = original_pw
                baseline[mode]["pw_off_max_abs"] = (velocity - off).abs().max().item()
        prefix = build_reasoner_prefix(net, processor, batch.history, batch.instruction)
        predicted = generate_physical_words(net, prefix)
        a = repack_physical_words(net, predicted.condition, data, t.cpu(), special)
        changed = predicted.class_ids.tolist()
        changed[1] = (changed[1] + 1) % len(config.vocabulary)
        b = repack_physical_words(net, prefix.with_words(tuple(changed), config.token_id_base), data, t.cpu(), special)
        a.to_cuda()
        b.to_cuda()
        va, repeat, vb = [net(p)["preds_vision"][0][:, :, 2:].float() for p in (a, a, b)]
        with mask_physical_word_suffix(net, a):
            ma, mb = [net(p)["preds_vision"][0][:, :, 2:].float() for p in (a, b)]

        def rms(x):
            return x.square().mean().sqrt().item()

        record(
            "joint",
            {
                "status": "PASS" if torch.equal(ma, mb) and baseline["native"]["pw_off_max_abs"] == 0 else "FAIL",
                "baselines": baseline,
                "predicted_class_ids": predicted.class_ids.tolist(),
                "intervention_rms": rms(va - vb),
                "relative_rms": rms(va - vb) / rms(va),
                "repeat_floor_rms": rms(va - repeat),
                "full_suffix_masked_intervention_rms": rms(ma - mb),
            },
        )
    del prefix, predicted, a, b, velocity, va, repeat, vb, ma, mb
    gc.collect()
    torch.cuda.empty_cache()
    if args.phase_a_step:
        print("One real PW/head-only Phase A update (no UND/GEN unfreezing)", flush=True)
        net.physword.float()
        cfg = PhaseConfig(phase="A", reasoner_layers=())
        opt = build_phase_optimizer(net, cfg)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=8) if args.checkpoint_roundtrip else None
        # Phase A needs no VAE on GPU; visual teacher and all base tensors stay frozen.
        step = PhysWordTrainingStep(net, processor, None, None, flow, special, "A")
        before = snapshot_parameters(net)
        try:
            out = step(batch)
            out["losses"]["total"].backward()
            opt.step()
            audit = optimizer_audit(net, opt, before)
            record(
                "phase_a_update",
                {
                    "status": "PASS",
                    "trainable": "PW embeddings + head only",
                    "losses": {k: v.detach().item() for k, v in out["losses"].items()},
                    "counts": out["counts"],
                    "groups": audit,
                    "frozen_unchanged": True,
                },
            )
            if scheduler is not None:
                scheduler.step()
                dataset = TrainingWindowDataset(args.task_dir / "windows.jsonl")
                next_batch = to_m3_batch(dataset[1], pw_config=config)
                next_batch = replace(
                    next_batch,
                    history=replace(next_batch.history, frames=resize(next_batch.history.frames)),
                    future_rgb=resize(next_batch.future_rgb),
                )
                del out
                real_checkpoint_roundtrip(
                    net, processor, opt, scheduler, cfg, step, batch, next_batch, data, special, args, config, record
                )
        except torch.cuda.OutOfMemoryError:
            record(
                "phase_a_update",
                {"status": "BLOCKED", "reason": "shared GPU free VRAM is insufficient for real backward"},
            )
    record(
        "memory",
        {
            "status": "PASS",
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "free_bytes_end": torch.cuda.mem_get_info()[0],
        },
    )


if __name__ == "__main__":
    main()
