"""Bounded M3 Phase B acceptance: independent losses, two updates and process resume.

This is a single-device 64x64 diagnostic, never an M4 fitting loop. The run mode
starts separate save/resume interpreters and enforces a total wall-time limit.
"""

import argparse
import gc
import hashlib
import json
import os
import random
import signal
import subprocess
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from cosmos_framework.data.generator.physword.sampling import StatefulWindowSampler
from cosmos_framework.model.generator.physword.checkpoint import load_checkpoint, save_checkpoint, training_protocol
from cosmos_framework.model.generator.physword.optimization import (
    PhaseConfig,
    build_phase_optimizer,
    optimizer_audit,
    snapshot_parameters,
)


def write_json(path, value):
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temp.replace(path)


def signature(value):
    """Exact per-tensor fingerprints, without another multi-GB oracle checkpoint."""
    if isinstance(value, torch.Tensor):
        tensor = value.detach().cpu().contiguous()
        return {
            "shape": list(tensor.shape),
            "dtype": str(tensor.dtype),
            "sha256": hashlib.sha256(tensor.reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest(),
        }
    if isinstance(value, dict):
        return {str(key): signature(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [signature(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"Unsupported state fingerprint value: {type(value)}")


def require_free_device():
    """Recheck after CPU loading; never mistake another job's low utilization for free."""
    uuid = os.environ.get("M3_EXPECTED_GPU_UUID")
    if not uuid:
        return  # Explicit interactive/Tiny execution owns device selection.
    result = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader,nounits"],
        check=True,
        capture_output=True,
        text=True,
        timeout=15,
    )
    foreign = [
        line
        for line in result.stdout.splitlines()
        if line.split(",")[0].strip() == uuid and int(line.split(",")[1]) != os.getpid()
    ]
    if foreign:
        raise RuntimeError(f"Selected GPU acquired by another process before allocation: {foreign}")
    result = subprocess.run(
        ["nvidia-smi", "-i", uuid, "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
        check=True,
        capture_output=True,
        text=True,
        timeout=15,
    )
    if int(result.stdout.strip()) < 70 * 1024:
        raise RuntimeError("Selected GPU has less than the queued 70 GiB free-memory requirement")


def prepare_masters(network, phase):
    """Use the existing exact selector; retain only its masters in FP32 on CPU."""
    selector = build_phase_optimizer(network.float(), phase)
    selected = [group["parameter_names"] for group in selector.param_groups]
    del selector
    for parameter in network.parameters():
        if not parameter.requires_grad:
            parameter.data = parameter.data.bfloat16()
    return selected


def build(args, *, resume):
    from cosmos_framework.model.generator.diffusion.rectified_flow import RectifiedFlow

    experiment_sha = None
    if args.tiny:
        from cosmos_framework.scripts.physword_m3_probe import SYNTHETIC_REVISIONS, synthetic_training

        net, proc, vae, phase, optimizer, batch, step, _ = synthetic_training()
        batches = [
            batch,
            replace(
                batch, word_labels=torch.tensor([[-100, 1, 1, 3]]), valid_word=torch.tensor([[False, True, True, True]])
            ),
        ]
        step.flow = RectifiedFlow(None, train_time_distribution="waver", shift=5, device="cuda")
        protocol = training_protocol(step, {"kind": "SYNTHETIC", "windows": ["tiny0", "tiny1"]})
        revisions = SYNTHETIC_REVISIONS

        def get_batch(index):
            return batches[index]

        def window_id(index):
            return f"tiny{index}"

        size, manifest_hash = 2, "synthetic-two-windows"
        loading = {"weights": "SYNTHETIC native Tiny; not real Edge"}
    else:
        from diffusers import AutoencoderKLWan

        from cosmos_framework.data.generator.physword.dataset import TrainingWindowDataset, to_m3_batch
        from cosmos_framework.data.generator.physword.task import config_from_manifest
        from cosmos_framework.model.generator.physword.loading import load_local_edge
        from cosmos_framework.model.generator.physword.preprocessing import PreprocessingConfig
        from cosmos_framework.model.generator.physword.training import LossConfig, PhysWordTrainingStep

        data = json.loads((args.task_dir / "protocol.json").read_text())
        manifest_hash = hashlib.sha256((args.task_dir / "windows.jsonl").read_bytes()).hexdigest()
        if data["windows_sha256"] != manifest_hash:
            raise ValueError("Actual task windows differ from the frozen dataset protocol")
        cfg = config_from_manifest(json.loads((args.task_dir / "vocabulary.json").read_text()))
        spec = json.loads((args.task_dir / "full_pw.json").read_text())
        experiment_sha = hashlib.sha256((args.task_dir / "full_pw.json").read_bytes()).hexdigest()
        if not spec["predicted_pw"] or spec["batch_size"] != 1 or spec["mode"] != "full_pw":
            raise ValueError("Expected the frozen predicted-PW B=1 experiment")
        expected_rf = {
            "continuous": True,
            "distribution": "waver",
            "loss_scale": 10,
            "normalize_by_active": False,
            "num_train_timesteps": 1000,
            "resolution": "480",
            "shift": 5,
            "weight": "uniform",
        }
        if spec["rf"] != expected_rf or spec["timing"] != data["timing"]:
            raise ValueError("This bounded acceptance requires the frozen native RF/timing protocol")
        phase = PhaseConfig(
            phase="B", **{**spec["optimizer"], "reasoner_layers": tuple(spec["optimizer"]["reasoner_layers"])}
        )
        net, proc, loading = load_local_edge(args.checkpoint, cfg, initialize_pw=not resume)
        if loading["revision"] != spec["revision"]:
            raise ValueError("Pinned experiment/base revision differs")
        # Resolve the existing exact selector on CPU, then keep ONLY selected
        # masters FP32. Rebuild the actual optimizer after moving to CUDA.
        selected = prepare_masters(net, phase)
        vae, info = AutoencoderKLWan.from_pretrained(
            str(args.checkpoint / "vae"),
            local_files_only=True,
            torch_dtype=torch.bfloat16,
            output_loading_info=True,
            low_cpu_mem_usage=False,
        )
        if any(info.get(name) for name in ("missing_keys", "unexpected_keys", "mismatched_keys")):
            raise ValueError(f"Real VAE checkpoint mismatch: {info}")
        require_free_device()
        net.cuda()
        optimizer = build_phase_optimizer(net, phase)
        if selected != [group["parameter_names"] for group in optimizer.param_groups]:
            raise AssertionError("CPU and CUDA exact parameter selectors disagree")
        vae.eval().requires_grad_(False).cuda()
        vae_cfg = json.loads((args.checkpoint / "vae/config.json").read_text())
        preprocessing = PreprocessingConfig(
            (480, 640),
            (64, 64),
            pixel_device="cuda",
            latent_mean=tuple(vae_cfg["latents_mean"]),
            latent_std=tuple(vae_cfg["latents_std"]),
        )
        special = {
            "eos_token_id": proc.tokenizer.eos_token_id,
            "start_of_generation": proc.tokenizer.convert_tokens_to_ids("<|vision_start|>"),
            "end_of_generation": proc.tokenizer.convert_tokens_to_ids("<|vision_end|>"),
        }
        flow = RectifiedFlow(None, train_time_distribution=spec["rf"]["distribution"], shift=5, device="cuda")

        def encode_raw(pixels):
            return vae.encode(pixels.bfloat16()).latent_dist.mode()

        step = PhysWordTrainingStep(
            net,
            proc,
            vae,
            encode_raw,
            flow,
            special,
            "B",
            LossConfig(
                spec["lambda_word"], spec["lambda_latent"], spec["rf"]["loss_scale"], spec["rf"]["normalize_by_active"]
            ),
            preprocessing=preprocessing,
            resolution=spec["rf"]["resolution"],
        )
        dataset = TrainingWindowDataset(args.task_dir / "windows.jsonl", split="train")
        size = len(dataset)

        def get_batch(index):
            return to_m3_batch(dataset[index], pw_config=cfg)

        def window_id(index):
            return dataset.records[index]["window_id"]

        protocol = training_protocol(step, data)
        protocol["inference_scheduler"] = json.loads((args.checkpoint / "scheduler/scheduler_config.json").read_text())
        revisions = {"base": loading["revision"], "data": manifest_hash}
    # All settings below are bounded validation settings, not M4 training defaults.
    protocol["acceptance"] = {
        "kind": "M3 full Phase B",
        "distinct_steps": 2,
        "size": 64,
        "ema": False,
        "grad_scaler": False,
        "num_workers": 0,
        "experiment_spec_sha256": experiment_sha,
    }
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=8)
    sampler = StatefulWindowSampler(size, manifest_hash, shuffle=False, seed=20261101)
    return (
        net,
        proc,
        vae,
        phase,
        optimizer,
        step,
        protocol,
        revisions,
        scheduler,
        sampler,
        get_batch,
        window_id,
        loading,
    )


def gradient_checks(net, vae, optimizer, output):
    reports = {}
    named = dict(net.named_parameters())
    last = len(net.language_model.model.layers) - 1

    def norm(name):
        gradient = named[name].grad
        return 0.0 if gradient is None else float(gradient.float().norm())

    for loss in ("video", "word", "latent"):
        optimizer.zero_grad(set_to_none=True)
        output["losses"][loss].backward(retain_graph=True)
        groups = optimizer_audit(net, optimizer)
        for group in groups:
            for item in group["parameters"]:
                if not np.isfinite(item["grad_norm"]):
                    raise AssertionError(f"Nonfinite {loss} gradient: {item['name']}")
        norms = {group["name"]: group["grad_norm"] for group in groups}
        assert norms["pw_embedding"] > 0 and norms["und"] > 0
        assert (norms["word_head"] > 0) == (loss == "word")
        assert all(parameter.grad is None for parameter in vae.parameters())
        assert all(parameter.grad is None for parameter in net.language_model.visual.parameters())
        if loss == "video":
            assert norms["gen"] > 0 and norms["gen_und_k_norm"] > 0
            for projection in ("k_proj", "v_proj"):
                assert norm(f"language_model.model.layers.{last}.self_attn.{projection}.weight") > 0
            assert norm("language_model.model.norm.weight") == 0
            assert norm(f"language_model.model.layers.{last}.mlp.up_proj.weight") == 0
        else:
            assert norms["gen"] == norms["gen_und_k_norm"] == 0
            assert norm("language_model.model.norm.weight") > 0
        reports[loss] = {"loss": float(output["losses"][loss].detach()), "groups": groups}
    optimizer.zero_grad(set_to_none=True)
    return reports


def worker(args):
    args.output.mkdir(parents=True, exist_ok=True)
    random.seed(20261101)
    np.random.seed(20261101)
    torch.manual_seed(20261101)
    bundle = build(args, resume=args.mode == "resume")
    net, proc, vae, phase, optimizer, step, protocol, revisions, scheduler, sampler, batch_at, window_at, loading = (
        bundle
    )
    if args.mode == "save":
        write_json(
            args.output / "setup.json",
            {"loading": loading, "protocol": protocol, "trainable_groups": optimizer_audit(net, optimizer)},
        )
    else:
        proc, manifest = load_checkpoint(
            args.output / "checkpoint",
            net,
            optimizer,
            phase,
            step.config,
            expected_revisions=revisions,
            scheduler=scheduler,
            experiment=protocol,
            vae=vae,
        )
        if manifest["step"] != 1 or manifest["training_state"]["accumulation_step"] != 0:
            raise AssertionError("Expected the first complete Phase B update boundary")
        step.processor = proc
        sampler.load_state_dict(manifest["training_state"]["sampler"])

    def update(iteration, independent=False):
        index = sampler.next_index()
        draws = [random.random(), float(np.random.random()), torch.rand(3), torch.rand(3, device="cuda")]
        before = snapshot_parameters(net)
        vae_before = signature(vae.state_dict())
        optimizer.zero_grad(set_to_none=True)
        result = step(batch_at(index), iteration=iteration)  # Fresh native sigma and epsilon.
        assert result["packed"].reasoner_condition.word_class_ids == tuple(result["prediction"].class_ids.tolist())
        assert not result["prediction"].prediction_hidden_states.requires_grad
        if independent:
            write_json(args.output / "independent_gradients.json", gradient_checks(net, vae, optimizer, result))
        result["losses"]["total"].backward()
        optimizer.step()
        scheduler.step()
        groups = optimizer_audit(net, optimizer, before)
        assert all(any(parameter["changed"] for parameter in group["parameters"]) for group in groups)
        assert signature(vae.state_dict()) == vae_before
        observation = {
            "iteration": iteration,
            "sample_index": index,
            "window_id": window_at(index),
            "losses": {key: float(value.detach()) for key, value in result["losses"].items()},
            "counts": result["counts"],
            "predicted_ids": result["prediction"].class_ids.tolist(),
            "teacher_logits": signature(result["teacher"]["logits"]),
            "velocity": signature(result["velocity"]),
            "continuous_timestep": signature(result["packed"].vision.timesteps),
            "rng_draws": signature(draws),
            "groups": groups,
            "frozen_parameters_unchanged": True,
            "vae_unchanged": True,
        }
        del result, before
        optimizer.zero_grad(set_to_none=True)
        gc.collect()
        return observation

    if args.mode == "save":
        first = update(0, independent=True)
        write_json(args.output / "step1.json", first)
        save_checkpoint(
            args.output / "checkpoint",
            net,
            proc,
            optimizer,
            phase,
            step.config,
            step=1,
            revisions=revisions,
            scheduler=scheduler,
            experiment=protocol,
            vae=vae,
            training_state={"global_step": 1, "accumulation_step": 0, "sampler": sampler.state_dict()},
        )
        observation = update(1)
    else:
        observation = update(1)
    state = signature(
        {
            "network": net.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "vae": vae.state_dict(),
            "sampler": sampler.state_dict(),
            "next_rng": [random.random(), float(np.random.random()), torch.rand(3), torch.rand(3, device="cuda")],
        }
    )
    actual = {"observation": observation, "state": state}
    if args.mode == "save":
        write_json(args.output / "oracle.json", actual)
    else:
        expected = json.loads((args.output / "oracle.json").read_text())
        write_json(args.output / "resumed.json", actual)
        if actual != expected:
            raise AssertionError("Independent-process next step differs; compare oracle.json and resumed.json")
        write_json(
            args.output / "resume_result.json",
            {
                "status": "PASS",
                "weights": "SYNTHETIC Tiny" if args.tiny else "REAL Edge",
                "phase": "full A+B+C",
                "independent_process": True,
                "all_parameter_optimizer_vae_scheduler_sampler_fingerprints_equal": True,
                "losses_logits_ids_velocity_and_rng_equal": True,
                "next_step": observation["iteration"],
                "next_window_id": observation["window_id"],
                "formal_training": False,
            },
        )
    print(f"{args.mode} PASS; CUDA peak {torch.cuda.max_memory_allocated() / 2**30:.3f} GiB", flush=True)


def run(args):
    args.output.mkdir(parents=True, exist_ok=False)
    deadline = time.monotonic() + args.timeout_seconds
    write_json(args.output / "result.json", {"status": "RUNNING", "formal_training": False})
    child = None

    def interrupted(signum, frame):
        raise TimeoutError(f"Acceptance controller interrupted by signal {signum}")

    old_handler = signal.signal(signal.SIGTERM, interrupted)
    try:
        for mode, seed in (("save", "1"), ("resume", "2")):
            command = [
                sys.executable,
                "-m",
                "cosmos_framework.scripts.physword_real_b_probe",
                "--mode",
                mode,
                "--output",
                str(args.output / "validation"),
            ]
            if args.tiny:
                command.append("--tiny")
            else:
                command += ["--checkpoint", str(args.checkpoint), "--task-dir", str(args.task_dir)]
            with (args.output / f"{mode}.log").open("w") as log:
                child = subprocess.Popen(
                    command,
                    env={**os.environ, "PYTHONHASHSEED": seed},
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
                try:
                    code = child.wait(timeout=max(1, deadline - time.monotonic()))
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGTERM)
                    try:
                        child.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        os.killpg(child.pid, signal.SIGKILL)
                        child.wait()
                    raise TimeoutError("Bounded M3 acceptance wall-time exceeded") from None
            if code:
                raise RuntimeError(f"{mode} interpreter exited {code}; inspect {mode}.log")
        result = json.loads((args.output / "validation/resume_result.json").read_text())
        write_json(args.output / "result.json", result)
        print(json.dumps(result, indent=2))
    except BaseException as error:
        write_json(args.output / "result.json", {"status": "FAIL", "reason": str(error), "formal_training": False})
        raise
    finally:
        # Workers have separate process groups; terminate only our own child
        # when an outer timeout/SSH controller interrupts the parent.
        if child is not None and child.poll() is None:
            os.killpg(child.pid, signal.SIGTERM)
            try:
                child.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait()
        signal.signal(signal.SIGTERM, old_handler)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("run", "save", "resume"), default="run")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--task-dir", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tiny", action="store_true", help="Exercise this SAME runner with synthetic native Tiny")
    parser.add_argument("--timeout-seconds", type=int, default=2700)
    args = parser.parse_args()
    if not args.tiny and (args.checkpoint is None or args.task_dir is None):
        parser.error("Real acceptance requires local --checkpoint and frozen --task-dir")
    if not 30 <= args.timeout_seconds <= 3600:
        parser.error("Acceptance wall-time must be between 30 seconds and one hour")
    if args.mode == "run":
        run(args)
    else:
        worker(args)


if __name__ == "__main__":
    main()
