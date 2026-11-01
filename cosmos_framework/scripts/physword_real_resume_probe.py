"""Independent interpreter continuation of the bounded real PW/head Phase A probe."""

import argparse
import hashlib
import json
from pathlib import Path

import torch

from cosmos_framework.data.generator.physword.dataset import TrainingWindowDataset, to_m3_batch
from cosmos_framework.data.generator.physword.task import config_from_manifest
from cosmos_framework.inference.physword import generate_physical_words, repack_physical_words
from cosmos_framework.model.generator.diffusion.rectified_flow import RectifiedFlow
from cosmos_framework.model.generator.physword.checkpoint import load_checkpoint, training_protocol
from cosmos_framework.model.generator.physword.loading import load_local_edge
from cosmos_framework.model.generator.physword.multimodal import build_reasoner_prefix
from cosmos_framework.model.generator.physword.optimization import PhaseConfig, build_phase_optimizer
from cosmos_framework.model.generator.physword.preprocessing import PreprocessingConfig
from cosmos_framework.model.generator.physword.training import PhysWordTrainingStep, rollout_eval
from cosmos_framework.model.generator.utils.data_and_condition import GenerationDataClean


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--task-dir", type=Path, required=True)
    parser.add_argument("--probe-output", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads((args.probe_output / "checkpoint/manifest.json").read_text())
    # Rebuild expected behavior from the local frozen experiment inputs, rather
    # than trusting only the saved manifest's behavior values.
    vae_config = json.loads((args.checkpoint / "vae/config.json").read_text())
    preprocessing = PreprocessingConfig(
        (480, 640),
        (64, 64),
        pixel_device="cuda",
        latent_mean=tuple(vae_config["latents_mean"]),
        latent_std=tuple(vae_config["latents_std"]),
    )
    data_protocol = json.loads((args.task_dir / "protocol.json").read_text())
    actual_sha = hashlib.sha256((args.task_dir / "windows.jsonl").read_bytes()).hexdigest()
    if actual_sha != data_protocol["windows_sha256"]:
        raise ValueError("Actual dataset manifest differs from the frozen split")
    config = config_from_manifest(json.loads((args.task_dir / "vocabulary.json").read_text()))
    net, proc, _ = load_local_edge(args.checkpoint, config, initialize_pw=False)
    net.physword.float()
    net.cuda()
    phase = PhaseConfig(phase="A", reasoner_layers=())
    optimizer = build_phase_optimizer(net, phase)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=8)
    special = {
        "eos_token_id": proc.tokenizer.eos_token_id,
        "start_of_generation": proc.tokenizer.convert_tokens_to_ids("<|vision_start|>"),
        "end_of_generation": proc.tokenizer.convert_tokens_to_ids("<|vision_end|>"),
    }
    flow = RectifiedFlow(None, train_time_distribution="waver", shift=5, device="cuda")
    step = PhysWordTrainingStep(net, proc, None, None, flow, special, "A", preprocessing=preprocessing)
    protocol = training_protocol(step, data_protocol)
    protocol["inference_scheduler"] = json.loads((args.checkpoint / "scheduler/scheduler_config.json").read_text())
    revisions = {"base": "344d602b128d1bbdacb43b08d0a3626f46343e29", "data": actual_sha}
    proc, restored = load_checkpoint(
        args.probe_output / "checkpoint",
        net,
        optimizer,
        phase,
        step.config,
        expected_revisions=revisions,
        scheduler=scheduler,
        experiment=protocol,
    )
    step.processor = proc
    oracle = torch.load(args.probe_output / "uninterrupted.pt", map_location="cpu", weights_only=True)
    draws = (torch.rand(3), torch.rand(3, device="cuda").cpu())
    assert all(torch.equal(a, b) for a, b in zip(draws, oracle["rng"], strict=True))
    dataset = TrainingWindowDataset(args.task_dir / "windows.jsonl", split="train")
    batch = to_m3_batch(dataset[0], pw_config=config)
    with torch.no_grad():
        logits = step(batch)["teacher"]["logits"].cpu()
        with rollout_eval(net), torch.autocast("cuda", dtype=torch.bfloat16):
            observed = preprocessing.batch(batch)
            prefix = build_reasoner_prefix(net, proc, observed.history, observed.instruction)
            prediction = generate_physical_words(net, prefix)
            data = GenerationDataClean(
                batch_size=1,
                is_image_batch=False,
                x0_tokens_vision=[x.cuda() for x in oracle["gen_latents"]],
                fps_vision=torch.tensor([10.0]),
            )
            pack = repack_physical_words(net, prediction.condition, data, torch.tensor([374.6]), special)
            pack.to_cuda()
            velocity = net(pack)["preds_vision"][0].cpu()
    torch.testing.assert_close(logits, oracle["logits"], atol=0, rtol=0)
    torch.testing.assert_close(velocity, oracle["velocity"], atol=0, rtol=0)
    assert torch.equal(prediction.class_ids.cpu(), oracle["ids"])
    next_index = restored["training_state"]["data_cursor"]
    next_batch = to_m3_batch(dataset[next_index], pw_config=config)
    output = step(next_batch)
    for key, loss in output["losses"].items():
        torch.testing.assert_close(loss.detach().cpu(), oracle["losses"][key], atol=0, rtol=0)
    output["losses"]["total"].backward()
    optimizer.step()
    scheduler.step()
    for name, expected in oracle["parameters"].items():
        torch.testing.assert_close(dict(net.named_parameters())[name].detach().cpu(), expected, atol=0, rtol=0)
    assert scheduler.state_dict() == oracle["scheduler"]
    assert restored["step"] == manifest["step"] == 1
    result = {
        "status": "PASS",
        "weights": "REAL Edge; PW/head-only A; 64x64 diagnosis",
        "independent_process": True,
        "schema": manifest["schema"],
        "word_logits_max_abs": 0,
        "velocity_max_abs": 0,
        "next_loss_max_abs": 0,
        "next_parameter_update_max_abs": 0,
        "rng_equal": True,
        "scheduler_equal": True,
        "predicted_ids_equal": True,
        "next_window_id": dataset[next_index]["window_id"],
        "preprocessing": preprocessing.contract(),
        "semantic_reinitialized": False,
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
