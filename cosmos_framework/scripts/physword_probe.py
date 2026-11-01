# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""M0/M1 tiny Edge velocity probe. Random weights; no model download or training.

This file can also run against the untouched checkout using PYTHONPATH and
--baseline-only: that mode never imports the new Physical Word implementation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from cosmos_framework.data.generator.sequence_packing.packers import pack_input_sequence
from cosmos_framework.data.generator.sequence_packing.sequence import SequencePlan
from cosmos_framework.model.generator.mot.cosmos3_vfm_network import Cosmos3VFMNetwork, Cosmos3VFMNetworkConfig
from cosmos_framework.model.generator.mot.unified_mot import Nemotron3DenseVLMoTConfig, Nemotron3DenseVLTextForCausalLM
from cosmos_framework.model.generator.utils.data_and_condition import GenerationDataClean

WORDS_A = ["approach", "pick", "move", "place"]
WORDS_B = ["approach", "push", "move", "place"]
SPECIAL_TOKENS = {"eos_token_id": 11, "start_of_generation": 20, "end_of_generation": 21}
SEED = 20261101


def make_network(*, physword: bool = False, device: str = "cuda") -> Cosmos3VFMNetwork:
    """Actual Edge dual-path classes, shrunk to two layers with GQA and UND K norm."""
    torch.manual_seed(SEED)
    config = Nemotron3DenseVLMoTConfig(
        config_dict=dict(
            vocab_size=64,
            hidden_size=128,
            intermediate_size=256,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=32,
            mrope_section=[6, 5, 5],
            attention_dropout=0.0,
            hidden_dropout=0.0,
            tie_word_embeddings=False,
        ),
        qk_norm_for_text=False,
        qk_norm_for_diffusion=True,
    )
    config.use_und_k_norm_for_gen = True
    language_model = Nemotron3DenseVLTextForCausalLM(config)
    kwargs = {}
    if physword:
        kwargs["physword"] = {"enabled": True}
    network = Cosmos3VFMNetwork(
        language_model,
        Cosmos3VFMNetworkConfig(
            vlm_config=language_model.config,
            latent_channel_size=48,
            latent_patch_size=2,
            latent_downsample_factor=16,
            max_latent_h=8,
            max_latent_w=8,
            max_latent_t=8,
            vision_gen=True,
            action_gen=False,
            sound_gen=False,
            joint_attn_implementation="two_way",
            **kwargs,
        ),
    )
    network.init_weights(buffer_device=torch.device("cpu"))
    return network.eval().to(device=device, dtype=torch.bfloat16 if device.startswith("cuda") else torch.float32)


def make_inputs(batch_size: int = 1) -> tuple[list[SequencePlan], list[list[int]], GenerationDataClean, torch.Tensor]:
    """Fixed synthetic RGB and noise. RGB→latent adapter is a fixture, NOT the Wan VAE."""
    generator = torch.Generator().manual_seed(SEED + 1)
    rgb = torch.randint(0, 256, (batch_size, 3, 32, 32), dtype=torch.uint8, generator=generator)
    history = F.adaptive_avg_pool2d(rgb.float() / 127.5 - 1, (4, 4)).repeat(1, 16, 1, 1)
    noise = torch.randn(batch_size, 48, 2, 4, 4, generator=generator)
    x0 = torch.randn(batch_size, 48, 2, 4, 4, generator=generator)
    latent = 0.5 * noise + 0.5 * x0
    latent[:, :, 0] = history
    data = GenerationDataClean(
        batch_size=batch_size,
        is_image_batch=False,
        x0_tokens_vision=[latent[i : i + 1].clone() for i in range(batch_size)],
        fps_vision=torch.full((batch_size,), 10.0),
    )
    plans = [
        SequencePlan(has_text=True, has_vision=True, condition_frame_indexes_vision=[0]) for _ in range(batch_size)
    ]
    captions = [[3, 4, 5], [6, 7, 8, 9]][:batch_size]
    return plans, captions, data, rgb


def make_pack(*, words: list[list[str]] | None = None, enabled: bool = False, batch_size: int = 1):
    plans, captions, data, _ = make_inputs(batch_size)
    kwargs = {}
    if enabled:
        kwargs = {"physword": {"enabled": True}, "physical_words": words}
    return pack_input_sequence(
        plans,
        captions,
        data,
        torch.full((batch_size,), 500.0),
        SPECIAL_TOKENS,
        latent_patch_size=2,
        **kwargs,
    )


def tensor_digest(tensor: torch.Tensor) -> str:
    return hashlib.sha256(tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


@torch.no_grad()
def run_probe(output: Path, *, baseline_only: bool = False, device: str = "cuda") -> dict:
    output.mkdir(parents=True, exist_ok=True)
    baseline_net = make_network(device=device)
    pack = make_pack()
    pack.to_cuda()
    baseline = baseline_net(pack)["preds_vision"][0].detach().cpu()
    torch.save(baseline, output / "baseline_velocity.pt")
    report = {
        "seed": SEED,
        "sigma": 0.5,
        "timestep": 500,
        "torch": torch.__version__,
        "device": device,
        "weights": "random tiny Edge: 2 layers, width128, GQA4:2, UND K norm",
        "rgb_latent_adapter": "deterministic fixture; real SigLIP2 and VAE not loaded",
        "rgb_sha256": tensor_digest(make_inputs()[3]),
        "baseline_sha256": tensor_digest(baseline),
        "velocity_shape": list(baseline.shape),
        "baseline_only": baseline_only,
    }
    if not baseline_only:
        from cosmos_framework.model.generator.physword.conditioning import physical_word_trace
        from cosmos_framework.model.generator.physword.config import PhysWordConfig

        net = make_network(physword=True, device=device)
        # All pretrained/shared parameters must be exactly the same as the no-PW model.
        shared = baseline_net.state_dict()
        for key, value in shared.items():
            torch.testing.assert_close(net.state_dict()[key], value, atol=0, rtol=0)
        results = []
        for words in (WORDS_A, WORDS_B):
            pw_pack = make_pack(words=[words], enabled=True)
            trace = physical_word_trace(pw_pack, PhysWordConfig(enabled=True))
            pw_pack.to_cuda()
            result = net(pw_pack)
            results.append(result["preds_vision"][0].detach().cpu())
            report.setdefault("traces", []).append(trace)
        for label, velocity in zip(("pick", "push"), results, strict=True):
            torch.save(velocity, output / f"{label}_velocity.pt")
        delta = (results[0].float() - results[1].float())[:, :, 1:]
        report.update(
            intervention_future_rms=delta.square().mean().sqrt().item(),
            intervention_future_max=delta.abs().max().item(),
            pick_sha256=tensor_digest(results[0]),
            push_sha256=tensor_digest(results[1]),
        )
        if report["intervention_future_rms"] == 0:
            raise AssertionError("Changing only PW produced no change in future velocity")
    (output / "probe.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--baseline-only", action="store_true")
    args = parser.parse_args()
    print(json.dumps(run_probe(args.output, baseline_only=args.baseline_only), indent=2))


if __name__ == "__main__":
    main()
