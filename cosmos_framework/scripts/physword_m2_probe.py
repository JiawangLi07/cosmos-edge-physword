# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Reproducible M2 tiny native processor/SigLIP2/Wan/MoT fixtures. No weight downloads."""

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import PreTrainedTokenizerFast
from transformers.models.siglip2.configuration_siglip2 import Siglip2VisionConfig

from cosmos_framework.data.generator.processors.cosmos3_edge_processing import (
    NemotronNanoV3BridgeProcessor,
    Qwen3VLVideoProcessorCustom,
    Siglip2ImageProcessorCustom,
)
from cosmos_framework.inference.physword import predict_physical_word_video_from_rgb
from cosmos_framework.model.generator.physword.multimodal import RGBHistory, prepare_history_video_data
from cosmos_framework.model.generator.reasoner.nemotron_3_dense_vl.vision_siglip2 import NemotronSiglip2VisionEncoder
from cosmos_framework.model.generator.tokenizers.wan2pt2_vae_4x16x16 import WanVAE_
from cosmos_framework.scripts.physword_probe import SEED, SPECIAL_TOKENS, make_network


def tiny_multimodal_fixture():
    """Random tiny versions of the REAL native classes; only tokenizer vocabulary is synthetic."""
    net = make_network(physword=True)
    torch.manual_seed(SEED + 2)
    vcfg = Siglip2VisionConfig(
        hidden_size=32, intermediate_size=64, num_hidden_layers=1, num_attention_heads=4, num_patches=16, patch_size=16
    )
    vcfg._attn_implementation = "eager"
    pcfg = SimpleNamespace(spatial_merge_size=2, input_hidden_size=32, merger_intermedia=64, out_hidden_size=128)
    net.language_model.visual = NemotronSiglip2VisionEncoder(vcfg, pcfg).cuda().bfloat16().eval().requires_grad_(False)
    special = {
        "<pad>": 0,
        "<s>": 1,
        "<unk>": 2,
        "<|im_start|>": 10,
        "<|im_end|>": 11,
        "<|video_pad|>": 18,
        "<|image_pad|>": 19,
        "<|vision_start|>": 20,
        "<|vision_end|>": 21,
    }
    vocab = {f"tok_{i}": i for i in range(64)}
    for token, index in special.items():
        del vocab[f"tok_{index}"]
        vocab[token] = index
    backend = Tokenizer(WordLevel(vocab, unk_token="<unk>"))
    backend.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend,
        unk_token="<unk>",
        bos_token="<s>",
        eos_token="<|im_end|>",
        pad_token="<pad>",
        additional_special_tokens=list(special),
    )
    template_path = Path(__file__).resolve().parents[2] / "tests/data/physword/edge_chat_template.jinja"
    # The actual checkpoint template is small metadata, already cloned locally.
    processor = NemotronNanoV3BridgeProcessor(
        image_processor=Siglip2ImageProcessorCustom(
            patch_size=16,
            merge_size=2,
            size={"shortest_edge": 4096, "longest_edge": 20480},
            image_mean=[0.5] * 3,
            image_std=[0.5] * 3,
        ),
        video_processor=Qwen3VLVideoProcessorCustom(
            patch_size=16,
            merge_size=2,
            temporal_patch_size=1,
            size={"shortest_edge": 4096, "longest_edge": 20480},
            image_mean=[0.5] * 3,
            image_std=[0.5] * 3,
            return_metadata=True,
        ),
        tokenizer=tokenizer,
        chat_template=template_path.read_text(),
    )
    net.language_model.config.image_token_id = 19
    net.language_model.config.video_token_id = 18
    net.language_model.config.vision_start_token_id = 20
    net.language_model.config.vision_config = SimpleNamespace(spatial_merge_size=2)
    frames = torch.randint(0, 256, (5, 3, 64, 64), dtype=torch.uint8, generator=torch.Generator().manual_seed(SEED + 3))
    return net, processor, RGBHistory(frames)


def tiny_vae():
    torch.manual_seed(SEED + 4)
    return (
        WanVAE_(dim=16, dec_dim=16, z_dim=48, dim_mult=[1, 2, 2, 2], num_res_blocks=1)
        .cuda()
        .eval()
        .requires_grad_(False)
    )


@torch.no_grad()
def history_video_latents(vae, history):
    return prepare_history_video_data(
        history, lambda pixels: encode_tiny_history(vae, pixels), tiny_future_noise(history)
    )


def encode_tiny_history(vae, pixels):
    # Native causal Wan implementation; identity normalization is ONLY for this random tiny fixture.
    return vae.encode(pixels.cuda(), (torch.zeros(48, device="cuda"), torch.ones(48, device="cuda")))


def tiny_future_noise(history):
    shape = (1, 48, 4, history.frames.shape[-2] // 16, history.frames.shape[-1] // 16)
    return torch.randn(shape, generator=torch.Generator().manual_seed(SEED + 5))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    net, processor, history = tiny_multimodal_fixture()
    vae = tiny_vae()
    prediction, packed, out = predict_physical_word_video_from_rgb(
        net,
        processor,
        history,
        "Pick the object and move it to the target.",
        lambda pixels: encode_tiny_history(vae, pixels),
        tiny_future_noise(history),
        torch.tensor([500.0]),
        SPECIAL_TOKENS,
    )
    args.output.mkdir(parents=True, exist_ok=True)
    report = {
        "weights": "RANDOM TINY: native SigLIP2, PatchMerger, WanVAE_, Edge MoT; synthetic tokenizer vocabulary",
        "seed": SEED,
        "timestep": 500.0,
        "sigma": 0.5,
        "instruction": "Pick the object and move it to the target.",
        "class_ids": prediction.class_ids.tolist(),
        "token_ids": prediction.token_ids.tolist(),
        "prediction_positions": prediction.prediction_positions.tolist(),
        "token_positions": prediction.token_positions.tolist(),
        "cache_length": prediction.cache_length,
        "und_length": packed.text_ids.numel(),
        "visual_shape": list(prediction.condition.visual_features.shape),
        "video_velocity_shape": list(out["preds_vision"][0].shape),
        "position_ids": packed.position_ids[:, packed.text_indexes].cpu().tolist(),
        "visual_indexes": prediction.condition.visual_indexes.tolist(),
        "text_global_indexes": packed.text_indexes.cpu().tolist(),
        "sample_lens": packed.sample_lens,
        "split_lens": packed.split_lens,
        "condition_mask": packed.vision.condition_mask[0].flatten().cpu().tolist(),
        "noisy_latent_frames": packed.vision.noisy_frame_indexes[0].cpu().tolist(),
        "timing": {
            "source_fps": history.timing.source_fps,
            "model_fps": history.timing.model_fps,
            "source_frame_indexes": list(history.source_frame_indexes),
            "source_pts_seconds": list(history.source_pts_seconds),
            "resolved_frames": history.timing.resolved_frames,
            "target_frames": list(history.timing.target_frames),
            "target_seconds": list(history.timing.target_seconds),
        },
    }
    (args.output / "probe.json").write_text(json.dumps(report, indent=2) + "\n")
    torch.save(out["preds_vision"][0].cpu(), args.output / "velocity.pt")
    print(json.dumps({k: v for k, v in report.items() if k != "position_ids"}, indent=2))


if __name__ == "__main__":
    main()
