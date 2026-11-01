"""Regression gates for the independent M3 review findings."""

import json
import os
import subprocess
import sys
from dataclasses import replace

import pytest
import torch

from cosmos_framework.inference.physword import generate_physical_words, repack_physical_words
from cosmos_framework.inference.physword_diagnostics import capture_und, compare_ar_repack
from cosmos_framework.model.generator.physword.checkpoint import load_checkpoint, save_checkpoint, training_protocol
from cosmos_framework.model.generator.physword.preprocessing import PreprocessingConfig
from cosmos_framework.model.generator.utils.data_and_condition import GenerationDataClean
from cosmos_framework.scripts.physword_m3_probe import SYNTHETIC_REVISIONS, synthetic_training
from cosmos_framework.scripts.physword_probe import SPECIAL_TOKENS


def test_independent_process_abc_resume(tmp_path):
    output = tmp_path / "abc"
    for mode, seed in (("save", "1"), ("resume", "2")):
        result = subprocess.run(
            [sys.executable, "-m", "cosmos_framework.scripts.physword_resume_probe", mode, "--output", str(output)],
            env={**os.environ, "PYTHONHASHSEED": seed},
            capture_output=True,
            text=True,
            timeout=120,
        )
        assert result.returncode == 0, result.stdout + result.stderr
    report = json.loads((output / "result.json").read_text())
    assert report["status"] == "PASS"


@pytest.mark.parametrize(
    "field,value",
    [
        ("input_size", (96, 96)),
        ("output_size", (96, 96)),
        ("interpolation", "bicubic"),
        ("align_corners", True),
        ("antialias", True),
        ("quantization", "floor"),
        ("pixel_scale", 255.0),
        ("pixel_offset", 0.0),
        ("pixel_device", "cuda"),
        ("latent_mean", (0.1,) * 48),
        ("latent_std", (2.0,) * 48),
    ],
)
def test_resume_rejects_actual_preprocessing_drift(tmp_path, field, value):
    net, proc, _, cfg, opt, _, step, _ = synthetic_training("A")
    protocol = training_protocol(step, {"split": "frozen"})
    save_checkpoint(
        tmp_path / "bundle",
        net,
        proc,
        opt,
        cfg,
        step.config,
        step=0,
        revisions=SYNTHETIC_REVISIONS,
        experiment=protocol,
    )
    step.preprocessing = replace(step.preprocessing, **{field: value})
    changed = training_protocol(step, {"split": "frozen"})
    with pytest.raises(ValueError, match="experiment"):
        load_checkpoint(
            tmp_path / "bundle", net, opt, cfg, step.config, expected_revisions=SYNTHETIC_REVISIONS, experiment=changed
        )


def test_preprocessing_executes_declared_normalization_and_rejects_external_resize():
    cfg = PreprocessingConfig((64, 64), (32, 32), latent_mean=(1.0,) * 48, latent_std=(2.0,) * 48)
    frames = torch.arange(5 * 3 * 64 * 64).remainder(256).byte().reshape(5, 3, 64, 64)
    pixels = cfg.rgb(frames)
    assert pixels.shape == (5, 3, 32, 32) and pixels.dtype == torch.uint8
    with pytest.raises(ValueError, match="input size"):
        cfg.rgb(pixels)
    torch.testing.assert_close(cfg.video_pixels(pixels), pixels.permute(1, 0, 2, 3)[None].float() / 127.5 - 1)
    torch.testing.assert_close(cfg.normalize_latents(torch.full((1, 48, 2, 4, 4), 3.0)), torch.ones(1, 48, 2, 4, 4))


def test_scheduler_ignores_only_internal_order_and_preserves_real_parameters():
    _, _, _, _, _, _, step, _ = synthetic_training("A")
    a = training_protocol(step, {})
    step.flow.noise_scheduler.register_to_config(_use_default_values=["b", "a"])
    assert training_protocol(step, {}) == a
    step.flow.noise_scheduler.register_to_config(shift=7.0)
    assert training_protocol(step, {}) != a


@pytest.mark.parametrize("modality", ["image", "video"])
def test_resume_rejects_visual_processor_normalization_drift(tmp_path, modality):
    net, proc, _, cfg, opt, _, step, _ = synthetic_training("A")
    protocol = training_protocol(step, {"split": "frozen"})
    save_checkpoint(
        tmp_path / "bundle",
        net,
        proc,
        opt,
        cfg,
        step.config,
        step=0,
        revisions=SYNTHETIC_REVISIONS,
        experiment=protocol,
    )
    getattr(proc, f"{modality}_processor").image_mean = [0.1] * 3
    changed = training_protocol(step, {"split": "frozen"})
    with pytest.raises(ValueError, match="experiment"):
        load_checkpoint(
            tmp_path / "bundle",
            net,
            opt,
            cfg,
            step.config,
            expected_revisions=SYNTHETIC_REVISIONS,
            experiment=changed,
        )


@pytest.mark.parametrize("mode", ["cached", "recompute"])
@torch.no_grad()
def test_t06_both_modes_and_explicit_no_cache_recompute(mode):
    net, _, _, _, _, batch, step, kwargs = synthetic_training()
    net.physword.config.decode_mode = mode
    net.eval()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        from cosmos_framework.model.generator.physword.multimodal import build_reasoner_prefix

        prefix = build_reasoner_prefix(net, step.processor, batch.history, batch.instruction)
        calls = []
        original = net.language_model.model.reasoner_forward

        def forward(*args, **kw):
            calls.append((kw["inputs_embeds"].shape[1], kw["cache"]))
            return original(*args, **kw)

        net.language_model.model.reasoner_forward = forward
        try:
            with capture_und(net) as ar:
                prediction = generate_physical_words(net, prefix, capture_cache=True)
        finally:
            net.language_model.model.reasoner_forward = original
        if mode == "recompute":
            assert [x[0] for x in calls] == list(range(prefix.context_length + 1, prefix.context_length + 6))
            assert all(cache is None for _, cache in calls[:-1])
        data = GenerationDataClean(
            batch_size=1,
            is_image_batch=False,
            x0_tokens_vision=[kwargs["noise"].cuda()],
            fps_vision=torch.tensor([10.0]),
        )
        pack = repack_physical_words(net, prediction.condition, data, kwargs["timesteps"].cpu(), SPECIAL_TOKENS)
        pack.to_cuda()
        with capture_und(net) as joint:
            out = net(pack)
        compare_ar_repack(net, prediction, ar, joint, out)
