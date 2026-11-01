# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""M2 native tiny integration checks; no pretrained weights, downloads or training."""

import json
from dataclasses import fields, replace
from pathlib import Path

import pytest
import torch

from cosmos_framework.inference.physword import (
    generate_physical_words,
    predict_physical_word_video_from_rgb,
    repack_physical_words,
)
from cosmos_framework.inference.physword_diagnostics import capture_und, compare_ar_repack
from cosmos_framework.model.generator.physword.multimodal import (
    FutureVisualTargets,
    PWTiming,
    ReasonerCondition,
    RGBHistory,
    build_reasoner_prefix,
    prepare_history_video_data,
    replay_physical_words,
)
from cosmos_framework.scripts.physword_m2_probe import history_video_latents, tiny_multimodal_fixture, tiny_vae
from cosmos_framework.scripts.physword_probe import SPECIAL_TOKENS


@pytest.fixture(scope="module")
def fixture():
    if not torch.cuda.is_available():
        pytest.skip("Native Edge MoT requires CUDA")
    net, processor, history = tiny_multimodal_fixture()
    vae = tiny_vae()
    prefix = build_reasoner_prefix(net, processor, history, "Pick the object and move it to the target.")
    data = history_video_latents(vae, history)
    return net, processor, history, vae, prefix, data


def pack(net, condition, data):
    result = repack_physical_words(net, condition, data, torch.tensor([500.0]), SPECIAL_TOKENS)
    result.to_cuda()
    return result


@torch.no_grad()
def test_t06_ar_repack_layerwise_parity_and_final_flush(fixture):
    net, _, _, _, prefix, data = fixture
    with capture_und(net) as ar:
        prediction = generate_physical_words(net, prefix, capture_cache=True)
    packed = pack(net, prediction.condition, data)
    size = prediction.condition.input_ids.numel()
    torch.testing.assert_close(
        packed.position_ids[:, packed.text_indexes[:size]].cpu(),
        prediction.condition.position_ids,
        atol=0,
        rtol=0,
        check_dtype=False,
    )
    assert not torch.equal(prefix.position_ids[0], prefix.position_ids[1])  # real multimodal positions
    assert prediction.cache_length == prefix.context_length + 5
    assert prediction.class_ids.shape == (4,)
    assert ((prediction.class_ids >= 0) & (prediction.class_ids < 16)).all()
    assert torch.equal(prediction.token_positions, prediction.prediction_positions + 1)
    with capture_und(net) as joint:
        out = net(packed)
    metrics = compare_ar_repack(net, prediction, ar, joint, out)
    destination = Path("/tmp/physword-m2-parity.json")
    destination.write_text(json.dumps(metrics, indent=2) + "\n")


@torch.no_grad()
@pytest.mark.parametrize("target", range(4))
def test_t03_shift_current_target_cannot_change_its_prediction(fixture, target):
    net, _, _, _, prefix, _ = fixture
    labels = [0, 1, 2, 3]
    changed = labels.copy()
    changed[target] = 10
    # Teacher replay contains start,w1,w2,w3; w4 is a target only.
    a = prefix.with_words(tuple(labels[:-1]), net.physword.config.token_id_base)
    b = prefix.with_words(tuple(changed[:-1]), net.physword.config.token_id_base)
    z_a, z_b = replay_physical_words(net, a), replay_physical_words(net, b)
    end = prefix.context_length + target + 1
    torch.testing.assert_close(a.input_ids[:end], b.input_ids[:end], atol=0, rtol=0)
    torch.testing.assert_close(z_a[:, :end], z_b[:, :end], atol=0, rtol=0)
    if target < 3:
        assert not torch.equal(z_a[:, end], z_b[:, end])


def test_t11_grad_and_no_grad_condition_velocity_parity(fixture):
    net, _, _, _, prefix, data = fixture
    prediction = generate_physical_words(net, prefix)
    replay = prefix.with_words(tuple(prediction.class_ids.tolist()), net.physword.config.token_id_base)
    a, b = pack(net, prediction.condition, data), pack(net, replay, data)
    for name in ("text_ids", "text_indexes", "position_ids"):
        torch.testing.assert_close(getattr(a, name), getattr(b, name), atol=0, rtol=0)
    assert a.sample_lens == b.sample_lens and a.split_lens == b.split_lens
    assert a.reasoner_condition.visual_features is b.reasoner_condition.visual_features
    with torch.no_grad():
        inference = net(a)["preds_vision"][0]
    try:
        net.train()
        net.language_model.visual.eval()
        with torch.enable_grad():
            differentiable = net(b)["preds_vision"][0]
    finally:
        net.eval()
    assert differentiable.requires_grad
    diff = (inference.float() - differentiable.float()).abs().max().item()
    Path("/tmp/physword-m2-t11.json").write_text(json.dumps({"velocity_max_abs": diff}) + "\n")
    torch.testing.assert_close(inference, differentiable.detach(), atol=0.02, rtol=0.02)


@torch.no_grad()
def test_t13_future_targets_cannot_enter_inference_and_rgb_is_used(fixture):
    net, processor, history, _, prefix, data = fixture
    first = FutureVisualTargets(torch.zeros((16, 3, 64, 64), dtype=torch.uint8), torch.zeros(4, dtype=torch.long))
    second = replace(
        first, frames=torch.full_like(first.frames, 255), word_class_ids=torch.full_like(first.word_class_ids, 15)
    )
    assert not torch.equal(first.frames, second.frames)
    rebuilt = build_reasoner_prefix(net, processor, history, "Pick the object and move it to the target.")
    for name in ("input_ids", "position_ids", "visual_indexes", "visual_features"):
        torch.testing.assert_close(getattr(prefix, name), getattr(rebuilt, name), atol=0, rtol=0)
    a, b = generate_physical_words(net, prefix), generate_physical_words(net, rebuilt)
    torch.testing.assert_close(a.token_ids, b.token_ids, atol=0, rtol=0)
    assert not {f.name for f in fields(ReasonerCondition)} & {"future_frames", "future_features", "future_targets"}
    with pytest.raises(TypeError):
        generate_physical_words(net, prefix, future_targets=second)
    assert not prefix.visual_features.requires_grad
    assert all(not p.requires_grad for p in net.language_model.visual.parameters())
    altered = build_reasoner_prefix(
        net, processor, replace(history, frames=255 - history.frames), "Pick the object and move it to the target."
    )
    assert not torch.equal(prefix.visual_features, altered.visual_features)
    # Fix PW IDs and GEN latents: this isolates the UND RGB feature path.
    condition_b = altered.with_words(tuple(a.class_ids.tolist()), net.physword.config.token_id_base)
    va = net(pack(net, a.condition, data))["preds_vision"][0]
    vb = net(pack(net, condition_b, data))["preds_vision"][0]
    assert (va.float() - vb.float()).square().mean() > 0


@torch.no_grad()
@pytest.mark.parametrize("temporal_window", [4, 68])
def test_t13_native_causal_vae_history_prefix(fixture, temporal_window):
    _, _, history, vae, _, _ = fixture
    vae.temporal_window = temporal_window
    history_rgb = history.frames.permute(1, 0, 2, 3).unsqueeze(0).cuda().float() / 127.5 - 1
    future = torch.randn((1, 3, 16, 64, 64), device="cuda", generator=torch.Generator(device="cuda").manual_seed(11))
    scale = (torch.zeros(48, device="cuda"), torch.ones(48, device="cuda"))
    historical = vae.encode(history_rgb, scale)
    full = vae.encode(torch.cat((history_rgb, future), 2), scale)
    changed = vae.encode(torch.cat((history_rgb, -future), 2), scale)
    assert historical.shape[2] == 2 and full.shape[2] == 6
    torch.testing.assert_close(historical, full[:, :, :2], atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(historical, changed[:, :, :2], atol=1e-5, rtol=1e-5)
    assert not torch.equal(full[:, :, 2:], changed[:, :, 2:])
    Path(f"/tmp/physword-m2-vae-{temporal_window}.json").write_text(
        json.dumps(
            {
                "history_full_max_abs": (historical - full[:, :, :2]).abs().max().item(),
                "future_change_history_max_abs": (historical - changed[:, :, :2]).abs().max().item(),
            }
        )
        + "\n"
    )


@torch.no_grad()
def test_history_encoder_sees_only_five_frames_and_noise_is_unconditioned(fixture):
    net, _, history, vae, prefix, _ = fixture
    seen = []

    def encode(pixels):
        seen.append(pixels.shape)
        return vae.encode(pixels.cuda(), (torch.zeros(48, device="cuda"), torch.ones(48, device="cuda")))

    noise = torch.randn((1, 48, 4, 4, 4))
    data = prepare_history_video_data(history, encode, noise)
    assert seen == [torch.Size([1, 3, 5, 64, 64])]
    torch.testing.assert_close(data.x0_tokens_vision[0][:, :, 2:].cpu(), noise, atol=0, rtol=0)
    pred = generate_physical_words(net, prefix)
    packed = pack(net, pred.condition, data)
    assert packed.vision.condition_mask[0].flatten().tolist() == [1, 1, 0, 0, 0, 0]
    assert packed.vision.noisy_frame_indexes[0].tolist() == [2, 3, 4, 5]
    assert packed.reasoner_condition.timing.target_frames == (8, 12, 16, 20)
    assert packed.reasoner_condition.timing.target_seconds == (0.8, 1.2, 1.6, 2.0)
    assert packed.reasoner_condition.timing.resolved_frames == 21


@torch.no_grad()
def test_predicted_word_intervention_and_und_cannot_read_gen(fixture):
    net, _, _, _, prefix, data = fixture
    prediction = generate_physical_words(net, prefix)
    labels = prediction.class_ids.tolist()
    labels[1] = (labels[1] + 1) % 16
    alternate = prefix.with_words(tuple(labels), net.physword.config.token_id_base)
    a, b = pack(net, prediction.condition, data), pack(net, alternate, data)
    torch.testing.assert_close(a.position_ids, b.position_ids, atol=0, rtol=0)
    torch.testing.assert_close(a.vision.tokens[0], b.vision.tokens[0], atol=0, rtol=0)
    out_a, out_b = net(a), net(b)
    delta = out_a["preds_vision"][0][:, :, 2:].float() - out_b["preds_vision"][0][:, :, 2:].float()
    assert delta.square().mean() > 0
    # Perturb every GEN input, including the historical latents. UND is unchanged.
    isolated = pack(net, prediction.condition, data)
    isolated.vision.tokens[0] = isolated.vision.tokens[0] + 3
    out_isolated = net(isolated)
    torch.testing.assert_close(
        out_a["last_hidden_state"][a.text_indexes],
        out_isolated["last_hidden_state"][isolated.text_indexes],
        atol=0,
        rtol=0,
    )
    Path("/tmp/physword-m2-intervention.json").write_text(
        json.dumps(
            {
                "future_velocity_rms": delta.square().mean().sqrt().item(),
                "future_velocity_max_abs": delta.abs().max().item(),
                "und_gen_perturbation_max_abs": 0.0,
            }
        )
        + "\n"
    )


@torch.no_grad()
def test_public_rgb_entry_calls_native_template_and_vision_once(fixture, monkeypatch):
    net, processor, history, vae, _, _ = fixture
    counts = {"template": 0, "vision": 0}
    template = processor.apply_chat_template
    visual = net.language_model.visual.get_image_features

    def count_template(*args, **kwargs):
        counts["template"] += 1
        return template(*args, **kwargs)

    def count_visual(*args, **kwargs):
        counts["vision"] += 1
        return visual(*args, **kwargs)

    monkeypatch.setattr(processor, "apply_chat_template", count_template)
    monkeypatch.setattr(net.language_model.visual, "get_image_features", count_visual)

    def encode(pixels):
        return vae.encode(pixels.cuda(), (torch.zeros(48, device="cuda"), torch.ones(48, device="cuda")))

    prediction, packed, out = predict_physical_word_video_from_rgb(
        net,
        processor,
        history,
        "Pick the object and move it to the target.",
        encode,
        torch.zeros((1, 48, 4, 4, 4)),
        torch.tensor([500.0]),
        SPECIAL_TOKENS,
    )
    assert counts == {"template": 1, "vision": 1}
    assert prediction.token_ids.numel() == 4 and out["preds_vision"][0].shape == (1, 48, 6, 4, 4)
    assert packed.reasoner_condition.visual_features is prediction.condition.visual_features


@torch.no_grad()
def test_instruction_is_part_of_the_shared_prefix(fixture):
    net, processor, history, _, _, _ = fixture
    a = build_reasoner_prefix(net, processor, history, "tok_30")
    b = build_reasoner_prefix(net, processor, history, "tok_31")
    assert not torch.equal(a.input_ids, b.input_ids)
    torch.testing.assert_close(a.visual_features, b.visual_features, atol=0, rtol=0)
    assert not torch.equal(replay_physical_words(net, a)[:, -1], replay_physical_words(net, b)[:, -1])


def test_m2_rejects_unsupported_timing_and_full_clip_inputs():
    with pytest.raises(ValueError):
        RGBHistory(torch.zeros((21, 3, 64, 64), dtype=torch.uint8))
    with pytest.raises(ValueError):
        PWTiming(model_fps=24)
    with pytest.raises(ValueError):
        PWTiming(blocks=5)


@torch.no_grad()
@pytest.mark.parametrize("invalid", ["batch", "views", "fps", "latent_length"])
def test_repack_rejects_unsupported_data(fixture, invalid):
    net, _, _, _, prefix, data = fixture
    condition = generate_physical_words(net, prefix).condition
    if invalid == "batch":
        bad = replace(data, batch_size=2)
    elif invalid == "views":
        bad = replace(data, x0_tokens_vision=data.x0_tokens_vision * 2)
    elif invalid == "fps":
        bad = replace(data, fps_vision=torch.tensor([24.0]))
    else:
        bad = replace(data, x0_tokens_vision=[data.x0_tokens_vision[0][:, :, :5]])
    with pytest.raises(ValueError):
        pack(net, condition, bad)


def test_source_fps_and_absolute_target_timestamps():
    timing = PWTiming(source_fps=30.0, start_seconds=2.0)
    history = RGBHistory(
        torch.zeros((5, 3, 64, 64), dtype=torch.uint8),
        timing,
        source_frame_indexes=(60, 63, 66, 69, 72),
        source_pts_seconds=(2.0, 2.1, 2.2, 2.3, 2.4),
    )
    assert history.timing.target_frames == (8, 12, 16, 20)
    assert history.timing.target_seconds == (2.8, 3.2, 3.6, 4.0)
