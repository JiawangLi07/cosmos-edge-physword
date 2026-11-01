# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Synthetic native tiny training acceptance, never evidence of real-data learning."""

import json
from dataclasses import replace
from pathlib import Path

import pytest
import torch

from cosmos_framework.inference.physword import generate_physical_words, repack_physical_words
from cosmos_framework.inference.physword_diagnostics import capture_und, compare_ar_repack, mask_physical_word_suffix
from cosmos_framework.model.generator.physword.checkpoint import (
    load_base_state,
    load_checkpoint,
    pw_mapping,
    save_checkpoint,
)
from cosmos_framework.model.generator.physword.losses import encode_future_targets, latent_cosine, word_ce
from cosmos_framework.model.generator.physword.multimodal import build_reasoner_prefix
from cosmos_framework.model.generator.physword.optimization import (
    PhaseConfig,
    build_phase_optimizer,
    optimizer_audit,
    snapshot_parameters,
)
from cosmos_framework.model.generator.physword.training import forward_teacher_words
from cosmos_framework.scripts.physword_m2_probe import history_video_latents, tiny_multimodal_fixture, tiny_vae
from cosmos_framework.scripts.physword_m3_probe import SYNTHETIC_REVISIONS, synthetic_training
from cosmos_framework.scripts.physword_probe import SPECIAL_TOKENS


@pytest.fixture(scope="module")
def native():
    if not torch.cuda.is_available():
        pytest.skip("Native attention requires CUDA")
    net, processor, history = tiny_multimodal_fixture()
    vae = tiny_vae()
    prefix = build_reasoner_prefix(net, processor, history, "Pick the object and move it to the target.")
    return net, processor, history, vae, prefix


@torch.no_grad()
def test_t08_full_suffix_mask_and_relative_intervention(native):
    net, _, history, vae, prefix = native
    data = history_video_latents(vae, history)
    prediction = generate_physical_words(net, prefix)
    labels = prediction.class_ids.tolist()
    labels[1] = (labels[1] + 1) % 16
    alternative = prefix.with_words(tuple(labels), net.physword.config.token_id_base)

    def pack(condition):
        p = repack_physical_words(net, condition, data, torch.tensor([500.0]), SPECIAL_TOKENS)
        p.to_cuda()
        return p

    a, b = pack(prediction.condition), pack(alternative)
    va = net(a)["preds_vision"][0][:, :, 2:].float()
    repeat = net(a)["preds_vision"][0][:, :, 2:].float()
    vb = net(b)["preds_vision"][0][:, :, 2:].float()
    with mask_physical_word_suffix(net, a):
        masked_a = net(a)["preds_vision"][0][:, :, 2:].float()
        masked_repeat = net(a)["preds_vision"][0][:, :, 2:].float()
        masked_b = net(b)["preds_vision"][0][:, :, 2:].float()

    def rms(x):
        return x.square().mean().sqrt().item()

    metrics = {
        "visible_velocity_rms": rms(va),
        "intervention_rms": rms(va - vb),
        "relative_rms": rms(va - vb) / rms(va),
        "repeat_floor_rms": rms(va - repeat),
        "masked_intervention_rms": rms(masked_a - masked_b),
        "masked_repeat_floor_rms": rms(masked_a - masked_repeat),
        "visible_masked_rms": rms(va - masked_a),
        "masked_suffix_und_indexes": list(range(prefix.context_length, a.text_ids.numel())),
    }
    assert metrics["intervention_rms"] > max(10 * metrics["repeat_floor_rms"], 1e-6)
    assert metrics["visible_masked_rms"] > 0
    torch.testing.assert_close(masked_a, masked_b, atol=0, rtol=0)
    Path("/tmp/physword-m3-visibility.json").write_text(json.dumps(metrics, indent=2) + "\n")


def test_t09_independent_loss_gradients():
    net, _, vae, cfg, optimizer, batch, step, kwargs = synthetic_training()
    result = step(batch, **kwargs)
    assert result["teacher"]["hidden"].shape == (1, 4, 128)
    assert result["teacher"]["logits"].shape == (1, 4, 16)
    assert not result["targets"].requires_grad
    assert not result["prediction"].prediction_hidden_states.requires_grad
    assert result["packed"].reasoner_condition.word_class_ids == tuple(result["prediction"].class_ids.tolist())
    assert result["packed"].reasoner_condition.visual_features is result["prefix"].visual_features
    reports = {}
    for loss_name in ("video", "word", "latent"):
        net.zero_grad(set_to_none=True)
        result["losses"][loss_name].backward(retain_graph=True)
        groups = optimizer_audit(net, optimizer)
        reports[loss_name] = {"loss": float(result["losses"][loss_name].detach()), "groups": groups}
        named = dict(net.named_parameters())

        def norm(name):
            return 0.0 if named[name].grad is None else named[name].grad.float().norm().item()

        assert norm("physword.embedding.weight") > 0
        assert norm("language_model.model.embed_tokens.weight") == 0
        assert all(p.grad is None for p in vae.parameters())
        assert all(p.grad is None for p in net.language_model.visual.parameters())
        if loss_name == "video":
            assert norm("language_model.model.layers.1.self_attn.k_proj.weight") > 0
            assert norm("language_model.model.layers.1.self_attn.v_proj.weight") > 0
            assert norm("language_model.model.layers.1.self_attn.k_norm_und_for_gen.weight") > 0
            assert norm("llm2vae.weight") > 0
            assert norm("physword.word_head.weight") == 0
            assert norm("language_model.model.norm.weight") == 0
            assert norm("language_model.model.layers.1.mlp.up_proj.weight") == 0
        else:
            assert norm("llm2vae.weight") == 0
            assert norm("language_model.model.layers.1.self_attn.k_norm_und_for_gen.weight") == 0
            assert norm("language_model.model.layers.1.mlp.up_proj.weight") > 0
            assert norm("language_model.model.norm.weight") > 0
            assert (norm("physword.word_head.weight") > 0) == (loss_name == "word")
    reports["losses"] = {k: float(v.detach()) for k, v in result["losses"].items()}
    reports["counts"] = result["counts"]
    Path("/tmp/physword-m3-gradients.json").write_text(json.dumps(reports, indent=2) + "\n")


@pytest.mark.parametrize("phase", ["A", "B"])
def test_t10_actual_optimizer_update_and_frozen_parameters(phase):
    net, _, vae, cfg, optimizer, batch, step, kwargs = synthetic_training(phase)
    before = snapshot_parameters(net)
    vae_before = snapshot_parameters(vae)
    optimizer.zero_grad(set_to_none=True)
    result = step(batch, **kwargs)
    result["losses"]["total"].backward()
    assert torch.isfinite(result["losses"]["total"])
    optimizer.step()
    groups = optimizer_audit(net, optimizer, before)
    assert all(any(p["changed"] for p in g["parameters"]) for g in groups)
    assert all(torch.equal(p.detach().cpu(), vae_before[n]) for n, p in vae.named_parameters())
    if phase == "A":
        assert result["prediction"] is None and result["velocity"] is None
        assert float(result["losses"]["video"].detach()) == 0
        assert {g["name"] for g in groups} == {"pw_embedding", "word_head", "und"}
    else:
        assert {g["name"] for g in groups} == {"pw_embedding", "word_head", "und", "gen", "gen_und_k_norm"}
    Path(f"/tmp/physword-m3-step-{phase}.json").write_text(
        json.dumps(
            {
                "losses": {k: float(v.detach()) for k, v in result["losses"].items()},
                "counts": result["counts"],
                "groups": groups,
                "frozen_unchanged": True,
                "vae_unchanged": True,
            },
            indent=2,
        )
        + "\n"
    )


def test_t14_checkpoint_tokenizer_optimizer_roundtrip(tmp_path):
    net, processor, _, cfg, optimizer, batch, step, kwargs = synthetic_training()
    result = step(batch, **kwargs)
    result["losses"]["total"].backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    expected = step(batch, **kwargs)
    save_checkpoint(
        tmp_path / "checkpoint", net, processor, optimizer, cfg, step.config, step=1, revisions=SYNTHETIC_REVISIONS
    )
    cpu_random, cuda_random = torch.rand(4), torch.rand(4, device="cuda")
    restored, _, _, cfg2, opt2, batch2, step2, kwargs2 = synthetic_training()
    proc2, manifest = load_checkpoint(
        tmp_path / "checkpoint", restored, opt2, cfg2, step2.config, expected_revisions=SYNTHETIC_REVISIONS
    )
    torch.testing.assert_close(torch.rand(4), cpu_random, atol=0, rtol=0)
    torch.testing.assert_close(torch.rand(4, device="cuda"), cuda_random, atol=0, rtol=0)
    assert manifest["step"] == 1 and manifest["mapping"] == pw_mapping(net) == pw_mapping(restored)
    assert len(manifest["mapping"]) == 17
    assert proc2.tokenizer.get_vocab() == processor.tokenizer.get_vocab()
    assert proc2.chat_template == processor.chat_template
    step2.processor = proc2
    actual = step2(batch2, **kwargs2)
    for key in ("hidden", "logits", "input_ids", "position_ids"):
        torch.testing.assert_close(actual["teacher"][key], expected["teacher"][key], atol=0, rtol=0)
    torch.testing.assert_close(actual["prediction"].token_ids, expected["prediction"].token_ids, atol=0, rtol=0)
    torch.testing.assert_close(actual["velocity"][0], expected["velocity"][0], atol=0, rtol=0)
    for key in expected["losses"]:
        torch.testing.assert_close(actual["losses"][key], expected["losses"][key], atol=0, rtol=0)
    for group, group2 in zip(optimizer.param_groups, opt2.param_groups, strict=True):
        for p, q in zip(group["params"], group2["params"], strict=True):
            for key in optimizer.state[p]:
                torch.testing.assert_close(optimizer.state[p][key], opt2.state[q][key], atol=0, rtol=0)
    expected["losses"]["total"].backward()
    actual["losses"]["total"].backward()
    optimizer.step()
    opt2.step()
    for name, value in net.state_dict().items():
        torch.testing.assert_close(value, restored.state_dict()[name], atol=0, rtol=0)
    Path("/tmp/physword-m3-roundtrip.json").write_text(
        json.dumps(
            {
                "mapping_count": 17,
                "word_logits_max_abs": 0.0,
                "predicted_ids_equal": True,
                "velocity_max_abs": 0.0,
                "optimizer_moments_equal": True,
                "next_update_equal": True,
                "rng_equal": True,
            }
        )
        + "\n"
    )


@pytest.mark.parametrize("corrupt", ["mapping", "tokenizer", "teacher"])
def test_t14_refuses_corrupt_or_incompatible_bundle(tmp_path, corrupt):
    net, processor, _, cfg, optimizer, _, step, _ = synthetic_training()
    path = tmp_path / "checkpoint"
    save_checkpoint(path, net, processor, optimizer, cfg, step.config, step=0, revisions=SYNTHETIC_REVISIONS)
    revisions = SYNTHETIC_REVISIONS
    if corrupt == "mapping":
        data = json.loads((path / "manifest.json").read_text())
        data["mapping"]["<pw_start>"] += 1
        (path / "manifest.json").write_text(json.dumps(data))
    elif corrupt == "tokenizer":
        with (path / "tokenizer.json").open("a") as f:
            f.write(" ")
    else:
        revisions = {**revisions, "vision": "wrong teacher"}
    with pytest.raises(ValueError):
        load_checkpoint(path, net, optimizer, cfg, step.config, expected_revisions=revisions)


def test_base_warm_start_only_allows_missing_pw_keys():
    net, _, _, _, _, _, _, _ = synthetic_training()
    state = {
        k: v for k, v in net.state_dict().items() if k not in {"physword.embedding.weight", "physword.word_head.weight"}
    }
    assert set(load_base_state(net, state).missing_keys) == {"physword.embedding.weight", "physword.word_head.weight"}
    bad = dict(state)
    del bad["llm2vae.weight"]
    with pytest.raises(ValueError):
        load_base_state(net, bad)
    bad = dict(state, **{"llm2vae.weight": torch.zeros(1)})
    with pytest.raises(ValueError):
        load_base_state(net, bad)


def test_invalid_targets_and_zero_valid_losses():
    logits = torch.randn(1, 4, 16, requires_grad=True)
    labels = torch.tensor([[2, -100, 1, -100]])
    mask = torch.tensor([[True, False, True, False]])
    loss, count = word_ce(logits, labels, mask)
    assert count == 2
    torch.testing.assert_close(
        loss, torch.nn.functional.cross_entropy(logits[:, [0, 2]].reshape(2, 16), labels[:, [0, 2]].flatten())
    )
    loss.backward()
    assert logits.grad[:, [1, 3]].abs().sum() == 0
    z = torch.randn(1, 4, 128, requires_grad=True)
    targets = torch.randn_like(z)
    targets[:, 1] = float("nan")
    targets[:, 2] = 0
    cosine, count = latent_cosine(z, targets, torch.ones(1, 4, dtype=torch.bool))
    assert count == 2 and cosine.dtype == torch.float32
    cosine.backward()
    assert torch.isfinite(z.grad).all() and z.grad[:, [1, 2]].abs().sum() == 0
    ce0, c0 = word_ce(logits, torch.full_like(labels, -100), torch.zeros_like(mask))
    z0, n0 = latent_cosine(z, targets, torch.zeros_like(mask))
    (ce0 + z0).backward()
    assert ce0 == 0 and z0 == 0 and c0 == n0 == 0


@torch.no_grad()
def test_t03_teacher_targets_do_not_enter_rollout_or_joint():
    net, _, _, _, _, batch, step, kwargs = synthetic_training()
    a = step(batch, **kwargs)
    changed = replace(batch, word_labels=torch.tensor([[9, 12, 2, 8]]), future_rgb=255 - batch.future_rgb)
    b = step(changed, **kwargs)
    torch.testing.assert_close(a["prediction"].token_ids, b["prediction"].token_ids, atol=0, rtol=0)
    torch.testing.assert_close(a["packed"].text_ids, b["packed"].text_ids, atol=0, rtol=0)
    torch.testing.assert_close(a["prefix"].visual_features, b["prefix"].visual_features, atol=0, rtol=0)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        target = encode_future_targets(net, step.processor, batch.future_rgb, batch.valid_latent)
    torch.testing.assert_close(a["targets"], target, atol=0, rtol=0)
    c = step(replace(batch, word_labels=changed.word_labels), **kwargs)
    torch.testing.assert_close(a["velocity"][0], c["velocity"][0], atol=0, rtol=0)
    for block in range(4):
        labels = batch.word_labels.clone()
        labels[0, block] = (int(labels[0, block]) + 5) % 16
        with torch.autocast("cuda", dtype=torch.bfloat16):
            teacher = forward_teacher_words(net, a["prefix"], labels, batch.valid_word)
        torch.testing.assert_close(
            a["teacher"]["logits"][:, : block + 1], teacher["logits"][:, : block + 1], atol=0, rtol=0
        )


def test_phase_selection_never_uses_substring_or_stale_groups():
    net, _, _, cfg, opt, _, step, _ = synthetic_training("A")
    named = dict(net.named_parameters())
    assert named["language_model.model.layers.1.self_attn.q_proj.weight"].requires_grad
    assert not named["language_model.model.layers.1.self_attn.q_proj_moe_gen.weight"].requires_grad
    b0 = PhaseConfig(phase="B", reasoner_layers=(), train_pw_embeddings=False, train_word_head=False)
    opt = build_phase_optimizer(net, b0)
    optimizer_audit(net, opt)
    assert not named["language_model.model.layers.1.self_attn.q_proj.weight"].requires_grad
    assert named["language_model.model.layers.1.self_attn.q_proj_moe_gen.weight"].requires_grad
    with pytest.raises(ValueError, match="Phase changed"):
        step(None)
    with pytest.raises(ValueError):
        build_phase_optimizer(net, replace(cfg, reasoner_layers=(99,)))


@torch.no_grad()
def test_future_targets_are_independent_native_spatial_means(monkeypatch):
    net, processor, _, _, _, batch, step, _ = synthetic_training("A")
    vision = net.language_model.visual
    native = vision.get_image_features
    collected = []

    def capture(pixels, grid):
        assert grid.shape == (1, 3) and int(grid[0, 0]) == 1
        result = native(pixels, grid)
        collected.append(result.float().mean(0))
        return result

    monkeypatch.setattr(vision, "get_image_features", capture)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        targets = encode_future_targets(net, processor, batch.future_rgb, batch.valid_latent)
    assert len(collected) == 4
    torch.testing.assert_close(targets[0], torch.stack(collected), atol=0, rtol=0)
    altered = batch.future_rgb.clone()
    altered[0] = 255 - altered[0]  # Original frame 5 is not a target frame.
    with torch.autocast("cuda", dtype=torch.bfloat16):
        same = encode_future_targets(net, processor, altered, batch.valid_latent)
    torch.testing.assert_close(targets, same, atol=0, rtol=0)
    collected.clear()
    mask = torch.tensor([[True, False, False, True]])
    with torch.autocast("cuda", dtype=torch.bfloat16):
        partial = encode_future_targets(net, processor, batch.future_rgb, mask)
    assert len(collected) == 2 and partial[:, [1, 2]].abs().sum() == 0
    assert all(not p.requires_grad for p in vision.parameters())


@torch.no_grad()
def test_video_native_reduction_scale_and_clean_history():
    net, _, _, _, _, batch, step, kwargs = synthetic_training()
    out = step(batch, **kwargs)
    observed = batch.history.frames.permute(1, 0, 2, 3).unsqueeze(0).float() / 127.5 - 1
    future = batch.future_rgb.permute(1, 0, 2, 3).unsqueeze(0).float() / 127.5 - 1
    with torch.autocast("cuda", dtype=torch.bfloat16):
        full = step.encode_video(torch.cat((observed, future), 2)).float()
        history = step.encode_video(observed).float()
    target = kwargs["noise"].to(full) - full
    pred = out["velocity"][0].float()
    # Existing Edge reduction includes clean frames in denominator; they contribute zero error.
    manual = (pred[:, :, 2:] - target[:, :, 2:]).square().sum() / pred.numel()
    torch.testing.assert_close(manual, out["losses"]["video_raw"], atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(out["losses"]["video"], manual * 10, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(out["packed"].vision.tokens[0][:, :, :2].float(), history, atol=0, rtol=0)
    sigma = step.flow.get_sigmas(kwargs["timesteps"].cuda(), {"device": "cuda", "dtype": torch.float32})
    expected = sigma * kwargs["noise"].to(full) + (1 - sigma) * full
    torch.testing.assert_close(out["packed"].vision.tokens[0][:, :, 2:].float(), expected[:, :, 2:], atol=0, rtol=0)


def test_frozen_und_still_propagates_pw_embedding_gradients():
    net, processor, vae, cfg, _, batch, step, kwargs = synthetic_training("A")
    cfg = replace(cfg, reasoner_layers=())
    optimizer = build_phase_optimizer(net, cfg)
    step = type(step)(net, processor, vae, step.encode_video, step.flow, SPECIAL_TOKENS, "A", step.config)
    out = step(batch, **kwargs)
    out["losses"]["total"].backward()
    assert net.physword.embedding.weight.grad.norm() > 0
    assert all(p.grad is None for p in net.language_model.model.parameters())
    assert {g["name"] for g in optimizer.param_groups} == {"pw_embedding", "word_head"}


def test_all_invalid_phase_a_uses_neutral_feedback_and_zero_losses():
    _, _, _, _, _, batch, step, kwargs = synthetic_training("A")
    batch = replace(
        batch,
        word_labels=torch.full_like(batch.word_labels, -100),
        valid_word=torch.zeros_like(batch.valid_word),
        valid_latent=torch.zeros_like(batch.valid_latent),
    )
    out = step(batch, **kwargs)
    assert out["counts"]["word"] == out["counts"]["latent"] == 0
    assert out["counts"]["word_samples"] == out["counts"]["latent_samples"] == 0
    assert out["losses"]["total"].detach() == 0
    assert out["teacher"]["input_ids"][-3:].tolist() == [step.net.physword.config.token_id_base] * 3
    out["losses"]["total"].backward()  # Empty supervision is a finite, differentiable zero.


def test_t06_fp32_master_bf16_autocast_ar_joint_parity():
    net, processor, vae, _, _, batch, _, _ = synthetic_training()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        prefix = build_reasoner_prefix(net, processor, batch.history, batch.instruction)
        with capture_und(net) as ar:
            prediction = generate_physical_words(net, prefix, capture_cache=True)
    data = history_video_latents(vae, batch.history)
    packed = repack_physical_words(net, prediction.condition, data, torch.tensor([500.0]), SPECIAL_TOKENS)
    packed.to_cuda()
    net.train()
    net.language_model.visual.eval()
    with torch.autocast("cuda", dtype=torch.bfloat16), capture_und(net) as joint:
        out = net(packed)  # grad-enabled varlen path versus no-grad AR
    metrics = compare_ar_repack(net, prediction, ar, joint, out)
    Path("/tmp/physword-m3-amp-parity.json").write_text(json.dumps(metrics, indent=2) + "\n")
