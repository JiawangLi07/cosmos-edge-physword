"""Pre-M4 fixes: semantic rows, dynamic vocab, matched baseline and full resume."""

import json
import random
from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from cosmos_framework.data.generator.physword.sampling import StatefulWindowSampler
from cosmos_framework.inference.physword_baselines import pack_video_baseline
from cosmos_framework.model.generator.physword.checkpoint import load_checkpoint, save_checkpoint, training_protocol
from cosmos_framework.model.generator.physword.config import PhysWordConfig
from cosmos_framework.model.generator.physword.initialization import initialize_physical_words
from cosmos_framework.model.generator.physword.optimization import PhaseConfig, build_phase_optimizer
from cosmos_framework.model.generator.physword.training import PhysWordBatch, PhysWordTrainingStep
from cosmos_framework.scripts.physword_m2_probe import history_video_latents, tiny_multimodal_fixture, tiny_vae
from cosmos_framework.scripts.physword_m3_probe import SYNTHETIC_REVISIONS, synthetic_training
from cosmos_framework.scripts.physword_probe import SPECIAL_TOKENS, make_network


@pytest.mark.parametrize("tied", [False, True])
def test_semantic_input_head_mean_pooling_actual_tying_and_order(tied):
    config = PhysWordConfig(enabled=True, vocabulary=("approach", "grasp"))
    net = make_network(physword=asdict(config), device="cpu")
    if tied:
        net.language_model.lm_head.weight = net.language_model.model.embed_tokens.weight
    tokenizer = SimpleNamespace(
        unk_token_id=2,
        get_vocab=lambda: {"<|im_start|>": 10},
        encode=lambda word, **_: {"approach": [3, 5], "grasp": [7]}[word],
    )
    expected_input = net.language_model.model.embed_tokens.weight[[3, 5]].float().mean(0).clone()
    expected_head = net.language_model.lm_head.weight[[3, 5]].float().mean(0).clone()
    result = initialize_physical_words(net, tokenizer, source={"kind": "SYNTHETIC fixture weights"})
    assert result["actual_weight_tying"] is tied
    torch.testing.assert_close(net.physword.embedding.weight[1], expected_input, atol=0, rtol=0)
    torch.testing.assert_close(net.physword.word_head.weight[0], expected_head, atol=0, rtol=0)
    assert net.physword.embedding.weight.shape == (3, 128) and net.physword.word_head.weight.shape == (2, 128)
    with pytest.raises(ValueError):
        initialize_physical_words(net, tokenizer, source={"kind": "SYNTHETIC"})
    with pytest.raises(RuntimeError):
        net.init_weights(buffer_device=torch.device("cpu"))


def test_dynamic_vocabulary_abc_and_masked_feedback():
    config = PhysWordConfig(enabled=True, vocabulary=("align", "grasp", "hold"), effector="left")
    net, processor, history = tiny_multimodal_fixture(pw_config=asdict(config))
    net.float()
    vae = tiny_vae()
    cfg = PhaseConfig(phase="B", reasoner_layers=(0, 1))
    opt = build_phase_optimizer(net, cfg)
    batch = PhysWordBatch(
        history,
        "Pick the object.",
        torch.zeros((16, 3, 64, 64), dtype=torch.uint8),
        torch.tensor([[0, -100, 2, 1]]),
        torch.tensor([[True, False, True, True]]),
        torch.ones(1, 4, dtype=torch.bool),
    )
    from cosmos_framework.model.generator.diffusion.rectified_flow import RectifiedFlow

    flow = RectifiedFlow(None, train_time_distribution="waver", device="cuda")

    def encode(x):
        return vae.encode(x.cuda(), (torch.zeros(48, device="cuda"), torch.ones(48, device="cuda")))

    step = PhysWordTrainingStep(net, processor, vae, encode, flow, SPECIAL_TOKENS, "B")
    out = step(batch, timesteps=torch.tensor([374.6]))
    assert out["teacher"]["logits"].shape == (1, 4, 3)
    assert out["counts"]["word"] == 3 and out["counts"]["latent"] == 4
    assert int(out["teacher"]["input_ids"][out["prefix"].context_length + 2]) == config.token_id_base
    assert int(out["prediction"].class_ids.max()) < 3
    out["losses"]["total"].backward()
    before = net.physword.embedding.weight.detach().clone()
    opt.step()
    assert not torch.equal(before, net.physword.embedding.weight)
    assert all(not p.requires_grad and p.grad is None for p in net.language_model.visual.parameters())
    assert all(not p.requires_grad and p.grad is None for p in vae.parameters())
    bad = replace(batch, word_labels=torch.tensor([[3, -100, 2, 1]]))
    with pytest.raises(ValueError, match="configured"):
        step(bad)


@torch.no_grad()
def test_native_matched_full_baselines_share_history_and_have_no_pw_in_matched():
    net, processor, history = tiny_multimodal_fixture()
    vae = tiny_vae()
    data = history_video_latents(vae, history)
    packs = {}
    for mode in ("native", "matched_rgb", "full_pw"):
        packs[mode] = pack_video_baseline(
            net, processor, history, "Pick the object.", data, torch.tensor([374.6]), SPECIAL_TOKENS, mode=mode
        )
        packs[mode].to_cuda()
        assert torch.isfinite(net(packs[mode])["preds_vision"][0]).all()
    a, b = packs["matched_rgb"].reasoner_condition, packs["full_pw"].reasoner_condition
    assert torch.equal(a.input_ids[: a.context_length], b.input_ids[: b.context_length])
    assert torch.equal(a.position_ids[:, : a.context_length], b.position_ids[:, : b.context_length])
    assert torch.equal(a.visual_features, b.visual_features)
    assert not a.word_class_ids and a.rgb_prefix_only
    assert not bool((a.input_ids >= net.physword.config.token_id_base).any())
    for p in packs.values():
        torch.testing.assert_close(p.vision.tokens[0], data.x0_tokens_vision[0], atol=0, rtol=0)


def test_complete_interrupted_resume_rng_scheduler_ema_scaler_and_data(tmp_path):
    net, processor, vae, cfg, opt, batch, step, _ = synthetic_training()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=8)
    ema = torch.optim.swa_utils.AveragedModel(
        net, device="cpu", multi_avg_fn=torch.optim.swa_utils.get_ema_multi_avg_fn(0.9)
    )
    scaler = torch.amp.GradScaler("cuda", init_scale=128)
    sampler = StatefulWindowSampler(3, "frozen SYNTHETIC manifest", seed=7)
    random.seed(3)
    np.random.seed(4)

    def update(network, optimizer, train_step, lr, avg, scale, data_sampler):
        index = data_sampler.next_index()
        # Stresses Python/NumPy/Torch CPU/CUDA RNG, RF time and fresh noise.
        py, np_value = random.random(), float(np.random.random())
        cpu, gpu = torch.rand(1), torch.rand(1, device="cuda")
        optimizer.zero_grad(set_to_none=True)
        out = train_step(batch)
        scale.scale(out["losses"]["total"]).backward()
        scale.step(optimizer)
        scale.update()
        lr.step()
        avg.update_parameters(network)
        return {k: v.detach().clone() for k, v in out["losses"].items()}, (index, py, np_value, cpu, gpu)

    update(net, opt, step, scheduler, ema, scaler, sampler)
    protocol = training_protocol(step, {"data": "SYNTHETIC", "split": "frozen"})
    save_checkpoint(
        tmp_path / "bundle",
        net,
        processor,
        opt,
        cfg,
        step.config,
        step=1,
        revisions=SYNTHETIC_REVISIONS,
        scheduler=scheduler,
        ema=ema,
        scaler=scaler,
        experiment=protocol,
        vae=vae,
        training_state={"global_step": 1, "accumulation_step": 0, "sampler": sampler.state_dict()},
    )
    expected_loss, expected_random = update(net, opt, step, scheduler, ema, scaler, sampler)
    restored, _, vae2, cfg2, opt2, _, step2, _ = synthetic_training()
    scheduler2 = torch.optim.lr_scheduler.CosineAnnealingLR(opt2, T_max=8)
    ema2 = torch.optim.swa_utils.AveragedModel(
        restored, device="cpu", multi_avg_fn=torch.optim.swa_utils.get_ema_multi_avg_fn(0.9)
    )
    scaler2 = torch.amp.GradScaler("cuda", init_scale=128)
    proc2, manifest = load_checkpoint(
        tmp_path / "bundle",
        restored,
        opt2,
        cfg2,
        step2.config,
        expected_revisions=SYNTHETIC_REVISIONS,
        scheduler=scheduler2,
        ema=ema2,
        scaler=scaler2,
        experiment=protocol,
        vae=vae2,
    )
    step2.processor = proc2
    sampler2 = StatefulWindowSampler(3, "frozen SYNTHETIC manifest", seed=999)
    sampler2.load_state_dict(manifest["training_state"]["sampler"])
    actual_loss, actual_random = update(restored, opt2, step2, scheduler2, ema2, scaler2, sampler2)
    for k in expected_loss:
        torch.testing.assert_close(actual_loss[k], expected_loss[k], atol=0, rtol=0)
    assert expected_random[:3] == actual_random[:3]
    for x, y in zip(expected_random[3:], actual_random[3:], strict=True):
        torch.testing.assert_close(x, y, atol=0, rtol=0)
    for name, p in net.state_dict().items():
        torch.testing.assert_close(p, restored.state_dict()[name], atol=0, rtol=0)
    for name, p in ema.state_dict().items():
        torch.testing.assert_close(p, ema2.state_dict()[name], atol=0, rtol=0)
    assert scheduler.state_dict() == scheduler2.state_dict() and scaler.state_dict() == scaler2.state_dict()
    metrics = {
        "loss_max_abs": 0,
        "next_update_max_abs": 0,
        "rng_equal": True,
        "data_order_equal": True,
        "scheduler_equal": True,
        "ema_equal": True,
        "scaler_equal": True,
        "frozen_vae_restored": True,
        "weights": "SYNTHETIC random native tiny",
    }
    Path("/tmp/physword-complete-resume.json").write_text(json.dumps(metrics, indent=2) + "\n")


@pytest.mark.parametrize("drift", ["rf", "data"])
def test_resume_rejects_rf_or_data_protocol_drift(tmp_path, drift):
    net, processor, _, cfg, opt, _, step, _ = synthetic_training()
    protocol = training_protocol(step, {"split_sha256": "A"})
    save_checkpoint(
        tmp_path / "bundle",
        net,
        processor,
        opt,
        cfg,
        step.config,
        step=0,
        revisions=SYNTHETIC_REVISIONS,
        experiment=protocol,
    )
    changed = json.loads(json.dumps(protocol))
    if drift == "rf":
        changed["rf"]["continuous_train_shift"] = 10
    else:
        changed["data"]["split_sha256"] = "B"
    with pytest.raises(ValueError, match="experiment"):
        load_checkpoint(
            tmp_path / "bundle", net, opt, cfg, step.config, expected_revisions=SYNTHETIC_REVISIONS, experiment=changed
        )


def test_mid_accumulation_resume_restores_gradients(tmp_path):
    net, processor, _, cfg, opt, batch, step, kwargs = synthetic_training()
    opt.zero_grad(set_to_none=True)
    (step(batch, **kwargs)["losses"]["total"] / 2).backward()
    save_checkpoint(
        tmp_path / "bundle",
        net,
        processor,
        opt,
        cfg,
        step.config,
        step=0,
        revisions=SYNTHETIC_REVISIONS,
        training_state={"global_step": 0, "accumulation_step": 1, "data_cursor": 1},
    )
    (step(batch, **kwargs)["losses"]["total"] / 2).backward()
    opt.step()
    restored, _, _, cfg2, opt2, batch2, step2, kwargs2 = synthetic_training()
    _, manifest = load_checkpoint(
        tmp_path / "bundle", restored, opt2, cfg2, step2.config, expected_revisions=SYNTHETIC_REVISIONS
    )
    assert manifest["training_state"]["accumulation_step"] == 1
    assert any(p.grad is not None for p in restored.parameters())
    (step2(batch2, **kwargs2)["losses"]["total"] / 2).backward()
    opt2.step()
    for name, p in net.state_dict().items():
        torch.testing.assert_close(p, restored.state_dict()[name], atol=0, rtol=0)
