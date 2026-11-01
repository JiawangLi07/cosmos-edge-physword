# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""M1 tests use the actual Edge two-path network and unmodified two-way attention."""

from dataclasses import replace

import pytest
import torch
from omegaconf import OmegaConf

from cosmos_framework.data.generator.sequence_packing.packers import pack_input_sequence
from cosmos_framework.data.generator.sequence_packing.runtime import from_all_seq, get_all_seq_unpadded
from cosmos_framework.model.generator.mot.attention import build_packed_sequence, dispatch_attention
from cosmos_framework.model.generator.mot.cosmos3_vfm_network import Cosmos3VFMNetworkConfig
from cosmos_framework.model.generator.physword.conditioning import (
    PW_TOKENS,
    PhysicalWordEmbedding,
    append_manual_words,
    manual_word_ids,
    physical_word_trace,
)
from cosmos_framework.model.generator.physword.config import PhysWordConfig, resolve_physword_config
from cosmos_framework.scripts.physword_probe import (
    SPECIAL_TOKENS,
    WORDS_A,
    WORDS_B,
    make_inputs,
    make_network,
    make_pack,
)


def test_off_preserves_text_objects_and_ignores_manual_protocol():
    plans, captions, _, _ = make_inputs()
    assert append_manual_words(captions, plans, None, PhysWordConfig()) is captions
    baseline = make_pack()
    _, _, data, _ = make_inputs()
    disabled = pack_input_sequence(
        plans,
        captions,
        data,
        torch.tensor([500.0]),
        SPECIAL_TOKENS,
        latent_patch_size=2,
        physword={"enabled": False},
        physical_words=[["invalid_but_disabled"]],
    )
    assert baseline.sample_lens == disabled.sample_lens
    assert baseline.split_lens == disabled.split_lens
    for field in ("text_ids", "text_indexes", "position_ids"):
        torch.testing.assert_close(getattr(baseline, field), getattr(disabled, field), atol=0, rtol=0)


def test_word_ids_and_global_indexes_for_unequal_packed_samples():
    config = PhysWordConfig(enabled=True)
    packed = make_pack(words=[WORDS_A, WORDS_B], enabled=True, batch_size=2)
    trace = physical_word_trace(packed, config)
    assert trace[0]["sample_start"] == 0
    assert trace[1]["sample_start"] == packed.sample_lens[0]
    for sample, (words, caption_length) in enumerate(zip((WORDS_A, WORDS_B), (3, 4), strict=True)):
        tokens = trace[sample]["tokens"]
        assert [t["token_id"] for t in tokens] == manual_word_ids(words, config)
        assert [t["sample_local_index"] for t in tokens] == list(range(caption_length, caption_length + 5))
        assert [t["global_index"] for t in tokens] == list(
            range(
                trace[sample]["sample_start"] + caption_length,
                trace[sample]["sample_start"] + caption_length + 5,
            )
        )
        assert packed.attn_modes[2 * sample] == "causal"
        assert packed.split_lens[2 * sample] == caption_length + 5 + 2  # PW start+4 words, EOS/BOV
    assert packed.vision is not None
    assert not set(t["global_index"] for row in trace for t in row["tokens"]) & set(
        packed.vision.sequence_indexes.tolist()
    )


def test_private_embedding_safe_lookup_and_no_base_mutation():
    base = torch.nn.Embedding(64, 8)
    old = base.weight.detach().clone()
    config = PhysWordConfig(enabled=True)
    table = PhysicalWordEmbedding(8, 64, config)
    ids = torch.tensor([3, *manual_word_ids(WORDS_A, config), 4])
    out = table(ids, base)
    torch.testing.assert_close(out[[0, -1]], base(torch.tensor([3, 4])), atol=0, rtol=0)
    torch.testing.assert_close(out[1], table.embedding.weight[0], atol=0, rtol=0)
    torch.testing.assert_close(base.weight, old, atol=0, rtol=0)
    assert table.embedding.num_embeddings == len(PW_TOKENS) == 17
    with pytest.raises(IndexError):
        table(torch.tensor([100]), base)


def test_config_can_be_enabled_through_structured_omegaconf():
    config = OmegaConf.structured(PhysWordConfig())
    assert not config.enabled
    config.enabled = True
    assert resolve_physword_config(config).enabled
    restored = Cosmos3VFMNetworkConfig(physword=OmegaConf.to_container(config))
    assert restored.to_dict()["physword"]["enabled"] is True


@pytest.mark.parametrize("kind", ["no_words", "unknown", "collision", "action"])
def test_manual_protocol_rejects_invalid_inputs(kind):
    plans, captions, _, _ = make_inputs()
    words = [WORDS_A]
    if kind == "no_words":
        words = None
    elif kind == "unknown":
        words = [["fly"]]
    elif kind == "collision":
        captions = [[1 << 30]]
    elif kind == "action":
        plans = [replace(plans[0], has_action=True)]
    with pytest.raises(ValueError):
        append_manual_words(captions, plans, words, PhysWordConfig(enabled=True))


@pytest.mark.parametrize(
    "overrides",
    [
        dict(action_gen=True),
        dict(sound_gen=True),
        dict(joint_attn_implementation="three_way"),
        dict(predict_text_tokens=True),
    ],
)
def test_network_rejects_modes_outside_m1(overrides):
    with pytest.raises(ValueError, match="M1 physword"):
        Cosmos3VFMNetworkConfig(physword={"enabled": True}, **overrides)


@pytest.fixture(scope="module")
def networks():
    if not torch.cuda.is_available():
        pytest.skip("Native Cosmos attention requires a CUDA backend")
    return make_network(), make_network(physword=True)


@pytest.mark.level(1)
@pytest.mark.gpus(1)
@torch.no_grad()
def test_pw_off_forward_is_exact_baseline(networks):
    baseline, enabled = networks
    for key, value in baseline.state_dict().items():
        torch.testing.assert_close(enabled.state_dict()[key], value, atol=0, rtol=0)
    packed = make_pack()
    packed.to_cuda()
    a = baseline(packed)["preds_vision"][0]
    b = enabled(packed)["preds_vision"][0]
    torch.testing.assert_close(a, b, atol=0, rtol=0)
    assert all(not key.startswith("physword.") for key in baseline.state_dict())


@pytest.mark.level(1)
@pytest.mark.gpus(1)
@torch.no_grad()
def test_intervention_changes_future_velocity_through_native_layers(networks):
    _, net = networks
    for layer in net.language_model.model.layers:
        assert layer.self_attn.dispatch_attention_fn is dispatch_attention
        assert layer.self_attn.k_norm_und_for_gen is not None
    a = make_pack(words=[WORDS_A], enabled=True)
    b = make_pack(words=[WORDS_B], enabled=True)
    assert a.sample_lens == b.sample_lens
    torch.testing.assert_close(a.position_ids, b.position_ids, atol=0, rtol=0)
    torch.testing.assert_close(a.vision.tokens[0], b.vision.tokens[0], atol=0, rtol=0)
    a.to_cuda()
    b.to_cuda()
    va = net(a)["preds_vision"][0]
    repeat = net(a)["preds_vision"][0]
    vb = net(b)["preds_vision"][0]
    torch.testing.assert_close(va, repeat, atol=0, rtol=0)
    assert (va[:, :, 1:].float() - vb[:, :, 1:].float()).square().mean().item() > 0


@pytest.mark.level(1)
@pytest.mark.gpus(1)
@torch.no_grad()
def test_pw_cannot_read_generator_and_earlier_pw_cannot_read_later_pw(networks):
    _, net = networks
    a = make_pack(words=[WORDS_A], enabled=True)
    b = make_pack(words=[WORDS_A], enabled=True)
    b.vision.tokens[0] = b.vision.tokens[0] + 2.0
    a.to_cuda()
    b.to_cuda()
    ha = net(a)["last_hidden_state"]
    hb = net(b)["last_hidden_state"]
    torch.testing.assert_close(ha[a.text_indexes], hb[b.text_indexes], atol=0, rtol=0)
    c = make_pack(words=[WORDS_B], enabled=True)
    c.to_cuda()
    hc = net(c)["last_hidden_state"]
    # Prefix, start, approach all precede the changed pick/push input (global index 5).
    torch.testing.assert_close(ha[:5], hc[:5], atol=0, rtol=0)
    assert not torch.equal(ha[5], hc[5])


@pytest.mark.level(1)
@pytest.mark.gpus(1)
@torch.no_grad()
def test_cross_sample_isolation_on_real_varlen_pack(networks):
    _, net = networks
    a = make_pack(words=[WORDS_A, WORDS_A], enabled=True, batch_size=2)
    b = make_pack(words=[WORDS_A, WORDS_B], enabled=True, batch_size=2)
    b.vision.tokens[1] = b.vision.tokens[1] + 3.0
    a.to_cuda()
    b.to_cuda()
    ha = net(a)
    hb = net(b)
    torch.testing.assert_close(
        ha["last_hidden_state"][: a.sample_lens[0]], hb["last_hidden_state"][: b.sample_lens[0]], atol=0, rtol=0
    )
    assert not torch.equal(ha["preds_vision"][1], hb["preds_vision"][1])


@pytest.mark.level(1)
@pytest.mark.gpus(1)
@torch.no_grad()
def test_native_two_way_matches_explicit_visibility_reference(networks):
    # Exercise the actual dispatcher/kernel, not a replacement attention implementation.
    generator = torch.Generator(device="cuda").manual_seed(73)
    values = [
        torch.randn(26, heads, 32, generator=generator, device="cuda", dtype=torch.bfloat16) for heads in (4, 2, 2)
    ]
    und = torch.tensor([*range(5), *range(12, 18)], device="cuda")
    gen = torch.tensor([*range(5, 12), *range(18, 26)], device="cuda")
    pack, meta, _ = build_packed_sequence(
        "two_way",
        packed_sequence=values[0],
        attn_modes=["causal", "full"] * 2,
        split_lens=[5, 7, 6, 8],
        sample_lens=[12, 14],
        packed_und_token_indexes=und,
        packed_gen_token_indexes=gen,
        num_heads=4,
        head_dim=32,
        num_layers=1,
    )
    # Native MoT derives Q/K/V from ONE layout. Share its offsets by identity,
    # which also lets two_way_attention select the correct DontCare causal type.
    packs = [pack, *(from_all_seq(value, pack) for value in values[1:])]
    actual, _ = dispatch_attention(*packs, meta)
    actual = get_all_seq_unpadded(actual).float()
    q, k, v = [value.float() for value in values]
    k, v = (value.repeat_interleave(2, dim=1) for value in (k, v))
    expected = []
    for start, u, g in ((0, 5, 7), (12, 6, 8)):
        size = u + g
        scores = torch.einsum("qhd,khd->hqk", q[start : start + size], k[start : start + size]) / 32**0.5
        mask = torch.ones(size, size, dtype=torch.bool, device="cuda")
        mask[:u] = False
        mask[:u, :u] = torch.ones(u, u, dtype=torch.bool, device="cuda").tril()
        weights = scores.masked_fill(~mask, -torch.inf).softmax(-1)
        expected.append(torch.einsum("hqk,khd->qhd", weights, v[start : start + size]).flatten(1))
    torch.testing.assert_close(actual, torch.cat(expected), atol=0.02, rtol=0.02)
