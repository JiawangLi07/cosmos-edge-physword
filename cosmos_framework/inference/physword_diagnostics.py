# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Temporary observational hooks for B=1 AR/joint parity. Attention math is untouched."""

from contextlib import contextmanager

import torch

from cosmos_framework.data.generator.sequence_packing.runtime import from_und_gen_splits, get_gen_seq, get_und_seq


@contextmanager
def capture_und(network):
    records = [{} for _ in network.language_model.model.layers]
    handles = []
    dispatchers = []

    def hooks(layer, record):
        residual = []

        def save(name, value):
            record.setdefault(name, []).append(value.detach().reshape(-1, value.shape[-1]).clone())

        def input_hook(module, args):
            save("input_hidden", args[0])

        def residual_hook(module, args):
            residual[:] = [args[0].detach().reshape(-1, args[0].shape[-1]).clone()]

        def mlp_hook(module, args, output):
            save("hidden", residual[0] + output.reshape_as(residual[0]))

        def k_hook(module, args, output):
            save("pre_rope_k", output)

        handles.extend(
            (
                layer.input_layernorm.register_forward_pre_hook(input_hook),
                layer.post_attention_layernorm.register_forward_pre_hook(residual_hook),
                layer.mlp.register_forward_hook(mlp_hook),
                layer.self_attn.k_proj.register_forward_hook(k_hook),
            )
        )
        original = layer.self_attn.dispatch_attention_fn

        def observe(q, k, v, *args, **kwargs):
            n = k["_num_causal_tokens"]
            record["raw_k"] = get_und_seq(k)[:n].detach().clone()
            record["raw_v"] = get_und_seq(v)[:n].detach().clone()
            normalized = kwargs.get("packed_key_states_normalized")
            record["gen_normalized_k"] = get_und_seq(normalized if normalized is not None else k)[:n].detach().clone()
            return original(q, k, v, *args, **kwargs)

        dispatchers.append((layer.self_attn, original))
        layer.self_attn.dispatch_attention_fn = observe

    try:
        for layer, record in zip(network.language_model.model.layers, records, strict=True):
            hooks(layer, record)
        yield records
    finally:
        for handle in handles:
            handle.remove()
        for module, original in dispatchers:
            module.dispatch_attention_fn = original


@torch.no_grad()
def compare_ar_repack(network, prediction, ar_records, joint_records, joint_output, *, atol=0.02, rtol=0.02):
    """Normalized AR K is derived PRE-RoPE; raw cache K must never be renormalized in place."""
    condition = prediction.condition
    size = condition.input_ids.numel()
    if prediction.debug_cache is None or prediction.cache_length != size:
        raise AssertionError("Capture the complete AR cache, including the last emitted word")
    metrics = {}

    def compare(name, a, b):
        difference = a.float() - b.float()
        metrics[name] = {
            "max_abs": difference.abs().max().item(),
            "rms": difference.square().mean().sqrt().item(),
            "shape": list(a.shape),
        }
        torch.testing.assert_close(a, b, atol=atol, rtol=rtol, msg=name)

    for index, (layer, ar, joint) in enumerate(
        zip(network.language_model.model.layers, ar_records, joint_records, strict=True)
    ):
        compare(f"layer_{index}.input_hidden", torch.cat(ar["input_hidden"]), torch.cat(joint["input_hidden"])[:size])
        compare(f"layer_{index}.hidden", torch.cat(ar["hidden"]), torch.cat(joint["hidden"])[:size])
        compare(f"layer_{index}.raw_k", prediction.debug_cache.keys[index][0], joint["raw_k"][:size])
        compare(f"layer_{index}.raw_v", prediction.debug_cache.values[index][0], joint["raw_v"][:size])
        pre_k = torch.cat(ar["pre_rope_k"]).view(size, layer.self_attn.num_key_value_heads, layer.self_attn.head_dim)
        norm_k = layer.self_attn.k_norm_und_for_gen(pre_k)
        cos, sin = network.language_model.model.rotary_emb(
            pre_k, position_ids=condition.position_ids.unsqueeze(1).to(pre_k.device)
        )
        _, gen_k = layer.self_attn._apply_rotary_pos_emb(pre_k, norm_k, cos[0], sin[0], unsqueeze_dim=1)
        compare(f"layer_{index}.gen_normalized_k", gen_k, joint["gen_normalized_k"][:size])
    joint_z = joint_output["last_hidden_state"][
        prediction.prediction_positions.to(joint_output["last_hidden_state"].device)
    ]
    compare("prediction_hidden_states", prediction.prediction_hidden_states[0], joint_z)
    return metrics


@contextmanager
def mask_physical_word_suffix(network, packed):
    """B=1 diagnostic only: remove ALL suffix keys from GEN in EVERY layer.

    Keep the native UND causal result, positions, lengths and queries unchanged.
    Recompute only GEN attention with existing PyTorch SDPA over [context; GEN].
    Never zero K/V (which would retain softmax mass). No persistent/cache support.
    """
    condition = packed.reasoner_condition
    if condition is None or len(packed.sample_lens) != 1 or not condition.boundaries:
        raise ValueError("Suffix ablation requires a finalized B=1 PW condition")
    context = condition.context_length
    und_length, gen_length = packed.split_lens
    if und_length - context != 7:
        raise ValueError("Mask must cover start + four words + EOS/BOV")
    originals = []

    def make_observer(original):
        def masked(q, k, v, *args, **kwargs):
            if kwargs.get("memory_value") is not None:
                raise ValueError("PW suffix diagnostics forbid cached memory")
            result, stored = original(q, k, v, *args, **kwargs)
            keys = kwargs.get("packed_key_states_normalized")
            keys = k if keys is None else keys
            q_gen = get_gen_seq(q)[:gen_length]
            # Prefix C is causally before P and cannot already contain PW information.
            kk = torch.cat((get_und_seq(keys)[:context], get_gen_seq(keys)[:gen_length]))
            vv = torch.cat((get_und_seq(v)[:context], get_gen_seq(v)[:gen_length]))
            repeats = q_gen.shape[1] // kk.shape[1]
            value = (
                torch.nn.functional.scaled_dot_product_attention(
                    q_gen.transpose(0, 1).unsqueeze(0),
                    kk.repeat_interleave(repeats, dim=1).transpose(0, 1).unsqueeze(0),
                    vv.repeat_interleave(repeats, dim=1).transpose(0, 1).unsqueeze(0),
                    dropout_p=0.0,
                )
                .squeeze(0)
                .transpose(0, 1)
                .flatten(1)
            )
            gen_result = torch.cat((value, get_gen_seq(result)[gen_length:]), dim=0)
            return from_und_gen_splits(get_und_seq(result), gen_result, result), stored

        return masked

    try:
        for layer in network.language_model.model.layers:
            attn = layer.self_attn
            originals.append((attn, attn.dispatch_attention_fn))
            attn.dispatch_attention_fn = make_observer(attn.dispatch_attention_fn)
        yield
    finally:
        for attn, original in originals:
            attn.dispatch_attention_fn = original
