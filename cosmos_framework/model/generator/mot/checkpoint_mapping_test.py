"""The shared loader must retain the native inference checkpoint ABI."""

import json
import re
import subprocess
from pathlib import Path

from cosmos_framework.model.generator.mot.checkpoint_mapping import _diffusers_to_net_key, _diffusers_weight_map


def test_key_mapping_is_identical_to_original_native_implementation():
    source = subprocess.check_output(["git", "show", "55f0554:cosmos_framework/inference/model.py"], text=True)
    block = source[source.index("_DIFFUSERS_ROOT_INDEX =") : source.index("_LM_VISUAL_KEY_MARKER =")]
    namespace = {"json": json, "re": re, "Path": Path}
    exec(block, namespace)
    keys = [
        "lm_head.weight",
        "embed_tokens.weight",
        "norm_moe_gen.weight",
        "proj_in.weight",
        "proj_out.bias",
        "time_embedder.linear_1.weight",
        "time_embedder.linear_2.bias",
        "action_proj_in.fc.weight",
        "action_proj_out.bias.weight",
        "model.visual.encoder.layers.0.self_attn.q_proj.weight",
    ]
    keys += [
        f"layers.18.self_attn.{part}.weight"
        for part in [
            "to_q",
            "to_k",
            "to_v",
            "to_out",
            "add_q_proj",
            "add_k_proj",
            "add_v_proj",
            "to_add_out",
            "norm_added_k",
        ]
    ]
    for key in keys:
        assert _diffusers_to_net_key(key, "transformer/model.safetensors") == namespace["_diffusers_to_net_key"](
            key, "transformer/model.safetensors"
        )


def test_root_and_generator_indexes_preserve_generator_only_norm(tmp_path):
    (tmp_path / "transformer").mkdir()
    key = "layers.0.self_attn.k_norm_und_for_gen.weight"
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "weight_map": {
                    "embed_tokens.weight": "transformer/first.safetensors",
                    key: "transformer/old-invalid.safetensors",
                }
            }
        )
    )
    (tmp_path / "transformer/diffusion_pytorch_model.safetensors.index.json").write_text(
        json.dumps(
            {"weight_map": {key: "second.safetensors", "layers.0.self_attn.add_k_proj.weight": "second.safetensors"}}
        )
    )
    result = _diffusers_weight_map(tmp_path)
    assert result[key] == "transformer/second.safetensors"
    assert result["embed_tokens.weight"] == "transformer/first.safetensors"
