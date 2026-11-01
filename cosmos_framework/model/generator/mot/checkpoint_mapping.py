# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Native Diffusers checkpoint key/index rules shared by inference and PW warm-start.

Moved verbatim from inference/model.py; no attention or tensor math here.
"""

import json
import re
from pathlib import Path

_DIFFUSERS_ROOT_INDEX = "model.safetensors.index.json"
_DIFFUSERS_MODEL_INDEX = "model_index.json"
_DIFFUSERS_DROP_WEIGHT_PATH_RES: tuple[re.Pattern[str], ...] = (re.compile(r"^(?!transformer/|vision_encoder/)"),)
_DIFFUSERS_DROP_KEY_RES: tuple[re.Pattern[str], ...] = (
    re.compile(r"^(?:feature_extractor|image_processor|scheduler|sound_tokenizer|text_encoder|tokenizer|vae)\."),
)
_DIFFUSERS_KEY_MAPPING_RES: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"^transformer\."), ""),
    (re.compile(r"^vision_encoder\."), ""),
    (re.compile(r"^model\.net\."), ""),
    (re.compile(r"^action_proj_in\."), "action2llm."),
    (re.compile(r"^action_proj_out\."), "llm2action."),
    (re.compile(r"^audio_proj_in\."), "sound2llm."),
    (re.compile(r"^audio_proj_out\."), "llm2sound."),
    (re.compile(r"^lidar_proj_in\."), "lidar2llm."),
    (re.compile(r"^lidar_proj_out\."), "llm2lidar."),
    (re.compile(r"^audio_modality_embed$"), "sound_modality_embed"),
    (re.compile(r"^proj_in\."), "vae2llm."),
    (re.compile(r"^proj_out\."), "llm2vae."),
    (re.compile(r"^time_embedder\.linear_1\."), "time_embedder.mlp.0."),
    (re.compile(r"^time_embedder\.linear_2\."), "time_embedder.mlp.2."),
    (re.compile(r"\.self_attn\.to_q\."), ".self_attn.q_proj."),
    (re.compile(r"\.self_attn\.to_k\."), ".self_attn.k_proj."),
    (re.compile(r"\.self_attn\.to_v\."), ".self_attn.v_proj."),
    (re.compile(r"\.self_attn\.to_out\."), ".self_attn.o_proj."),
    (re.compile(r"\.self_attn\.norm_q\."), ".self_attn.q_norm."),
    (re.compile(r"\.self_attn\.norm_k\."), ".self_attn.k_norm."),
    (re.compile(r"\.self_attn\.add_q_proj\."), ".self_attn.q_proj_moe_gen."),
    (re.compile(r"\.self_attn\.add_k_proj\."), ".self_attn.k_proj_moe_gen."),
    (re.compile(r"\.self_attn\.add_v_proj\."), ".self_attn.v_proj_moe_gen."),
    (re.compile(r"\.self_attn\.to_add_out\."), ".self_attn.o_proj_moe_gen."),
    (re.compile(r"\.self_attn\.norm_added_q\."), ".self_attn.q_norm_moe_gen."),
    (re.compile(r"\.self_attn\.norm_added_k\."), ".self_attn.k_norm_moe_gen."),
    (re.compile(r"^model\.lm_head\."), "language_model.lm_head."),
    (re.compile(r"^lm_head\."), "language_model.lm_head."),
    (re.compile(r"^model\.visual\."), "language_model.visual."),
    (re.compile(r"^visual\."), "language_model.visual."),
    (
        re.compile(r"^(blocks\.|deepstack_merger_list\.|merger\.|patch_embed\.|pos_embed\.)(.*)$"),
        r"language_model.visual.\1\2",
    ),
    (
        re.compile(
            r"^language_model\.(?!model\.|lm_head\.|visual\.)(embed_tokens\.|layers\.|norm(?:_moe_gen)?\.)(.*)$"
        ),
        r"language_model.model.\1\2",
    ),
    (
        re.compile(r"^model\.(embed_tokens\.|layers\.|norm(?:_moe_gen)?\.)(.*)$"),
        r"language_model.model.\1\2",
    ),
    (
        re.compile(r"^(embed_tokens\.|layers\.|norm(?:_moe_gen)?\.)(.*)$"),
        r"language_model.model.\1\2",
    ),
)
_DIFFUSERS_NET_KEY_PREFIXES: tuple[str, ...] = (
    "action2llm.",
    "action_pos_embed.",
    "language_model.",
    "latent_pos_embed.",
    "lidar2llm.",
    "llm2action.",
    "llm2lidar.",
    "llm2sound.",
    "llm2vae.",
    "rig_view_embed.",
    "sound2llm.",
    "time_embedder.",
    "vae2llm.",
)
_DIFFUSERS_NET_KEYS: frozenset[str] = frozenset(
    {
        "action_modality_embed",
        "latent_pos_embed",
        "sound_modality_embed",
    }
)


def _should_drop_diffusers_weight_path(path: str) -> bool:
    path = path.replace("\\", "/")
    return bool(path) and any(pattern.search(path) is not None for pattern in _DIFFUSERS_DROP_WEIGHT_PATH_RES)


def _should_drop_diffusers_key(name: str) -> bool:
    return any(pattern.search(name) is not None for pattern in _DIFFUSERS_DROP_KEY_RES)


def _is_diffusers_model_weight_path(path: str) -> bool:
    return bool(path) and not _should_drop_diffusers_weight_path(path)


def _apply_diffusers_key_mapping(name: str) -> str:
    for pattern, replacement in _DIFFUSERS_KEY_MAPPING_RES:
        name = pattern.sub(replacement, name)
    return name


def _is_loadable_diffusers_net_key(name: str) -> bool:
    return name in _DIFFUSERS_NET_KEYS or name.startswith(_DIFFUSERS_NET_KEY_PREFIXES)


def _diffusers_to_net_key(name: str, weight_path: str = "") -> str | None:
    """Rename a diffusers checkpoint key to its OmniMoTModel.net subtree key.

    Returns None for non-Cosmos model components in a full diffusers pipeline.
    """
    if _should_drop_diffusers_weight_path(weight_path) or _should_drop_diffusers_key(name):
        return None

    net_key = _apply_diffusers_key_mapping(name)
    if _should_drop_diffusers_key(net_key) or not _is_loadable_diffusers_net_key(net_key):
        return None
    return net_key


def _read_safetensors_index(index_path: Path) -> dict[str, str]:
    index = json.loads(index_path.read_text(encoding="utf-8"))
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict):
        raise ValueError(f"{index_path} does not contain a safetensors weight_map.")

    result: dict[str, str] = {}
    for key, value in weight_map.items():
        if not isinstance(key, str) or not isinstance(value, str):
            raise TypeError(f"{index_path} weight_map must contain string keys and values.")
        result[key] = value
    return result


def _diffusers_weight_map(checkpoint_path: Path) -> dict[str, str]:
    index_path = checkpoint_path / _DIFFUSERS_ROOT_INDEX
    weight_map = _read_safetensors_index(index_path) if index_path.exists() else {}
    # The generator-only K norm (`k_norm_und_for_gen`) belongs to the transformer
    # component and must be sourced from its own index under the diffusers name
    # (`layers.N...`). The converter is supposed to exclude it from the root index
    # (see _convert_model_to_diffusers._remap_language_model_state_dict), but
    # checkpoints exported before that fix (e.g. nvidia/Cosmos3-Edge-Policy-DROID)
    # leak it into the root index under the raw model-internal name
    # (`layers.layers.N...`), which points at tensors absent from the shards and
    # breaks loading. Drop any root-index k-norm entries and let the transformer
    # index below supply the correctly named ones.
    weight_map = {k: v for k, v in weight_map.items() if ".k_norm_und_for_gen." not in k}
    # The root index may be a reasoner manifest (e.g. Cosmos3-Edge) that lists only the
    # shared understanding-pathway weights and the vision tower. The transformer
    # component's own index holds the complete DiT state — including the
    # generation-pathway tensors (`*_moe_gen`, `add_*`) absent from such a root index.
    # Merge it (root entries win) so the full generator can be loaded.
    transformer_index_path = checkpoint_path / "transformer" / "diffusion_pytorch_model.safetensors.index.json"
    if transformer_index_path.exists():
        for key, rel_path in _read_safetensors_index(transformer_index_path).items():
            weight_map.setdefault(key, f"transformer/{rel_path}")
    if not weight_map:
        raise FileNotFoundError(f"Diffusers safetensors index not found at {index_path} or {transformer_index_path}")
    return weight_map


def _diffusers_files_to_keys(weight_map: dict[str, str]) -> dict[str, list[str]]:
    files_to_keys: dict[str, list[str]] = {}
    for diff_key, rel_path in weight_map.items():
        if _should_drop_diffusers_weight_path(rel_path):
            continue
        files_to_keys.setdefault(rel_path, []).append(diff_key)
    return files_to_keys


def _is_diffusers_checkpoint(checkpoint_path: Path) -> bool:
    index_path = checkpoint_path / _DIFFUSERS_ROOT_INDEX
    transformer_index_path = checkpoint_path / "transformer" / "diffusion_pytorch_model.safetensors.index.json"
    if (checkpoint_path / _DIFFUSERS_MODEL_INDEX).exists() and (index_path.exists() or transformer_index_path.exists()):
        return True
    if not index_path.exists():
        return False
    return any(_is_diffusers_model_weight_path(path) for path in _read_safetensors_index(index_path).values())
