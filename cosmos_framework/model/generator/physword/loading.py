"""Local-only strict Edge warm-start using the native checkpoint key mapping."""

import hashlib
import json
import re
import subprocess
from dataclasses import asdict
from pathlib import Path

import torch
from safetensors import safe_open

from cosmos_framework.data.generator.processors.cosmos3_edge_processing import build_cosmos3_edge_processor
from cosmos_framework.model.generator.mot.checkpoint_mapping import _diffusers_to_net_key, _diffusers_weight_map
from cosmos_framework.model.generator.mot.cosmos3_vfm_network import Cosmos3VFMNetwork, Cosmos3VFMNetworkConfig
from cosmos_framework.model.generator.mot.unified_mot import Nemotron3DenseVLMoTConfig, Nemotron3DenseVLTextForCausalLM
from cosmos_framework.model.generator.physword.initialization import initialize_physical_words
from cosmos_framework.scripts.physword_preflight import inspect_checkpoint


def verify_resource_hashes(checkpoint, revision):
    """Use the pinned git LFS objects as the offline authoritative hash manifest."""
    names = [
        "tokenizer.json",
        "vision_encoder/model.safetensors",
        "vae/diffusion_pytorch_model.safetensors",
        "transformer/diffusion_pytorch_model-00001-of-00002.safetensors",
        "transformer/diffusion_pytorch_model-00002-of-00002.safetensors",
    ]
    result = {}
    for name in names:
        pointer = subprocess.check_output(["git", "-C", str(checkpoint), "show", f"{revision}:{name}"], text=True)
        expected = re.search(r"oid sha256:([a-f0-9]{64})", pointer)
        if expected is None:
            raise ValueError(f"Missing pinned LFS hash for {name}")
        h = hashlib.sha256()
        with (checkpoint / name).open("rb") as f:
            for chunk in iter(lambda: f.read(8 << 20), b""):
                h.update(chunk)
        if h.hexdigest() != expected[1]:
            raise ValueError(f"Official checkpoint SHA256 mismatch: {name}")
        result[name] = h.hexdigest()
    return result


def load_local_edge(checkpoint, pw_config, *, initialize_pw=True):
    """Load all UND/GEN/vision/projector weights on CPU; no implicit downloads.

    RGB-only intentionally omits the checkpoint's action heads. Vision is loaded
    strictly by the existing Edge local-bundle loader, not silently ignored.
    Semantic initialization runs after base loading, before any optimizer exists.
    """
    checkpoint = Path(checkpoint).resolve()
    preflight = inspect_checkpoint(checkpoint)
    if not preflight["files_complete"]:
        raise RuntimeError("Real Edge resources are incomplete; run physword_preflight first")
    verified_resources = verify_resource_hashes(checkpoint, preflight["revision"])
    top = json.loads((checkpoint / "config.json").read_text())
    generator = json.loads((checkpoint / "transformer/config.json").read_text())
    text = dict(top["text_config"])
    # Public Transformers export uses Qwen-style names; the native Nemotron
    # config exposes rms_norm_eps as a read-only alias of layer_norm_epsilon.
    text["layer_norm_epsilon"] = text.pop("rms_norm_eps")
    text["mlp_hidden_act"] = text.pop("hidden_act")
    text.pop("model_type", None)
    text.pop("rope_parameters", None)
    text["torch_dtype"] = text.pop("dtype", "bfloat16")
    text["mrope_section"] = generator["rope_scaling"]["mrope_section"]
    text["rope_theta"] = generator["rope_theta"]
    text["tie_word_embeddings"] = top["tie_word_embeddings"]
    text["hidden_dropout"] = 0.0
    wrapper = Nemotron3DenseVLMoTConfig(
        config_dict=text, qk_norm_for_text=generator["qk_norm_for_text"], qk_norm_for_diffusion=True
    )
    wrapper.use_und_k_norm_for_gen = generator["use_und_k_norm_for_gen"]
    # CPU has ample memory. Materialize once, then strictly fill all base tensors.
    language = Nemotron3DenseVLTextForCausalLM(wrapper)
    net = Cosmos3VFMNetwork(
        language,
        Cosmos3VFMNetworkConfig(
            vlm_config=language.config,
            vision_gen=True,
            action_gen=False,
            sound_gen=False,
            latent_channel_size=generator["latent_channel"],
            latent_patch_size=generator["latent_patch_size"],
            latent_downsample_factor=16,
            temporal_compression_factor_vision=generator["temporal_compression_factor"],
            timestep_scale=generator["timestep_scale"],
            enable_fps_modulation=generator["enable_fps_modulation"],
            base_fps=generator["base_fps"],
            max_latent_h=64,
            max_latent_w=64,
            max_latent_t=64,
            joint_attn_implementation="two_way",
            fp32_derived_buffers=True,
            physword=asdict(pw_config),
        ),
    )
    net.init_weights(buffer_device=torch.device("cpu"))
    # Cast parameters only. Module.bfloat16() would also round non-persistent
    # rotary/timestep frequencies; casting them back to FP32 cannot undo that.
    # Native RF embeds continuous time in FP32 and rebuilds these buffers in FP32.
    for parameter in net.parameters():
        parameter.data = parameter.data.bfloat16()
    language._local_checkpoint_dir = str(checkpoint)
    language._ensure_vision_tower()
    target = net.state_dict()
    loaded, ignored_action, ignored_vision, source_keys = set(), [], [], {}
    paths = {}
    for source, rel in _diffusers_weight_map(checkpoint).items():
        if rel.startswith("vision_encoder/"):
            ignored_vision.append(source)  # Already loaded, strictly, by _ensure_vision_tower.
            continue
        dest = _diffusers_to_net_key(source, rel)
        if dest is None:
            raise ValueError(f"Unknown checkpoint model key: {source}")
        if dest not in target:
            if dest.startswith(("action2llm.", "llm2action.", "action_modality_embed")):
                ignored_action.append(source)
                continue
            raise ValueError(f"Unexpected checkpoint tensor: {source} -> {dest}")
        if dest in source_keys:
            raise ValueError(f"Duplicate source tensor for {dest}")
        source_keys[dest] = source
        paths.setdefault(rel, []).append((source, dest))
    with torch.no_grad():
        for rel, pairs in paths.items():
            with safe_open(checkpoint / rel, framework="pt", device="cpu") as f:
                for source, dest in pairs:
                    value = f.get_tensor(source)
                    if value.shape != target[dest].shape:
                        raise ValueError(
                            f"Shape mismatch: {source}: {tuple(value.shape)} vs {tuple(target[dest].shape)}"
                        )
                    target[dest].copy_(value)
                    loaded.add(dest)
    vision_keys = {k for k in target if k.startswith("language_model.visual.")}
    missing = target.keys() - loaded - vision_keys - {"physword.embedding.weight", "physword.word_head.weight"}
    if missing:
        raise ValueError(f"Missing required Edge weights: {sorted(missing)}")
    net.time_embedder.float()  # Native timestep MLP runs FP32 outside autocast.
    processor = build_cosmos3_edge_processor(str(checkpoint))
    vocabulary = processor.tokenizer.get_vocab()
    largest = max(vocabulary.values())
    if largest >= language.model.embed_tokens.num_embeddings or largest >= language.lm_head.weight.shape[0]:
        raise ValueError("Tokenizer ID exceeds pretrained embedding/LM-head rows")
    for key in ("image_token_id", "video_token_id", "vision_start_token_id"):
        if top[key] not in vocabulary.values():
            raise ValueError(f"Missing tokenizer special ID: {key}")
    initialization = None
    if initialize_pw:
        net.physword.float()
        initialization = initialize_physical_words(
            net,
            processor.tokenizer,
            source={
                "verified_pretrained": True,
                "revision": preflight["revision"],
                "checkpoint": str(checkpoint),
                "resource_sha256": verified_resources,
                "input_source_key": source_keys["language_model.model.embed_tokens.weight"],
                "head_source_key": source_keys["language_model.lm_head.weight"],
            },
        )
    report = {
        "resource_sha256": verified_resources,
        "revision": preflight["revision"],
        "loaded_base_tensors": len(loaded),
        "vision_tensors_strictly_loaded": len(vision_keys),
        "missing_required": [],
        "unexpected": [],
        "shape_mismatch": [],
        "explicitly_unused_action_keys": ignored_action,
        "vision_source_keys_loaded_by_native_bundle_loader": len(ignored_vision),
        "tokenizer_max_id": largest,
        "input_embedding_rows": language.model.embed_tokens.num_embeddings,
        "lm_head_rows": language.lm_head.weight.shape[0],
        "initialization": initialization,
    }
    return net.eval().requires_grad_(False), processor, report
