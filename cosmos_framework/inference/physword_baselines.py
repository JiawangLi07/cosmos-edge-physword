"""Reproducible single-step Native / Matched RGB-prefix / Full PW conditions."""

from dataclasses import asdict, replace

from cosmos_framework.data.generator.sequence_packing.packers import pack_input_sequence
from cosmos_framework.data.generator.sequence_packing.sequence import SequencePlan
from cosmos_framework.inference.physword import generate_physical_words, repack_physical_words
from cosmos_framework.model.generator.physword.multimodal import build_reasoner_prefix


def pack_video_baseline(network, processor, history, instruction, gen_data, timesteps, special_tokens, *, mode):
    """Caller fixes weights/history/noise/sigma; only the UND condition changes.

    Repacking remains differentiable. AR alone is no-grad. The returned native
    network forward can be used for evaluation or the existing native video loss.
    """
    if mode not in {"native", "matched_rgb", "full_pw"}:
        raise ValueError("Expected native, matched_rgb or full_pw baseline mode")
    if mode == "native":
        return pack_input_sequence(
            sequence_plans=[SequencePlan(has_text=True, has_vision=True, condition_frame_indexes_vision=[0, 1])],
            input_text_indexes=[processor.tokenizer.encode(instruction, add_special_tokens=False)],
            gen_data_clean=gen_data,
            input_timesteps=timesteps.cpu(),
            special_tokens=special_tokens,
            latent_patch_size=network.config.latent_patch_size,
            physword=asdict(replace(network.physword.config, enabled=False)),
            unified_3d_mrope_temporal_modality_margin=15000,
            enable_fps_modulation=True,
            base_fps=24.0,
            temporal_compression_factor=4,
        )
    prefix = build_reasoner_prefix(network, processor, history, instruction)
    if mode == "matched_rgb":
        return repack_physical_words(
            network, prefix.without_physical_words(), gen_data, timesteps, special_tokens, rgb_prefix_only=True
        )
    prediction = generate_physical_words(network, prefix)
    return repack_physical_words(network, prediction.condition, gen_data, timesteps, special_tokens)
