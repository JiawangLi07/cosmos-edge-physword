"""Continuous native RF parity, including fractional time and real gradient paths."""

import ast
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from cosmos_framework.inference.physword import repack_physical_words
from cosmos_framework.model.generator.algorithm.loss.flow_matching import compute_flow_matching_loss
from cosmos_framework.model.generator.diffusion.rectified_flow import RectifiedFlow
from cosmos_framework.model.generator.utils.data_and_condition import GenerationDataClean
from cosmos_framework.scripts.physword_m3_probe import synthetic_training
from cosmos_framework.scripts.physword_probe import SPECIAL_TOKENS


def original_native_time():
    """Execute the original NVIDIA method, with a minimal self, without its trainer imports."""
    source = subprocess.check_output(
        ["git", "show", "55f0554:cosmos_framework/model/generator/omni_mot_model.py"], text=True
    )
    cls = next(x for x in ast.parse(source).body if isinstance(x, ast.ClassDef) and x.name == "OmniMoTModel")
    method = next(x for x in cls.body if isinstance(x, ast.FunctionDef) and x.name == "_get_train_noise_level_vision")
    code = ast.Module(
        body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), method],
        type_ignores=[],
    )
    namespace = {"torch": torch}
    exec(compile(ast.fix_missing_locations(code), "original_native_time", "exec"), namespace)
    return namespace[method.name]


@pytest.mark.parametrize("distribution", ["waver", "logitnormal", "uniform"])
@pytest.mark.parametrize("resolution,shift", [("256", 3), ("480", 5), ("720", 10)])
def test_continuous_sampling_equals_original_native(distribution, resolution, shift):
    flow = RectifiedFlow(None, train_time_distribution=distribution, shift=shift, device="cpu")
    cfg = SimpleNamespace(use_discrete_rf=False, shift={"256": 3, "480": 5, "720": 10})
    kwargs = {"device": "cpu", "dtype": torch.float32}
    obj = SimpleNamespace(
        rectified_flow_video=flow,
        tensor_kwargs_fp32=kwargs,
        config=SimpleNamespace(rectified_flow_training_config=cfg, causal_training_strategy="none"),
    )
    torch.manual_seed(20261101)
    original_t, original_s = original_native_time()(obj, 8, False, [6] * 8, resolutions=[resolution] * 8)
    torch.manual_seed(20261101)
    actual_t, actual_s = flow.sample_continuous_train_time(8, torch.full((8,), shift), kwargs)
    torch.testing.assert_close(actual_s, original_s, atol=0, rtol=0)
    torch.testing.assert_close(actual_t, original_t, atol=0, rtol=0)


@pytest.mark.parametrize("sigma", [0.12345, 0.37460, 0.82731])
def test_fractional_sigma_full_step_native_values_and_gradients(sigma):
    net, _, _, _, _, batch, step, _ = synthetic_training()
    sigma_tensor = torch.tensor([sigma], device="cuda")
    timestep = sigma_tensor * 1000
    noise = torch.randn((1, 48, 6, 4, 4), generator=torch.Generator().manual_seed(12))
    captured = []
    embeddings = []
    hook = net.time_embedder.register_forward_pre_hook(lambda _, args: captured.append(args[0].detach().clone()))
    output_hook = net.time_embedder.register_forward_hook(lambda _, args, out: embeddings.append(out.detach().clone()))
    try:
        result = step(batch, sigmas=sigma_tensor, noise=noise)
    finally:
        hook.remove()
        output_hook.remove()
    with torch.no_grad(), torch.autocast("cuda", enabled=False):
        reference_embedding = net.time_embedder(timestep * net.timestep_scale)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        observed = batch.history.frames.permute(1, 0, 2, 3).unsqueeze(0).float() / 127.5 - 1
        future = batch.future_rgb.permute(1, 0, 2, 3).unsqueeze(0).float() / 127.5 - 1
        history = step.encode_video(observed).float()
        full = step.encode_video(torch.cat((observed, future), 2)).float()
        clean = torch.cat((history, full[:, :, 2:]), 2)
        epsilon = noise.to(clean)
        mask = result["packed"].vision.condition_mask[0]
        # Original OmniMoT _add_noise_to_input protocol, independent of the new helper.
        effective = sigma_tensor.view(-1, 1, 1) * (1 - mask)
        native_xt, native_vt = step.flow.get_interpolation([epsilon], [clean], [effective])
        native_data = GenerationDataClean(
            batch_size=1, is_image_batch=False, x0_tokens_vision=native_xt, fps_vision=torch.tensor([10.0])
        )
        packed = repack_physical_words(net, result["prediction"].condition, native_data, timestep.cpu(), SPECIAL_TOKENS)
        packed.to_cuda()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        output = net(packed)
        native_loss, _ = compute_flow_matching_loss(
            pred=[x.float() for x in output["preds_vision"]],
            target=native_vt,
            condition_mask=packed.vision.condition_mask,
            timesteps=timestep.reshape(1, 1),
            has_valid_tokens=True,
            rectified_flow=step.flow,
            tensor_kwargs_fp32={"device": "cuda", "dtype": torch.float32},
        )
        native_loss = native_loss * 10
    names = [
        "physword.embedding.weight",
        "language_model.model.layers.0.self_attn.k_proj.weight",
        "language_model.model.layers.0.self_attn.v_proj.weight",
        "llm2vae.weight",
        "time_embedder.mlp.0.weight",
    ]
    parameters = [dict(net.named_parameters())[n] for n in names]
    actual_grads = torch.autograd.grad(result["losses"]["video"], parameters)
    native_grads = torch.autograd.grad(native_loss, parameters)
    metrics = {
        "sigma": sigma,
        "noised_latents_max_abs": (result["packed"].vision.tokens[0] - native_xt[0]).abs().max().item(),
        "embedder_input_max_abs": (torch.cat(captured) - timestep * net.timestep_scale).abs().max().item(),
        "timestep_embedding_max_abs": (embeddings[0] - reference_embedding).abs().max().item(),
        "velocity_target_max_abs": (native_vt[0] - (epsilon - clean)).abs().max().item(),
        "video_loss_max_abs": (result["losses"]["video"] - native_loss).abs().item(),
        "gradient_max_abs": {
            n: (a - b).abs().max().item() for n, a, b in zip(names, actual_grads, native_grads, strict=True)
        },
        "atol": 0,
        "rtol": 0,
    }
    assert all(metrics[k] == 0 for k in metrics if k.endswith("max_abs") and k != "gradient_max_abs")
    assert all(x == 0 for x in metrics["gradient_max_abs"].values())
    Path(f"/tmp/physword-rf-{sigma}.json").write_text(json.dumps(metrics, indent=2) + "\n")
