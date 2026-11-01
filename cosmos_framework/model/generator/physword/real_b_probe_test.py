"""The queued real acceptance entry itself, exercised with native Tiny weights."""

import json
import os
import subprocess
import sys

import torch

from cosmos_framework.model.generator.physword.optimization import build_phase_optimizer, optimizer_audit
from cosmos_framework.scripts.physword_m3_probe import synthetic_training
from cosmos_framework.scripts.physword_real_b_probe import prepare_masters, signature


def test_state_fingerprints_detect_parameter_dtype_shape_and_optimizer_changes():
    value = torch.arange(8, dtype=torch.float32)
    expected = signature({"parameter": value, "optimizer": {0: {"step": torch.tensor(1)}}})
    assert expected == signature({"parameter": value.clone(), "optimizer": {0: {"step": torch.tensor(1)}}})
    assert signature(value) != signature(value.reshape(2, 4))
    assert signature(value) != signature(value.bfloat16())
    changed = value.clone()
    changed[0] += 0.001
    assert signature(value) != signature(changed)
    assert expected != signature({"parameter": value, "optimizer": {0: {"step": torch.tensor(2)}}})


def test_real_path_cpu_master_promotion_and_post_cuda_optimizer_identity():
    net, _, _, phase, _, batch, step, _ = synthetic_training()
    net.cpu()
    selected = prepare_masters(net, phase)
    assert all(p.dtype == (torch.float32 if p.requires_grad else torch.bfloat16) for p in net.parameters())
    net.cuda()
    optimizer = build_phase_optimizer(net, phase)
    assert selected == [group["parameter_names"] for group in optimizer.param_groups]
    result = step(batch)
    result["losses"]["total"].backward()
    groups = optimizer_audit(net, optimizer)
    assert all(group["grad_norm"] > 0 for group in groups)


def test_queued_runner_independent_losses_update_and_new_interpreter_resume(tmp_path):
    output = tmp_path / "bounded"
    result = subprocess.run(
        [sys.executable, "-m", "cosmos_framework.scripts.physword_real_b_probe", "--tiny", "--output", str(output)],
        env=os.environ.copy(),
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert result.returncode == 0, result.stdout + result.stderr + (output / "save.log").read_text()
    report = json.loads((output / "result.json").read_text())
    assert report["status"] == "PASS" and report["phase"] == "full A+B+C"
    validation = output / "validation"
    gradients = json.loads((validation / "independent_gradients.json").read_text())
    assert set(gradients) == {"video", "word", "latent"}
    first = json.loads((validation / "step1.json").read_text())
    oracle = json.loads((validation / "oracle.json").read_text())
    resumed = json.loads((validation / "resumed.json").read_text())
    assert first["sample_index"] == 0 and oracle["observation"]["sample_index"] == 1
    assert oracle["observation"]["counts"]["word"] == 3
    assert oracle == resumed
