"""Optional CPU contract verification against the installed M3 branch; never trains.

Run from repo root: python -m tools.physword_data.check_m3_bridge --manifest ...
The optional --m3-root only extends the model physword package search path for
read-only integration checks before merging independent worktrees.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from cosmos_framework.data.generator.physword.common import write_json
from cosmos_framework.data.generator.physword.dataset import (
    TrainingWindowDataset,
    m3_target_payload,
    make_dataloader,
    to_m3_batch,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--m3-root")
    parser.add_argument("--check-vae", action="store_true")
    args = parser.parse_args()
    start = time.perf_counter()
    torch.set_num_threads(1)
    if args.m3_root:
        import cosmos_framework.model.generator.physword as model_package

        model_package.__path__.append(str(Path(args.m3_root).resolve() / "cosmos_framework/model/generator/physword"))
    dataset = TrainingWindowDataset(args.manifest)
    if not len(dataset):
        raise ValueError("Need at least one training window")
    sample = dataset[0]
    batch = to_m3_batch(sample)
    payload = m3_target_payload(sample)
    assert payload["target_rgb"].shape[0] == 4
    assert torch.equal(payload["target_rgb"], torch.as_tensor(sample["future_rgb"])[[3, 7, 11, 15]])
    loaded = next(iter(make_dataloader(args.manifest, batch_size=2, num_workers=0)))
    assert loaded["history_rgb"].dtype == torch.uint8
    assert batch.word_labels.shape == batch.valid_word.shape == (1, 4)
    report = {
        "native_m3_type": type(batch).__name__,
        "history_shape": list(batch.history.frames.shape),
        "future_shape": list(batch.future_rgb.shape),
        "word_shape": list(batch.word_labels.shape),
        "per_arm_target_shape": list(loaded["pw_arm_class_targets"].shape),
        "effectors": sample["pw_effectors"],
        "dataloader_history_shape": list(loaded["history_rgb"].shape),
        "torch_version": torch.__version__,
        "training_started": False,
        "device": "cpu",
        "vae": [],
    }
    if args.check_vae:
        from cosmos_framework.model.generator.tokenizers.wan2pt2_vae_4x16x16 import WanVAE_

        torch.manual_seed(20261101)
        vae = (
            WanVAE_(dim=16, dec_dim=16, z_dim=48, dim_mult=[1, 2, 2, 2], num_res_blocks=1).eval().requires_grad_(False)
        )
        history = torch.randn((1, 3, 5, 32, 32))
        future = torch.randn((1, 3, 16, 32, 32))
        scale = torch.zeros(48), torch.ones(48)
        with torch.no_grad():
            for temporal_window in (4, 68):
                vae.temporal_window = temporal_window
                observed = vae.encode(history, scale)
                full = vae.encode(torch.cat((history, future), 2), scale)
                changed = vae.encode(torch.cat((history, -future), 2), scale)
                assert observed.shape[2] == 2 and full.shape[2] == 6
                torch.testing.assert_close(observed, full[:, :, :2], atol=1e-5, rtol=1e-5)
                torch.testing.assert_close(observed, changed[:, :, :2], atol=1e-5, rtol=1e-5)
                report["vae"].append(
                    {
                        "weights": "random tiny native WanVAE_, NOT checkpoint validation",
                        "temporal_window": temporal_window,
                        "history_latents": list(observed.shape),
                        "full_latents": list(full.shape),
                        "history_full_max_abs": (observed - full[:, :, :2]).abs().max().item(),
                        "changed_future_history_max_abs": (observed - changed[:, :, :2]).abs().max().item(),
                    }
                )
    report["elapsed_seconds"] = time.perf_counter() - start
    write_json(args.output, report)
    print(json.dumps(report))


if __name__ == "__main__":
    main()
