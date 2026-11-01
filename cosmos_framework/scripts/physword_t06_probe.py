"""Real Edge, bounded K=4 recompute/joint parity. No optimizer or model download."""

import argparse
import json
from pathlib import Path

import torch

from cosmos_framework.data.generator.physword.dataset import TrainingWindowDataset, to_m3_batch
from cosmos_framework.data.generator.physword.task import config_from_manifest
from cosmos_framework.inference.physword import generate_physical_words, repack_physical_words
from cosmos_framework.inference.physword_diagnostics import capture_und, compare_ar_repack
from cosmos_framework.model.generator.physword.loading import load_local_edge
from cosmos_framework.model.generator.physword.multimodal import build_reasoner_prefix
from cosmos_framework.model.generator.physword.preprocessing import PreprocessingConfig
from cosmos_framework.model.generator.utils.data_and_condition import GenerationDataClean


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--task-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--windows", type=int, nargs="+", default=[0, 1, 100])
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Choose a fresh T06 evidence file")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(20261101)
    cfg = config_from_manifest(json.loads((args.task_dir / "vocabulary.json").read_text()))
    net, proc, loading = load_local_edge(args.checkpoint, cfg)
    net.cuda().eval()
    dataset = TrainingWindowDataset(args.task_dir / "windows.jsonl", split="train")
    preprocessing = PreprocessingConfig((480, 640), (64, 64))
    special = {
        "eos_token_id": proc.tokenizer.eos_token_id,
        "start_of_generation": proc.tokenizer.convert_tokens_to_ids("<|vision_start|>"),
        "end_of_generation": proc.tokenizer.convert_tokens_to_ids("<|vision_end|>"),
    }
    result = {"weights": "REAL Edge", "loading": loading, "windows": [], "status": "RUNNING"}
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for index in args.windows:
            sample = dataset[index]
            batch = preprocessing.batch(to_m3_batch(sample, pw_config=cfg))
            prefix = build_reasoner_prefix(net, proc, batch.history, batch.instruction)
            with capture_und(net) as ar:
                prediction = generate_physical_words(net, prefix, capture_cache=True)
            data = GenerationDataClean(
                batch_size=1,
                is_image_batch=False,
                x0_tokens_vision=[torch.randn(1, 48, 6, 4, 4, device="cuda")],
                fps_vision=torch.tensor([10.0]),
            )
            packed = repack_physical_words(net, prediction.condition, data, torch.tensor([374.6]), special)
            packed.to_cuda()
            with capture_und(net) as joint:
                output = net(packed)
            metrics = compare_ar_repack(net, prediction, ar, joint, output)  # Original tolerances.
            result["windows"].append(
                {
                    "window_id": sample["window_id"],
                    "decode_mode": prediction.decode_mode,
                    "class_ids": prediction.class_ids.tolist(),
                    "metrics": metrics,
                    "all_max_abs": max(m["max_abs"] for m in metrics.values()),
                }
            )
            args.output.write_text(json.dumps(result, indent=2) + "\n")
            print(sample["window_id"], "PASS", result["windows"][-1]["all_max_abs"], flush=True)
    result.update(
        status="PASS",
        atol=0.02,
        rtol=0.02,
        rgb="REAL history resized 64x64",
        gen_latents="Synthetic fixed shape; real GEN weights; VAE isolation",
        peak_cuda_bytes=torch.cuda.max_memory_allocated(),
    )
    args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
