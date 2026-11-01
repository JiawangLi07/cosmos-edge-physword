# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Offline M0 inventory. Uses stdlib only, never loads or downloads weights."""

from __future__ import annotations

import argparse
import json
import struct
import subprocess
from pathlib import Path

LFS_MAGIC = b"version https://git-lfs.github.com/spec/v1"


def inspect_file(path: Path, *, safetensors: bool = False) -> dict:
    if not path.is_file():
        return {"status": "missing", "path": str(path)}
    size = path.stat().st_size
    with path.open("rb") as stream:
        prefix = stream.read(256)
        if prefix.startswith(LFS_MAGIC):
            fields = dict(line.split(" ", 1) for line in prefix.decode().splitlines() if " " in line)
            return {
                "status": "lfs_pointer",
                "path": str(path),
                "disk_bytes": size,
                "expected_bytes": int(fields.get("size", 0)),
                "oid": fields.get("oid"),
            }
        if not safetensors:
            try:
                json.loads(path.read_text())
            except (ValueError, UnicodeError):
                return {"status": "invalid_json", "path": str(path)}
            return {"status": "present", "path": str(path), "disk_bytes": size}
        try:
            if len(prefix) < 8:
                raise ValueError("missing header length")
            header_size = struct.unpack("<Q", prefix[:8])[0]
            if not 2 <= header_size <= min(64 * 1024 * 1024, size - 8):
                raise ValueError("invalid header size")
            stream.seek(8)
            header = json.loads(stream.read(header_size))
            tensors = {key: item for key, item in header.items() if key != "__metadata__"}
            for item in tensors.values():
                start, end = item["data_offsets"]
                if not 0 <= start <= end <= size - 8 - header_size:
                    raise ValueError("tensor data is truncated")
            return {
                "status": "present",
                "path": str(path),
                "disk_bytes": size,
                "tensor_count": len(tensors),
                "shapes": {key: item["shape"] for key, item in tensors.items()},
            }
        except (ValueError, KeyError, TypeError, struct.error) as exc:
            return {"status": "invalid_safetensors", "path": str(path), "error": str(exc)}


def inspect_checkpoint(root: Path) -> dict:
    components = {}
    for name, index_name in (
        ("reasoner", "model.safetensors.index.json"),
        ("generator", "transformer/diffusion_pytorch_model.safetensors.index.json"),
    ):
        index_path = root / index_name
        index_status = inspect_file(index_path)
        entries = [index_status]
        if index_status["status"] == "present":
            index = json.loads(index_path.read_text())
            entries += [
                inspect_file(index_path.parent / file, safetensors=True)
                for file in sorted(set(index["weight_map"].values()))
            ]
        components[name] = entries
    components["siglip2"] = [inspect_file(root / "vision_encoder/model.safetensors", safetensors=True)]
    components["vae"] = [
        inspect_file(root / "vae/diffusion_pytorch_model.safetensors", safetensors=True),
        inspect_file(root / "vae/config.json"),
    ]
    components["tokenizer"] = [
        inspect_file(root / file)
        for file in (
            "tokenizer.json",
            "tokenizer_config.json",
            "special_tokens_map.json",
            "processor_config.json",
            "preprocessor_config.json",
            "video_preprocessor_config.json",
        )
    ]
    compatibility = {}
    try:
        reasoner = json.loads((root / "config.json").read_text())
        gen = json.loads((root / "transformer/config.json").read_text())
        vae = json.loads((root / "vae/config.json").read_text())
        compatibility = {
            "reasoner_generator_dimensions_match": all(
                reasoner["text_config"][k] == gen[k]
                for k in ("hidden_size", "num_hidden_layers", "head_dim", "num_attention_heads", "num_key_value_heads")
            ),
            "vae_channels_match": vae["z_dim"] == gen["latent_channel"],
            "vae_time_compression_matches": vae["scale_factor_temporal"] == gen["temporal_compression_factor"],
            "patch_width_matches": gen["patch_latent_dim"] == gen["latent_channel"] * gen["latent_patch_size"] ** 2,
            "edge_gen_und_k_norm_enabled": gen["use_und_k_norm_for_gen"],
            "weight_shape_validation": "not_run_until_real_shards_present",
            "tokenizer_id_validation": "not_run_until_real_tokenizer_present",
        }
    except (OSError, ValueError, KeyError) as exc:
        compatibility["metadata_error"] = str(exc)
    try:
        revision = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
    except subprocess.CalledProcessError:
        revision = None
    missing = [item for entries in components.values() for item in entries if item["status"] != "present"]
    metadata_ok = bool(compatibility) and all(
        value is True
        for key, value in compatibility.items()
        if key not in ("weight_shape_validation", "tokenizer_id_validation")
    )
    return {
        "checkpoint": str(root.resolve()),
        "revision": revision,
        "components": components,
        "compatibility": compatibility,
        "files_complete": not missing,
        "metadata_compatible": metadata_ok,
        "missing_or_invalid": missing,
        "real_edge_forward_status": "ready_for_runtime_validation" if not missing and metadata_ok else "blocked",
        "downloads_performed": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = inspect_checkpoint(args.checkpoint)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {
                "files_complete": result["files_complete"],
                "metadata_compatible": result["metadata_compatible"],
                "missing_or_invalid_count": len(result["missing_or_invalid"]),
                "real_edge_forward_status": result["real_edge_forward_status"],
            }
        )
    )


if __name__ == "__main__":
    main()
