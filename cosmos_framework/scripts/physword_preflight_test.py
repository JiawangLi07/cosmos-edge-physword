# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

import json
import struct

from cosmos_framework.scripts.physword_preflight import inspect_checkpoint, inspect_file


def test_lfs_pointer_never_counts_as_weights(tmp_path):
    path = tmp_path / "model.safetensors"
    path.write_text("version https://git-lfs.github.com/spec/v1\noid sha256:abc\nsize 8000000000\n")
    result = inspect_file(path, safetensors=True)
    assert result["status"] == "lfs_pointer"
    assert result["expected_bytes"] == 8000000000


def test_missing_and_truncated_weights_are_rejected(tmp_path):
    path = tmp_path / "model.safetensors"
    assert inspect_file(path, safetensors=True)["status"] == "missing"
    header = json.dumps({"weight": {"dtype": "F32", "shape": [2], "data_offsets": [0, 8]}}).encode()
    path.write_bytes(struct.pack("<Q", len(header)) + header + b"\0" * 4)
    assert inspect_file(path, safetensors=True)["status"] == "invalid_safetensors"
    path.write_bytes(struct.pack("<Q", len(header)) + header + b"\0" * 8)
    assert inspect_file(path, safetensors=True)["status"] == "present"


def test_missing_checkpoint_cannot_be_reported_ready(tmp_path):
    report = inspect_checkpoint(tmp_path)
    assert not report["files_complete"]
    assert not report["metadata_compatible"]
    assert report["real_edge_forward_status"] == "blocked"
    assert report["downloads_performed"] is False
