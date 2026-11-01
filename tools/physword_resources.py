"""Fetch only the five missing Edge resources, pinned and SHA256 verified.

No snapshots, examples, or datasets are downloaded. Existing non-pointer files
must already match; downloads are staged and only replace verified LFS pointers.
"""

import argparse
import hashlib
import json
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests

REPOSITORY = "nvidia/Cosmos3-Edge"
REVISION = "344d602b128d1bbdacb43b08d0a3626f46343e29"
FILES = (
    "tokenizer.json",
    "vision_encoder/model.safetensors",
    "vae/diffusion_pytorch_model.safetensors",
    "transformer/diffusion_pytorch_model-00001-of-00002.safetensors",
    "transformer/diffusion_pytorch_model-00002-of-00002.safetensors",
)


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--download", action="store_true")
    parser.add_argument(
        "--metadata", type=Path, help="Previously fetched official revision metadata (offline fallback)"
    )
    parser.add_argument("--transport", choices=["requests", "curl"], default="curl")
    args = parser.parse_args()
    if args.metadata:
        saved = json.loads(args.metadata.read_text())
        metadata = {"sha": saved["sha"], "siblings": saved["files"]}
    else:
        response = requests.get(
            f"https://huggingface.co/api/models/{REPOSITORY}/revision/{REVISION}",
            params={"blobs": "true"},
            timeout=30,
        )
        response.raise_for_status()
        metadata = response.json()
    if metadata["sha"] != REVISION:
        raise ValueError("Official snapshot revision mismatch")
    records = {x["rfilename"]: x for x in metadata["siblings"]}

    def obtain(name):
        record = records[name]
        expected = record["lfs"]
        path = args.checkpoint / name
        result = {"file": name, "expected_bytes": expected["size"], "sha256": expected["sha256"]}
        if path.exists() and path.stat().st_size == expected["size"]:
            result["status"] = "verified_existing" if sha256(path) == expected["sha256"] else "hash_mismatch"
            return result
        if path.exists():
            with path.open("rb") as f:
                pointer = f.read(200).startswith(b"version https://git-lfs.github.com/spec/v1")
            if not pointer:
                return {**result, "status": "refused_existing_nonpointer"}
        if not args.download:
            return {**result, "status": "missing_or_lfs_pointer"}
        staged = args.checkpoint / ".physword-download" / (name.replace("/", "__") + ".partial")
        staged.parent.mkdir(parents=True, exist_ok=True)
        try:
            url = f"https://huggingface.co/{REPOSITORY}/resolve/{REVISION}/{name}"
            if args.transport == "curl":
                command = [
                    "curl",
                    "--fail",
                    "--location",
                    "--silent",
                    "--show-error",
                    "--connect-timeout",
                    "20",
                    "--max-time",
                    "900",
                    "--continue-at",
                    "-",
                    "--output",
                    str(staged),
                    url + "?download=true",
                ]
                if os.environ.get("HTTPS_PROXY"):
                    command[1:1] = ["--proxy", os.environ["HTTPS_PROXY"]]
                result_curl = subprocess.run(command, capture_output=True)
                if result_curl.returncode:
                    return {**result, "status": "blocked_curl", "exit_code": result_curl.returncode}
            else:
                with requests.get(url, params={"download": "true"}, stream=True, timeout=(20, 60)) as r:
                    if not r.ok:
                        return {**result, "status": "blocked_http", "http_status": r.status_code}
                    with staged.open("wb") as f:
                        for chunk in r.iter_content(chunk_size=8 << 20):
                            f.write(chunk)
            if staged.stat().st_size != expected["size"] or sha256(staged) != expected["sha256"]:
                return {**result, "status": "download_integrity_failed", "received_bytes": staged.stat().st_size}
            path.parent.mkdir(parents=True, exist_ok=True)
            os.replace(staged, path)
            return {**result, "status": "downloaded_verified"}
        except requests.RequestException as e:
            # Never log signed redirect URLs or authentication material.
            return {**result, "status": "blocked_network", "exception": type(e).__name__}

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(obtain, FILES))
    report = {"repository": REPOSITORY, "revision": REVISION, "download_authorized": args.download, "files": results}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    for row in results:
        print(row["file"], row["status"], flush=True)


if __name__ == "__main__":
    main()
