"""Wait for an actually unused GPU on 121--129, then run ONE bounded M3 probe.

Standard library only. No CUDA allocations, GPU holder, process termination or
changes to other users' queues. This is opportunistic polling, not a reservation.
"""

import argparse
import concurrent.futures
import csv
import fcntl
import io
import json
import os
import shlex
import shutil
import subprocess
import time
from pathlib import Path

PRIORITY = (127, 125, 123, 121, 124, 128, 129, 122, 126)
SSH = ("ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=6", "-o", "StrictHostKeyChecking=yes")


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def parse_snapshot(gpus, processes):
    occupied = set()
    for row in csv.reader(io.StringIO(processes)):
        if row:
            if len(row) != 2 or not row[1].strip().isdigit():
                raise ValueError("Invalid compute process query; refusing to assume GPUs are free")
            occupied.add(row[0].strip())
    result = []
    for row in csv.reader(io.StringIO(gpus)):
        if not row:
            continue
        if len(row) != 6:
            raise ValueError("Invalid GPU snapshot")
        index, uuid, used, free, total, utilization = (item.strip() for item in row)
        result.append(
            {
                "index": int(index),
                "uuid": uuid,
                "used_mib": int(used),
                "free_mib": int(free),
                "total_mib": int(total),
                "utilization": int(utilization),
                "has_process": uuid in occupied,
            }
        )
    if not result or len({gpu["uuid"] for gpu in result}) != len(result):
        raise ValueError("Missing/duplicate GPUs; refusing to assume device availability")
    return result


def is_idle(gpu):
    return (
        not gpu["has_process"] and gpu["used_mib"] <= 1024 and gpu["free_mib"] >= 70 * 1024 and gpu["utilization"] <= 5
    )


def query_node(node):
    host = f"10.254.30.{node}"

    def query(fields):
        command = shlex.join(["nvidia-smi", fields, "--format=csv,noheader,nounits"])
        return subprocess.run([*SSH, host, command], capture_output=True, text=True, check=True, timeout=15).stdout

    try:
        # Query processes AFTER memory. A disappearing process is conservative;
        # a newly arriving process prevents dispatch immediately.
        gpus = query("--query-gpu=index,uuid,memory.used,memory.free,memory.total,utilization.gpu")
        processes = query("--query-compute-apps=gpu_uuid,pid")
        return {"node": node, "gpus": parse_snapshot(gpus, processes)}
    except (OSError, ValueError, subprocess.SubprocessError) as error:
        return {"node": node, "error": str(error), "gpus": []}


def update_streaks(snapshots, previous):
    current = {}
    for snapshot in snapshots:
        for gpu in snapshot["gpus"]:
            key = (snapshot["node"], gpu["uuid"])
            current[key] = previous.get(key, 0) + 1 if is_idle(gpu) else 0
    # Failed/missing nodes lose their streak; stale evidence cannot qualify.
    return current


def choose(snapshots, streaks, samples):
    for node in PRIORITY:
        snapshot = next((item for item in snapshots if item["node"] == node), None)
        if snapshot:
            for gpu in sorted(snapshot["gpus"], key=lambda item: item["index"]):
                if is_idle(gpu) and streaks.get((node, gpu["uuid"]), 0) >= samples:
                    return node, gpu
    return None


def launch_command(args, gpu):
    environment = {
        "CUDA_VISIBLE_DEVICES": gpu["uuid"],
        "M3_EXPECTED_GPU_UUID": gpu["uuid"],
        "I4_ATTN_BACKENDS": "flash3",
        "COSMOS_TRAINING": "0",
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "OMP_NUM_THREADS": "8",
        "PYTHONPATH": str(args.checkout),
        "PYTHONNOUSERSITE": "1",
    }
    command = [
        "timeout",
        "--signal=TERM",
        "--kill-after=15s",
        "2850s",
        "env",
        *[f"{key}={value}" for key, value in environment.items()],
        str(args.python),
        "-m",
        "cosmos_framework.scripts.physword_real_b_probe",
        "--checkpoint",
        str(args.checkpoint),
        "--task-dir",
        str(args.task_dir),
        "--output",
        str(args.output / "run"),
        "--timeout-seconds",
        "2700",
    ]
    # Check the immutable checkout immediately before launch on the remote host.
    verify = shlex.join(["git", "-C", str(args.checkout), "rev-parse", "HEAD"])
    check = f'test "$({verify})" = {shlex.quote(args.commit)}'
    clean = shlex.join(["git", "-C", str(args.checkout), "diff", "--quiet", "HEAD", "--"])
    return f"{check} && {clean} && cd {shlex.quote(str(args.checkout))} && exec {shlex.join(command)}"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkout", type=Path, required=True)
    parser.add_argument("--commit", required=True)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--task-dir", type=Path, required=True)
    parser.add_argument("--poll-seconds", type=int, default=60)
    parser.add_argument("--idle-samples", type=int, default=3)
    parser.add_argument("--wait-hours", type=float, default=24)
    parser.add_argument("--scan-once", action="store_true", help="Read-only snapshot; never submit")
    args = parser.parse_args()
    if args.poll_seconds < 10 or args.idle_samples < 2 or not 0 < args.wait_hours <= 48:
        parser.error("Require poll>=10s, samples>=2, finite wait<=48h")
    args.output.mkdir(parents=True, exist_ok=True)
    # Protect against accidentally starting the same queue twice.
    with (args.output / "controller.lock").open("a") as controller_lock:
        fcntl.flock(controller_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (args.output / "run").exists():
            raise FileExistsError("A previous acceptance already used this queue output; choose a new directory")
        write_json(
            args.output / "controller.json",
            {"pid": os.getpid(), "commit": args.commit, "checkout": str(args.checkout), "started_unix": time.time()},
        )
        deadline, streaks = time.monotonic() + args.wait_hours * 3600, {}
        while time.monotonic() < deadline and not (args.output / "STOP").exists():
            with concurrent.futures.ThreadPoolExecutor(max_workers=9) as executor:
                snapshots = list(executor.map(query_node, PRIORITY))
            streaks = update_streaks(snapshots, streaks)
            status = {
                "status": "WAITING",
                "updated_unix": time.time(),
                "commit": args.commit,
                "idle_samples_required": args.idle_samples,
                "minimum_free_gib": 70,
                "poll_seconds": args.poll_seconds,
                "snapshots": snapshots,
                "idle_streaks": {f"{node}/{uuid}": n for (node, uuid), n in streaks.items() if n},
            }
            write_json(args.output / "status.json", status)
            if args.scan_once:
                return
            candidate = choose(snapshots, streaks, args.idle_samples)
            if candidate and shutil.disk_usage(args.output).free >= 128 * 2**30:
                node, gpu = candidate
                locks = args.output.parent / "gpu_locks"
                locks.mkdir(exist_ok=True)
                with (locks / f"{node}-{gpu['uuid']}.lock").open("a") as device_lock:
                    try:
                        fcntl.flock(device_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    except BlockingIOError:
                        streaks[(node, gpu["uuid"])] = 0
                        continue
                    fresh = query_node(node)
                    if not any(item["uuid"] == gpu["uuid"] and is_idle(item) for item in fresh["gpus"]):
                        streaks[(node, gpu["uuid"])] = 0
                        continue
                    status.update(status="RUNNING", selected_node=node, selected_gpu=gpu, started_unix=time.time())
                    write_json(args.output / "status.json", status)
                    try:
                        with (args.output / "remote.log").open("w") as log:
                            result = subprocess.run(
                                [*SSH, f"10.254.30.{node}", launch_command(args, gpu)],
                                stdout=log,
                                stderr=subprocess.STDOUT,
                                timeout=3000,
                            )
                        report_path = args.output / "run/result.json"
                        report = json.loads(report_path.read_text()) if report_path.exists() else {}
                        passed = result.returncode == 0 and report.get("status") == "PASS"
                        status.update(
                            status="PASS" if passed else "FAIL",
                            remote_exit_code=result.returncode,
                            result=report,
                            finished_unix=time.time(),
                        )
                    except (OSError, ValueError, subprocess.SubprocessError) as error:
                        status.update(status="FAIL", error=str(error), finished_unix=time.time())
                    write_json(args.output / "status.json", status)
                    return  # Exactly ONE attempt. No automatic repeated training.
            time.sleep(args.poll_seconds)
        write_json(
            args.output / "status.json",
            {
                "status": "STOPPED" if (args.output / "STOP").exists() else "EXPIRED",
                "finished_unix": time.time(),
                "commit": args.commit,
            },
        )


if __name__ == "__main__":
    main()
