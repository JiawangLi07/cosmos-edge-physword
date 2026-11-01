"""Availability counterexamples for the read-only GPU polling queue."""

from argparse import Namespace
from pathlib import Path

import pytest

from tools.physword_gpu_queue import choose, is_idle, launch_command, parse_snapshot, update_streaks


def gpu(uuid="GPU-test", **changes):
    return {
        "index": 0,
        "uuid": uuid,
        "used_mib": 0,
        "free_mib": 81000,
        "total_mib": 81559,
        "utilization": 0,
        "has_process": False,
        **changes,
    }


@pytest.mark.parametrize(
    "changes",
    [
        {"has_process": True},
        {"used_mib": 72000},
        {"free_mib": 26000},
        {"utilization": 80},
    ],
)
def test_busy_or_reserved_cards_never_qualify(changes):
    assert not is_idle(gpu(**changes))
    assert is_idle(gpu())


def test_actual_process_query_overrides_zero_utilization():
    parsed = parse_snapshot("0, GPU-a, 0, 81000, 81559, 0\n", "GPU-a, 123\n")
    assert not is_idle(parsed[0])


@pytest.mark.parametrize(
    "gpus,processes",
    [("", ""), ("0, GPU-a, N/A, 81000, 81559, 0", ""), ("0, GPU-a, 0, 81000, 81559, 0", "query error")],
)
def test_bad_queries_fail_closed(gpus, processes):
    with pytest.raises(ValueError):
        parse_snapshot(gpus, processes)


def test_three_fresh_observations_and_node_failure_resets_streak():
    snapshots = [{"node": 127, "gpus": [gpu()]}]
    streaks = {}
    for _ in range(2):
        streaks = update_streaks(snapshots, streaks)
        assert choose(snapshots, streaks, 3) is None
    streaks = update_streaks(snapshots, streaks)
    assert choose(snapshots, streaks, 3)[0] == 127
    streaks = update_streaks([{"node": 127, "gpus": [], "error": "SSH timeout"}], streaks)
    streaks = update_streaks(snapshots, streaks)
    assert choose(snapshots, streaks, 3) is None


def test_priority_only_among_available_cards_and_no_nodes_outside_range():
    snapshots = [{"node": node, "gpus": [gpu(uuid=f"GPU-{node}")]} for node in (120, 125, 127)]
    streaks = {(node, f"GPU-{node}"): 3 for node in (120, 125, 127)}
    assert choose(snapshots, streaks, 3)[0] == 127
    snapshots[-1]["gpus"][0]["has_process"] = True
    assert choose(snapshots, streaks, 3)[0] == 125


def test_dispatch_is_pinned_single_gpu_offline_and_bounded():
    args = Namespace(
        checkout=Path("/aifs/test checkout"),
        commit="a" * 40,
        python=Path("/aifs/env/bin/python"),
        checkpoint=Path("/aifs/base"),
        task_dir=Path("/aifs/task"),
        output=Path("/aifs/job"),
    )
    command = launch_command(args, gpu())
    for literal in (
        "CUDA_VISIBLE_DEVICES=GPU-test",
        "M3_EXPECTED_GPU_UUID=GPU-test",
        "HF_HUB_OFFLINE=1",
        "2850s",
        "2700",
        "physword_real_b_probe",
        args.commit,
        "diff --quiet HEAD",
    ):
        assert literal in command
    assert "train.py" not in command and "kill " not in command
