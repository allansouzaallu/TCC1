"""Run one live N20 lab scenario on an already provisioned CUDA runner."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import signal
import sys
import tempfile
import time

from sistema.scripts import runpod_worker as worker

ROOT = Path(__file__).resolve().parents[3]
PROFILE = Path(__file__).with_name("profile.json")


def load_profile():
    p = json.loads(PROFILE.read_text(encoding="utf-8"))
    if (p["schema"], p["nodes"], p["rounds"], p["gpu_count"], p["models_per_gpu"],
        p["inference_layout"], p["quantization"]) != (
        "tcc-live-n20/v1", 20, 50, 2, 10, "per_node", "nf4"
    ):
        raise ValueError("Invalid fixed N20 CUDA profile")
    if p["timeout_minutes"] > 110:
        raise ValueError("The runtime limit must not exceed 110 minutes")
    taskfile = ROOT / p["task_file"]
    raw = taskfile.read_bytes().replace(b"\r\n", b"\n")  # Canonical Git LF
    if hashlib.sha256(raw).hexdigest() != p["task_sha256"]:
        raise ValueError("Task file digest mismatch")
    tasks = json.loads(raw)["tarefas"]
    if len(tasks) != 50 or len({t["id"] for t in tasks}) != 50:
        raise ValueError("Need 50 distinct fixed tasks")
    scenarios = {s["id"]: s for s in p["scenarios"]}
    grid = {(s["malicious_nodes"], s["noise_mode"]) for s in scenarios.values()}
    if len(p["scenarios"]) != 6 or len(scenarios) != 6 or grid != {
        (0, "relative"), (4, "multiplicative"), (8, "multiplicative"),
        (16, "multiplicative"), (8, "relative"), (16, "relative"),
    } or p["seeds"] != list(range(5)):
        raise ValueError("Unexpected N20 lab scenario grid")
    return p, scenarios


def scenario_args(profile, scenario, seed):
    if seed not in profile["seeds"]:
        raise ValueError("Seed outside audited grid")
    protocol = profile["protocol"]
    config = {
        "nodes": profile["nodes"], "gpu_count": profile["gpu_count"],
        "models_per_gpu": profile["models_per_gpu"], "model": profile["model"],
        "model_revision": profile["model_revision"], "task_bank": profile["task_file"],
        "max_new_tokens": profile["max_new_tokens"], "temperature": profile["temperature"],
        "top_p": profile["top_p"], "noise_mode": scenario["noise_mode"],
        "noise_distance": protocol["relative_noise_distance"],
        **{k: protocol[k] for k in ("degree", "topology", "signal", "consensus_mode", "consensus_center",
                                   "mechanism", "defense", "audit_mode", "malicious_strategy", "collusion_value")},
    }
    entry = {
        "id": f"live-n20-r50-{scenario['id']}-s{seed}",
        "rounds": profile["rounds"], "semente": seed,
        "fracao_maliciosos": scenario["malicious_nodes"] / profile["nodes"],
        "nota": protocol["score"], "honest_rounds": protocol["honest_rounds"], "audit_p": protocol["audit_p"],
    }
    return config, entry


def check_mapping(profile, scenario, seed, root, snapshot):
    config, entry = scenario_args(profile, scenario, seed)
    processes, protocol = worker.scenario_configuration(config, entry, root, snapshot)
    keys = list(processes)
    per_gpu = [sum(slot == gpu for slot, _ in processes.values()) for gpu in range(profile["gpu_count"])]
    if keys != [f"node-{i}" for i in range(20)] or per_gpu != [10, 10]:
        raise ValueError("Bad node or GPU mapping")
    if len({node["url"] for node in protocol["nodes"]}) != 20:
        raise ValueError("Endpoints must be distinct")
    malicious = 0
    for node_id, (_, env) in processes.items():
        if any(env.get(k) != v for k, v in {
            "MODEL_RUN_ID": entry["id"], "MODEL_DEVICE": "cuda", "MODEL_QUANTIZATION": "nf4",
            "INFERENCE_MODE": "model", "MODEL_CACHE_DIR": str(root / "node-models" / node_id),
        }.items()):
            raise ValueError("Unexpected node inference runtime")
        malicious += env.get("PROFILE") == "malicious"
    if malicious != scenario["malicious_nodes"]:
        raise ValueError("Bad malicious node count")


def preflight(profile, scenarios):
    with tempfile.TemporaryDirectory() as folder:
        root = Path(folder)
        for scenario in scenarios.values():
            for seed in profile["seeds"]:
                check_mapping(profile, scenario, seed, root, root)
    print("30 combinations validated offline; 20 model processes; 10 per GPU; 50 rounds each.")


def live(profile, scenario, seed, snapshot, out):
    if os.getenv("TCC_LIVE_N20_APPROVED") != "true":
        raise RuntimeError("Set explicit live execution approval before running.")
    if sys.platform != "linux":
        raise RuntimeError("Live launch requires a dedicated Linux CUDA runner.")
    snapshot = snapshot.resolve(strict=True)
    if snapshot.name != profile["model_revision"] or not (snapshot / "config.json").is_file() or not list(snapshot.glob("*.safetensors")):
        raise ValueError("Select the preinstalled snapshot pinned by profile.json.")
    import torch
    n = profile["gpu_count"]
    if not torch.cuda.is_available() or torch.cuda.device_count() != n:
        raise RuntimeError("Select exactly two dedicated CUDA devices.")
    devices = os.getenv("CUDA_VISIBLE_DEVICES", "0,1").split(",")
    if len(devices) != n or len(set(devices)) != n or not all(devices):
        raise RuntimeError("Invalid CUDA_VISIBLE_DEVICES mapping.")
    for i in range(n):
        memory_free, _ = torch.cuda.mem_get_info(i)
        if memory_free < 40 * 1024 ** 3:
            raise RuntimeError(f"CUDA device {i} must have at least 40 GiB free.")
    config, entry = scenario_args(profile, scenario, seed)
    out = out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    check_mapping(profile, scenario, seed, out, snapshot)
    metadata = {"profile": profile, "scenario": entry, "started": datetime.now(timezone.utc).isoformat(),
                "devices": [torch.cuda.get_device_name(i) for i in range(n)],
                "expected_own_generations": 1000, "completed": False, "git_sha": os.getenv("GITHUB_SHA", "")}
    record = out / "live-run.json"
    record.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    deadline = time.monotonic() + 60 * profile["timeout_minutes"]
    interrupted = False

    def stop(_signum, _frame):
        nonlocal interrupted
        interrupted = True

    def check():
        if interrupted or time.monotonic() >= deadline:
            raise InterruptedError("Live run cancelled or deadline reached.")

    last = None

    def progress(state, **fields):
        nonlocal last
        item = (state, tuple(fields.items())) if state != "running" else (state, fields["rounds_completed"])
        if item != last:
            print(state, fields, flush=True)
            last = item

    previous = {s: signal.signal(s, stop) for s in (signal.SIGINT, signal.SIGTERM)}
    try:
        worker.execute_scenario(out, entry, config, snapshot, devices, check, progress)
    finally:
        for signum, handler in previous.items():
            signal.signal(signum, handler)
    metadata.update(completed=True, verified_own_generations=1000, finished=datetime.now(timezone.utc).isoformat())
    record.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")


def main():
    profile, scenarios = load_profile()
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="operation", required=True)
    sub.add_parser("preflight")
    run = sub.add_parser("run")
    run.add_argument("--scenario", choices=scenarios, required=True)
    run.add_argument("--seed", choices=profile["seeds"], type=int, required=True)
    run.add_argument("--snapshot", type=Path, required=True)
    run.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.operation == "preflight":
        preflight(profile, scenarios)
    else:
        live(profile, scenarios[args.scenario], args.seed, args.snapshot, args.output)


if __name__ == "__main__":
    main()
