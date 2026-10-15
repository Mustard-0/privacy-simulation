"""Portable privacy cells. Planning never starts training or contacts GitHub."""
from __future__ import annotations

import argparse
import copy
import csv
import ctypes
import hashlib
import json
import os
import platform
from pathlib import Path
import shutil
import subprocess
import sys
import threading
import time

import privacy_checks as checks

ROOT = Path(__file__).resolve().parent
STAGES = ("screen", "formal_training", "full_evaluation")


def hashes(folder):
    return {p.name: checks.sha(p) for p in sorted(Path(folder).glob("*.py"))}


def configuration():
    return checks.read(ROOT / "config.json")


def fingerprint():
    return dict(model=hashes(ROOT / "model"), config=checks.sha(ROOT / "config.json"),
                runner={name: checks.sha(ROOT / name) for name in ("cloud.py", "privacy_checks.py")})


def verify_gate(path):
    gate = checks.read(path)
    if not gate.get("passed") or gate.get("package") != fingerprint():
        raise ValueError("A passing screen from this exact source/config is required")
    if gate["baseline_choice"]["lr"] not in (0, 1):
        raise ValueError("Invalid validation-only baseline selection")
    return gate


def jobs_for(stage, gate=None):
    config = configuration()
    if stage == "screen":
        return checks.screen_jobs(config, 0)
    if gate is None:
        raise ValueError("Formal work requires the audited screen gate")
    return [checks.job(method, epsilon,
                       gate["baseline_choice"]["lr"] if method == checks.METHODS[2] else 0,
                       config["training"]["seed"])
            for epsilon in checks.EPSILONS for method in checks.METHODS]


def prepare(out, stage, gate=None):
    config = copy.deepcopy(configuration())
    config["training"]["updates"] = 30 if stage == "screen" else 2000
    if stage == "full_evaluation":
        config["training"]["test_seeds"] = checks.RESERVED_TEST_SEEDS
    metadata = dict(config=configuration(), model_hashes=hashes(ROOT / "model"),
                    package=fingerprint(), reserved_test_seeds=checks.RESERVED_TEST_SEEDS,
                    selection_basis="TMAPO/NoAmp LR1 predeclared; convergence unfinished",
                    scope="Independent CPU/cloud execution; never merge into old CUDA runs")
    frozen = out / "pipeline.json"
    if frozen.exists() and checks.read(frozen) != metadata:
        raise ValueError("Refusing changed source/config; preserve this run")
    checks.save(frozen, metadata)
    source = out / "model_source"
    if not source.exists():
        shutil.copytree(ROOT / "model", source, ignore=shutil.ignore_patterns("__pycache__"))
    checks.verify_source(source, metadata["model_hashes"])
    folder = out / stage
    plan = dict(kind="privacy", jobs=jobs_for(stage, gate), formal_eval=stage == "full_evaluation")
    for name, data in (("config.json", config), ("plan.json", plan),
                       ("source_hashes.json", metadata["model_hashes"])):
        path = folder / name
        if path.exists() and checks.read(path) != data:
            raise ValueError("Saved stage plan/config changed")
        checks.save(path, data)
    return folder, config, plan["jobs"]


def available_ram_gib():
    if os.name == "nt":
        class Memory(ctypes.Structure):
            _fields_ = [("length", ctypes.c_ulong), ("load", ctypes.c_ulong)] + [
                (key, ctypes.c_ulonglong) for key in
                ("total", "available", "page_total", "page_free", "virtual_total", "virtual_free", "extended")]
        memory = Memory()
        memory.length = ctypes.sizeof(memory)
        if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(memory)):
            raise OSError("Memory query failed")
        return memory.available / 2**30
    entries = dict(line.split(":", 1) for line in Path("/proc/meminfo").read_text().splitlines())
    return float(entries["MemAvailable"].split()[0]) / 2**20


class ResourceGuard:
    """Request an update-boundary checkpoint, never terminate a process."""
    def __init__(self, folder, device):
        self.folder, self.device = folder, device
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self.watch, daemon=True)

    def watch(self):
        while not self.stop.is_set():
            try:
                state = dict(ram_free_gib=available_ram_gib(), time=time.time())
                danger = state["ram_free_gib"] < .5
                if self.device == "cuda":
                    result = subprocess.run(["nvidia-smi", "--query-gpu=temperature.gpu,memory.free",
                                             "--format=csv,noheader,nounits"], check=True,
                                            capture_output=True, text=True, timeout=10)
                    temperature, free = map(float, result.stdout.splitlines()[0].split(","))
                    state.update(temperature_c=temperature, gpu_free_mib=free)
                    danger |= temperature > 78 or free < 1024
                if danger:
                    checks.save(self.folder / "STOP_AFTER_UPDATE", dict(reason="resource risk", **state))
                    return
            except Exception as exc:
                checks.save(self.folder / "STOP_AFTER_UPDATE", dict(reason="resource query failure", error=repr(exc)))
                return
            self.stop.wait(5)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *args):
        self.stop.set()
        self.thread.join(timeout=15)


def run_cell(out, stage, job_id, device, seconds, gate=None):
    folder, config, jobs = prepare(out, stage, gate)
    candidate = next((j for j in jobs if j["id"] == job_id), None)
    if candidate is None:
        raise ValueError("Unknown exact job ID")
    target = folder / job_id
    if (target / "complete.json").exists():
        if checks.read(target / "complete.json")["job"] != candidate:
            raise ValueError("Completed cell does not match this job")
        return True
    if (target / "failure.json").exists():
        raise ValueError("Failure evidence requires review, not blind retry")
    if (folder / "STOP_AFTER_UPDATE").exists():
        raise ValueError("Saved safety stop requires resource review before recovery")
    target.mkdir(parents=True, exist_ok=True)
    lock = target / "RUNNING.lock"
    with lock.open("x", encoding="utf-8") as handle:
        handle.write(str(os.getpid()))
    try:
        checks.model_path(out)
        with (target / "runtime_history.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(dict(time=time.time(), platform=platform.platform(),
                python=platform.python_version(), device=device, package=fingerprint()))+"\n")
        if stage == "full_evaluation":
            evaluate_cell(out, candidate, target, device)
            return True
        from run import train_job
        if available_ram_gib() < 2:
            raise RuntimeError("One worker requires 2 GiB launch headroom; no other local cell will be launched")
        if device == "cuda":
            result = subprocess.run(["nvidia-smi", "--query-gpu=temperature.gpu,memory.free",
                                     "--format=csv,noheader,nounits"], check=True,
                                    capture_output=True, text=True, timeout=10)
            temperature, free = map(float, result.stdout.splitlines()[0].split(","))
            if temperature >= 75 or free < 1536:
                raise RuntimeError("GPU is not ready for a safe launch")
        checks.save(out / "state.json", dict(pid=os.getpid(), worker_pids=[], stage=stage, status="running"))
        with ResourceGuard(folder, device):
            done = train_job(config, candidate, target, device, wall_seconds=seconds, plot_progress=False)
        checks.save(out / "state.json", dict(pid=os.getpid(), worker_pids=[], stage=stage,
                                             status="complete" if done else "paused"))
        return done
    except Exception as exc:
        checks.save(target / "failure.json", dict(job=candidate, error=repr(exc)))
        raise
    finally:
        lock.unlink()


def evaluate_cell(out, candidate, target, device):
    import torch
    from common import seed_all
    from evaluation import evaluate
    from learners import make_policy
    source = out / "formal_training" / candidate["id"]
    formal = out / "formal_training"
    if not checks.read(formal / "audit.json").get("passed") or not checks.read(formal / "batch_audit.json").get("passed"):
        raise ValueError("Audit the complete formal grid before opening reserved tests")
    if not (source / "complete.json").exists():
        raise ValueError("Full evaluation requires completed formal training")
    manifest = checks.read(source / "manifest.json")
    if manifest["source_hashes"] != hashes(ROOT / "model") or manifest["device"] != device:
        raise ValueError("Evaluation source/device must match the trained cell")
    if available_ram_gib() < 6:
        raise RuntimeError("Full evaluation requires 6 GiB headroom")
    config = copy.deepcopy(manifest["config"])
    config["training"]["test_seeds"] = checks.RESERVED_TEST_SEEDS
    seed_all(candidate["seed"], config["training"]["torch_threads"])
    checkpoint = source / "best_validation.pt"
    saved = torch.load(checkpoint, map_location=device, weights_only=False)
    policy = make_policy(candidate["method"], config, device)
    policy.load(saved["policy"])
    # Each reserved trace is written immediately; do not retain five task cohorts.
    for name in ("test_raw.csv", "test_tasks.csv", "test_batches.csv"):
        if (target / name).exists():
            raise ValueError("Partial evaluation requires review; do not append duplicate reserved traces")
    for seed in checks.RESERVED_TEST_SEEDS:
        tests, tasks, batches = evaluate(policy, config, candidate["method"], [seed], True, collect=True)
        for name, data in (("test_raw.csv", tests), ("test_tasks.csv", tasks), ("test_batches.csv", batches)):
            if data:
                path = target / name
                exists = path.exists()
                with path.open("a", newline="", encoding="utf-8" if exists else "utf-8-sig") as handle:
                    writer = csv.DictWriter(handle, list(data[0]))
                    if not exists:
                        writer.writeheader()
                    writer.writerows(data)
        del tests, tasks, batches
    checks.save(target / "complete.json", dict(job=candidate, config=config, checkpoint=str(checkpoint.resolve()),
        checkpoint_hash=checks.sha(checkpoint), selected_update=saved["update"], training_updates_performed=0,
        model_hashes=hashes(ROOT / "model"), evaluation="five reserved 300-second traces; one training seed"))


def review(out, stage, gate=None):
    folder, config, jobs = prepare(out, stage, gate)
    checks.model_path(out)
    for candidate in jobs:
        path = folder / candidate["id"]
        saved = checks.read(path / ("complete.json" if stage == "full_evaluation" else "manifest.json"))
        expected = copy.deepcopy(config)
        expected["environment"].update(users_per_region=candidate["users_per_region"], epsilon_req=candidate["epsilon_req"])
        expected["training"]["actor_lr"], expected["training"]["critic_lr"] = expected["experiments"]["learning_rates"][candidate["lr"]]
        if saved["config"] != expected or saved["job"] != candidate:
            raise ValueError("Raw cell manifest/config differs from the fixed protocol")
    if stage == "full_evaluation":
        checks.audit_full(out, folder)
    else:
        from audit_comparison import audit
        audit(folder)
    checks.audit_batches(out, folder)
    if stage != "screen":
        from plots import plot_run
        plot_run(folder)
    if stage == "screen":
        from evaluation import quality_check
        probe = out / "probe"
        probe.mkdir(exist_ok=True)
        quality_check(config, probe, 20000)
        result = checks.screen_gate(folder, probe)
        result["package"] = fingerprint()
        checks.save(out / "screen_gate.json", result)
        return result
    return checks.read(folder / "audit.json")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("plan", "run", "review"))
    parser.add_argument("--stage", choices=STAGES, default="screen")
    parser.add_argument("--out", type=Path, default=ROOT / "runs")
    parser.add_argument("--gate", type=Path)
    parser.add_argument("--job")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--max-seconds", type=float, default=14400)
    args = parser.parse_args()
    if args.max_seconds <= 0:
        parser.error("max-seconds must be positive")
    if args.stage != "screen" and not args.gate:
        parser.error("Formal work requires --gate from an audited screen")
    gate = verify_gate(args.gate) if args.gate else None
    if args.command == "plan":
        print(json.dumps(dict(include=jobs_for(args.stage, gate)), separators=(",", ":")))
    elif args.command == "review":
        print(json.dumps(review(args.out, args.stage, gate), indent=2))
    else:
        if not args.job:
            parser.error("run requires an exact --job from plan")
        done = run_cell(args.out, args.stage, args.job, args.device, args.max_seconds, gate)
        print("CELL_STATUS=" + ("complete" if done else "paused"), flush=True)


if __name__ == "__main__":
    main()
