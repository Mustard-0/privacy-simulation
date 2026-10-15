"""Short execution benchmark with raw-outcome digests, never paper evidence."""
import argparse
import cProfile
import hashlib
import io
import json
from pathlib import Path
import pstats
import sys
import time

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", choices=("model", "reference_model"), required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--method", default="TMAPO")
    parser.add_argument("--slots", type=int, default=200)
    parser.add_argument("--mode", choices=("deterministic", "training"), default="deterministic")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(ROOT / args.source))
    import numpy as np
    import torch
    from common import seed_all
    from environment import Environment, GreedyPolicy
    from learners import make_policy

    cfg = json.loads((ROOT / "config.json").read_text(encoding="utf-8-sig"))
    seed_all(2026, 1)
    policy = GreedyPolicy(cfg["environment"]) if args.method == "Greedy" else make_policy(args.method, cfg, args.device)
    if args.checkpoint:
        saved = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
        policy.load(saved["policy"])
    policy.deterministic = args.mode == "deterministic"
    c = cfg["environment"]
    env = Environment(c, 71005, args.method, 15., measure_start=0., measure_end=args.slots*c["dt_s"], collect=True)
    profiler = cProfile.Profile()
    if args.device == "cuda":
        torch.cuda.synchronize()
    clock = time.perf_counter()
    profiler.enable()
    for slot in range(args.slots + int(c["lifetime_s"]/c["dt_s"])+3):
        reward, records, context, glob = env.step(policy, arrivals=env.slot < args.slots)
        if args.mode == "training" and slot < args.slots:
            policy.remember(env.agent_rewards(cfg["training"].get("local_credit_fraction", 0.)), records, False)
            if (slot+1) % cfg["training"]["rollout_steps"] == 0:
                context, glob, *_ = env.observations()
                policy.set_progress((slot+1)//cfg["training"]["rollout_steps"], 2000)
                policy.update(context, glob)
        if env.slot >= args.slots and not env.tasks:
            break
    else:
        raise RuntimeError("Benchmark cohort did not drain")
    profiler.disable()
    if args.device == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter()-clock
    def canonical(value):
        if isinstance(value, torch.Tensor):
            return value.detach().cpu().numpy().tolist()
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, dict):
            return {str(k): canonical(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [canonical(v) for v in value]
        return value
    payload = dict(metrics=env.metrics(), tasks=env.rows, batches=env.batch_rows, trace=env.arrival_hash)
    if args.mode == "training":
        payload["trained_policy"] = canonical(policy.state(for_evaluation=True))
    serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
    text = io.StringIO()
    pstats.Stats(profiler, stream=text).strip_dirs().sort_stats("cumulative").print_stats(22)
    result = dict(source=args.source, method=args.method, device=args.device, mode=args.mode, requested_slots=args.slots,
                  measured_seconds=elapsed, cohort_slots=env.slot, raw_digest=hashlib.sha256(serialized.encode()).hexdigest(),
                  metrics=payload["metrics"], torch=str(torch.__version__), numpy=np.__version__,
                  scope="execution-only short benchmark; not training or formal evaluation")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    args.out.with_suffix(".profile.txt").write_text(text.getvalue(), encoding="utf-8")
    print(json.dumps(result, indent=2))
    print(text.getvalue())


if __name__ == "__main__":
    main()
