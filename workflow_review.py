"""Do not mistake a successful checkpoint upload for a completed experiment."""
import os
from pathlib import Path

import cloud


def main():
    stage = os.environ["STAGE"]
    out = Path("runs")
    gate = cloud.verify_gate(Path("review/screen_gate.json")) if stage != "screen" else None
    jobs = cloud.jobs_for(stage, gate)
    missing, failed = [], []
    for job in jobs:
        folder = out / stage / job["id"]
        if (folder / "failure.json").exists() or (folder / "RUNNING.lock").exists():
            failed.append(job["id"])
        elif not (folder / "complete.json").exists():
            missing.append(job["id"])
    cloud.checks.save(out / "workflow_status.json", dict(stage=stage, total=len(jobs),
        incomplete=missing, needs_review=failed, all_complete=not missing and not failed))
    if failed:
        raise RuntimeError("Inspect failures/interrupted locks before any recovery: " + str(failed))
    if missing:
        print("Grid incomplete; checkpoints retained. No audit pass or formal gate is declared.")
        return
    result = cloud.review(out, stage, gate)
    if not result["passed"]:
        raise RuntimeError("Mechanism screen failed. Evidence retained; no formal training may start.")


if __name__ == "__main__":
    main()
