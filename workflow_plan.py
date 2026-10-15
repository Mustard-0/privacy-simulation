"""Bound the aggregate timeout allowance before dispatching a matrix."""
import json
import os
from pathlib import Path

import cloud


def budget(stage, count, minutes):
    if not isinstance(minutes, int) or minutes < 300 or minutes > 2000:
        raise ValueError("This workflow permits 300..2000 aggregate minutes; no unbounded paid grid")
    # 5 plan + 20 review minutes, and each matrix job's hard timeout, are all included.
    timeout = min(290, (minutes - 25) // count)
    if timeout < 20:
        raise ValueError("Insufficient dispatch allowance")
    seconds = max(60, (timeout - 15) * 60)
    return dict(timeout=timeout, seconds=seconds, maximum_minutes=count*timeout+25)


def main():
    if os.environ.get("REPOSITORY_PRIVATE") != "false":
        raise ValueError("Only free public standard-runner execution is permitted")
    for key in ("RESUME_RUN", "GATE_RUN"):
        if os.environ.get(key) and not os.environ[key].isdigit():
            raise ValueError("Run IDs must contain digits only")
    stage = os.environ["STAGE"]
    gate = cloud.verify_gate(Path("review/screen_gate.json")) if stage != "screen" else None
    jobs = cloud.jobs_for(stage, gate)
    allowance = budget(stage, len(jobs), int(os.environ["APPROVED_MINUTES"]))
    output = dict(matrix=json.dumps(dict(include=jobs), separators=(",", ":")),
                  timeout=str(allowance["timeout"]), seconds=str(allowance["seconds"]))
    with Path(os.environ["GITHUB_OUTPUT"]).open("a", encoding="utf-8") as handle:
        for key, value in output.items():
            handle.write(key+"="+value+"\n")
    print(json.dumps(allowance))
    print("Actual usage/storage must be checked in GitHub Billing before dispatch. No billing settings are changed.")


if __name__ == "__main__":
    main()
