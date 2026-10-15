# Privacy Simulation Benchmark

Reproducible simulated task routing and conditional batch privacy benchmarks.
No external dataset or private checkpoint is required.

The manual Actions workflow runs four independent standard Ubuntu CPU machines.
It refuses private-repository execution and uses no GPU/larger runner, no Actions
artifact storage, no cache and no paid service. Reproducibility bundles are versioned
project Release assets, below 2 GiB each. Source and simulated outputs are public.

## Protocol

- Three methods, four epsilon budgets (0.5/1/2/4), delta 1e-6; three regions of
  100 users and one training seed, 2026.
- Screen: 13 cells x 30 updates, including the adapted baseline's second LR at
  epsilon 1 only. TMAPO and NoAmp share predeclared LR1, not a completed LR search.
- Formal training: 12 cells x 2000 updates x 128 slots, only after the complete
  screen passes raw-task/batch, service, stability and actual-amplification checks.
- Validation selects the checkpoint, including update 0 if appropriate. Preserve
  unfavorable results, overlaps, failures and pauses. Do not tune on final tests.
- Separate reserved evaluation: five new seeds, 30 s warmup + 300 s measurement
  each, after all formal cells and audits. Short windows are diagnostic only.
- QL is report quality loss, not attack inference error. Accounting is conditional
  on public fixed batch membership, not whole-trajectory privacy. One training
  seed does not establish multi-seed superiority.

## Execution

Use Python 3.11.9, CPU PyTorch 2.5.1 and `requirements.txt`.

```bash
python cloud.py plan
python cloud.py run --stage screen --job TMAPO_M100_eps0.5_LR1_s2026 --device cpu
python cloud.py review --stage screen
python cloud.py plan --stage formal_training --gate runs/screen_gate.json
python -m unittest test_cloud test_archives
python -m unittest discover -s model -p 'test_*.py'
```

Actions defaults: `stage=screen`, `max_parallel=4`, `approved_minutes=600`.
The minute input bounds runtime, not monetary spending. Subsequent chunks use
the same commit and `resume_run`; formal work also requires `gate_run`.
Exact checkpoints retain optimizer, RNG, environment, normalizer, raw rows and
the fixed total LR schedule. Upload success is not experiment completion.
Inspect safety stops, failure evidence and interrupted locks before recovery.

`reference_model/` preserves pre-optimization code. `benchmark.py` compares raw
task/batch and trained-policy digests. Optimizations skip unused deterministic
probabilities and share only immutable observation-time calculations. Routing
remains sequential. New manifests permit kernel patch differences but retain
OS/architecture/libc, Python, NumPy, PyTorch, device, config and code checks.

Do not upload real personal data or credentials. No billing settings are changed.

[Runner specs](https://docs.github.com/en/actions/reference/runners/github-hosted-runners)
| [Billing](https://docs.github.com/en/billing/concepts/product-billing/github-actions)
| [Release assets](https://docs.github.com/en/repositories/releasing-projects-on-github/about-releases)
