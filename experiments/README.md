# Experiments (Phase 2)

Empty until the protocol in [`../notes/03-experiment-plan.md`](../notes/03-experiment-plan.md) is agreed.

Planned layout:
```
experiments/
  harness/        conversation replay + context strategies (with unit tests)
  configs/        one config per experiment/condition
  spikes/         small feasibility checks
  results/raw/    immutable per-run measurements (+ metadata)
  analysis/       notebooks/scripts producing figures from raw data
```
