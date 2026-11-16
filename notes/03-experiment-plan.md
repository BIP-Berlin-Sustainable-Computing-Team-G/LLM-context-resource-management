# 03 — Experiment Plan (draft)

Status: **draft for the poster** (Phase 1). Becomes the binding protocol for
Phase 2 once hardware and model choices are settled (see `open-questions.md`).

## Overview

| Exp | Answers | Setting | Main factor(s) |
|---|---|---|---|
| E1 | RQ1 | Local, vLLM, full history | conversation length × prefix caching on/off |
| E2 | RQ2 | Local, vLLM | context strategy × prefix caching on/off |
| E3 | RQ3 | Commercial API + EcoLogits (+ local calibration) | prompt caching on/off; estimated vs measured |

Shared building block: a **conversation harness** that replays scripted
multi-turn conversations turn by turn against an OpenAI-compatible endpoint
(vLLM serves one; most APIs do too), applies a context strategy before each
call, and logs per-turn tokens, timings, cache hits and energy.

## Common setup

- **Models:** 1–2 open-weight instruct models of different sizes (e.g. ~3B and
  ~8B) — size depends on available GPU.
- **Engine:** vLLM with `enable_prefix_caching` on/off (the on/off switch is the
  cleanest "with vs without cache reuse" manipulation we have).
- **Decoding:** `temperature=0`, fixed `max_tokens`; for energy-only runs consider
  fixed output length (e.g. `ignore_eos`) so decode work is constant across conditions.
- **Conversations:** two sources
  1. *Synthetic, controlled* — scripted user turns; specific facts planted at
     turn *i*, asked about at turn *j* (controls information position, cf. Lost in the Middle).
  2. *Realistic* — an existing multi-turn dataset/benchmark (candidate list in `02-literature.md`).
- **Lengths:** T ∈ {1, 5, 10, 20, 40} turns (up to the model's context limit).
- **Repetitions:** ≥5 per condition, randomised order, warm-up runs discarded.
- **Isolation:** one request at a time (batch size 1) for the main experiments,
  so per-request energy is attributable; optionally a concurrent-load variant later.

## Metrics

| Metric | Source |
|---|---|
| Energy per turn and per conversation (J), idle-subtracted | NVML / CodeCarbon (GPU), RAPL (CPU) |
| Prefill tokens computed vs cached | vLLM metrics (prefix cache hit rate), API usage fields |
| TTFT, decode time, total latency | harness timestamps / vLLM metrics |
| Peak KV-cache usage | vLLM metrics |
| Answer quality | accuracy on planted-fact questions; benchmark score; optionally LLM-as-judge |

## E1 — Energy growth with conversation length (RQ1)

- Full history, prefix caching **off** vs **on**.
- Plot: energy per turn vs turn index; cumulative energy vs T.
- Expected shape (hypothesis): off → per-turn energy rises ~linearly; on → much
  flatter. Fit simple models (linear vs quadratic) to cumulative energy.
- Also split energy into prefill vs decode share (via timing) to explain the gap
  between token savings and energy savings.

## E2 — Context strategies (RQ2)

- Strategies: full history · sliding window (k last turns, k ∈ {2, 5}) ·
  summarised history (summary refreshed every turn vs every k turns) ·
  optional cache-friendly hybrid.
- Each × prefix caching on/off.
- Summarisation calls are part of the measured energy.
- Output: **energy–quality Pareto plot** per conversation length.

## E3 — API caching vs estimation tools (RQ3)

1. Run the same conversations against one commercial API with prompt caching
   active vs disabled/avoided (e.g. by perturbing the prefix). Record reported
   cached tokens, latency, cost.
2. Compute EcoLogits estimates for both conditions. Does the estimate change?
3. **Calibration:** for an open model run locally, compare EcoLogits' estimate
   for that model size with our measured energy, with and without prefix caching.
- Deliverable: table "reported cached tokens / billed cost / EcoLogits energy /
  measured energy (local)".

## Analysis

- Medians + IQR / bootstrap CIs; non-parametric tests (Mann–Whitney / ART ANOVA)
  and effect sizes (Cliff's delta), as in the vLLM-configurations paper.
- Report energy, not only CO₂ (carbon depends on grid assumptions).

## Threats to validity (to mention on the poster)

- Power-sampling resolution of `nvidia-smi` (see literature) and short requests.
- Batch size 1 is not how production serves requests (pessimistic utilisation).
- Single hardware platform; small models only.
- API: no ground-truth energy; provider caching is a black box.
- Quality metric validity (synthetic facts vs real conversations; LLM-judge bias).

## Phase 2 development plan (rough, to refine)

| When | Milestone |
|---|---|
| Late Nov | Hardware access confirmed; vLLM + energy measurement working on one prompt |
| Early Dec | Conversation harness + context strategies + unit tests; synthetic dataset |
| Mid Dec | E1 runs; pilot E2 |
| Early Jan | E2 + E3 runs; analysis |
| Mid/late Jan | Report (8–10 pages) + final presentation |
