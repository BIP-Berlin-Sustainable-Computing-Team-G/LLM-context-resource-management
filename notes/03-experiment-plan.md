# 03 — Experiment Plan (draft)

Status: **draft for the poster** (Phase 1). Becomes the binding protocol for
Phase 2. RQ/H numbers refer to [`chosen-research-questions.md`](chosen-research-questions.md).
How it runs on the server (containers, fallbacks, timeline):
[`06-implementation-plan.md`](06-implementation-plan.md).

## Energy experiment canvas (course template)

| Field | Our answer |
|---|---|
| Research question | RQ1–RQ8 with H1–H8 |
| System under test | vLLM (pinned version) serving Qwen3-4B in bf16 on one RTX 4080 16 GB; prefix caching on vs off |
| Workload / input set | Our scripted conversations (1–40 turns, planted facts, mixed task types); LoCoMo subset as external check |
| Energy metric | GPU energy per turn and per conversation (J), idle-subtracted; CPU energy only if RAPL is readable |
| Further metrics | Prefill tokens computed vs cached, TTFT, decode time, KV usage, accuracy on graded turns, energy per correct answer |
| Varied parameters | Prefix caching on/off; conversation length; context strategy; fact position; (stretch) number of concurrent users |
| Fixed parameters | Model and revision, vLLM version and flags, GPU, greedy decoding, `max_tokens`, seed, system prompt |
| Source of energy data | CodeCarbon, cross-checked with the NVML energy counter; EcoLogits as the estimate we compare against |
| Repetitions and run order | ≥ 5 per energy condition, randomised order, warm-up discarded, cache reset between runs, idle baseline before each block |
| Threats to validity | See the end of this note |

## Overview

| Exp | Answers | Design |
|---|---|---|
| E1 | RQ1, RQ2 | full history; conversation length × cache on/off |
| E2 | RQ3–RQ6 | 5 context strategies × cache on/off × fact position |
| E3 | RQ8 | EcoLogits formula applied to every E1/E2 turn vs measurement (no extra runs) |
| E4 (stretch) | RQ7 | number of concurrent users × cache on/off |

All experiments use one **conversation harness**. It replays scripted
conversations turn by turn against vLLM's OpenAI-compatible server, applies the
context strategy before each request, and logs per-turn tokens, cached tokens,
timings and energy.

## System under test

- **Engine: vLLM**, pinned version, OpenAI-compatible server. Two of our four
  papers are about it, and its prefix caching is the mechanism we study.
- **Only two server configurations**, differing in one flag:
  `--enable-prefix-caching` vs `--no-enable-prefix-caching`. The default
  differs between versions, so the flag is always set explicitly. All other
  flags are identical and recorded (Zine et al.: options can even change accuracy).
- **Context strategies live in the harness, not in the server.** The server
  only sees the prompt it gets. So each strategy runs against both server
  configurations: a full factorial, strategy × caching. This contains every
  "isolated vs combined" comparison, e.g. sliding window without cache vs
  chunked window with cache.
- **One GPU per run**, no tensor parallelism (Fernandez et al.: 2 GPUs cut
  latency by 40 % but raise energy by 29 %). The two 4080s run independent
  runs in parallel; the GPU is recorded and balanced across conditions.
- **Model:** Qwen3-4B in bf16 (proposed: fits one 16 GB card, also used by
  Zine et al., not gated). KV cache 144 KiB per token → ≈ 39k tokens fit next
  to the weights (calculation, `01-theory.md` §3). Fallback: Llama-3.2-3B.
- **Mac:** harness development, unit tests, dry runs against a fake server. No
  reported numbers come from the Mac.

## Context strategies (E2)

S = system prompt, k = 5 turns (one turn = one user message + one answer).

| # | Strategy | Prompt at each turn | Prefix stable? | Extra LLM calls |
|---|---|---|---|---|
| 1 | Full history | S + all turns | yes, append-only | none |
| 2 | Sliding window | S + last k turns | no, shifts every turn | none |
| 3 | Chunked window (cache-aware, ours) | S + last k…2k−1 turns; the oldest k are dropped at once | breaks once every k turns | none |
| 4 | Batch summary | when the prompt exceeds a token budget: S + summary of older turns + last k turns | breaks at each new summary, stable in between | one summary call per threshold |
| 5 | Rolling summary | S + summary updated after every turn + last turn | no, rewritten every turn | one summary call per turn |

- Summaries are written by the **same model on the same server**, with a fixed
  summary prompt and a token limit. Their energy is part of the strategy's
  energy (CLAUDE.md §6).
- Expected failure modes (to test, not results): full history — memory grows,
  facts in the middle may be missed (Liu et al.); sliding/chunked window —
  early facts are gone; summaries — details and numbers get lost, and errors
  compound in a rolling summary.

## Workload: our own conversation set

Built from Jan's buzz-group task types. The task types set the **workload**;
the history-dependent questions set the **quality** measure.

| Task type | Role | Output length | Graded? |
|---|---|---|---|
| Text generation | workload, decode-heavy | long | no |
| Code writing | workload | medium–long | no |
| Tokenization (e.g. count letters in a word) | workload + quality | very short | yes, exact answer |
| Reasoning (e.g. area of a hexagon) | workload + quality | short–medium | yes, numeric answer |
| **Recall from an earlier turn** | quality | short | yes, exact match |

- Some turns **plant facts** (a name, a number, a constraint); later turns ask
  about them. Following Liu et al., the planted fact sits in:
  - *early* turns (1–3) — retention of initial information,
  - *middle* turns (10–15 in a 30–40-turn chat) — the U-shape's weak spot,
  - *recent* turns (last 2–3) — recency.
- General-knowledge questions are not graded: they do not depend on context.
- **History replay.** The assistant replies stored in the history are fixed:
  generated once with Qwen3-4B (full history, greedy) and frozen in the
  conversation file. In every run the model generates an answer to each turn
  (that is what we measure), but the frozen reply goes into the history. So
  every condition sees exactly the same prompts, and only the strategy and
  the cache setting differ. Consequence: the KV of the model's own answer is
  never reused, so the cache saving is a lower bound (limitation).
- External check: a subset of **LoCoMo** (`datasets.md`), for quality only. Its
  questions list the turns that hold the answer, so we can see whether a
  strategy still had the needed turn.

## Common settings

- Greedy decoding (`temperature=0`), fixed seed.
- **Energy runs:** fixed output length per turn (`max_tokens` with
  `ignore_eos`), so decode work is identical across conditions.
- **Quality runs:** natural answers (stop at end of answer), graded turns only.
  Greedy decoding on a single request is close to deterministic, so 2
  repetitions are enough to check this; energy needs ≥ 5.
- Conversation lengths T ∈ {1, 5, 10, 20, 40} turns.
- Randomised order, warm-up discarded, prefix cache reset between runs.
- Single user (one request at a time) for E1–E3. Concurrency only in E4.

## Metrics

| Metric | Source |
|---|---|
| Energy per turn and per conversation (J), idle-subtracted | NVML energy counter read before/after each request; CodeCarbon per run as cross-check |
| Prefill tokens computed vs cached | vLLM `/metrics`: `vllm:prefix_cache_queries`, `vllm:prefix_cache_hits` (deltas per request) |
| TTFT, decode time | harness timestamps (streaming), `vllm:time_to_first_token_seconds` |
| KV-cache usage, preemptions | `vllm:kv_cache_usage_perc`, `vllm:num_preemptions` |
| Answer quality | accuracy on graded turns; for LoCoMo token-F1 |
| Energy per correct answer | derived |
| EcoLogits estimate | formula from its methodology page, applied per turn |

## E1 — Energy growth with conversation length (RQ1, RQ2)

- Full history, cache off vs on, T up to 40, two answer lengths (short ≈ 50,
  long ≈ 300 tokens).
- Plots: energy per turn vs turn; cumulative energy vs T; energy saving ÷
  token saving vs T (break-even = where the CIs separate).

## E2 — Context strategies and break-even (RQ3–RQ6)

- 5 strategies × cache off/on × fact position, T ∈ {10, 20, 40}.
- RQ3/RQ5: energy–quality Pareto plot per T; ranking by energy per correct answer.
- RQ4: `E_total(T) = Σ E_turn(t) + Σ E_summary` per summary strategy vs full
  history with cache; T\* = where the curves cross.
- RQ6: accuracy by fact position × strategy.

## E3 — EcoLogits vs measurement (RQ8)

- For every E1/E2 turn: EcoLogits estimate (model size, output tokens) vs our
  measured energy. Plot measured ÷ estimated vs turn index, cache on/off.
- Optional: if a free API (Google AI Studio, GroqCloud) reports cached tokens
  and is supported by EcoLogits, show that its cached-token count changes
  between a stable and a perturbed prefix while the estimate does not.

## E4 (stretch) — Concurrent users (RQ7)

- 1, 2, 4, 8 simulated users with think-time between turns, one vLLM server.
- Energy per request, cache hit rate, evictions and preemptions as N grows.

## Analysis

- Medians + IQR or bootstrap CIs; ART ANOVA, Cliff's δ, Holm correction (as
  Zine et al.).
- Energy is the primary metric; CO₂ only as a derived number with a stated grid
  intensity.

## Threats to validity

- Power-sampling resolution: `nvidia-smi` on A100 samples ≈ 25 % of the time
  (Zine et al.) → per-turn energy from the NVML energy counter, runs long
  enough, very short turns aggregated.
- Shared server: other users' jobs can add power → foreign GPU processes are
  logged and such runs discarded (`06-implementation-plan.md`).
- Single user, one consumer GPU type, small model: not a production server.
- History replay never reuses the KV of the model's own answers → lower bound.
- Scripted conversations are controlled but not natural chat.
- APIs are a black box: no ground-truth energy for RQ8.
