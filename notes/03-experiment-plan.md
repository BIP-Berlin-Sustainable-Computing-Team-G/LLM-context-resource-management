# 03 — Experiment Plan (draft)

Status: **draft for the poster** (Phase 1). Becomes the binding protocol for
Phase 2. Question IDs (B1, F1, O1, …) refer to `research-questions.md`.

## Overview

| Exp | Answers | Main factor(s) |
|---|---|---|
| E1 | B1, F1, F2, F3 | conversation length × prefix caching on/off (full history) |
| E2 | B2, O1, O2, O4 | context strategy (full history, sliding window, summary, rolling summary) × caching on/off |
| E3 | B3 | API caching vs EcoLogits estimate (free tier or documentation only) |
| E4 (stretch) | F4 | number of concurrent users × caching on/off |

All experiments use one **conversation harness**. It replays scripted multi-turn
conversations turn by turn against vLLM's OpenAI-compatible server, applies a
context strategy before each call, and logs per-turn tokens, cached tokens,
timings and energy.

## System under test & container architecture

- **Engine: vLLM** on the server (2 × RTX 4080 16 GB, CUDA). We chose vLLM
  because two of our four papers are about it (PagedAttention, vLLM
  configurations), and because its prefix caching is the mechanism we study.
- **One GPU per run**, no tensor parallelism. Fernandez et al. show tensor
  parallelism raises total energy. The second GPU can run a separate
  experiment, but then CPU energy cannot be attributed to either run.
- **Models:** one family, small instruct models that fit one 16 GB GPU in bf16.
  Candidates are Qwen3-4B or Llama-3.2-3B (same as Zine et al., so we can
  compare) or Gemma 3 4B. A quantised ~8B model is optional as a second size.
- **Mac (Apple Silicon):** harness development, unit tests and dry runs only.
  No reported numbers come from the Mac: Docker there has no GPU access, and
  the engine would differ.

### Container deployment & ablation matrix

To ensure rigorous reproducibility and eliminate environment drift, vLLM instances
are deployed as isolated Docker containers monitored via CodeCarbon and NVML.

1. **Core 2-container baseline comparison:**
   - **Container 0 (Unoptimized Baseline):** vLLM running without prefix caching
     (`--no-enable-prefix-caching`). Naive full recomputation on every turn.
   - **Container 1 (Optimized LLM):** vLLM running with automatic prefix caching
     (`--enable-prefix-caching`). Reuses KV-cache blocks across turns for stable prefixes.

2. **Multi-container ablation setup (isolated vs. combined optimizations):**
   To dissect how much energy each individual mechanism saves vs. costs, we extend
   to an ablation suite:
   - **Container A (Prefix Caching only):** Prefix caching ON, full history sent.
   - **Container B (Sliding Window Context):** Fixed sliding window (last 5 messages),
     caching OFF. Demonstrates pure token-truncation savings without cache reuse.
   - **Container C (Context Summarization):** Periodic / rolling summary manager,
     caching OFF. Demonstrates context compression with summarisation energy tax.
   - **Container D ("Mega-optimized" / Combined):** Prefix caching ON combined with
     a cache-aware context manager (e.g. chunked sliding window or append-only
     rolling summary that avoids invalidating KV-cache prefix blocks).
   - *Goal:* Contrast the combined optimization against each isolated option to
     reveal synergies, overheads, and trade-offs.

## Workload: our own conversation set

Built from Jan's buzz-group task types. The task types define the
**workload**; the history-dependent questions define the **quality** measure.

| Task type | Role | Output length | Graded? |
|---|---|---|---|
| Text generation | workload, decode-heavy | long | no |
| Code writing | workload | medium–long | no (code quality depends on the model, not the runtime) |
| Tokenization (e.g. count letters in a word) | workload + quality | very short | yes, exact answer |
| Reasoning (e.g. area of a hexagon) | workload + quality | short–medium | yes, numeric answer |
| **Recall from an earlier turn** | quality | short | yes, exact match |

- Each conversation mixes these turns. Some early turns **plant facts** (a name,
  a number, a constraint), and later turns ask about them.
- **Positioning following Liu et al. (Lost in the Middle):** The position of the
  planted fact is varied systematically across turns:
  - *Early turns* (turns 1–3) — tests retention of initial instructions.
  - *Middle turns* (turns 10–15) — directly targets the U-shaped degradation blind spot.
  - *Recent turns* (last 2–3 turns) — tests recency bias and short-term window retention.
- General-knowledge questions (answered from training data) are left out of
  the quality score, because they do not depend on the context.
- The mix of output lengths matters for energy. Decode-heavy turns should
  reduce the relative saving from caching (F1).
- Optional: one existing multi-turn benchmark as an external check. [pick one,
  check licence and size]

## Common settings

- Greedy decoding (`temperature=0`), fixed seed, fixed `max_tokens`. For
  energy-only sweeps use a fixed output length so decode work is constant.
- Conversation lengths T ∈ {1, 5, 10, 20, 40} turns, up to the context limit.
- At least 5 repetitions per condition, randomised order, warm-up discarded,
  cache reset between runs.
- Single user (one request at a time) for E1–E3. Concurrency only in E4.

## Metrics

| Metric | Source |
|---|---|
| Energy per turn and per conversation (J), idle-subtracted | CodeCarbon, cross-checked with NVML energy counter [check support on RTX 4080] |
| Prefill tokens computed vs cached | vLLM metrics (prefix-cache queries/hits), usage fields |
| TTFT, decode time, total latency | harness timestamps, vLLM metrics |
| KV-cache usage, preemptions | vLLM metrics |
| Answer quality | accuracy on graded turns (recall, tokenization, reasoning) |
| Energy per correct answer | derived (O4) |

## E1 — Energy growth with conversation length

- Full history, prefix caching off vs on (Container 0 vs Container 1).
- Plots: energy per turn vs turn index; cumulative energy vs T; prefill vs
  decode share per turn.
- Hypotheses: without caching, per-turn energy rises with history length; with
  caching it stays flatter. The relative energy saving stays well below the
  relative token saving (F1). The curves separate only above some history
  length (F2). The TTFT gain is larger than the energy gain (F3).

## E2 — Context strategies & Break-Even Analysis

Evaluates the 4 primary context strategies under both caching regimes:
1. **Full history:** Naive baseline. Re-sends complete transcript every turn.
   - *Hypothesis:* Cache-friendly (stable prefix), but memory grows linearly and
     middle-placed facts suffer severe degradation (Liu et al.).
2. **Sliding window (last 5 messages):** Strict cut-off discarding older messages.
   - *Hypothesis:* Low, constant memory; avoids middle-context degradation for recent
     facts, but amnesic to early planted facts and destroys prefix caching (100% cache miss).
3. **Summarised history (one-time batch):** Compresses dialogue up to turn $t$
   into a single narrative summary upon hitting a token budget.
   - *Hypothesis:* Bounded prompt length, but causes an energy/latency spike during
     summarisation and invalidates subsequent KV cache.
4. **Rolling summary (incremental):** Continuously merges previous summary with
     the latest exchange.
   - *Hypothesis:* Continuous decode energy overhead; causes abstraction drift.

### Core Research Question for E2 (O2):
> **What is the net energy overhead of context summarization, and at what conversation
> length does it reach a break-even point compared to KV-cache reuse?**
> *(PL: Jaki jest narzut energetyczny ciągłego streszczania kontekstu i przy jakiej
> długości rozmowy osiąga ono próg opłacalności w porównaniu z ponownym użyciem pamięci
> podręcznej KV?)*

- **Break-even methodology:**
  - Track total cumulative energy: $E_{\text{total}}(T) = \sum_{t=1}^T E_{\text{inference}}(t) + \sum E_{\text{summarise}}$.
  - In short conversations ($T < T^*$), the extra decode energy required by summarisation
    calls plus the loss of prefix caching will make summarisation consume **more net energy**
    than uncompressed prefix-cached full history.
  - Determine empirically at what turn length $T^*$ the curves cross (break-even point).
- Output: energy–quality Pareto plot per conversation length; ranking by energy
  per correct answer (O4).

## E3 — API caching vs estimation tools (zero budget)

- If a free provider (Google AI Studio, GroqCloud) reports cached tokens and is
  supported by EcoLogits: run the same conversations with a stable prefix vs a
  perturbed prefix, and compare the reported cached tokens with the EcoLogits
  estimate.
- Otherwise: a documentation analysis of how EcoLogits models input and cached
  tokens, plus a comparison of its estimate for our local model with our
  measured energy, cache on vs off.

## E4 (stretch) — Concurrent users

- N simulated users with think-time between turns, against one vLLM server.
- Measure energy per request, cache hit rate, evictions and preemptions as N
  grows. The expectation is that the saving collapses once KV memory is full
  (Kwon et al. did not keep chat caches between rounds for this reason).

## Analysis

- Medians + IQR or bootstrap CIs; ART ANOVA and Cliff's δ with Holm correction,
  as in Zine et al.
- Report energy, not only CO₂, since carbon depends on grid assumptions.

## Threats to validity

- Power-sampling resolution (Zine et al.: `nvidia-smi` on A100 can miss short
  spikes). Very short turns are measured badly.
- A single user does not reflect production batching (E4 partly addresses this).
- One GPU type (consumer RTX 4080), small models only.
- Our own conversation set: controlled, but not natural chat.
- API: no ground-truth energy; provider caching is a black box.

## Phase 2 timeline (rough)

| When | Milestone |
|---|---|
| Late Nov | vLLM running on the server; energy measured for one prompt; NVML/CodeCarbon checked on the 4080 |
| Early Dec | Harness + context strategies + unit tests (developed on the Mac); conversation set v1 |
| Mid Dec | E1 runs; E2 pilot; free-API check for E3 |
| Early Jan | E2 + E3 runs (E4 if time allows); analysis |
| Mid/late Jan | Report (8–10 pages) + final presentation |
