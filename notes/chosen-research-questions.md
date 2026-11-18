# Chosen research questions

The 8 questions this project answers, picked from the candidates in
[`research-questions.md`](research-questions.md) (IDs in brackets). Each has one
hypothesis. **These are hypotheses, not results.** Experiments E1–E4 are
defined in [`03-experiment-plan.md`](03-experiment-plan.md).

> **Main question:** How much energy can context management and KV-cache reuse
> save in multi-turn conversations, and what does it cost in answer quality?

## Overview

| RQ | Short form | Exp | Priority | Brief sub-question |
|---|---|---|---|---|
| RQ1 | Energy growth per turn, cache off vs on | E1 | core | 1 |
| RQ2 | Energy saving vs token saving; break-even length | E1 | core | 1 |
| RQ3 | Context strategies: energy and quality | E2 | core | 2 |
| RQ4 | When does summarisation pay for itself? | E2 | core | 2 |
| RQ5 | Cache-aware (chunked) window | E2 | core | 2 |
| RQ6 | Position of the needed fact × strategy | E2 | core | 2 |
| RQ7 | Concurrent users and the KV memory limit | E4 | stretch | topic ("limited resource") |
| RQ8 | EcoLogits estimate vs measured energy | E3 | from E1/E2 data | 3 |

Why these 8: RQ1–RQ2 measure the energy side, RQ3–RQ6 the energy–quality
trade-off, RQ7–RQ8 the two limits — KV memory on the server and energy
reporting by users. Not chosen: F3 (TTFT vs energy) is logged anyway and goes
to the discussion; O3 (cache-hostile prompts) is predictable; O4 (energy per
correct answer) is the metric for RQ3–RQ6, not a question.

---

## Energy growth (E1)

**RQ1 — How does energy per turn grow with conversation length when the full
history is re-sent, with and without prefix caching?** [B1]
- **H1:** Without caching, energy per turn grows about linearly with the history
  length (conversation total ≈ quadratic). With caching it grows much more slowly.
- Basis: token model in `01-theory.md` §4 (poster Fig. 1); Kwon et al.
- Measured: energy per turn vs turn index, 1–40 turns, cache on/off.

**RQ2 — Is the energy saved by caching smaller than the share of prefill tokens
it skips, and below which history length is it zero?** [F1, F2]
- **H2:** The relative energy saving is much smaller than the relative token
  saving, because decode work stays. Below a break-even history length the
  saving is not measurable, because short prefills under-use the GPU.
- Basis: Fernandez et al. (decode dominates energy; sub-linear prefill energy at
  small inputs, Fig. 2); Zine et al. (≈ 1–2 % saving for Llama-3B on 5-turn WildChat).
- Measured: energy saving ÷ token saving per turn and per conversation, split by
  answer length; the first history length where cache-on and cache-off
  confidence intervals separate.

## Energy vs quality (E2)

**RQ3 — How do the context strategies compare in energy and answer quality,
with and without caching?** [B2]
- Strategies: full history, sliding window, chunked window, batch summary,
  rolling summary (defined in `03-experiment-plan.md`).
- **H3:** With caching on, full history needs the least prefill compute; a
  sliding window holds less context but recomputes more, because its prefix
  shifts every turn. Ranked by **energy per correct answer**, no strategy wins
  at every conversation length.
- Basis: `01-theory.md` §5–6 (calculation: 16k vs 74k prefilled tokens over 40
  turns, poster Fig. 2).
- Measured: energy per conversation, accuracy on graded turns, energy per
  correct answer.

**RQ4 — What is the net energy cost of summarisation, and after how many turns
does it break even with prefix-cached full history?** [O2]
- **H4:** Each summary call is decode-heavy, and a rewritten summary breaks the
  cached prefix. In short conversations summarisation therefore costs more
  energy than it saves; it pays off only beyond a break-even length T\*.
  A rolling summary (rewritten every turn) reaches T\* later than a batch
  summary (rewritten rarely), or never.
- Basis: Fernandez et al. (decode costs more per token than prefill); vLLM
  prefix matching.
- Measured: cumulative energy `E_total(T) = Σ E_turn + Σ E_summary` for each
  summary strategy vs full history with cache; T\* = where the curves cross.

**RQ5 — Does a cache-aware context strategy keep most of the energy saving and
most of the quality?** [O1]
- **H5:** A chunked window (drop the 5 oldest turns at once, every 5 turns)
  needs far less prefill than a sliding window (calculation: 26k vs 74k tokens)
  and holds far less context than full history (4.2k vs 16.2k tokens at turn
  40), so it lies on the energy–quality Pareto front.
- Basis: our own idea, from how vLLM matches cached blocks (Kwon et al.; vLLM docs).
- Measured: as RQ3; Pareto plot of energy vs accuracy.

**RQ6 — How does accuracy depend on which turn holds the needed fact, for each
strategy?** [L1]
- **H6:** Truncation fails on early facts by design. Full history is not the
  upper bound for facts in the middle. 3–4B models rely mostly on recent turns,
  so their overall accuracy loss from truncation is smaller than the share of
  history dropped.
- Basis: Liu et al. (U-shaped accuracy; Llama-2 7B only recency-biased, App. E).
- Measured: accuracy by fact position (early / middle / recent) × strategy ×
  conversation length.

## Limits (E4, E3)

**RQ7 — With several users chatting at once, up to how many users does keeping
their caches still save energy?** [F4]
- **H7:** Only while all active histories fit in KV memory (≈ 39k tokens for
  Qwen3-4B on one 4080 — about two 40-turn chats). Beyond that, evictions and
  preemptions erase the saving.
- Basis: Kwon et al. §6.5 (they did not keep chat caches between rounds for
  this reason); `01-theory.md` §3 (calculation).
- Measured: energy per request, prefix-cache hit rate, preemptions for 1, 2, 4
  and 8 simulated users with think-time between turns.
- Stretch goal: only if E1 and E2 are done by early January.

**RQ8 — How far is the EcoLogits estimate from our measured energy across turns
and cache settings?** [B3]
- **H8:** The EcoLogits formula has no input or cached tokens (methodology page,
  checked 2026-11-17). For the same output length it gives the same estimate at
  turn 1 and turn 40, cache on or off, so measured ÷ estimated changes strongly
  with turn index and cache setting.
- Basis: EcoLogits docs; Fernandez et al. (estimates differ from measurement).
- Measured: the EcoLogits formula applied to every E1/E2 turn vs our measurement.
