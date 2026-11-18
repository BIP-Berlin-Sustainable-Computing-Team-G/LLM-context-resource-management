# Research questions

Main question (course brief):

> How much energy can context management and KV-cache reuse save in multi-turn
> conversations, and what does it cost in answer quality?

Everything below the questions is a **hypothesis or a calculation**, not a
result. Setup: [`implementation.md`](implementation.md).

## RQ1: Energy growth

How does energy per turn grow with conversation length when the full history
is re-sent, with and without cache reuse?

- **H1:** Without reuse, energy per turn grows about linearly with history
  length, so a whole conversation costs about quadratically more. With
  reuse, energy per turn stays nearly flat.
- **Measured:** energy per turn vs turn index (1–40), full history,
  prompt cache on/off.

## RQ2: Strategies compared

How do context strategies (full history, sliding window, summarised history)
compare in energy and answer quality?

- Strategies: full history, sliding window (last k = 5 turns), batch summary,
  rolling summary.
- **H2:** Truncation and summaries lower energy only when the prompt cache is
  off. With the cache on, a sliding window can cost more prefill than full
  history, because its prefix changes every turn. Quality: windows lose early
  facts; summaries lose details.
- **Measured:** energy per conversation, accuracy on graded turns, per strategy
  × cache on/off.

## RQ3: Best combination

Which combination of context strategy and KV-cache reuse gives the best
trade-off between energy efficiency and answer quality?

- **H3:** No single combination wins at every conversation length. Full
  history with cache wins for short and medium chats; a summary with cache
  wins only for long ones.
- **Measured:** energy per correct answer per combination, and an
  energy–accuracy Pareto plot for each conversation length (10, 20, 40 turns).

## RQ4: Summarisation break-even

What is the net energy cost of summarisation, and after how many turns does it
break even with prefix-cached full history?

- **H4:** Each summary is extra decode work, and a rewritten summary breaks the
  cached prefix. So summarisation costs more than it saves in short chats and
  pays off only after a break-even length T\*. A rolling summary reaches T\*
  later than a batch summary, or never.
- **Measured:** cumulative energy `E(T) = Σ E_turn + Σ E_summary` for each
  summary strategy vs cached full history; T\* is where the curves cross.

---

## Comments

**Why these four.** RQ1 and RQ2 are taken from the brief almost word for word.
RQ3 combines them into one answer to the main question. RQ4 isolates the one
strategy that spends energy to save energy. The brief's third question
(EcoLogits vs API caching) is dropped: we run no API, and we measure energy
directly.

**RQ1.** Back-of-envelope (calculation): with a 500-token system prompt and 100
user + 300 answer tokens per turn, 40 turns mean 336k prefilled tokens without
reuse and 16k with reuse. Energy will not drop by the same factor (Fernandez
et al.): decode work stays the same, and short prefills do not use the chip
fully. The interesting result is how big the gap really is.

**RQ2.** Here "summarised history" is two strategies: batch (summarise once the
prompt passes a token budget) and rolling (update the summary every turn).
They break the cache at different rates, and that is the point of RQ4. We fix
k = 5 and treat other k values as a limitation.

**RQ3.** "Best" needs one number, so we use **energy per correct answer**. The
Pareto plot shows trade-offs that one number hides. Liu et al. suggest that
full history is not automatically the most accurate, which is why quality is
measured rather than assumed.

**RQ4.** Summary calls are part of the strategy's energy (CLAUDE.md §6).
Without them, summarisation would always look cheap.

**Hybrid model.** Qwen3.5-4B mixes linear-attention layers with a few
full-attention layers (see `implementation.md`). Its cache is mostly a
fixed-size state, not a growing KV cache. This may make reuse work
differently from the papers' pure-transformer models. If the pilot shows
that, it becomes a finding we discuss, not a reason to change the RQs.
