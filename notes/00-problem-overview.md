# 00 — Problem Overview

## The problem in one paragraph

LLMs are stateless: in a chat, the **entire conversation history is re-sent with
every new message**. The model must process ("prefill") all of those tokens again
before it can generate the answer. The number of tokens processed per turn
therefore grows with conversation length, and the *total* work over a
conversation grows roughly **quadratically** in the number of turns. The
**key–value (KV) cache** stores the intermediate attention state of already
processed tokens so it can be reused instead of recomputed — but it costs GPU
memory (a scarce resource) and can only be reused if the prompt prefix is
*exactly* identical. Context-management strategies (truncation, summarisation)
reduce the number of tokens but may **break cache reuse** and may **lose
information** the model needs to answer correctly.

So there is a three-way trade-off: **energy ↔ answer quality ↔ memory (cache)**.

## Overarching research question

> How much energy can context management and KV-cache reuse save in multi-turn
> conversations, and what does it cost in answer quality?

## Research questions

We answer 8 research questions, RQ1–RQ8, each with one hypothesis:
[`chosen-research-questions.md`](chosen-research-questions.md). They cover the
three sub-questions of the course brief:

| Brief sub-question | Our RQs |
|---|---|
| 1. How does energy per turn grow with conversation length, with and without cache reuse? | RQ1, RQ2 |
| 2. How do context strategies (full history, sliding window, summarised history) compare in energy and answer quality? | RQ3–RQ6 |
| 3. Do API prompt caching and energy-estimation tools agree: does EcoLogits account for cached tokens? | RQ8 |
| (topic: the KV cache is a limited resource) | RQ7 |

## Why this matters (sustainability angle)

- Inference, not training, dominates the lifetime compute of deployed LLMs
  (the vLLM-configurations paper cites >90% of workload; see `02-literature.md`).
- Chat is the dominant usage pattern, and conversations get long (agents, coding
  assistants, long chats).
- Providers already use prompt caching for **cost**; we ask how much **energy**
  it actually saves and whether users/tools can see that.
- Energy estimation tools are what users and companies use for reporting. If they
  ignore caching, reported footprints are biased.

## Scope (decisions in `open-questions.md`)

- Text-only multi-turn chat, no tool use or RAG. Single user; concurrent users
  only as a stretch goal.
- One small open-weight model family (3–4B) served by **vLLM** on our own
  RTX 4080 GPUs. For RQ3, we apply EcoLogits' formula to our own runs; a free API tier
  only if one reports cached tokens and is supported by EcoLogits.
- Energy = GPU (+ CPU if measurable) energy during inference, idle-subtracted.
  Embodied emissions are out of scope (mentioned as a limitation).
- Quality = accuracy on our own conversation set: recall of facts planted in
  earlier turns, plus exactly gradable tasks (tokenization, reasoning).

## Key terms (glossary)

- **Prefill** — processing all input tokens in parallel to build the KV cache; compute-bound.
- **Decode** — generating output tokens one by one; memory-bandwidth-bound.
- **KV cache** — stored key/value vectors for every processed token in every layer.
- **Prefix caching** (vLLM: *automatic prefix caching*) — reusing KV blocks across
  requests that share an identical token prefix.
- **API prompt caching** — the provider-side equivalent; reported as cached input
  tokens and billed at a discount.
- **TTFT** — time to first token (mostly prefill). **TPOT/ITL** — time per output token (decode).
- **Context strategies** — full history, sliding window (e.g. last 5 messages / $k$ turns),
  summarised history (batch summary at threshold), rolling summary (incremental updates).
