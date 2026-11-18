# Open questions and decisions

Record a decision with its date. If it is still open, say what is blocking it.

## Decided (2026-11-17)

- **Hardware.** Server with 2 × RTX 4080 (16 GB each) and 32 GB RAM for all
  measurements. Mac (Apple Silicon, 48 GB) for harness development only.
- **Engine.** vLLM on the server. Two of the four required papers are about
  vLLM, so the exam will focus on it, and its prefix caching is what we study.
  Prefix caching on vs off is run as two server configurations (containers).
- **Models.** One small family, 3–8B parameters. 3–4B fits one 4080 in bf16
  (Qwen3-4B, Llama-3.2-3B or Gemma 3 4B); 8B only quantised. One GPU per run.
- **Energy tool.** CodeCarbon, cross-checked with direct NVML readings.
- **Benchmark.** Our own conversation set (see `03-experiment-plan.md`). Jan's
  task types (text generation, code, tokenization, reasoning) provide the
  workload; facts planted in earlier turns and asked about later provide the
  quality measure.
- **API budget.** Zero. Only free tiers (Google AI Studio, GroqCloud), or B3 is
  answered from documentation.
- **Poster.** Printed by the course staff.
- **Poster ownership.** Jan: research questions, method, experiment design.
  Asia, Lena, Valeria: takeaways from their papers. See `04-poster-plan.md`.

## Checked in the docs (2026-11-17)

- **EcoLogits and cached tokens.** Its formula has no input or cached tokens
  (`01-theory.md` §9). RQ8 is now about the size of the error (E3).
- **vLLM switch.** `--enable-prefix-caching` / `--no-enable-prefix-caching`;
  only full blocks are cached; LRU eviction. The default differs between
  versions, so we always set the flag.
- **Qwen3-4B KV size.** 144 KiB per token → ≈ 39k tokens of KV on one 4080
  (calculation, `01-theory.md` §3).

## For the team to decide (before the poster, Thursday)

- **Model.** Proposal: Qwen3-4B (not gated, used by Zine et al.), Llama-3.2-3B
  as fallback.
- **Quality set.** Proposal: our own planted-fact conversations as the main set,
  a LoCoMo subset as external check (`03-experiment-plan.md`).
- **Poster.** Decided: standard A0 portrait, made in Canva (`04-poster-plan.md`).

## Still open (Phase 2, on the server)

- **NVML on the RTX 4080.** Check that the energy counter works (not only
  instantaneous power) and at what resolution.
- **vLLM version.** Pin it, run `vllm serve --help` to see the prefix-caching
  default, read the real KV capacity from the start-up log.
- **Free API for RQ8 (optional).** A provider that reports cached tokens and is
  supported by EcoLogits (Google AI Studio, GroqCloud); check free-tier limits.
- **LoCoMo licence and category names (1–5).**
