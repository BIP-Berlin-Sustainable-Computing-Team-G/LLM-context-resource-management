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

## Still open

- **Final model.** Qwen3-4B / Llama-3.2-3B (comparable with Zine et al.) or
  Gemma 3 4B. Check that vLLM supports it and how big its KV cache per token is.
- **NVML on the RTX 4080.** Check that the energy counter works (not only
  instantaneous power) and at what resolution.
- **Free API for B3.** Find a provider that both reports cached tokens in its
  usage data and is supported by EcoLogits. Check Google AI Studio (Gemma vs
  Gemini models) and GroqCloud, including the limits of the free tier.
- **vLLM defaults.** Check whether prefix caching is on by default in the
  version we pin, and how to switch it off explicitly.
