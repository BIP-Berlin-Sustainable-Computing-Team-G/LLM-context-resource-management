# 02 — Literature

One summary per paper in `notes/papers/`, written by the team member who owns it.

| Paper | Owner | PDF | Summary | Why it matters to us |
|---|---|---|---|---|
| Fernandez et al., ACL 2025 — energy of LLM inference optimisations | Jan | `papers/2025.acl-long.1563.pdf` | [fernandez-2025](papers/fernandez-2025.md) | Prefill vs decode energy; latency ≠ energy; FLOP estimates fall short |
| Liu et al., TACL 2024 — Lost in the Middle | Asia | `papers/2024.tacl-1.9.pdf` | — | More context ≠ better answers; position of the needed fact matters |
| Zine, Coignion et al., arXiv:2607.09172 — vLLM configurations | Lena | `papers/2607.09172v2.pdf` | — | Closest related work: prefix caching energy/accuracy, incl. short multi-turn chat |
| Kwon et al., SOSP 2023 — PagedAttention / vLLM | Valeria | `papers/2309.06180v1.pdf` | — | How KV cache is stored and shared; why it is a limited resource |

## Still to find (verify before citing)

- EcoLogits methodology documentation (for B3).
- CodeCarbon methodology (how it reads GPU/CPU power).
- Multi-turn memory benchmarks for quality → see [datasets.md](datasets.md).
- Context compression / summarisation for chat agents.
- Provider docs on prompt caching (OpenAI, Anthropic, Google).
- Yang et al., "Part-time Power Measurements: Nvidia-smi's Lack of Attention", SC24 (cited by Zine et al.) — for our measurement setup.
