# 02 — Literature

One summary per paper in `notes/papers/`, written by the team member who owns it.

| Paper | Owner | PDF | Summary | Why it matters to us |
|---|---|---|---|---|
| Fernandez et al., ACL 2025 — energy of LLM inference optimisations | Jan | `papers/2025.acl-long.1563.pdf` | [fernandez-2025](papers/fernandez-2025.md) | Prefill vs decode energy; latency ≠ energy; FLOP estimates fall short |
| Liu et al., TACL 2024 — Lost in the Middle | Asia | `papers/2024.tacl-1.9.pdf` | [liu-2024](papers/liu-2024.md) | More context ≠ better answers; position of the needed fact matters; U-shaped performance curve |
| Zine, Coignion et al., arXiv:2607.09172 — vLLM configurations | Lena | `papers/2607.09172v2.pdf` | [zine-2026](papers/zine-2026.md) | Closest related work: prefix caching energy/accuracy, incl. short multi-turn chat |
| Kwon et al., SOSP 2023 — PagedAttention / vLLM | Valeria | `papers/2309.06180v1.pdf` | [kwon-2023](papers/kwon-2023.md) | How KV cache is stored and shared; why it is a limited resource |

## Tool and provider documentation (checked 2026-11-17)

| Source | What we use it for |
|---|---|
| vLLM docs, [prefix caching design](https://docs.vllm.ai/en/latest/design/prefix_caching.html) and [automatic prefix caching](https://docs.vllm.ai/en/latest/features/automatic_prefix_caching.html) | Block hash, full blocks only, LRU eviction, `--enable-prefix-caching` / `--no-enable-prefix-caching` |
| EcoLogits, [LLM inference methodology](https://ecologits.ai/latest/methodology/llm_inference/) | Energy formula behind RQ8 (no input or cached tokens in it) |
| OpenAI, [prompt caching](https://developers.openai.com/api/docs/guides/prompt-caching) | Automatic caching, `cached_tokens` usage field |
| Anthropic, [prompt caching](https://platform.claude.com/docs/en/build-with-claude/prompt-caching) | `cache_control`, 5 min / 1 h TTL, read 0.1× price |
| Google, [context caching](https://aistudio.google.com/docs/caching) | Implicit caching on by default for Gemini 2.5+ |
| Qwen3-4B [`config.json`](https://huggingface.co/Qwen/Qwen3-4B/raw/main/config.json) | Layers / KV heads / head dim for the KV-size calculation |
| Maharana et al., *Evaluating Very Long-Term Conversational Memory of LLM Agents*, arXiv:2402.17753 (LoCoMo) | Quality benchmark candidate, see [datasets.md](datasets.md) |

## Still to find

- CodeCarbon methodology (how it reads GPU/CPU power).
- Yang et al., "Part-time Power Measurements: Nvidia-smi's Lack of Attention", SC24 (cited by Zine et al.) — for our measurement setup.
- Context compression / summarisation for chat agents (one paper is enough, for RQ4).
