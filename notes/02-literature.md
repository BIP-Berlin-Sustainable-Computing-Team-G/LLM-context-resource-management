# 02 — Literature

PDFs are in `papers/`. For each paper: what it says, and what **we** take from it.
Facts below were checked against the PDFs unless marked `[UNVERIFIED]`.

---

## Fernandez et al. — Energy Considerations of LLM Inference and Efficiency Optimizations (ACL 2025)
doi:10.18653/v1/2025.acl-long.1563 · `papers/2025.acl-long.1563.pdf`

**Says**
- Systematic analysis of energy for common inference optimisations across
  workloads incl. **conversational AI** and code generation; covers software
  frameworks, decoding strategies, GPU architectures, online vs offline serving,
  model parallelism.
- Models real workloads by **binning input/output token distributions** and batch sizes.
- **Naive estimates based on FLOPs or theoretical GPU utilisation significantly
  underestimate real energy.**
- Properly applied optimisations reduce total energy by **up to 73%** vs an
  unoptimised baseline (PyTorch vs optimised vLLM).

**We take**
- Motivation: software/serving choices matter a lot for energy.
- Methodological warning: measure, don't just compute tokens/FLOPs.
- Input vs output length must be controlled separately (prefill vs decode).

---

## Kwon et al. — Efficient Memory Management for LLM Serving with PagedAttention (SOSP 2023)
doi:10.1145/3600006.3613165 · `papers/2309.06180v1.pdf`

**Says**
- KV cache is huge and dynamic; existing systems waste **60–80%** of KV memory
  through fragmentation and over-reservation.
- PagedAttention: block-based KV cache with block tables (OS paging analogy);
  waste <4%; **2–4× throughput** at same latency vs FasterTransformer/Orca.
- Enables **sharing** of KV blocks across sequences (parallel sampling, beam
  search, shared prefixes) with copy-on-write.

**We take**
- The technical background for "KV cache is a limited resource" and for how
  prefix reuse works in practice (vLLM = our likely engine).

---

## Liu et al. — Lost in the Middle: How Language Models Use Long Contexts (TACL 2024)
doi:10.1162/tacl_a_00638 · `papers/2024.tacl-1.9.pdf`

**Says**
- Multi-document QA and key–value retrieval tasks: performance is **highest when
  relevant information is at the beginning or end** of the input, and degrades
  significantly in the middle (U-shape).
- Extended-context models are not necessarily better at *using* their context.

**We take**
- Quality is not monotonic in context length → full history is not a free
  quality upper bound; truncation/summarisation could help or hurt.
- Our quality benchmark must vary the **position (turn) of the needed fact**.

---

## Zine, Coignion et al. — Attention to Detail: Evaluating Energy, Performance, and Accuracy Trade-offs Across vLLM Configurations (arXiv:2607.09172)
`papers/2607.09172v2.pdf`

**Says**
- Controlled study of three vLLM options: **attention kernel** (FlashAttention-2,
  FlashAttention-3, FlashInfer), **prefix caching** (on/off), **chunked prefill**
  (on/off); 5 open-weight LLMs × 5 tasks, 9,000 runs.
- Attention type and **prefix caching are the main levers**; chunked prefill has
  limited effect.
- Prefix caching reduces energy in most cases, but **modestly and strongly
  task-dependent**; clearest effect on TTFT; a few cases show increases.
- Enabling prefix caching can change vLLM's KV block allocation path even when
  no prefix is reused → effects even without cache hits.
- **Inference options can change accuracy** (prefix caching significantly altered
  accuracy for six model–benchmark pairs).
- Model choice dominates global trade-offs; config tuning gives local Pareto improvements.
- Method: CPU via `perf`, GPU via `nvidia-smi` at 1 Hz; ART ANOVA + Cliff's delta;
  4× A100-40GB nodes. Limitation noted: `nvidia-smi` on A100 samples only ~25%
  of runtime → up to 65% error on spiky workloads.

**We take**
- Closest related work. Their tasks are mostly single-turn; **our gap: multi-turn
  conversations where the reusable prefix grows every turn**, and the interaction
  with context strategies that break the prefix.
- Reuse their statistical approach (non-parametric, effect sizes) and
  measurement caveats.
- We must check accuracy even for "pure performance" optimisations.

---

## To find / read (candidates — verify before citing)
- EcoLogits methodology documentation (for RQ3). `[TODO]`
- CodeCarbon methodology (how it reads GPU/CPU power). `[TODO]`
- Multi-turn / long-conversation memory benchmarks for quality (e.g. LoCoMo,
  LongMemEval, MT-Bench-101). `[UNVERIFIED — check existence, licence, size]`
- Work on context compression / summarisation for chat agents. `[TODO]`
- Provider docs on prompt caching (OpenAI, Anthropic, Google). `[TODO]`
