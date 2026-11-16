# 05 — Oral Exam Prep

Short answers; details in `01-theory.md`. Each team member should be able to
answer all of these without notes.

1. **Why does a chat resend the whole history?** LLMs are stateless; the model
   only "remembers" what is in the prompt of the current request.
2. **What exactly is stored in the KV cache, and why can it be reused?** Keys and
   values of every token in every layer. With causal attention, past tokens'
   K/V never change when new tokens are appended.
3. **How big is the KV cache?** `2 × layers × kv_heads × head_dim × bytes` per
   token; Llama-3-8B ≈ 128 KiB/token → ~1 GiB for 8k tokens.
4. **Prefill vs decode — which is compute-bound, which memory-bound, and which does caching help?**
   Prefill compute-bound, decode memory-bandwidth-bound; prefix caching removes
   prefill work for the reused prefix only.
5. **Why won't token savings translate 1:1 into energy savings?** Decode work is
   unchanged, attention still reads the whole cached context, idle/static power,
   GPU utilisation effects (Fernandez et al.: FLOP-based estimates underestimate energy).
6. **Why does a sliding window break prefix caching?** Reuse needs an identical
   prefix; dropping the oldest turn shifts the start, so every block hash changes.
7. **Why is the KV cache a "limited resource"? What happens when it is full?**
   GPU memory is shared by weights and all sequences; cached prefixes get evicted
   (LRU), fewer concurrent requests fit → lower throughput.
8. **What is PagedAttention?** Block-wise KV memory with block tables (like OS
   paging), reducing fragmentation (60–80% waste → <4%) and enabling block sharing.
9. **What does "Lost in the Middle" mean for our study?** More context ≠ better
   use of context; accuracy is U-shaped over position, so we control fact position.
10. **How will you measure energy? What are the pitfalls?** NVML/nvidia-smi or
    CodeCarbon for GPU, RAPL for CPU; sampling resolution, idle baseline,
    warm-up, repetitions, randomised order.
11. **How does EcoLogits estimate energy, and why might it miss caching?**
    From model size, output tokens and infrastructure assumptions; if input/cached
    tokens are not modelled, caching has no effect on the estimate (to verify).
12. **How do you measure answer quality?** Planted facts at controlled turns +
    a multi-turn benchmark; accuracy (and possibly LLM-as-judge).
13. **Energy vs carbon?** Carbon = energy × grid intensity; we report energy as
    primary because carbon depends on location/time assumptions.
14. **Biggest threats to validity?** Measurement resolution, batch-size-1 setup,
    small models / single GPU type, black-box APIs.
15. **What is new compared to the vLLM-configurations paper?** Multi-turn,
    growing prefixes, interaction of context strategies with caching, and the
    estimation-tool question.
