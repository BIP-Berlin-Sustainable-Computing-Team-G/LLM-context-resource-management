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
8. **What is PagedAttention?** The KV cache is stored in fixed-size blocks (16
   tokens by default) mapped through a block table, like OS paging. Older
   systems used only 20–38 % of KV memory for real token states; vLLM uses
   96 % (Kwon et al., Fig. 2). Blocks can be shared with copy-on-write.
9. **What does "Lost in the Middle" mean for our study?** More context ≠ better
   use of context; accuracy is U-shaped over position, so we control fact position.
10. **How will you measure energy? What are the pitfalls?** CodeCarbon,
    cross-checked with the NVML energy counter, on an RTX 4080; RAPL for CPU.
    Pitfalls: sampling resolution, idle baseline, warm-up, repetitions,
    randomised order.
11. **How does EcoLogits estimate energy, and why might it miss caching?**
    From model size, output tokens and infrastructure assumptions; if input/cached
    tokens are not modelled, caching has no effect on the estimate (to verify).
12. **How do you measure answer quality?** With our own conversation set.
    Facts are planted at early, middle or recent turns and asked about later,
    and tokenization and reasoning tasks have exact answers. Text and code
    turns are workload only and are not graded.
13. **Energy vs carbon?** Carbon = energy × grid intensity; we report energy as
    primary because carbon depends on location/time assumptions.
14. **Biggest threats to validity?** Measurement resolution, the single-user
    setup, small models on one consumer GPU type, our own (synthetic)
    conversation set, black-box APIs.
15. **What is new compared to the vLLM-configurations paper?** They already
    tested prefix caching on 5-turn WildChat, batched across 200 conversations,
    and found small savings. We look at long conversations turn by turn,
    context strategies that break or keep the prefix, a single user vs. many,
    and quality on questions that depend on the history.

## vLLM (two of the four papers — expect detailed questions)

16. **Why vLLM and not llama.cpp or HF Transformers?** It is the engine our
    papers study, it has automatic prefix caching we can switch on and off, and
    Fernandez et al. show that plain PyTorch/HF wastes a lot of energy compared
    to vLLM.
17. **How does automatic prefix caching find a reusable block?** Each full block
    is identified by a hash of its own tokens plus the hash of everything before
    it. A new request reuses blocks from the start as long as the hashes match;
    the first mismatch ends reuse. [check against current vLLM docs]
18. **What happens to cached blocks when memory is needed?** Blocks no request
    is using stay cached until space is needed, then the least recently used
    ones are evicted. [check docs] Running requests can be preempted: their KV
    is swapped to CPU RAM or recomputed later (Kwon et al. §7.3: recomputation
    overhead never exceeds 20 % of swapping's latency).
19. **Why did Kwon et al. not keep the chat cache between rounds?** It would take
    memory from other requests. That is our trade-off: reuse saves compute but
    holds scarce KV memory.
20. **Can turning on prefix caching change results without any cache hit?**
    Yes. Zine et al. saw an energy drop on LongBench, which has no shared
    prefixes, probably through a different block-allocation path. They also saw
    accuracy shifts in 6 model–benchmark pairs, likely from floating-point
    reduction order under sampling. That is why we use greedy decoding and
    measure quality in every condition.
21. **What is continuous batching and why does it matter for energy?** Finished
    requests leave the batch and new ones join at every iteration, so the GPU
    stays busy. Fernandez et al. measured up to ~5 % energy saving at large
    batch sizes, and overhead at small ones.
22. **Why one GPU per run and not tensor parallelism over both 4080s?**
    Fernandez et al.: going to 2 GPUs cuts latency 40 % but raises energy 29 %.
    A 3–4B model fits one 16 GB card.
