# Study guide

What every team member must be able to explain without notes. Course
material (CodeCarbon, workshops) goes into the last section once we have the
slides.

## Definitions

| Term | Meaning |
|---|---|
| Token | Unit of text the model reads and writes (≈ ¾ of an English word). |
| Context window | Maximum number of tokens the model can see in one request. |
| Stateless model | The model keeps nothing between requests. A chat "remembers" only because the client sends the history again. |
| Prefill | Processing all prompt tokens in parallel before the first output token. Compute-bound. Sets the time to first token (TTFT). |
| Decode | Generating output one token at a time. Memory-bandwidth-bound: the weights are read once per token. |
| KV cache | Keys and values of every processed token in every attention layer. They are kept, so the next token does not recompute them. |
| Prefix / prompt caching | Reusing the KV cache of a previous request when the new prompt starts with exactly the same tokens. Only the new suffix is prefilled. |
| Cache hit / miss | Hit: the prefix is reused. Miss: it is recomputed. One changed token early in the prompt makes everything after it a miss. |
| Slot (llama-server) | One conversation's cache on the server. llama-server compares the new prompt with the slot's previous one and reuses the common prefix. |
| PagedAttention (vLLM) | KV cache stored in fixed blocks, like OS memory pages. Little memory is wasted, and blocks can be shared between requests (Kwon et al.). |
| Linear attention / Gated DeltaNet | An attention variant that keeps a fixed-size state instead of a growing KV cache. Cost per token does not grow with context. Used in 24 of Qwen3.5-4B's 32 layers. |
| Hybrid model | Mixes linear-attention and full-attention layers (Qwen3.5: 3 : 1). |
| Context checkpoint | A saved recurrent state at a given position. A hybrid model can only resume from one of these, not from any token. |
| GQA | Grouped-query attention: several query heads share one KV head, so the KV cache is smaller. |
| Quantisation (Q4_K_M, Q8_0) | Storing weights with fewer bits (≈ 4.5 / 8.5 bits). Smaller and faster, but some quality loss. |
| GGUF | llama.cpp's model file format; it holds the weights and the metadata. |
| Sliding window | Keep only the last k turns. Bounded context, but early facts are lost and the prefix shifts every turn. |
| Batch / rolling summary | Replace old turns with an LLM-written summary, either once over a budget (batch) or after every turn (rolling). |
| Lost in the middle | Models use information at the start and end of the context better than in the middle (Liu et al.). |
| Energy vs power | Power (W) is a rate. Energy (J = W·s) is power over time. We report energy. |
| Idle baseline | Power the machine draws doing nothing. Subtracted so we count only the inference. |
| CodeCarbon | Python library that estimates energy and CO₂ of code. On Apple Silicon it reads powermetrics. |
| powermetrics | macOS tool (sudo) that samples CPU, GPU and ANE power. |
| EcoLogits | Estimates LLM energy from model size and output tokens. Input and cached tokens are not in its formula. |
| Pareto front | The set of options where you cannot improve energy without losing quality, or the other way round. |
| Break-even length T\* | The conversation length where a strategy's extra cost equals its savings. |

## Key mechanics in five lines

1. Every turn re-sends the whole history, so prefilled tokens per turn grow
   linearly and the conversation total grows quadratically.
2. The cache removes prefill only for an identical prefix. Decode is
   unchanged.
3. So energy savings < token savings (Fernandez et al.: decode dominates
   energy).
4. Shortening the prompt (window, summary) changes the prefix and can cancel
   the cache's savings.
5. Fewer tokens can also mean lost information. Quality must be measured.

## Likely exam questions

1. **Why does a chat re-send its history?** The model is stateless. Context
   exists only inside the prompt.
2. **What is in the KV cache, and why can it be reused?** K and V of every past
   token. With causal attention they do not change when tokens are appended.
3. **Prefill vs decode: which one does caching help?** Only prefill, and only
   for the reused prefix.
4. **Why will energy savings be smaller than token savings?** Decode stays
   the same, idle and static power stay, and short prefills under-use the
   chip.
5. **Why can a sliding window cost more than full history?** Its start moves
   every turn, so the prefix never matches and everything is prefilled again.
   Full history only appends.
6. **When does summarisation pay off?** When the prefill it saves on later
   turns exceeds the decode cost of writing the summaries plus the cache
   misses it causes. That point is T\* (RQ4).
7. **How do you measure energy on a Mac? Pitfalls?** powermetrics per turn,
   CodeCarbon per run. Pitfalls: sampling interval, background processes,
   heat, battery and power mode, idle baseline, too few repetitions.
8. **Why llama.cpp and not vLLM, when two papers are about vLLM?** vLLM needs
   CUDA, and we measure on a Mac. The mechanism is the same: reuse of the
   common prefix. vLLM does it per hashed block across requests, llama-server
   per slot.
9. **What is special about Qwen3.5-4B?** It is a hybrid: most layers keep a
   fixed-size state, so the KV cache is small (32 KiB/token), but a
   cache can only resume from checkpoints. We check that reuse works before
   measuring.
10. **Does Q4 quantisation invalidate the results?** Not the comparisons: every
    condition uses the same file. Absolute numbers are not comparable with
    bf16 results in the papers.
11. **How do you grade quality?** Facts planted in early, middle or recent turns
    and asked about later, graded by exact match. A LoCoMo subset as an
    external check.
12. **Why freeze the assistant replies?** So every condition sees identical
    prompts. Cost: the cache saving is a lower bound.
13. **What does "Lost in the Middle" mean for you?** Full history is not
    automatically the best for quality, so we vary where the needed fact is.
14. **What is energy per correct answer?** Total energy ÷ correctly answered
    graded turns. It joins both axes of RQ3 into one number.
15. **Main threats to validity?** One machine, one small model, a single user,
    scripted chats, powermetrics accuracy, frozen history.
16. **What would change on a real server?** Many users share the cache memory.
    Kwon et al. did not keep chat caches between turns because of this.

## From the course

To add: CodeCarbon and workshop slides.
