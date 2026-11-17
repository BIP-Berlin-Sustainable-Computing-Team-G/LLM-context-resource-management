# 01 — Theory

Goal: every team member can explain each section in 1–2 minutes at the exam.

## 1. Autoregressive generation and self-attention

- A decoder-only transformer generates one token at a time. Each new token is
  conditioned on **all previous tokens**.
- In each layer, every token is projected into a **query** `q`, **key** `k` and
  **value** `v`. Attention for token *i*: `softmax(q_i · K^T / sqrt(d)) · V`,
  where `K`, `V` contain the keys/values of tokens `1..i`.
- Keys and values of past tokens **do not change** when new tokens are appended
  (causal masking). → They can be computed once and stored: the **KV cache**.

## 2. Two phases of inference: prefill vs decode

| | Prefill | Decode |
|---|---|---|
| What | Process the whole prompt (n tokens) at once | Generate output tokens one by one |
| Parallelism | High (matrix–matrix) | Low (matrix–vector per sequence) |
| Bottleneck | **Compute-bound** | **Memory-bandwidth-bound** (weights + KV cache read every step) |
| Cost scaling | Linear layers ~O(n), attention ~O(n²) | Per token ~O(context length) for attention reads |
| Latency metric | TTFT (time to first token) | TPOT / inter-token latency |

Why it matters for us: prompt caching only removes **prefill** work for the
reused prefix. Decode work stays. So energy savings < token savings.

## 3. KV-cache size — worked example

Bytes per token = `2 (K and V) × n_layers × n_kv_heads × head_dim × bytes_per_value`

- Llama-3-8B (32 layers, 8 KV heads via GQA, head_dim 128, FP16 = 2 bytes):
  `2 × 32 × 8 × 128 × 2 = 131,072 B ≈ 128 KiB per token`
  → 8k-token conversation ≈ **1 GiB** of KV cache, per sequence.
- OPT-13B example from the PagedAttention paper: ~800 KB per token; a 2048-token
  sequence ≈ 1.6 GB.
- **Grouped-query attention (GQA)** shares K/V heads across query heads → smaller
  cache. (Exam hook: architecture choices directly change cache memory.)

KV cache is a **limited resource**: GPU memory not used for weights is shared by
all concurrent sequences. Bigger caches → fewer concurrent requests (smaller
batch) → lower throughput → more energy per request. Cached prefixes that are
kept around for reuse compete with active requests and get evicted (vLLM: LRU
eviction of unused blocks).

## 4. Why multi-turn chat is expensive (back-of-envelope)

Let: system prompt `S` tokens, each turn adds a user message `u` and an assistant
answer `a` (`m = u + a`), `T` turns.

Prompt length at turn *t*: `L_t = S + (t−1)·m + u`

- **No cache reuse** — total prefilled tokens over the conversation:
  `Σ L_t = T·S + m·T(T−1)/2 + T·u` → **O(T²)**
- **Perfect prefix reuse** — only new tokens are prefilled each turn:
  `≈ S + u + (T−1)·m` → **O(T)**

Example: `S=500, u=100, a=300, T=20` → 88,000 vs ≈ 8,200 prefilled tokens
(~10.7× fewer). Attention FLOPs inside prefill grow even faster (quadratic per
prompt), so without reuse cumulative attention work is ~cubic in T.

Caveat (important for the poster): this is a **token** argument, not an
**energy** result. Real energy depends on decode share, batch size, hardware
utilisation, and idle power — Fernandez et al. show FLOP-based estimates
underestimate real energy. That gap is exactly what we want to measure.

## 5. PagedAttention and prefix caching (vLLM)

- Old serving systems pre-allocated a contiguous KV region per request for the
  maximum length → internal/external fragmentation; Kwon et al. report 60–80% of
  KV memory wasted in existing systems.
- **PagedAttention**: KV cache split into fixed-size **blocks** (like OS virtual
  memory pages), mapped via a block table; near-zero waste (<4%), 2–4× throughput.
- Blocks can be **shared** between sequences (copy-on-write) → foundation for
  **prefix sharing / automatic prefix caching**: blocks are identified by a hash
  of their tokens *and all preceding tokens*. A new request reuses every block
  whose full prefix matches.
- Consequence: reuse works **only for an identical prefix**. Changing one token
  early in the prompt invalidates all following blocks.

## 6. Context-management strategies and their interaction with caching

| Strategy | Tokens per turn | Prefix stable? | Information loss | Extra cost |
|---|---|---|---|---|
| **1. Full history** | grows linearly | **Yes** (append-only) → cache-friendly | none (but subject to "lost in the middle") | memory footprint grows $O(T)$ |
| **2. Sliding window** (e.g. last 5 messages) | bounded | **No** — start shifts every turn → full cache miss | drops all turns older than window | none |
| **3. Summarised history** (one-time batch) | bounded | **No** — prefix rewrites at compression boundary | lossy abstraction, loss of detail | one-off high decode energy spike |
| **4. Rolling summary** (incremental) | bounded | **No** (unless chunked / append-only) | compounding abstraction drift | continuous decode energy overhead per turn |

### Detailed breakdown of strategies & failure modes:

1. **Full history:** The naive conversational standard. Remembers every word verbatim across all turns.
   - *Problems:* Heavy GPU VRAM consumption ($O(T)$ KV-cache size, risking evictions or out-of-memory under concurrency); severe **"Lost in the Middle"** retrieval degradation for facts in intermediate turns (Liu et al., 2024); quadratic token work and energy waste if prefix caching is inactive.
2. **Sliding window (e.g. last 5 messages / $k$ turns):** Retains only the most recent conversation slice.
   - *Problems:* **Immediate amnesia** for any fact or constraint mentioned prior to the window; **destroys prefix caching** completely because the prompt prefix shifts forward on every turn, causing a 100% cache miss despite processing fewer tokens.
   - *Benefit:* Strictly bounded memory; eliminates middle-context dilution for recent interactions.
3. **Summarised history (one-time batch):** When context reaches a threshold, a single compression prompt condenses all preceding history into a synthesized overview.
   - *Problems:* Large one-time **latency and energy spike** (an entire long transcript is read and decoded into summary tokens); loss of fine details, numbers, or exact quotes; the summarisation prompt itself is vulnerable to "Lost in the Middle".
4. **Rolling summary (incremental):** Continuously merges new incoming dialogue into an evolving summary state.
   - *Problems:* **Compounding information drift and hallucinations** over long horizons; incurs an ongoing **energy tax** on every turn due to extra LLM generation; rewrites the summary prefix each turn, destroying KV-cache reuse unless structured as append-only deltas.

Key insight for the poster: **truncation/summarisation and caching pull in opposite directions.**
Fewer tokens do not guarantee lower energy if those tokens destroy KV-cache reuse. Furthermore, summarisation incurs an extra decode-heavy generation cost. This establishes a **break-even threshold**: summarisation only saves net energy if the conversation becomes long enough that the per-turn prefill savings of a condensed context outweigh the continuous decode energy overhead of generating the summaries.

## 7. Long context ≠ well-used context ("Lost in the Middle")

Liu et al. (2024): model accuracy is highest when relevant information is at the
**beginning or end** of the context and drops when it is in the **middle**
(U-shaped curve), even for long-context models. Key implications:
- Full history is not automatically the quality upper bound; retaining everything
  buries intermediate facts in the model's blind spot.
- Summarisation/truncation might hurt less than expected — or even help — when
  the relevant information would otherwise sit in the middle.
- Our quality test must control *where* (which turn) the needed information is.

## 8. API prompt caching (provider side)

- Providers (e.g. OpenAI, Anthropic, Google) cache prompt prefixes and report
  cached input tokens in the API response usage fields; cached tokens are billed
  at a large discount. Details differ: automatic vs explicit cache markers,
  minimum cacheable length, cache lifetime (minutes), pricing of cache writes.
  `[UNVERIFIED — check and cite current official docs with date before the poster]`
- We cannot measure provider energy directly → we rely on **estimation tools**.

## 9. Measuring and estimating energy

**Measurement (our own hardware):**
- GPU: NVML / `nvidia-smi` power readings, integrated over time. Caveat from the
  vLLM-configurations paper: on A100, `nvidia-smi` samples power for only ~25% of
  the runtime and interpolates → errors up to 65% for spiky workloads. Prefer
  NVML energy counters where available, and long-enough runs.
- CPU: RAPL (Linux, via `perf` or CodeCarbon). Apple Silicon: `powermetrics`.
- **CodeCarbon** (codecarbon.io): Python library wrapping these sources; outputs
  energy (kWh) and CO₂eq using grid carbon intensity.
- Containerised runs + monitoring (idea from the course intro) to compare two
  code variants under identical conditions.
- Always subtract idle baseline and report hardware.

**Estimation (APIs, no hardware access):**
- **EcoLogits**: estimates energy and environmental impacts of API calls from
  model (active parameter count, possibly estimated for closed models), number of
  generated tokens, and infrastructure assumptions (PUE, GPU count, electricity
  mix). Whether/how **input or cached tokens** enter the formula is the core of
  RQ3. `[UNVERIFIED — read EcoLogits methodology page and cite version]`

**Energy → carbon:** `CO₂eq = energy × grid carbon intensity (gCO₂/kWh)`.
Location and time of day matter; we report energy as the primary metric.
