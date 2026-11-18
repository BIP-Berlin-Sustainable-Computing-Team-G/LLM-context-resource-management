# 04 — Poster Plan

**Deadline:** poster submitted **Thursday 17:30**; poster session + oral exam **Friday**.
**Purpose:** the research problem, theory, research questions and our
experiment/development plan. **No results** — only hypotheses and design.

This note holds **all content of the poster**. The poster itself is made in
**Canva**: one shared design that all four of us can edit. Text is changed here
first, then copied into Canva, so the repository stays the source of truth.

## Format and layout (Canva)

- **Size:** standard A0 portrait (841 × 1189 mm), custom size in Canva.
- **Grid:** title bar across the top, three columns, references along the bottom.

| Column 1 | Column 2 | Column 3 |
|---|---|---|
| 1 Motivation + Fig. 1 | 3 Research questions | 5 Method |
| 2 Background + paper takeaways | 4 The tension + Fig. 2 | 6 Threats · 7 Timeline |

- **Figures:** upload `poster/figures/fig1-token-growth.svg` and
  `fig2-strategies.svg` (SVG stays sharp at A0; PNG at 300 dpi as fallback).
  Regenerate with `python3 poster/figures/make_figures.py`. The method diagram
  (§5) is drawn directly in Canva.
- **Type:** body text ≥ 24 pt, headings ≥ 48 pt, title ≥ 90 pt (readable from
  1.5 m). Two colours for "no cache / cache" used the same way in every figure
  (red / blue, as in the figures).
- **Export:** Canva → Download → *PDF Print* → save as `poster/poster.pdf`.
- Every box is labelled as **calculation**, **hypothesis** or **plan**. Nothing
  is shown as a result.

---

## Title

**Context Windows and KV-Cache Reuse: How Much Energy Does Remembering a Conversation Cost?**

Asia · Jan · Lena · Valeria — Sustainable Computing, TU Berlin, 2026/27

---

## 1. Motivation

- An LLM is stateless: in a chat, **the whole history is sent again with every
  message**, and the model processes it again before answering.
- Tokens processed per turn grow linearly, so the total over a conversation
  grows **quadratically** (Fig. 1).
- The **KV cache** can keep the processed history and reuse it, but it takes
  scarce GPU memory and only works if the prompt prefix is *identical*.
- Context management (cutting or summarising history) saves tokens, but can
  break cache reuse and lose information.
- Inference is the main share of the compute of deployed LLMs (Zine et al.),
  and chats keep getting longer (assistants, coding agents).

**Fig. 1** — *Tokens the model must prefill per turn and in total, without and
with prefix reuse. Calculation for a 500-token system prompt and 100 + 300
tokens per turn: after 40 turns 336k vs 16k tokens. Not a measurement.*

---

## 2. Background

**Prefill vs decode.** Prefill processes the prompt in parallel (compute-bound,
sets the time to first token). Decode generates one token at a time
(memory-bound). Caching removes only prefill work.

**KV cache.** For every token and layer the model stores a key and a value
vector. Size per token = 2 × layers × KV heads × head dim × bytes.
Qwen3-4B: 2 × 36 × 8 × 128 × 2 B = **144 KiB per token**. On one 16 GB RTX 4080
this leaves room for ≈ 39k tokens — about two 40-turn chats (calculation).

**PagedAttention / vLLM.** The KV cache is stored in 16-token blocks, like OS
memory pages. Waste drops from 60–80 % to < 4 %, and blocks can be shared.
Prefix caching identifies each full block by a hash of its tokens *and
everything before it* → one changed token invalidates every block after it.

**Takeaways from the literature**

| Paper | Takeaway for us |
|---|---|
| Fernandez et al., ACL 2025 | Decode dominates energy; faster ≠ cheaper (2 GPUs: −40 % latency, +29 % energy); even optimised vLLM uses 27–64 % more energy than a FLOP-based estimate |
| Kwon et al., SOSP 2023 | KV memory limits how many requests run at once; they did *not* keep chat caches between turns because it takes memory from other users |
| Zine, Coignion et al., 2026 | Prefix caching saved only ≈ 1–2 % energy on 5-turn chats (Llama-3B); TTFT varies far more than energy across configurations (324 % vs 38.8 %) |
| Liu et al., TACL 2024 | "Lost in the middle": accuracy is U-shaped over the position of the needed information; Llama-2 7B shows only recency bias |

**Energy estimation.** EcoLogits estimates energy from output tokens and model
size. Input and cached tokens are not in its formula (methodology page,
checked 2026-11-17).

---

## 3. Research questions and hypotheses

> **How much energy can context management and KV-cache reuse save in
> multi-turn conversations, and what does it cost in answer quality?**

| | Question | Hypothesis |
|---|---|---|
| RQ1 | How does energy per turn grow with conversation length, cache off vs on? | Off: ≈ linear per turn (quadratic total); on: much flatter |
| RQ2 | Is the energy saved smaller than the prefill tokens skipped? | Much smaller — decode stays; no saving below a break-even length |
| RQ3 | How do context strategies compare in energy and quality? | With cache, full history needs the least prefill; no strategy wins at every length |
| RQ4 | When does summarisation pay for its own calls? | Only after a break-even length T\*; a rolling summary later or never |
| RQ5 | Does a cache-aware (chunked) window keep both savings? | Yes — far less prefill than a sliding window, far less memory than full history |
| RQ6 | Does it matter which turn holds the needed fact? | Truncation fails on early facts; small models rely on recent turns, so the loss is smaller than expected |
| RQ7 | With many users, until when does keeping caches help? | Only while all histories fit in KV memory (≈ 39k tokens) |
| RQ8 | How far is EcoLogits from measured energy? | Its estimate ignores history and cache, so the error grows with both |

*All hypotheses. Results follow in January.* RQ7 is a stretch goal.

---

## 4. The tension: fewer tokens ≠ less work

**Fig. 2** — *With caching on, full history prefills only the new tokens each
turn. A sliding window holds 6× less context but prefills 4.6× more tokens
over 40 turns, because its prefix changes every turn. A chunked window breaks
the prefix only once every 5 turns. Calculation, not a measurement.*

| Strategy | Context held | Prefix stable? | Information lost | Extra cost |
|---|---|---|---|---|
| Full history | grows | yes | none (but "lost in the middle") | KV memory |
| Sliding window (last 5 turns) | bounded | no | early turns | — |
| Chunked window (ours) | bounded | 4 of 5 turns | early turns | — |
| Batch summary | bounded | between summaries | details | a summary call per threshold |
| Rolling summary | bounded | no | details, drift | a summary call per turn |

---

## 5. Method

**Setup.** vLLM serving Qwen3-4B (bf16) on one RTX 4080 16 GB per run (Science
Club server, 2 GPUs). Only two server configurations: prefix caching on / off.
Context strategies are applied by our harness, so every strategy runs against
both (full factorial).

**Method diagram (draw in Canva):**
`Scripted conversation → context strategy → vLLM (cache on | off) → answer`
with side arrows to `NVML energy · /metrics cache hits · grading` → `one raw record per run`.

**Conversations.** Our own scripted set, up to 40 turns: text, code,
tokenization and reasoning tasks. Facts are planted in early, middle or recent
turns and asked about later (exact-match grading). Assistant replies in the
history are frozen, so every condition gets identical prompts. A LoCoMo subset
as an external check.

| Exp | Answers | Varied |
|---|---|---|
| E1 | RQ1, RQ2 | conversation length × cache on/off |
| E2 | RQ3–RQ6 | 5 strategies × cache on/off × fact position |
| E3 | RQ8 | EcoLogits estimate vs measured energy, per turn |
| E4 (stretch) | RQ7 | 1–8 concurrent users × cache on/off |

**Measurement.** GPU energy per turn from the NVML energy counter, CodeCarbon as
a cross-check; idle baseline subtracted; warm-up discarded; ≥ 5 repetitions in
random order; cache reset between runs; greedy decoding. Main metric for the
trade-off: **energy per correct answer**. Statistics as in Zine et al.
(medians, ART ANOVA, Cliff's δ, Holm correction).

**Robust on a shared server.** No root needed: vLLM runs in Apptainer or a plain
Python environment; runs are resumable; runs during which another user's job
appears on our GPU are discarded and repeated. Estimated ≈ 40 GPU hours.

---

## 6. Threats to validity

- Power sampling misses short spikes (`nvidia-smi` on A100 samples ≈ 25 % of
  the time; Zine et al.) → NVML energy counter, long enough runs.
- Single user on one consumer GPU with a small model: not a production server.
- Frozen history: the KV of the model's own answers is never reused → our
  saving is a lower bound.
- Scripted conversations are controlled but not natural chat.
- Provider APIs are a black box: no ground-truth energy for RQ8.

---

## 7. Timeline

| When | Milestone |
|---|---|
| Late Nov | Server set up; vLLM serves Qwen3-4B; energy of one request measured |
| Early Dec | Harness, 5 strategies, unit tests; conversation set v1 |
| Mid Dec | Pilot → number of repetitions; E1 |
| Early Jan | E2 + E3, analysis (E4 if time allows) |
| Late Jan | Report (8–10 pages) and final presentation |

---

## References

1. J. Fernandez et al. Energy Considerations of Large Language Model Inference and Efficiency Optimizations. ACL 2025. doi:10.18653/v1/2025.acl-long.1563
2. W. Kwon et al. Efficient Memory Management for Large Language Model Serving with PagedAttention. SOSP 2023. doi:10.1145/3600006.3613165
3. N. F. Liu et al. Lost in the Middle: How Language Models Use Long Contexts. TACL 12, 2024. doi:10.1162/tacl_a_00638
4. N. Zine, T. Coignion et al. Attention to Detail: Evaluating Energy, Performance, and Accuracy Trade-offs Across vLLM Configurations. arXiv:2607.09172, 2026.
5. A. Maharana et al. Evaluating Very Long-Term Conversational Memory of LLM Agents (LoCoMo). arXiv:2402.17753, 2024.
6. vLLM documentation (automatic prefix caching); EcoLogits methodology (LLM inference). Accessed 2026-11-17.

---

## Owners

| Section | Owner | Status |
|---|---|---|
| 1 Motivation + Fig. 1 | Jan | text + figure ready |
| 2 Background + takeaway row per paper | Asia, Lena, Valeria, Jan (each checks own row against the PDF) | text ready, check pending |
| 3 Research questions | Jan | ready |
| 4 Tension + Fig. 2 | Jan, Asia (strategies) | ready |
| 5 Method + diagram | Jan, Valeria (server), Lena (measurement) | text ready, diagram in Canva |
| 6–7 Threats, timeline, references | Lena | ready |
| Canva layout | Asia, Valeria | to do |

## Checklist

- [x] Text for all sections (this note)
- [x] Fig. 1 and Fig. 2, labelled as calculations
- [x] Poster size: standard A0 portrait
- [ ] Canva design created and shared with all four
- [ ] Method diagram drawn in Canva
- [ ] Each paper owner checked their takeaway row against the PDF
- [ ] No result-like claims; hypotheses labelled as such
- [ ] Every member can answer `05-exam-prep.md`
- [ ] PDF exported to `poster/poster.pdf`
