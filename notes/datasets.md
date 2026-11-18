# Candidate datasets for benchmarking

Shortlist of datasets for the experiments. Nothing here is decided yet.
Entries marked `[UNVERIFIED]` are from memory: check the arXiv ID, licence and
size before citing or downloading.

## What we need

| Need | Used for | Requirement |
|---|---|---|
| Realistic multi-turn transcripts | RQ1–RQ5 (energy) | Real turn lengths and conversation lengths; text, not only token counts |
| Questions with known answers about earlier turns | RQ3–RQ6 (quality) | Ground truth, and the answer sits in a known turn |
| Fits our hardware | all | 2 × RTX 4080 (16 GB) → 3–4B model; KV cache must fit next to the weights, so roughly ≤ 8–16k tokens of context (our assumption, to check) |
| Usable licence | all | Allows research use |

## 1. Realistic multi-turn conversations (energy workloads)

No ground-truth answers, so not usable for quality.

| Dataset | Source | What it is | Status |
|---|---|---|---|
| ShareGPT | various HF mirrors | Real user–ChatGPT conversations; used by vLLM's benchmark scripts | `[UNVERIFIED]` |
| LMSYS-Chat-1M | arXiv:2309.11998 | 1M real conversations with many models | `[UNVERIFIED]` |
| WildChat | arXiv:2405.01470 | ~1M real ChatGPT conversations, many multi-turn | `[UNVERIFIED]` |
| UltraChat | arXiv:2305.14233 | Synthetic multi-turn dialogues | `[UNVERIFIED]` |

## 2. Multi-turn quality benchmarks (RQ3–RQ6)

| Dataset | Source | What it tests | Fit for us | Status |
|---|---|---|---|---|
| LoCoMo | arXiv:2402.17753 | Very long conversations with questions about earlier turns | Good: ≈10–21k tokens per conversation, evidence turns annotated — see [details below](#locomo-in-detail) | Data checked 2026-11-17; paper: Maharana et al., arXiv:2402.17753 (checked 2026-11-17) |
| LongMemEval | arXiv:2410.10813 | Long-term chat memory across sessions | Histories ~100k+ tokens → only a subset or shortened version fits | `[UNVERIFIED]` |
| MT-Bench-101 | arXiv:2402.14762 | Fine-grained multi-turn abilities (recall, follow-up, …) | Possible | `[UNVERIFIED]` |
| MT-Eval | arXiv:2401.16745 | Multi-turn abilities incl. recollection | Possible | `[UNVERIFIED]` |
| MT-Bench | arXiv:2306.05685 | Two-turn chat quality | Too short (2 turns) | `[UNVERIFIED]` |
| Lost in Conversation | Laban et al., arXiv:2505.06120 | Instructions spread over several turns | Related work rather than dataset | `[UNVERIFIED]` |

### LoCoMo in detail

Source: `data/locomo10.json` in the GitHub repo `snap-research/locomo`,
downloaded and inspected 2026-11-17. All numbers below are counted from that
file unless marked otherwise.

**Size**

| | Value |
|---|---|
| Conversations | 10, each between two fictional people (e.g. Caroline and Melanie) |
| Sessions per conversation | 19–32 (one chat on one day, with date and time) |
| Turns per conversation | 369–689 (one turn = one utterance) |
| Words per conversation | 8,019–16,165 |
| Tokens per conversation | ≈10–21k (calculation: words × ~1.3; to be counted with our tokenizer) |
| Questions | 1,986 in total, 105–260 per conversation |

**Fields per conversation**

| Field | Content |
|---|---|
| `sample_id` | e.g. `conv-26` |
| `conversation` | `speaker_a`, `speaker_b`, `session_1 … session_N`, each with `session_N_date_time` (e.g. "1:56 pm on 8 May, 2023") |
| `qa` | Questions with answers (see below) |
| `session_summary` | Ready-made text summary per session |
| `observation` | Facts per speaker and session, each linked to its source turn, e.g. `["Caroline attended an LGBTQ support group…", "D1:3"]` |
| `event_summary` | Main life events per speaker and session, with date |

A turn:
```json
{"speaker": "Caroline", "dia_id": "D1:1", "text": "Hey Mel! Good to see you! How have you been?"}
```
`D1:3` = session 1, turn 3. Turns that share an image additionally have
`img_url`, `blip_caption` (automatic text description of the image) and
`query`. In the first conversation, 116 of 419 turns have a caption.

**Questions (`qa`)**

Each entry has `question`, `answer`, `evidence` (list of turn IDs) and
`category` (1–5).

| Cat. | Count | Example from the file |
|---|---|---|
| 1 | 282 | "What did Caroline research?" → *Adoption agencies* |
| 2 | 321 | "When did Caroline go to the LGBTQ support group?" → *7 May 2023* |
| 3 | 96 | "What fields would Caroline be likely to pursue in her education?" → *Psychology, counseling certification* |
| 4 | 841 | "What did the charity race raise awareness for?" → *mental health* |
| 5 | 446 | "What did Caroline realize after her charity race?" — adversarial: no `answer`, only `adversarial_answer`; the correct response is "not mentioned" |

The file does not name the categories. Commonly cited mapping:
1 = multi-hop, 2 = temporal, 3 = open-domain, 4 = single-hop, 5 = adversarial
`[UNVERIFIED — check in the paper]`.

**What this means for us** (our interpretation)

- **Evidence turn IDs** let us check, per strategy, whether the turn needed for
  an answer was still in the context → explains *why* quality drops (RQ6).
- **`session_summary`** can serve as a reference summary. For energy we still
  generate our own summaries, otherwise their cost is not counted (CLAUDE.md §6).
- **Fits our hardware, but not by much** (calculation): ≈21k tokens ×
  ≈112 KiB/token (3B model with 28 layers, 8 KV heads, head dim 128, bf16 —
  `[UNVERIFIED]`, check model config) ≈ 2.3 GiB KV cache + ≈6.4 GB weights
  → fits on one 16 GB RTX 4080. vLLM `max_model_len` must be ≥ ~24k.
  With Qwen3-4B (144 KiB/token, checked): ≈ 2.9 GiB, still fits, but it uses
  over half of the ≈ 39k-token KV budget (`01-theory.md` §3).
- **Category 5** cannot be scored with token-F1 (correct answer is "not
  mentioned") → score separately or exclude.
- **Not a user–assistant chat:** two people, short turns (≈20–25 words per
  turn, calculation: words ÷ turns per conversation). Real assistant replies are
  longer, so the prefill/decode ratio differs from real chat use.
- **Licence:** not checked yet (the README link returned 404).

## 3. Long-context stress tests (single prompt, not a conversation)

Useful as a reference for "can the model use a fact at position X", not as chat workloads.

| Dataset | Source | Status |
|---|---|---|
| Needle in a Haystack | (method, many implementations) | — |
| RULER | arXiv:2404.06654 | `[UNVERIFIED]` |
| LongBench | arXiv:2308.14508 | `[UNVERIFIED]` |
| ∞Bench | arXiv:2402.13718 | `[UNVERIFIED]` |
| Lost in the Middle data | Liu et al., TACL 2024, `papers/2024.tacl-1.9.pdf` | in `papers/` |

## 4. Serving workload traces

Only request lengths and timestamps, no text.

| Dataset | Source | Status |
|---|---|---|
| BurstGPT | Wang et al., arXiv:2401.17644 | Cited in Fernandez et al. (2025), Tables 1–2 |
| Azure LLM inference traces (chat, code) | Stojkovic et al. (2024) | Cited in Fernandez et al. (2025), Tables 1–2 |

## Option: synthetic conversations with planted facts

Build conversations ourselves: filler turns (e.g. taken from ShareGPT/WildChat)
plus one fact planted at a chosen turn, then a question about it at the end.

- **Pro:** we control conversation length and the position of the fact exactly
  → clean analysis for RQ6, easy to explain in the exam.
- **Con:** less realistic; generator code needs unit tests (CLAUDE.md §6).

## Current suggestion (to discuss)

- **Energy (RQ1–RQ5):** replay transcripts from ShareGPT or WildChat.
- **Quality (RQ3–RQ6):** LoCoMo, or synthetic planted-fact conversations.

## To check before deciding

- [ ] arXiv IDs, licences and download sources of all `[UNVERIFIED]` entries
- [ ] Token lengths of LoCoMo / LongMemEval conversations with our tokenizer
  (LoCoMo word counts done, see above)
- [ ] LoCoMo licence and category names (1–5)
- [ ] Max context that fits on one RTX 4080 for the chosen model (weights + KV cache)
