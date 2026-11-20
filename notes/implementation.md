# Implementation

How we answer [`research-questions.md`](research-questions.md). Facts about
tools were checked on 2026-11-18. Links are in the comments.

## Setup

| | Choice |
|---|---|
| Machine | MacBook, Apple M5, 48 GB unified memory (record the exact chip) |
| Backup | Science Club server (2 × RTX 4080), only if the Mac is not enough |
| Engine | `llama-server` (llama.cpp), Metal backend, one pinned build |
| Model | Qwen3.5-4B, GGUF Q4_K_M, one pinned file (SHA256 recorded) |
| Options | text only (`--no-mmproj`), thinking off (`enable_thinking: false`), one slot (`-np 1`), KV cache f16 (default) |
| Energy | `powermetrics` (CPU, GPU, ANE power) per turn; CodeCarbon per run |
| Harness | Python script that replays conversations against llama-server's OpenAI API |

## Design

```
conversation file → strategy (harness) → llama-server (cache on | off) → answer
                                               │
                     powermetrics + CodeCarbon + timings.cache_n → one JSON per run
```

- **Cache on/off** = two server configurations: default vs `--no-cache-prompt`.
  Nothing else changes.
- **Strategies live in the harness.** The server only sees the prompt. Every
  strategy runs with cache on and off.

| Strategy | Prompt at turn t | Prefix stable? | Extra calls |
|---|---|---|---|
| Full history | system + all turns | yes | none |
| Sliding window | system + last 5 turns | no | none |
| Batch summary | system + summary + last 5 turns, re-summarised when over a token budget | between summaries | one per re-summary |
| Rolling summary | system + summary (updated every turn) + last turn | no | one per turn |

- **Workload:** our own scripted conversations, up to 40 turns: text, code,
  tokenization, reasoning. Facts are planted in early, middle or recent turns
  and asked about later (exact-match grading). A LoCoMo subset is the
  external quality check.
- **Frozen history:** assistant replies in the history are generated once and
  stored. Every condition gets the same prompts.
- **Conditions:** 4 strategies × cache on/off × T ∈ {10, 20, 40}.
  RQ1 uses full history only.

## Measurement

- **Energy runs:** fixed answer length (`max_tokens` + `ignore_eos`), so
  decode work is equal across conditions. **Quality runs:** natural answers.
- Greedy decoding (`temperature 0`), fixed seed.
- Per turn: energy (powermetrics, integrated over the request), `prompt_n`,
  `cache_n`, `predicted_n`, timings (from llama-server's `timings`).
- Per run: CodeCarbon total; build, model hash, flags, macOS version, power
  state.
- ≥ 5 repetitions per energy condition, random order, warm-up discarded, slot
  cache erased between runs (`POST /slots/0?action=erase`), idle baseline
  subtracted.
- Mac on AC power, battery full, other apps closed, same power mode, cool-down
  between runs.

## Day-1 checks (pilot)

1. Prefix reuse works for this model: with full history, `cache_n` ≈ previous
   prompt length from turn 2 on. If not, tune context checkpoints
   (`--ctx-checkpoints`, `--checkpoint-min-step`) or use Qwen3-4B instead.
2. `--no-cache-prompt` really gives `cache_n = 0`.
3. powermetrics sampling interval and noise at idle; CodeCarbon runs with sudo.
4. Thinking is off, and answers have the expected length.
5. Variance of a few runs → final number of repetitions.

---

## Comments

**llama-server, not Ollama.** Ollama is built on llama.cpp but hides cache
control: no erase endpoint, and no per-turn `cache_n`. llama-server exposes
both. Neither engine "compacts" a chat for us. Windows and summaries are
written in the harness in any case.

**Why not vLLM any more.** It needs CUDA, and the Mac has none. Kwon et al.
and Zine et al. still explain the mechanism: llama.cpp also reuses the common
prefix of the previous request. It works per slot rather than per hashed
block, so the theory carries over. Our numbers are not directly comparable
with Zine et al.

**Qwen3.5-4B is a hybrid model.** 32 layers: 24 Gated DeltaNet (linear
attention, fixed-size state) and 8 full-attention layers with 4 KV heads ×
256 dim. KV cache in f16 = 2 × 8 × 4 × 256 × 2 B = **32 KiB per token**, so
16k tokens take ≈ 0.5 GiB (calculation, from the model card). Memory is not
our bottleneck on 48 GB.
- Risk: a recurrent state cannot be cut back to an arbitrary position.
  llama-server therefore reuses it only from saved checkpoints, and users
  report full re-processing with Qwen3.5 (llama.cpp issues
  [#24714](https://github.com/ggml-org/llama.cpp/issues/24714),
  [#21383](https://github.com/ggml-org/llama.cpp/issues/21383)). Frozen history
  makes this worse: the stored reply differs from the generated one, so the
  server must roll back to the end of the previous prompt. Hence day-1 check 1.
- If reuse fails, the fallback is Qwen3-4B, a plain transformer with a
  normal KV cache.

**Q4_K_M.** Exists for macOS / llama-server, e.g.
[bartowski/Qwen_Qwen3.5-4B-GGUF](https://huggingface.co/bartowski/Qwen_Qwen3.5-4B-GGUF)
(≈ 3 GB). There is no official Qwen GGUF, so we pin one community file.
Effect on results (our reasoning):
- All conditions use the same file, so comparisons between strategies and
  cache settings stay valid.
- Absolute energy and quality differ from a bf16 model and from the papers.
  4-bit hurts small models more.
- The KV cache stays f16, so the cache mechanism itself is not quantised.
- With 48 GB, Q8_0 (≈ 4.5 GB) or bf16 (≈ 9 GB) would also fit and remove this
  question. Worth deciding before the pilot.

**Energy on a Mac.** No NVML here. `powermetrics` reports CPU, GPU and ANE
power separately and needs sudo. CodeCarbon uses powermetrics on Apple
Silicon and also needs sudo, otherwise it falls back to a constant estimate
([CodeCarbon methodology](https://docs.codecarbon.io/latest/explanation/methodology/)).
powermetrics is an estimate with undocumented accuracy, so we report the same
source in every condition and compare relative differences. We report SoC
energy (CPU + GPU + ANE), because llama.cpp also uses the CPU.

**Determinism.** llama.cpp docs: with `cache_prompt` on, outputs are not
guaranteed to be bit-identical, because the batch size differs. Quality is
therefore graded per condition, not copied from one condition to another.

**Frozen history is a lower bound.** The model's own generated tokens are
never reused, so the measured cache saving is smaller than in a real chat.

**Thinking mode.** Qwen3.5 thinks by default. We switch it off with
`chat_template_kwargs: {"enable_thinking": false}`; otherwise answer length,
and with it energy, varies wildly.
