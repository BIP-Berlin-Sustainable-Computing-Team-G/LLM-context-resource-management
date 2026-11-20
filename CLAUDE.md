# CLAUDE.md — Context Windows & KV-Cache Reuse (Sustainable Computing, TU Berlin, Erasmus)

Project rules for humans and AI assistants working in this repository.
These rules override generic/global defaults where they conflict.

## 1. What this project is

A 4-person student research project for the course on sustainable computing.
Topic: **context windows and KV-cache reuse** in multi-turn LLM conversations.

**Overarching research question:**
> How much energy can context management and KV-cache reuse save in multi-turn
> conversations, and what does it cost in answer quality?

The four research questions (RQ1–RQ4) and their hypotheses are in
[`notes/research-questions.md`](notes/research-questions.md), the setup in
[`notes/implementation.md`](notes/implementation.md).
The original course brief is in [`BIP-note-topic-requirements.txt`](BIP-note-topic-requirements.txt) — it is the source of truth for requirements.

## 2. Two phases — always know which one you are in

| Phase | Deliverable | Deadline | Nature |
|---|---|---|---|
| **Phase 1 — Research & Poster** | Poster + oral exam defence: theory, research questions, hypotheses, experiment design, development plan | Poster submitted **Thursday 17:30**, session/exam **Friday** (week of 2026-11-16) | Theoretical. **No results are expected.** |
| **Phase 2 — Implementation & Report** | Experiment harness, measurements, analysis, 8–10 page report + final presentation | **January 2027** | Empirical. |

Rules that follow from this:
- During Phase 1, do not spend time on production-quality code. Small throwaway
  scripts (e.g. a back-of-envelope calculation, a quick feasibility check of a
  tool) are fine and go to `scratch/` (git-ignored) or `experiments/spikes/`.
- Never put fabricated or "expected" numbers on the poster as if they were
  results. Hypotheses must be clearly labelled as hypotheses; calculations as
  calculations (with the formula and assumptions shown).
- Everything we claim on the poster must be defensible orally by **every** team
  member. If only the AI understands it, it does not go on the poster.

## 3. Language

- **Everything in the repository is written in English**: notes, poster text,
  code, comments, commit messages, issue/PR text, file names.
- Chat with the AI assistant may be in any language (Polish, German, …); the
  assistant answers in the user's language, but anything it writes to files is
  English.

## 4. Repository layout

```
BIP-note-topic-requirements.txt   course brief (source of truth)
CLAUDE.md                         this file
README.md                         short entry point for the team
notes/
  research-questions.md           RQ1–RQ4, hypotheses, comments
  implementation.md               setup, design, measurement, comments
  study-guide.md                  definitions + likely exam questions
papers/                           PDFs + one summary per paper (<firstauthor>-<year>.pdf/.md)
poster/                           figures (+ script) and exported PDF (Phase 1)
experiments/                      Phase 2: harness/, configs/, spikes/, results/raw/, analysis/
report/                           Phase 2: final report
scratch/, temp/                   personal throwaway space (git-ignored)
```

Keep `notes/` to these few files; extend them instead of adding new ones. Folders are created when
there is real content for them — no placeholder files.

### Team and paper ownership

| Member | Paper | Summary |
|---|---|---|
| Jan | Fernandez et al., ACL 2025 | `papers/fernandez-2025.md` |
| Asia | Liu et al., TACL 2024 (Lost in the Middle) | `papers/liu-2024.md` |
| Lena | Zine, Coignion et al., arXiv:2607.09172 (vLLM configurations) | `papers/zine-2026.md` |
| Valeria | Kwon et al., SOSP 2023 (PagedAttention) | `papers/kwon-2023.md` |

Each paper summary has the same sections: *In short · Setup · Key findings ·
Limitations · What it means for our project*. Every number cites a figure,
table or section of the paper. Our own interpretation is marked as such.

### Hardware

- Experiments and development: MacBook, Apple M5, 48 GB unified memory.
  Energy from `powermetrics` and CodeCarbon (both need sudo).
- Backup only: Science Club server, 2 × NVIDIA RTX 4080 (16 GB each).
- **Engine: `llama-server` (llama.cpp)**, model Qwen3.5-4B GGUF Q4_K_M
  (decided 2026-11-18). Prompt cache on vs off = two server configurations.
  Details and risks: `notes/implementation.md`.

## 5. Working with AI (allowed — but we must understand everything)

The course allows AI use. We are examined orally, so understanding is the point.
Rules for the assistant:

1. **Explain, don't just produce.** When writing theory, include the *why* and
   the intuition, not only the conclusion. Prefer short derivations and worked
   numeric examples (e.g. KV-cache size for a concrete model).
2. **No invented sources.** Cite only papers that are in `papers/` or that have a
   verifiable DOI/arXiv ID. If a claim is from memory and not verified against a
   source, mark it `[UNVERIFIED]`. Never invent numbers, author names, or quotes.
3. **Separate fact / hypothesis / our assumption** explicitly in notes
   (e.g. "Paper X reports…", "We hypothesise…", "We assume…").
4. **Flag uncertainty and push back.** If a research question is not measurable
   with our resources, say so and propose a scoped-down alternative.
5. **Tool and API facts age quickly** (pricing, cache TTLs, minimum cacheable
   prompt length, EcoLogits methodology). Link the official docs and record the
   date checked.
6. Keep notes concise and poster-oriented: bullet points, formulas, small tables.
7. **No filler in the repository.** Write only what a team member will actually
   read: no README per folder, no boilerplate sections, no restating the same
   content in several files, no "summary of the summary". Brainstorm and
   working drafts go to `scratch/` (git-ignored) until the team agrees on them.
   Prefer editing an existing note over creating a new file.

## 6. Research & experiment rules (mainly Phase 2, but design them now)

- **Protocol before code.** Every experiment is first described in
  `notes/implementation.md` (hypothesis, independent/dependent variables,
  controls, metrics, number of repetitions, analysis) before implementation.
- **Reproducibility:** fixed seeds, `temperature=0` (or fixed sampling params),
  pinned package versions, record model name + revision, hardware (GPU/CPU model,
  driver, CUDA), inference engine version and config for every run.
- **Energy measurement hygiene:** warm-up runs, ≥5 repetitions per condition
  (more if variance is high), randomised run order, measure and subtract idle
  baseline power, report medians + spread, note the sampling rate of the meter
  (e.g. `powermetrics` interval; `nvidia-smi` limitations in Zine et al., §VII).
- **Raw data is immutable.** Raw measurements go to `experiments/results/raw/`
  (one file per run, with metadata); analysis reads raw data and writes derived
  data/figures elsewhere. Never hand-edit results.
- **Tests for measurement code.** Token counting, context-strategy
  transformations (sliding window, summarisation), and metric/aggregation code get
  unit tests — a bug there silently invalidates the results.
- **Secrets:** API keys only via environment variables / `.env` (git-ignored).
  Never commit keys. Track API spending.
- **Count all energy.** If a strategy uses extra LLM calls (e.g. summarisation),
  their cost is part of that strategy's energy.

## 7. Git workflow

- `main` holds the agreed state. Work on short-lived branches:
  `notes/<topic>`, `poster/<section>`, `exp/<experiment>`, `fix/<thing>`.
- Small, focused commits with imperative English messages
  (`notes: add KV-cache size derivation`).
- Pull before you start, push when you stop — four people edit the same notes.
- No AI attribution lines in commits or PRs.
- Do not commit large binaries other than the reference PDFs and the final poster
  export; never commit raw model weights or big result dumps (use `.gitignore`).

## 8. Definition of done

- **Phase 1:** poster PDF in `poster/`, every section traceable to a note in
  `notes/`, every team member can answer the questions in `notes/study-guide.md`.
- **Phase 2:** experiments reproducible from `experiments/README.md` (written
  once the harness exists) with one
  command per experiment, figures regenerated from raw data, report in `report/`.
