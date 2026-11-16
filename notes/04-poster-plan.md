# 04 — Poster Plan

**Deadline:** poster submitted **Thursday 17:30**; poster session + oral exam **Friday**.
**Purpose:** summary of the research problem, theory, research questions, and
our experiment/development plan. **No results** — only hypotheses and design.

## Proposed sections (reading order)

1. **Title + team** — "Context Windows and KV-Cache Reuse: Energy vs. Quality in Multi-Turn LLM Chat" (working title)
2. **Motivation** — chat resends full history every turn; inference dominates
   LLM compute; one striking figure: prefilled tokens vs turn number (quadratic
   vs linear, from the back-of-envelope in `01-theory.md`, clearly labelled as a calculation).
3. **Background** — prefill vs decode; what the KV cache is (size formula + one
   number); prefix caching / PagedAttention; API prompt caching; Lost in the Middle.
4. **Research questions + hypotheses** — RQ1–RQ3 table from `00-problem-overview.md`.
5. **Key insight / tension** — truncation & summarisation reduce tokens but break
   the cached prefix (small diagram: append-only prompt vs shifting window).
6. **Method** — harness diagram; E1/E2/E3 overview table; factors, metrics, tools
   (vLLM, CodeCarbon/NVML, EcoLogits).
7. **Expected outcomes & threats to validity** — clearly marked as expectations.
8. **Timeline to January.**
9. **References** — the four core papers + tool docs.

## Owners (fill in)

| Section | Owner | Status |
|---|---|---|
| Motivation + figure | | |
| Background / theory | | |
| RQs + hypotheses | | |
| Method + diagrams | | |
| Threats + timeline + references | | |
| Layout / design | | |

## Checklist

- [ ] Poster format/size confirmed with course (A0? A1? portrait?) — see `open-questions.md`
- [ ] Every number on the poster has a source or a shown calculation
- [ ] No result-like claims; hypotheses labelled as such
- [ ] All [UNVERIFIED] items in notes resolved or left off the poster
- [ ] Every member can answer `05-exam-prep.md`
- [ ] PDF exported to `poster/`
