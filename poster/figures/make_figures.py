"""Poster figures. Calculations only, no measurements.

Token model (see notes/01-theory.md §4): system prompt S, each turn adds a user
message u and an answer a. A context strategy decides which earlier turns stay
in the prompt. With prefix caching, the tokens that can be reused are the
longest common prefix with the previous turn's prompt (conservative: the
previous answer is prefilled again, as in the §4 formula).

Run: python3 poster/figures/make_figures.py
"""

from pathlib import Path

import matplotlib.pyplot as plt

S, U, A = 500, 100, 300  # system prompt, user message, answer (tokens)
M = U + A
T = 40  # turns
W, K = 5, 5  # sliding window size; chunked window drops K turns at once

OUT = Path(__file__).parent


def window_start(t, strategy):
    """Index of the first earlier turn kept in the prompt at turn t (0-based turns)."""
    if strategy == "full":
        return 0
    if strategy == "sliding":
        return max(0, t - W)
    if strategy == "chunked":
        # keep between W and W+K-1 earlier turns; drop K at once
        return 0 if t < W + K else ((t - W) // K) * K
    raise ValueError(strategy)


def simulate(strategy, cache):
    """Per-turn (prefilled tokens, prompt length)."""
    prefilled, length = [], []
    prev_start = None
    for t in range(T):
        start = window_start(t, strategy)
        n_hist = t - start
        prompt = S + n_hist * M + U
        if not cache or t == 0:
            reused = 0
        elif start == prev_start:
            # previous prompt was S + (n_hist-1)*M + U; it is a prefix of this one
            reused = S + (n_hist - 1) * M + U
        else:
            reused = S  # window moved: only the system prompt still matches
        prefilled.append(prompt - reused)
        length.append(prompt)
        prev_start = start
    return prefilled, length


def check():
    # worked example from notes/01-theory.md §4 (T = 20)
    global T
    saved, T = T, 20
    assert sum(simulate("full", cache=False)[0]) == 88_000
    assert sum(simulate("full", cache=True)[0]) == 8_200
    T = saved


ACCENT, GREY, BLUE, ORANGE = "#c0392b", "#7f8c8d", "#2c6fbb", "#e67e22"
plt.rcParams.update({"font.size": 13, "axes.spines.top": False, "axes.spines.right": False})


def fig_growth():
    turns = range(1, T + 1)
    no, _ = simulate("full", cache=False)
    yes, _ = simulate("full", cache=True)
    cum = lambda xs: [sum(xs[: i + 1]) / 1000 for i in range(len(xs))]
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(turns, [x / 1000 for x in no], color=ACCENT, lw=2.5, label="no cache reuse")
    ax[0].plot(turns, [x / 1000 for x in yes], color=BLUE, lw=2.5, label="perfect prefix reuse")
    ax[0].set(xlabel="turn", ylabel="tokens prefilled in this turn (k)", title="Per turn")
    ax[1].plot(turns, cum(no), color=ACCENT, lw=2.5, label="no cache reuse  ~ T²")
    ax[1].plot(turns, cum(yes), color=BLUE, lw=2.5, label="perfect prefix reuse  ~ T")
    ax[1].set(xlabel="turn", ylabel="cumulative tokens prefilled (k)", title="Whole conversation")
    ax[1].legend(frameon=False)
    fig.text(0.5, -0.02, f"Calculation, not a measurement: S={S}, u={U}, a={A} tokens per turn.",
             ha="center", fontsize=11, color=GREY)
    fig.tight_layout()
    save(fig, "fig1-token-growth")


def fig_strategies():
    turns = range(1, T + 1)
    # drawn in this order so the full-history line (dashed, on top) stays visible
    styles = {"chunked": (f"chunked window (drop {K} turns at once)", ORANGE, "-"),
              "sliding": (f"sliding window (last {W} turns)", ACCENT, "-"),
              "full": ("full history", BLUE, "--")}
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.4))
    for s, (label, color, ls) in styles.items():
        pre, length = simulate(s, cache=True)
        total = sum(pre) / 1000
        ax[0].plot(turns, [x / 1000 for x in pre], color=color, lw=2.4, ls=ls,
                   label=f"{label}: {total:.1f}k prefilled in total")
        ax[1].plot(turns, [x / 1000 for x in length], color=color, lw=2.4, ls=ls)
    ax[0].set(xlabel="turn", ylabel="tokens prefilled (k)", title="Compute: prefill per turn, cache on")
    ax[1].set(xlabel="turn", ylabel="prompt length (k tokens)", title="Memory + decode: context held")
    fig.legend(*ax[0].get_legend_handles_labels(), loc="upper center", ncol=1, frameon=False,
               fontsize=11, bbox_to_anchor=(0.5, 1.13))
    fig.text(0.5, -0.02, "Calculation, not a measurement. Same token model as Fig. 1; "
             "block granularity (16 tokens) ignored.", ha="center", fontsize=11, color=GREY)
    fig.tight_layout()
    save(fig, "fig2-strategies")


def save(fig, name):
    fig.savefig(OUT / f"{name}.svg", bbox_inches="tight")
    fig.savefig(OUT / f"{name}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    check()
    fig_growth()
    fig_strategies()
    for s in ("full", "sliding", "chunked"):
        for c in (False, True):
            pre, length = simulate(s, c)
            print(f"{s:8s} cache={c!s:5s} total prefilled={sum(pre):>7,}  final prompt={length[-1]:>6,}")
