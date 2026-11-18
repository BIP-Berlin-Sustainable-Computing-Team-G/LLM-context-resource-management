"""Poster figures. Calculations and hypothesis sketches only, no measurements.

Token model: system prompt S, each turn adds a user message U and an answer A.
Assistant replies are frozen, so with prefix reuse a turn prefills only the
previous answer and the new user message.

Run: python3 poster/make_figures.py   (writes PNGs at 300 dpi next to this file)
"""

from pathlib import Path

import matplotlib.pyplot as plt

OUT = Path(__file__).parent
S, U, A, T, K = 500, 100, 300, 40, 5

DARK, GREEN, MINT, AMBER, GREY = "#17322A", "#3FB26F", "#9DDDB7", "#E09F3E", "#8A9A94"

plt.rcParams.update({
    "font.family": "Arial", "font.size": 22, "axes.edgecolor": DARK,
    "axes.labelcolor": DARK, "xtick.color": DARK, "ytick.color": DARK,
    "axes.spines.top": False, "legend.labelcolor": DARK, "axes.spines.right": False, "axes.linewidth": 2,
    "xtick.major.width": 2, "ytick.major.width": 2,
})


def prompt_len(t, window=None):
    kept = t - 1 if window is None else min(t - 1, window)
    return S + kept * (U + A) + U


def prefill(t, window=None, cache=True):
    """Tokens the model must process at turn t (1-based)."""
    if not cache or t == 1:
        return prompt_len(t, window)
    if window is None or t - 1 <= window:
        return U + A  # append-only: previous answer + new user message
    return prompt_len(t, window) - S  # window moved: only the system prompt matches


def save(fig, name):
    fig.savefig(OUT / name, dpi=300, transparent=True, bbox_inches="tight")
    plt.close(fig)


# RQ1: prefilled tokens per turn, full history
turns = range(1, T + 1)
fig, ax = plt.subplots(figsize=(5.6, 3.9))
ax.plot(turns, [prefill(t, cache=False) for t in turns], color=DARK, lw=4, ls="--", label="cache off")
ax.plot(turns, [prefill(t) for t in turns], color=GREEN, lw=5, label="cache on")
ax.set_xlabel("turn")
ax.set_ylabel("tokens prefilled per turn")
ax.set_ylim(0, 17000)
ax.yaxis.set_major_formatter(lambda v, _: f"{v/1000:.0f}k")
ax.legend(frameon=False, loc="upper left")
save(fig, "fig-rq1.png")

# RQ2: total prefill over 40 turns, full history vs sliding window
tot = {
    (w, c): sum(prefill(t, w, c) for t in turns)
    for w in (None, K) for c in (False, True)
}
fig, ax = plt.subplots(figsize=(5.6, 3.9))
x = [0, 1]
ax.bar([i - 0.2 for i in x], [tot[(None, False)], tot[(K, False)]], 0.38, color=DARK, label="cache off")
ax.bar([i + 0.2 for i in x], [tot[(None, True)], tot[(K, True)]], 0.38, color=GREEN, label="cache on")
for i, w in enumerate((None, K)):
    for dx, c in ((-0.2, False), (0.2, True)):
        v = tot[(w, c)]
        ax.text(i + dx, v + 6000, f"{v/1000:.0f}k", ha="center", color=DARK, fontsize=19)
ax.set_xticks(x, ["full history", "sliding window\n(last 5 turns)"])
ax.set_ylabel("tokens prefilled, 40 turns")
ax.yaxis.set_major_formatter(lambda v, _: f"{v/1000:.0f}k")
ax.set_ylim(0, 380000)
ax.legend(frameon=False, loc="upper right")
save(fig, "fig-rq2.png")

# RQ3: hypothesis sketch of the energy-quality trade-off (no data)
fig, ax = plt.subplots(figsize=(5.6, 3.9))
pts = {  # (energy, accuracy, colour, label side)
    "full + cache": (1.0, 0.84, GREEN, "right"),
    "full, no cache": (4.3, 0.84, DARK, "left"),
    "batch summary + cache": (1.6, 0.72, GREEN, "right"),
    "rolling summary + cache": (3.3, 0.61, GREEN, "left"),
    "window + cache": (2.2, 0.49, GREEN, "left"),
    "window, no cache": (2.7, 0.49, DARK, "right"),
}
for name, (e, q, c, side) in pts.items():
    ax.scatter(e, q, s=380, color=c, zorder=3)
    dx = 0.2 if side == "right" else -0.2
    ax.text(e + dx, q, name, ha="left" if side == "right" else "right", va="center", fontsize=16, color=DARK)
ax.set_xlabel("energy per conversation  →")
ax.set_ylabel("accuracy  →")
ax.set_xticks([])
ax.set_yticks([])
ax.set_xlim(0.5, 4.6)
ax.set_ylim(0.42, 0.9)
save(fig, "fig-rq3.png")


# RQ4: toy model of cumulative energy (assumed costs, shape only).
# With prefix caching, full history prefills only U + A new tokens per turn, so a
# summary can only win through cheaper decode over a shorter context. That makes
# the break-even length T* long, possibly beyond our 40-turn conversations.
TOY = range(1, 121)


def decode_cost(ctx):  # per output token; grows with the context it attends to
    return 4 * (1 + ctx / 8000)


def run_full():
    e, out = 0.0, []
    for t in TOY:
        e += prefill(t) + A * decode_cost(prompt_len(t))
        out.append(e)
    return out


def run_batch(budget=6000, summary=300):
    e, out, ctx, broken = 0.0, [], S + U, False
    for t in TOY:
        if ctx > budget:  # summarise old turns into one block, keep the last K turns
            e += 50 + summary * decode_cost(ctx)
            ctx, broken = S + summary + K * (U + A) + U, True
        e += (ctx - S if broken else (U + A if t > 1 else ctx)) + A * decode_cost(ctx)
        broken = False
        out.append(e)
        ctx += U + A
    return out


def run_rolling(summary=300):
    e, out = 0.0, []
    for t in TOY:
        ctx = S + summary + (U + A) + U
        e += (summary + U + A + 50) + summary * decode_cost(summary + U + A)  # rewrite summary
        e += (ctx - S) + A * decode_cost(ctx)  # prefix broken after S every turn
        out.append(e)
    return out


full, batch, roll = run_full(), run_batch(), run_rolling()
tstar = next((t for t, f, b in zip(TOY, full, batch) if t > 10 and b < f), None)
fig, ax = plt.subplots(figsize=(5.6, 3.9))
ax.axvspan(1, T, color=MINT, alpha=0.35, lw=0)
ax.text(T + 3, max(full) * 0.97, "← our runs\n   (≤ 40 turns)", ha="left", va="top", color=DARK, fontsize=16)
ax.plot(TOY, full, color=GREEN, lw=5, label="full history + cache")
ax.plot(TOY, batch, color=DARK, lw=4, label="batch summary + cache")
ax.plot(TOY, roll, color=GREY, lw=4, ls=":", label="rolling summary + cache")
if tstar:
    ax.axvline(tstar, color=AMBER, lw=3, ls="--")
    ax.text(tstar + 2, max(full) * 0.05, "T*", color=AMBER, fontsize=26, fontweight="bold")
ax.set_xlabel("turn")
ax.set_ylabel("cumulative energy  →")
ax.set_yticks([])
ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.24), ncol=1, fontsize=17)
save(fig, "fig-rq4.png")

print("prefill totals over 40 turns:", tot, "| toy T*:", tstar)
