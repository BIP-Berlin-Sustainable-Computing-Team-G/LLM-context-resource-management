# Research questions — candidates and scores

The long list we choose the poster questions from. Each entry has one score out
of 10, based on three things:

- how directly it answers the main question (energy saved vs. quality lost),
- whether we can do it by January on our server (2 × RTX 4080 16 GB, vLLM),
- whether it adds something beyond our four papers.

Everything below is a question or a hypothesis. None of it is a result.

---

## From the course brief

**B1. How does energy per turn grow with conversation length when the full
history is re-sent, with and without cache reuse?** — **9/10**
This is the core experiment. Without caching, every turn prefills the whole
history again. With prefix caching, only the new tokens are prefilled. Zine et al.
cover only 5 turns batched across 200 conversations. We would look at long
conversations, one turn at a time, and split each turn into prefill and decode.

**B2. How do context strategies (full history, sliding window, summarised
history) compare in energy and answer quality?** — **9/10**
This is where the trade-off becomes visible. Trimming the history saves tokens,
but it changes the prompt prefix and destroys cache reuse, and it can drop
information the model needs. None of our four papers looks at this combination.
It costs the most effort, because we need a quality test where the answer
depends on an earlier turn. Following Liu et al., the position of the needed
fact should vary. Our conversation set (`03-experiment-plan.md`) does this:
facts planted in early turns, plus gradable tasks.

**B3. Do API prompt caching and energy-estimation tools agree — does EcoLogits
account for cached tokens?** — **6/10**
It matters for how footprints get reported. The EcoLogits documentation may
already answer it without an experiment. We also cannot measure the energy of a
commercial API, so we could only compare the estimator against our own local
measurement. Our budget is zero, so it only works on a free tier (Google AI
Studio, GroqCloud) that reports cached tokens. That makes it a good side
question rather than a main one.

## From Fernandez et al. (Jan)

**F1. In single-user chat, prefix caching saves much less energy than the share
of tokens it skips.** — **8/10**
Fernandez et al. show that decode dominates energy except for very short
outputs, and caching removes only prefill work. Zine et al. measured about 1–2 %
saving for Llama-3B on 5-turn WildChat, which points the same way. Test: per-turn
energy split into prefill and decode, with cache on and off; compare the
relative energy saving with the relative token saving.

**F2. There is a break-even history length below which caching saves almost no
energy.** — **7/10**
At short inputs and batch size 1 the GPU is under-utilised, so prefill energy
grows sub-linearly (Fernandez, Fig. 2), and skipping those tokens is nearly free.
Savings should appear only once the history is long enough to keep the GPU busy.
Test: a sweep over history length; find where the cache-on and cache-off curves
separate. This is cheap, because it comes from the B1 runs.

**F3. The TTFT improvement from caching overstates its energy improvement.** — **6/10**
In Fernandez et al., tensor parallelism cuts latency by 61 % and raises energy
by 55 %. Across vLLM configurations, Zine et al. found median differences of 324 % in
TTFT but 38.8 % in energy. It
costs nothing extra, since we log TTFT anyway, but it is partly known already, so
it serves best as a supporting point.

**F4. With several concurrent users, keeping conversation caches pays off only
while KV memory lasts; beyond that, evictions erase the saving.** — **7/10**
Fernandez et al. show that batch size changes the effect of every optimisation.
Kwon et al. deliberately did *not* keep chat caches between rounds because it
takes memory from other requests. On a 16 GB GPU this limit is easy to hit.
Test: N simulated users with think-time between turns, measuring energy per
request, cache hit rate and evictions as N grows. This needs a load generator
and the vLLM server mode, so it is more work than F1–F3.

## Further candidates (our own)

**O1. Can a cache-aware context strategy keep most of both the saving and the
quality?** — **9/10**
For example, drop old history in large chunks only every k turns instead of
sliding by one turn, or append to a summary instead of rewriting it. A plain
sliding window changes the prefix every turn, so nothing gets reused. A
cache-aware variant keeps the prefix stable for k turns. This is the most
original idea we have, and it extends B2 directly.

**O2. What is the net energy overhead of context summarization, and at what
conversation length does it reach a break-even point compared to KV-cache reuse?** — **9/10**
*(PL: Jaki jest narzut energetyczny ciągłego streszczania kontekstu i przy jakiej
długości rozmowy osiąga ono próg opłacalności w porównaniu z ponownym użyciem pamięci
podręcznej KV?)*
Summarisation compresses dialogue to keep input lengths small, but each summarisation
call requires extra autoregressive generation (decode-heavy work, which Fernandez
et al. show consumes far more energy per token than prefill). Furthermore, updating
a summary alters the prompt prefix, destroying prefix-cache blocks. In short
dialogues ($T < T^*$), the decode energy overhead of summarisation exceeds the
energy saved by shortening prefill. This question determines the exact break-even
turn count $T^*$ where summarisation becomes energetically viable compared to
caching the full history.

**O3. What does a "cache-hostile" prompt cost?** — **6/10**
Examples are a timestamp or user ID at the top of the system prompt, or an
edited earlier message. Because caching needs an identical prefix in whole
blocks, one changed token at the start should cost about as much as having no
cache. It is a small experiment with a practical message, but the result is
fairly predictable.

**O4. Does the ranking of strategies change when energy is counted per
*correct* answer instead of per turn?** — **8/10**
This is the metric that ties energy and quality into one number, which is what
the main question asks for. It is not a separate experiment, just how we report
B2/O1.

---

## Proposal for the poster

The main line is **B1 + F1/F2**, then **B2 + O1 + O2**, reported with **O4**.
F4 is the stretch goal. B3 is mentioned as a side question. F3 and O3
go in the discussion only.
