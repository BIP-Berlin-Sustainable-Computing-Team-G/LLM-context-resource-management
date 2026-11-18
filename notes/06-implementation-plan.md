# 06 — Implementation Plan (Phase 2)

How we run [`03-experiment-plan.md`](03-experiment-plan.md) on the Science Club
server: **2 × RTX 4080 16 GB, 32 GB RAM, shared with other users, probably no
root.** Every step has a fallback that needs no root.

Design rules:
1. **Assume no root.** Nothing in the plan needs `sudo`. Things that do (driver
   update, power limits, RAPL) are recorded, not changed.
2. **One way to start the server, four ways to run it.** `serve.sh` hides
   whether vLLM runs in Apptainer, Podman, Docker or a plain venv. Same
   pinned vLLM version and the same flags in every case.
3. **Every run can crash.** Runs are small, independent and resumable; a crash
   costs one run, not a night.
4. **Raw data is never edited.** One JSON file per run, written atomically.

## 1. Day-1 check (`scripts/preflight.sh`, no root)

Run once on the server before choosing anything. Output goes to
`experiments/results/raw/preflight/<date>.txt` so the setup is documented.

| Check | Command | Why |
|---|---|---|
| GPUs, driver, max CUDA version | `nvidia-smi` | decides which vLLM version we can use (we cannot update the driver) |
| Other users on the GPUs | `nvidia-smi --query-compute-apps=pid,used_memory --format=csv` | shared server |
| NVML energy counter | `python -c` with `nvidia-ml-py`: `nvmlDeviceGetTotalEnergyConsumption` read twice | per-turn energy (Volta and newer; check that it updates and how often) |
| Power limit, clocks, persistence mode | `nvidia-smi -q -d POWER,CLOCK,PERFORMANCE` | record only; changing needs root |
| CPU energy (RAPL) | `cat /sys/class/powercap/intel-rapl:0/energy_uj` | usually root-only on recent kernels → then GPU energy only, as Fernandez et al. |
| Container runtimes | `which apptainer singularity podman docker`; `docker info`; `podman info` | choose the runtime (§2) |
| Disk and quota | `df -h $HOME /scratch* /tmp`; `quota -s` | need ≈ 50 GB (§4) |
| RAM | `free -g` | 32 GB shared by two vLLM processes |
| Internet / Hugging Face | `curl -I https://huggingface.co` | otherwise copy model and images from the laptop |
| Session tools | `which tmux screen` | long runs must survive logout |
| Python | `python3 --version`; `uv --version` (installable into `$HOME` without root) | venv fallback |

## 2. Runtime: first one that works

| Order | Runtime | Needs | How vLLM runs |
|---|---|---|---|
| A | **Apptainer / Singularity** (common on shared machines) | installed by admin; no root to use | `apptainer pull vllm.sif docker://vllm/vllm-openai:<pinned tag>` once, then `apptainer run --nv vllm.sif …` |
| B | Podman rootless | subuid/subgid and NVIDIA CDI set up by admin | `podman run --device nvidia.com/gpu=0 vllm/vllm-openai:<tag> …` |
| C | Docker | membership in the `docker` group (root-equivalent, often not given) | `docker run --gpus device=0 …` |
| D | **Plain venv** (always possible) | only the NVIDIA driver | `uv venv` in our project dir, `uv pip install vllm==<pinned>`, `vllm serve …` |

- A and D are the realistic options without root; D is the guaranteed fallback.
- **Driver constraint:** each vLLM release is built for a CUDA version, and the
  driver must support it (`nvidia-smi` shows the maximum). If the newest vLLM
  needs a newer driver, we pin an older vLLM release that matches. Decided on
  day 1, then frozen for the whole project.
- The chosen runtime, image digest or wheel version is written into every run
  record.
- If nothing works (no Apptainer, driver too old for any usable vLLM): ask the
  Science Club admins for Apptainer or a driver update. That is the only
  blocker that needs someone else.

## 3. Processes on the server

```
 GPU 0                                   GPU 1
 ┌──────────────────────────────┐        ┌──────────────────────────────┐
 │ vLLM server (container/venv) │        │ vLLM server (container/venv) │
 │ 127.0.0.1:8100               │        │ 127.0.0.1:8101               │
 │ cache ON or OFF (per block)  │        │ cache ON or OFF (per block)  │
 └──────────────▲───────────────┘        └──────────────▲───────────────┘
                │ HTTP (OpenAI API, /metrics, /health)  │
 ┌──────────────┴───────────────────────────────────────┴───────────────┐
 │ Harness (host venv, in tmux): runner → strategy → client → recorder  │
 │ Energy meter thread: NVML counter + power samples for GPU 0 and 1    │
 └──────────────────────────────────┬───────────────────────────────────┘
                                    ▼
            experiments/results/raw/<exp>/<run_id>.json  → git + rsync to laptop
```

- **One vLLM process per GPU** (`CUDA_VISIBLE_DEVICES=0` / `=1`), each bound to
  `127.0.0.1` on a high port. No ports open to the network; we connect with
  `ssh -L`.
- A **block** = one server configuration on one GPU. Blocks are started and
  stopped by the runner. Cache-on and cache-off blocks are spread over both
  GPUs (GPU = blocking factor), so a difference between the two cards cannot
  look like a caching effect.
- The harness runs **outside** the container: it needs no GPU, only NVML
  (`nvidia-ml-py` talks to the host driver without root).
- Server flags (identical in all blocks except the caching flag):
  `--enable-prefix-caching` | `--no-enable-prefix-caching`,
  `--model Qwen/Qwen3-4B --revision <commit>`, `--dtype bfloat16`,
  `--max-model-len 24576`, `--gpu-memory-utilization 0.90`, `--seed 0`,
  `--host 127.0.0.1 --port <p>`. Environment: `VLLM_SERVER_DEV_MODE=1` (enables
  `POST /reset_prefix_cache`), `HF_HOME=<project dir>/hf`.
- RAM: two processes with an 8 GB model each fit in 32 GB, but CPU swap
  space for preempted requests is set low (check the flag name in the pinned
  version) so the two servers plus the harness stay under ≈ 24 GB.

## 4. Disk layout on the server

Everything inside one project directory (`$PROJECT`, on the disk with the most
quota), never in `/tmp`:

| Path | Content | Size (estimate) |
|---|---|---|
| `$PROJECT/repo` | git clone of this repository | small |
| `$PROJECT/hf` | model weights (`HF_HOME`), pinned revision | ≈ 8 GB |
| `$PROJECT/images` | `vllm.sif` + `APPTAINER_CACHEDIR` / `APPTAINER_TMPDIR` | ≈ 20 GB during pull |
| `$PROJECT/venv` | harness venv (+ vLLM venv for runtime D) | ≈ 10 GB |
| `repo/experiments/results/raw` | run records (JSON, KB each) | < 1 GB |

No internet on the server → `huggingface-cli download Qwen/Qwen3-4B
--revision <commit>` and the image/wheels on another machine, then `rsync`.

## 5. Harness (Python, developed on the Mac)

Repository layout (created when the code exists, no placeholders):

```
experiments/
  harness/          conversations.py  strategies.py  client.py  energy.py
                    vllm_metrics.py   runner.py      store.py   ecologits.py
  tests/            unit tests (pytest), fake OpenAI server for dry runs
  conversations/    v1 scripted conversations (JSON), frozen assistant replies
  configs/          e1.yaml, e2.yaml, e4.yaml: conditions, repetitions, seeds
  scripts/          preflight.sh, serve.sh, run.sh
  results/raw/      one JSON per run (immutable)
  analysis/         reads raw/, writes figures and tables
  pyproject.toml + uv.lock
```

| Module | Does | Tested by |
|---|---|---|
| `strategies` | full, sliding, chunked, batch summary, rolling summary → list of messages | unit tests: which turns are kept, prefix stability, token budget |
| `conversations` | load scripts, planted facts, graded questions, expected answers | schema test, fact-position test |
| `client` | streaming request; TTFT and end time; `ignore_eos` + `max_tokens` for energy runs | fake server |
| `energy` | NVML counter before/after each request (mJ), 10 Hz power samples as cross-check, temperature, foreign processes; idle baseline | fake NVML |
| `vllm_metrics` | `/metrics` before/after each request: prefix-cache queries/hits, KV usage, preemptions | parser test on a saved sample |
| `runner` | builds the randomised run list from a config (seeded), skips runs whose file exists, restarts the server on failure | dry run end to end against the fake server |
| `store` | write `<run_id>.json.tmp`, then rename (atomic) | unit test |
| `ecologits` | EcoLogits formula per turn (RQ8) | test against a hand calculation |

Cached tokens are taken from the `/metrics` deltas. With one request at a time
they belong to that request. The usage field `cached_tokens`
(`--enable-prompt-tokens-details`) is logged too, but it is reported empty in
some vLLM V1 versions.

**Run record** (one per run): run ID, experiment, condition, conversation ID,
repetition, GPU index, git commit, runtime, vLLM version, image digest or
wheel, model revision, all server flags, driver and CUDA version, idle power,
start/end time; and per turn: prompt tokens, cached tokens, output tokens,
TTFT, duration, energy (counter delta), mean power, temperature, foreign GPU
processes (yes/no), answer text, graded result.

## 6. Run procedure

1. Start the block's server; wait for `/health`; read the KV capacity from
   the start-up log.
2. Idle baseline: 60 s with the model loaded, no requests.
3. Warm-up conversation (discarded).
4. For each run in the block (randomised order): `POST /reset_prefix_cache`;
   check that the first request shows 0 cache hits; wait until the GPU is below
   a set temperature; replay the conversation; write the run file.
5. Before and after every run: check `nvidia-smi` for other users' processes.
   If one appears, the run is marked invalid and repeated later.
6. Stop the server; next block.

Long runs: `tmux new -s exp` → `scripts/run.sh configs/e2.yaml`. After a
logout, crash or reboot, the same command continues where it stopped.

## 7. What can go wrong and what we do

| Risk | How we notice | Fallback |
|---|---|---|
| No Docker / no root | preflight | Apptainer (A) or venv (D) |
| Driver too old for the newest vLLM | preflight: max CUDA in `nvidia-smi` | pin an older vLLM release |
| NVML energy counter missing or coarse | preflight: counter does not update per second | integrate 10 Hz power samples; longer turns; aggregate per conversation |
| RAPL not readable | preflight | report GPU energy only (as Fernandez et al.); say so in the report |
| Other users on our GPUs | foreign-process check per run | book time slots with the Science Club; discard and repeat affected runs |
| vLLM crashes / out of memory | `/health` fails, process exits | runner restarts the server, retries the run twice, then logs it as failed |
| `/reset_prefix_cache` missing or silent | first-request hits ≠ 0 | restart the server between runs (slower, same result) |
| `cached_tokens` empty in usage | field is null | `/metrics` deltas (primary anyway) |
| Server reboot, SSH drop | runner stops | tmux; resumable runner; rerun the same command |
| No internet on the server | preflight | copy model, image and wheels with `rsync` |
| Disk quota | preflight / pull fails | all caches in `$PROJECT`; delete the image pull cache after building the SIF |
| GPU heat changes power | temperature in each record | cool-down before each run; randomised order spreads the effect |
| Data loss | — | results committed to git and `rsync`ed to a laptop after every session |
| Server unavailable for weeks | — | E1 is small enough (≈ 4 h) to run on any single CUDA GPU; RQ8 needs no GPU. Last resort: a Windows PC with an NVIDIA GPU, see §10 |

## 8. GPU time (calculation, to be checked in the pilot)

Assumed decode speed ≈ 60–70 tokens/s for a 4B model in bf16 on one 4080
(upper bound from memory bandwidth: 717 GB/s ÷ 8 GB of weights ≈ 90 reads/s).
A 40-turn run with 300-token answers then takes ≈ 4 min.

| Experiment | Runs | GPU hours |
|---|---|---|
| E1: 5 scripts × 2 cache × 2 answer lengths × 5 reps | 100 | ≈ 4 |
| E2 energy: 5 strategies × 2 cache × 6 scripts × 5 reps | 300 | ≈ 22 (+ summary calls) |
| E2 quality: 5 strategies × 2 cache × 6 scripts × 2 reps + LoCoMo subset | ≈ 150 | ≈ 8 |
| E4 (stretch) | — | ≈ 5 |
| **Total** | | **≈ 40 GPU h ≈ 20 h on two GPUs** |

Shorter conversations (T = 10, 20) come from the same runs: cumulative energy
up to turn T.

## 9. Milestones and owners (proposal)

| When | Milestone | Go / no-go |
|---|---|---|
| Week after the poster | Server account; `preflight.sh` run; runtime chosen | if no runtime works → ask the admins |
| Late Nov | Qwen3-4B served; one request measured with NVML; KV capacity read | energy counter usable? else power-sample integration |
| Early Dec | Harness + strategies + tests (Mac); conversation set v1 with frozen replies (server) | dry run passes end to end |
| Mid Dec | Pilot: 1 rep per E1/E2 cell → set repetitions from the variance; E1 full | variance small enough for ≥ 5 reps in budget |
| Early Jan | E2 + E3 analysis; E4 if time | — |
| Late Jan | Report (8–10 pages) + final presentation | — |

| Area | Proposed owner |
|---|---|
| Server, runtime, `serve.sh`, vLLM flags | Valeria (PagedAttention / vLLM) |
| Energy measurement, idle baseline, statistics | Lena (vLLM configurations paper: same method) |
| Conversation set, fact positions, grading, summary prompts | Asia (Lost in the Middle, strategies) |
| Harness core, runner, analysis plots, EcoLogits formula | Jan (Fernandez et al.) |

## 10. Fallback: Windows PC (16 GB RAM)

Only if the server is unavailable for weeks. Checked 2026-11-17.

- **vLLM does not run natively on Windows**; the supported route is **WSL2**
  (Ubuntu) with the Windows NVIDIA driver. Inside WSL2 no Linux GPU driver is
  installed. The rest of this plan (venv runtime D, harness, run procedure)
  stays the same.
- **The GPU decides, not the RAM.** With ≥ 16 GB VRAM: same setup as the
  server. With 12 GB: Qwen3-4B fits with only ≈ 2 GB of KV → short
  conversations only. With 8 GB: a smaller model (Qwen3-1.7B) → a different
  experiment, not comparable.
- **16 GB RAM is tight.** WSL2 gets 50 % of the RAM by default (8 GB) and swap
  of 25 % (4 GB). Loading an 8 GB model needs more → set `memory=12GB` and
  `swap=16GB` in `%UserProfile%\.wslconfig`; run one vLLM process only, close
  other programs.
- **Energy:** NVML inside WSL2 supports only part of the queries (NVIDIA *CUDA
  on WSL* guide). Read the energy counter on the Windows side instead
  (`nvidia-smi.exe` or `nvidia-ml-py` in Windows Python) — check on day 1.
- **Noise:** the same GPU drives the Windows desktop → no other use of the PC
  during runs, monitor off or on the integrated GPU if there is one; Windows
  updates and sleep switched off for the run period.
- Results from this PC are reported as a separate platform, not mixed with
  4080 server numbers.
