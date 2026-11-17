# Fernandez et al. 2025 — Energy Considerations of LLM Inference and Efficiency Optimizations

J. Fernandez, C. Na, V. Tiwari, Y. Bisk, S. Luccioni, E. Strubell. ACL 2025 (long),
pp. 32556–32569. doi:10.18653/v1/2025.acl-long.1563 · `papers/2025.acl-long.1563.pdf`
Owner: Jan

## In short

The paper measures how much GPU energy LLM inference uses and how much common
optimisations save, across input/output lengths, batch sizes, decoding
strategies, frameworks (PyTorch/HF vs vLLM) and GPUs. The main message: the
effect of an optimisation depends on the shape of the workload, the software
stack and the hardware. Measured energy is far above what a FLOPs-based estimate
predicts. Applied properly, the optimisations save up to 73 % on simulated
real-world workloads.

## Setup

- Models: Llama-3.1-8B (Base/Instruct), Qwen-1.5-32B, OLMo 1B/7B vs OLMoE 1B-7B.
- GPUs: RTX A6000, A6000 Ada, A100 80GB PCIe; tensor parallelism up to 4 GPUs.
- Controlled sweeps of input length (up to 32k) and output length (up to 4k), batch 1–1024.
- **GPU energy only**, via NVML/CodeCarbon. Warm-up, then **mean of 3 runs**.
- Real workloads (BurstGPT, Azure chat/code traces) modelled offline by *binning*
  requests on (input length, output length) and measuring one full batch per bin (§4.1).
- "Ideal" baseline: TDP / peak FLOPS × FLOPs of the workload (§4.2).

## Key findings

1. **Prefill vs decode behave differently** (Fig. 2). At short inputs and small
   batches, prefill energy grows *sub-linearly* with input length because the GPU
   is under-utilised. Decode energy grows linearly with output length even at
   small batch, and decode costs more energy per token than prefill.
2. **Decode dominates total energy** except for tasks with very short outputs,
   e.g. classification (§3.1).
3. **Faster is not the same as cheaper.** Going from 1 to 2 or 4 GPUs with tensor
   parallelism cuts latency by 40 % / 61 % but raises energy by 29 % / 55 %
   (Fig. 6). Speculative decoding saves energy only at batch ≤ 16 and costs
   +25.65 % at batch 128 (Fig. 3).
4. **Batch size changes the picture for every optimisation.** CUDA graphs help
   most at small batch. Continuous batching has overhead at small batch and
   saves up to about 5 % at large batch (App. D). Long inputs limit the maximum
   batch size because of memory.
5. **Gains depend on the GPU.** `torch.compile` saves 29.9 % on A100 but 2 % on
   A6000 (§3.4).
6. **Theory underestimates reality.** Even optimised vLLM uses 27–64 % more
   energy than the FLOPs × TDP estimate, and unoptimised PyTorch 103–507 % more
   (Table 4). vLLM saves 73 % / 38 % / 72 % compared to PyTorch on BurstGPT /
   Azure Code / Azure Chat.
7. **Real chat requests have long inputs and short outputs.** For Azure Chat the
   median input is 928 tokens and the median output 41 (Tables 1–2).

## Limitations

- GPU energy only. CPU, RAM and cooling are excluded (stated by the authors).
- Offline batch processing. A chat service is online and latency-bound, so the
  73 % does not carry over to it directly.
- Every request is treated independently: **no KV-cache reuse, no multi-turn
  structure**. For chat traces the long inputs are mostly re-sent history,
  which is exactly what caching would skip.
- 3 runs, means only, no spread and no significance tests.
- Answer quality is not measured.

## What it means for our project

- It gives us the method: separate prefill and decode, and vary input and output
  length independently. With caching, a turn's prefill only covers the new
  tokens, so the same decomposition applies **per turn**.
- Finding 1 → caching probably saves almost nothing until the history is long
  enough to load the GPU. There should be a break-even history length.
- Finding 2 → caching only removes prefill work, so its energy saving should be
  much smaller than its token saving.
- Finding 3 → a lower TTFT does not mean lower energy. We have to measure both.
- Finding 4 → single-user (batch 1) and multi-user results can differ, and KV
  memory is the shared constraint.
- Finding 6 → token- or FLOP-based estimators (EcoLogits) need checking against
  measurement.
