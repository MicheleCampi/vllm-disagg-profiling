# vLLM Disaggregated Serving — Prefill/Decode Profiling

An A/B profiling experiment comparing **aggregated** (tensor-parallel) vs
**disaggregated** (1-prefiller / 1-decoder) LLM serving on equal hardware
(2 GPUs), measuring the per-device power/utilisation asymmetry between the
prefill and decode phases under load.

## Method

- **Arm A — Aggregated (TP=2):** one vLLM instance, tensor-parallel across 2 GPUs.
- **Arm B — Disaggregated (1P1D):** prefiller on GPU0 + decoder on GPU1 + proxy, KV cache transferred over NIXL/LMCache.
- **Load:** `vllm bench serve`, three regimes (prefill-heavy / balanced / decode-heavy), identical parameters across arms.
- **Measurement (three views):** client-side (throughput, TTFT, TPOT, p50/p99), server-side per-device GPU (inferscope `--sample-only --gpu`), and KV-transfer cost where observable.

Model: Llama-3.1-8B-Instruct. Both arms use `--enforce-eager` for comparability.

## Hypothesis

Disaggregation isolates prefill (compute-bound, ~70–100% GPU power) from
decode (memory-bound, ~20–40% GPU power). The experiment measures whether,
at equal hardware, disaggregation wins — and what the per-device view reveals
about hardware-utilisation cost (a potentially under-used decode GPU).

## Scripts

- `setup.sh` — one-time install on the GPU node (vLLM + LMCache + NIXL + inferscope build + model download).
- `arm_a_aggregated.sh` — launches the aggregated TP=2 server.
- `arm_b_disaggregated.sh` — launches the disaggregated 1P1D servers (reuses upstream vLLM/LMCache example).
- `run_benchmark.sh` — measurement orchestrator: runs inferscope + `vllm bench serve`, identical for both arms.

## Status

Work in progress — setup and scripts prepared; GPU execution pending.

## Tooling

Profiling by [inferscope](https://github.com/MicheleCampi/inferscope).
