# vLLM Disaggregated Serving — Prefill/Decode Profiling

An A/B profiling experiment comparing **aggregated** (tensor-parallel) vs
**disaggregated** (1-prefiller / 1-decoder) LLM serving on equal hardware
(2× H100 80GB), measuring the per-device power/utilisation asymmetry between
the prefill and decode phases under load.

## Method

- **Arm A — Aggregated (TP=2):** one vLLM instance, tensor-parallel across 2 GPUs.
- **Arm B — Disaggregated (1P1D):** prefiller on GPU0 + decoder on GPU1 + proxy, KV cache transferred over NIXL/LMCache (`LMCacheConnectorV1`).
- **Load:** `vllm bench serve`, random dataset, 200 prompts at 3.6 req/s, three regimes (prefill-heavy 7500/200, balanced 2000/2000, decode-heavy 500/4000), identical parameters across arms.
- **Measurement (three views):** client-side (throughput, TTFT, TPOT, p99), server-side per-device GPU (inferscope `--sample-only --gpu`), and the gap between them.

Engine: vLLM 0.22.1 (cu12), as stated in commit `6ffd953`; the installed version is not logged in `results/`. Model: Qwen2.5-7B-Instruct. Both arms use `--enforce-eager` for comparability.

## Results

Disaggregation imposes a latency tax in every regime and never improves
throughput at this scale (1P1D, equal hardware). What it buys is **phase
isolation** — visible only per-device as a power asymmetry that grows
monotonically with the decode share of the workload.

| Regime | Throughput (disagg vs agg) | TTFT mean (agg → disagg) | Per-device power ratio (disagg) |
|---|---|---|---|
| prefill-heavy | tie (~26.7k tok/s) | 148 → 443 ms (p99 → 1056 ms) | 1.22× |
| balanced | −1.6% | 62 → 142 ms | 2.48× |
| decode-heavy | **−11%** | 43 → 80 ms | **3.66×** |

In decode-heavy the disaggregated prefiller drops to **1% utilisation / 127 W**
(idle) while the decoder runs at **70% / 466 W** — a stranded GPU that appears
nowhere in the client output. The aggregated arm stays symmetric (1.00×) in all
regimes. Zero failed requests across all six runs.

Full analysis: see the [write-up](https://github.com/MicheleCampi/vllm-disagg-profiling).
Raw results in `results/` (six inferscope JSON + benchmark logs).

## Scripts

- `setup.sh` — one-time install on the GPU node (vLLM + LMCache + NIXL + inferscope build + model download).
- `arm_a_aggregated.sh` — launches the aggregated TP=2 server.
- `arm_b_disaggregated.sh` — launches the disaggregated 1P1D servers (reuses upstream vLLM/LMCache example).
- `run_benchmark.sh` — measurement orchestrator: runs inferscope + `vllm bench serve`, identical for both arms.

## Reproducibility

Validated on a single Lambda Cloud node, **2× H100 80GB SXM5**, CUDA 12.x.

Verified stack (see `setup.sh`):
- **vLLM 0.22.1 (cu12)**, installed via `uv pip install "vllm" --torch-backend=auto`. Note: do *not* pin `0.21.0` — its only PyPI wheel is cu13 and breaks on a cu12 stack.
- **Qwen2.5-7B-Instruct** (public, no HF token).
- **DeepGEMM disabled** (`VLLM_USE_DEEP_GEMM=0`): Qwen runs in bf16, and the FP8 backend probe otherwise aborts startup. Applied to the upstream disaggregation launcher by `setup.sh`.
- **NIXL forced to the cu12 variant**: installing LMCache pulls cu13 variants that conflict with the cu12 stack.

Flow: `clone → setup.sh → arm_a_aggregated.sh | arm_b_disaggregated.sh → run_benchmark.sh <arm> <port> <regime>`.

> Honesty note: the NIXL→cu12 step is the verified cure for the LMCache-cu13 conflict, but the exact resolution was not re-tested end-to-end after the experiment. If the disaggregated arm fails to start on a fresh node, check the NIXL variant first.

> Not in this repository: the server logs (`aggregated_server.log`, `prefiller.log`, `decoder.log`, `proxy.log`) are not committed, and environment variables or arguments set by hand on the node, outside these scripts, would leave no trace here.

## Status

Complete — both arms executed on 2× H100, six runs profiled. Results in `results/`.

## Tooling

Profiling by [inferscope](https://github.com/MicheleCampi/inferscope).
