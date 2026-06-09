#!/bin/bash
# ARM B — Disaggregated 1P1D: prefiller GPU0 + decoder GPU1 + proxy.
# Adattato da examples/disaggregated/lmcache/disagg_prefill_lmcache_v1/disagg_example_nixl.sh
# Avvia SOLO i server (prefiller+decoder+proxy); benchmark e inferscope li lancia l'orchestratore.
set -e

# Path agli script upstream del setup LMCache/NIXL (gia' clonati in ~/vllm-src)
LMCACHE_DIR="$HOME/vllm-src/examples/disaggregated/lmcache/disagg_prefill_lmcache_v1"

PIDS=()
cleanup() {
    echo "Stopping disaggregated servers..."
    trap - INT TERM
    kill -- -$$ 2>/dev/null || true
    wait 2>/dev/null || true
    exit 0
}
trap cleanup INT TERM USR1

wait_for_server() {
  local port=$1
  echo "Waiting for server on port $port..."
  timeout 1200 bash -c "until curl -s localhost:${port}/v1/completions > /dev/null; do sleep 2; done"
}

cd "$LMCACHE_DIR"

echo "=== ARM B: Disaggregated 1P1D (prefiller GPU0 + decoder GPU1 + proxy:9000) ==="
echo "Launching prefiller, decoder, proxy. Logs: prefiller.log decoder.log proxy.log"

bash disagg_vllm_launcher.sh prefiller Qwen/Qwen2.5-7B-Instruct > >(tee prefiller.log) 2>&1 &
PIDS+=("$!")

bash disagg_vllm_launcher.sh decoder Qwen/Qwen2.5-7B-Instruct > >(tee decoder.log) 2>&1 &
PIDS+=("$!")

python3 disagg_proxy_server.py \
    --host localhost --port 9000 \
    --prefiller-host localhost --prefiller-port 8100 \
    --decoder-host localhost --decoder-port 8200 \
    > >(tee proxy.log) 2>&1 &
PIDS+=("$!")

wait_for_server 8100
wait_for_server 8200
wait_for_server 9000

echo "All disaggregated servers up. Proxy on port 9000."
echo "Prefiller PID=${PIDS[0]} (GPU0) | Decoder PID=${PIDS[1]} (GPU1) | Proxy PID=${PIDS[2]}"
echo "Ora lanciare inferscope --sample-only --gpu (vede entrambe le GPU per-device), poi il benchmark sulla porta 9000."

# Mantieni i server vivi finche' non interrotto (l'orchestratore gestisce benchmark + stop)
wait
