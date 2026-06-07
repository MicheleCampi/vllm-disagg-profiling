#!/bin/bash
# ARM A — Aggregated TP=2: un solo vLLM, tensor-parallel su 2 GPU,
# prefill+decode insieme. Confronto a parità di hardware (2 GPU) con l'arm B (1P1D).
# Parametri di carico IDENTICI all'esempio disaggregated per confronto equo.
set -e

MODEL="meta-llama/Llama-3.1-8B-Instruct"

# Carico: passato come argomenti, default = regime prefill-heavy (come esempio upstream)
INPUT_LEN=${INPUT_LEN:-7500}
OUTPUT_LEN=${OUTPUT_LEN:-200}
NUM_PROMPTS=${NUM_PROMPTS:-200}
REQUEST_RATE=${REQUEST_RATE:-3.6}
BURSTINESS=${BURSTINESS:-100}

echo "=== ARM A: Aggregated TP=2 | input=$INPUT_LEN output=$OUTPUT_LEN rate=$REQUEST_RATE ==="

# Avvia vLLM in tensor-parallel su 2 GPU, --enforce-eager per coerenza con arm B
CUDA_VISIBLE_DEVICES=0,1 vllm serve "$MODEL" \
    --host 0.0.0.0 \
    --port 8000 \
    --tensor-parallel-size 2 \
    --enforce-eager \
    --gpu-memory-utilization 0.8 \
    > >(tee aggregated_server.log) 2>&1 &
SERVER_PID=$!

# Attendi che il server risponda
echo "Waiting for aggregated server on port 8000..."
timeout 1200 bash -c 'until curl -s localhost:8000/v1/completions > /dev/null; do sleep 2; done'
echo "Server up. PID=$SERVER_PID"

echo "ARM A pronto. Lanciare inferscope --sample-only su questo PID/GPU, poi il benchmark."
echo "SERVER_PID=$SERVER_PID"
