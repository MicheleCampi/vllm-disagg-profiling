#!/bin/bash
# ORCHESTRATORE di misura: lancia inferscope (sample-only, per-device GPU) + vllm bench serve.
# Usato IDENTICO per entrambe le arm. Cambia solo --port (8000=aggregated, 9000=disaggregated).
#
# Uso: ./run_benchmark.sh <arm_label> <port> <regime_label>
#   es: ./run_benchmark.sh aggregated 8000 prefill_heavy
#       ./run_benchmark.sh disaggregated 9000 prefill_heavy
set -e

ARM_LABEL=$1          # aggregated | disaggregated
PORT=$2               # 8000 | 9000
REGIME=$3             # prefill_heavy | balanced | decode_heavy

if [[ -z "$ARM_LABEL" || -z "$PORT" || -z "$REGIME" ]]; then
    echo "Uso: $0 <arm_label> <port> <regime_label>"
    exit 1
fi

MODEL="meta-llama/Llama-3.1-8B-Instruct"

# Parametri di carico (default = prefill-heavy come esempio upstream; override via env)
INPUT_LEN=${INPUT_LEN:-7500}
OUTPUT_LEN=${OUTPUT_LEN:-200}
NUM_PROMPTS=${NUM_PROMPTS:-200}
REQUEST_RATE=${REQUEST_RATE:-3.6}
BURSTINESS=${BURSTINESS:-100}

# Durata finestra inferscope: stima generosa, ritagliamo in analisi.
# A request-rate 3.6 e 200 prompt -> ~55s di invio + coda/decode. 300s copre con margine.
SAMPLE_DURATION=${SAMPLE_DURATION:-300}

OUTDIR="$HOME/disagg-experiment/results/${ARM_LABEL}_${REGIME}"
mkdir -p "$OUTDIR"

echo "=== RUN: arm=$ARM_LABEL port=$PORT regime=$REGIME ==="
echo "    input=$INPUT_LEN output=$OUTPUT_LEN prompts=$NUM_PROMPTS rate=$REQUEST_RATE"
echo "    inferscope window=${SAMPLE_DURATION}s -> $OUTDIR"

# 1) Avvia inferscope in sample-only sul sistema (per-device GPU vede entrambe le GPU).
#    Si attacca a un PID per la finestra; --gpu abilita il campionamento NVML per-device.
#    NB: il PID di un server vLLM e' passato via env PROBE_PID (lo fornisce l'orchestratore arm).
inferscope --sample-only \
    --pid "${PROBE_PID:-1}" \
    --include-descendants \
    --gpu \
    --duration-secs "$SAMPLE_DURATION" \
    > "$OUTDIR/inferscope_${ARM_LABEL}_${REGIME}.json" 2>"$OUTDIR/inferscope.err" &
INFERSCOPE_PID=$!
echo "inferscope avviato (PID=$INFERSCOPE_PID), campiona per ${SAMPLE_DURATION}s"

# 2) Lancia il benchmark vLLM contro la porta dell'arm in esame.
cd "$HOME/vllm-src/benchmarks"
vllm bench serve --port "$PORT" --seed "$(date +%s)" \
    --model "$MODEL" \
    --dataset-name random \
    --random-input-len "$INPUT_LEN" \
    --random-output-len "$OUTPUT_LEN" \
    --num-prompts "$NUM_PROMPTS" \
    --burstiness "$BURSTINESS" \
    --request-rate "$REQUEST_RATE" \
    | tee "$OUTDIR/benchmark_${ARM_LABEL}_${REGIME}.log"

echo "Benchmark finito. Attendo chiusura finestra inferscope..."
wait $INFERSCOPE_PID 2>/dev/null || true
echo "=== RUN completo. Risultati in $OUTDIR ==="
ls -la "$OUTDIR"
