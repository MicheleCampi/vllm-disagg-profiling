#!/bin/bash
# SETUP — da eseguire UNA VOLTA sul nodo GPU appena acceso (2x A100, CUDA 12.x).
# Installa: vLLM + LMCache + NIXL + (build) inferscope, e scarica il modello.
# NB: LMCache wheel e' linkato a CUDA 12 (libcudart.so.12). L'ambiente DEVE essere CUDA 12.x.
set -e

echo "================ SETUP DISAGG EXPERIMENT ================"

# --- 0) Precondizioni ---------------------------------------------------
echo "--- Verifica GPU e CUDA ---"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || { echo "nvidia-smi non disponibile"; exit 1; }
NUM_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
echo "GPU trovate: $NUM_GPUS"
if [ "$NUM_GPUS" -lt 2 ]; then echo "ERRORE: servono almeno 2 GPU."; exit 1; fi

echo "--- Versione CUDA driver/toolkit ---"
nvcc --version 2>/dev/null | grep release || echo "nvcc non in PATH (ok se il driver e' CUDA 12.x)"
echo ">>> ATTENZIONE: LMCache richiede runtime CUDA 12.x. Se l'ambiente e' CUDA 13, fermarsi e cambiare immagine."

# --- 1) HF_TOKEN --------------------------------------------------------
if [ -z "$HF_TOKEN" ] || [[ "$HF_TOKEN" != hf_* ]]; then
    echo "ERRORE: HF_TOKEN non impostato o non valido (deve iniziare con hf_)."
    echo "Impostalo con: export HF_TOKEN=hf_xxx   (necessario per scaricare Llama-3.1-8B)"
    exit 1
fi
echo "HF_TOKEN ok."

# --- 2) Ambiente Python isolato ----------------------------------------
echo "--- Creo venv Python 3.12 ---"
if ! command -v uv >/dev/null 2>&1; then
    echo "Installo uv (gestore pacchetti veloce)..."
    pip install uv
fi
uv venv --python 3.12 ~/disagg-venv
source ~/disagg-venv/bin/activate

# --- 3) vLLM (CUDA 12.9 wheel index) -----------------------------------
echo "--- Installo vLLM (wheel CUDA 12.x) ---"
uv pip install vllm \
    --extra-index-url https://download.pytorch.org/whl/cu129 \
    --index-strategy unsafe-best-match

# --- 4) LMCache + NIXL --------------------------------------------------
echo "--- Installo LMCache ---"
uv pip install lmcache
echo "--- Installo NIXL ---"
uv pip install nixl

# --- 5) Dipendenze di supporto per il benchmark ------------------------
echo "--- Installo pandas, datasets, quart (richieste dagli script vLLM) ---"
uv pip install pandas datasets quart

# --- 6) inferscope (build da sorgente, feature GPU NVIDIA) -------------
echo "--- Build inferscope con gpu-nvidia ---"
if [ ! -d ~/inferscope-src ]; then
    git clone --depth 1 https://github.com/MicheleCampi/inferscope.git ~/inferscope-src
fi
if ! command -v cargo >/dev/null 2>&1; then
    echo "Installo Rust toolchain..."
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
    source "$HOME/.cargo/env"
fi
cd ~/inferscope-src
cargo build --release --features gpu-nvidia
echo "inferscope built: $(ls -la target/release/inferscope)"

# --- 7) Pre-download del modello (evita di pagare GPU idle durante il download) ---
echo "--- Pre-scarico Llama-3.1-8B-Instruct ---"
source ~/disagg-venv/bin/activate
python3 -c "from huggingface_hub import snapshot_download; snapshot_download('meta-llama/Llama-3.1-8B-Instruct')" || echo "Download fallito o gia' presente"

echo "================ SETUP COMPLETO ================"
echo "venv:        ~/disagg-venv (source ~/disagg-venv/bin/activate)"
echo "inferscope:  ~/inferscope-src/target/release/inferscope"
echo "vLLM repo:   ~/vllm-src (per benchmarks/ ed esempi disaggregated)"
echo "Prossimo: lanciare le arm via gli script in ~/disagg-experiment/"
