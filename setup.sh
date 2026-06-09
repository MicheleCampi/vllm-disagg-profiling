#!/bin/bash
# SETUP — da eseguire UNA VOLTA sul nodo GPU appena acceso (2 GPU NVIDIA, CUDA 12.x).
# Validato su 2x H100 80GB SXM5 (Lambda Cloud): vLLM 0.22.1 cu12, Qwen2.5-7B-Instruct.
# Installa: vLLM + LMCache + NIXL(cu12) + (build) inferscope, e scarica il modello.
# NB: LMCache e' linkato a CUDA 12 (libcudart.so.12). L'ambiente DEVE essere CUDA 12.x.
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
echo ">>> ATTENZIONE: lo stack (vLLM/torch/LMCache) e' cu12. Se l'ambiente e' CUDA 13, fermarsi e cambiare immagine."

# --- 1) Ambiente Python isolato ----------------------------------------
echo "--- Creo venv Python 3.12 ---"
if ! command -v uv >/dev/null 2>&1; then
    echo "Installo uv (gestore pacchetti veloce)..."
    pip install uv
fi
uv venv --python 3.12 ~/disagg-venv
source ~/disagg-venv/bin/activate

# --- 2) vLLM (variante cu12 coerente col nodo) -------------------------
# NB: NON pinnare 0.21.0 (la sua unica wheel PyPI e' cu13 e rompe su stack cu12).
# --torch-backend=auto fa rilevare a uv il CUDA del nodo e scegliere la wheel cu12
# coerente sia per torch sia per vLLM. Risolve il mismatch libcudart.so.13.
echo "--- Installo vLLM (torch-backend auto, cu12) ---"
uv pip install --no-cache "vllm" --torch-backend=auto
echo "--- versione vLLM installata ---"
python3 -c "import vllm; print('vllm', vllm.__version__)"
python3 -c "import vllm._C; print('vllm._C OK')"

# --- 2b) vLLM source tree (esempi disaggregated + benchmarks) ----------
# arm_b usa examples/disaggregated/lmcache/..., run_benchmark usa benchmarks/.
# Allineo il clone alla versione runtime effettiva (v0.22.1): esempi <-> API coerenti.
echo "--- Clono vLLM source @ v0.22.1 ---"
if [ ! -d ~/vllm-src ]; then
    git clone --depth 1 --branch v0.22.1 https://github.com/vllm-project/vllm.git ~/vllm-src
else
    echo "~/vllm-src gia' presente (salto il clone)"
fi

# --- 2c) Patch DeepGEMM al launcher upstream ---------------------------
# vLLM 0.22.1 all'avvio sonda il backend FP8 DeepGEMM; se assente, crasha.
# Qwen gira in bf16 => DeepGEMM inutile. Lo disabilito via env davanti a
# CUDA_VISIBLE_DEVICES nei due blocchi (prefiller + decoder).
# Il launcher e' upstream (non versionabile nel repo): la patch va riapplicata
# qui ad ogni setup. Idempotente (skip se gia' patchato).
F=~/vllm-src/examples/disaggregated/lmcache/disagg_prefill_lmcache_v1/disagg_vllm_launcher.sh
if [ -f "$F" ] && ! grep -q "VLLM_USE_DEEP_GEMM=0" "$F"; then
    echo "--- Patch DeepGEMM al launcher ---"
    cp "$F" "$F.bak"
    sed -i 's|CUDA_VISIBLE_DEVICES=0 \\|VLLM_USE_DEEP_GEMM=0 VLLM_DEEP_GEMM_WARMUP=skip CUDA_VISIBLE_DEVICES=0 \\|; s|CUDA_VISIBLE_DEVICES=1 \\|VLLM_USE_DEEP_GEMM=0 VLLM_DEEP_GEMM_WARMUP=skip CUDA_VISIBLE_DEVICES=1 \\|' "$F"
    grep -n "VLLM_USE_DEEP_GEMM" "$F"
else
    echo "Launcher gia' patchato o assente (skip)"
fi

# --- 3) LMCache + NIXL (cu12) ------------------------------------------
echo "--- Installo LMCache ---"
uv pip install lmcache
# NIXL forzato a cu12: l'install di LMCache tira varianti cu13 (nixl-cu13,
# cupy-cuda13x) che confliggono con lo stack cu12 e causano libcudart.so.13.
# Rimuovo le cu13 e installo nixl-cu12.
# NOTA ONESTA: cura verificata per QUESTO mismatch (LMCache->cu13), ma la sequenza
# esatta sul ramo 0.22.1 non e' stata ri-testata post-esperimento. Se l'arm B non
# sale, e' qui che va guardato per primo.
echo "--- Forzo NIXL su variante cu12 ---"
uv pip uninstall nixl nixl-cu13 cupy-cuda13x 2>/dev/null || true
uv pip install nixl-cu12

# --- 4) Dipendenze di supporto per il benchmark ------------------------
echo "--- Installo pandas, datasets, quart (richieste dagli script vLLM) ---"
uv pip install pandas datasets quart

# --- 5) inferscope (build da sorgente, feature GPU NVIDIA) -------------
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
# inferscope su PATH per gli script di benchmark
mkdir -p ~/.local/bin
ln -sf ~/inferscope-src/target/release/inferscope ~/.local/bin/inferscope
echo "inferscope linkato in ~/.local/bin (assicurarsi sia in PATH prima di run_benchmark)"

# --- 6) Pre-download del modello (evita GPU idle durante il download) ---
echo "--- Pre-scarico Qwen2.5-7B-Instruct (pubblico, nessun token) ---"
source ~/disagg-venv/bin/activate
python3 -c "from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen2.5-7B-Instruct')" || echo "Download fallito o gia' presente"

echo "================ SETUP COMPLETO ================"
echo "venv:        ~/disagg-venv (source ~/disagg-venv/bin/activate)"
echo "inferscope:  ~/inferscope-src/target/release/inferscope (linkato in ~/.local/bin)"
echo "vLLM repo:   ~/vllm-src (esempi disaggregated + benchmarks, @ v0.22.1)"
echo "Prossimo: lanciare le arm via gli script in ~/disagg-experiment/"
