#!/bin/bash
#SBATCH --job-name=translate_en_es_qwen
#SBATCH --qos=standby
#SBATCH --requeue
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --partition=m13l
# 2 GPUs: the first GPU on m13l nodes is sometimes unavailable, which makes
# device_map="auto" silently fall back to (very slow) CPU inference instead
# of erroring — see honesty/run_honesty.sh history. Also matches the GPU
# count the Qwen2.5-32B honesty job (12727650) actually needed to fit.
#SBATCH --gres=gpu:l40s:2
#SBATCH --output=%x_%j.out
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=koltonrbaldwin@gmail.com

# --- print job info for debugging ---
echo "Job ID:    $SLURM_JOB_ID"
echo "Node:      $SLURMD_NODENAME"
echo "Started:   $(date)"

# --- environment setup ---
PYTHON="$HOME/.conda/envs/repe/bin/python"

# --- model cache: use scratch if available, otherwise fall back to home ---
if [ -d "/scratch/$USER" ]; then
    export HF_HOME="/scratch/$USER/hf_cache"
elif [ -d "/scratch" ] && [ -w "/scratch" ]; then
    export HF_HOME="/scratch/$USER/hf_cache"
else
    export HF_HOME="$HOME/.cache/huggingface"
fi
mkdir -p "$HF_HOME"
echo "HF_HOME: $HF_HOME"

# compute nodes have no internet — use only what's already cached
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

# --- run ---
REPO="$HOME/matrix/representation-engineering"
cd "$REPO/examples/translation_confidence"
"$PYTHON" translate.py \
      --model "Qwen/Qwen2.5-32B-Instruct" \
      --input-dir "./input" \
      --output-dir "./output-qwen-32b"

echo "Finished: $(date)"
