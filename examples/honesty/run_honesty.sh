#!/bin/bash
#SBATCH --job-name=honesty_repe
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
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

# --- run ---
REPO="$HOME/matrix/representation-engineering"
cd "$REPO/examples/honesty"
"$PYTHON" honesty.py \
    --model "mistralai/Mistral-7B-Instruct-v0.1" \
    --output-dir "./output-mistral-test" \
    --batch-size 16

echo "Finished: $(date)"
