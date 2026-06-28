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

# --- cache models to scratch so they don't eat your home quota ---
export HF_HOME="/scratch/$USER/hf_cache"
mkdir -p "$HF_HOME"

# --- run ---
REPO="$HOME/representation-engineering"
cd "$REPO/examples/honesty"
"$PYTHON" honesty.py \
    --model "mistralai/Mistral-7B-Instruct-v0.1" \
    --output-dir "./output-mistral-test" \
    --batch-size 16

echo "Finished: $(date)"
