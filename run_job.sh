#!/bin/bash
#SBATCH --job-name=liew_replication
#SBATCH --account=def-hrouhani
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=/scratch/yasir071/output/liew_replication/%x_%j.out
#SBATCH --error=/scratch/yasir071/output/liew_replication/%x_%j.err

# ─────────────────────────────────────────
# SETUP
# ─────────────────────────────────────────
module load python/3.10
module load cuda/11.8

# Create output directory
mkdir -p /scratch/yasir071/output/liew_replication

# Activate virtual environment
source /scratch/yasir071/venv/bin/activate

# ─────────────────────────────────────────
# RUN
# ─────────────────────────────────────────
cd /scratch/yasir071/Joint-Moment-Prediction

echo "Starting DNN model..."
python src/run_loso.py --model dnn

echo "Starting Transfer Learning model..."
python src/run_loso.py --model tl

echo "Done!"