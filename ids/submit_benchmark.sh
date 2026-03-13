#!/bin/bash
# ============================================================
# SLURM job script — IEC104 Benchmark Training
# Submit with:  sbatch /scratch/j99tang/xai-datasets/ids/submit_benchmark.sh
# Monitor with: squeue -u j99tang
#               tail -f /scratch/j99tang/logs/benchmark_JOBID.log
# ============================================================

#SBATCH --job-name=iec104-benchmark
#SBATCH --output=/scratch/j99tang/logs/benchmark_%j.log   # %j = job ID
#SBATCH --time=06:00:00       # 6-hour wall-clock limit
#SBATCH --partition=compute
#SBATCH --gpus-per-node=1
#SBATCH --nodes=1

# ── Create output directories if they don't exist ────────────────────────
mkdir -p /scratch/j99tang/results
mkdir -p /scratch/j99tang/logs

# ── Load Python module ────────────────────────────────────────────────────
# Check available versions with:  module spider python
module load python/3.11
module load gcc arrow/17.0.0 

# ── Activate your virtual environment ────────────────────────────────────
source /scratch/j99tang/envs/xai-env/bin/activate

# ── Print environment info (useful for debugging) ────────────────────────
echo "Job started at: $(date)"
echo "Running on node: $(hostname)"
echo "Python: $(which python)"
echo "Python version: $(python --version)"
echo ""

# ── Run the training script ───────────────────────────────────────────────
python /scratch/j99tang/xai-datasets/ids/train_benchmark.py \
    --data-dir  /scratch/j99tang/data/raw/iec104/iec104 \
    --output-dir /scratch/j99tang/results

echo ""
echo "Job finished at: $(date)"
