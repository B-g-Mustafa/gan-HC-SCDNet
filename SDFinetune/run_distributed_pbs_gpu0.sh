#!/bin/bash
#PBS -N sd35-lora-gpu0
#PBS -P personal-birul001
#PBS -q normal
#PBS -l select=1:ncpus=8:ngpus=1:mem=40GB
#PBS -l walltime=100:00:00
#PBS -j oe
#PBS -o logs/gpu0_job_output.log

# ==========================================
# PBS Job Script for GPU 0 (Main Process)
# Stable Diffusion 3.5 LoRA Distributed Training
# ==========================================

HOME_PATH="/home/users/ntu/birul001/"
PROJECT_PATH="${HOME_PATH}gan-project/git/gan-HC-SCDNet"
OUTPUT_PATH="${HOME_PATH}gan-project/outputs"

# --- Load modules and activate environment ---
module load python/3.9.12
source "${HOME_PATH}/birul001-python/bin/activate"

# --- Set API keys ---
export HF_TOKEN="hf_nq"
export WANDB_API_KEY="d572"

# --- Print job information ---
echo "=========================================="
echo "GPU 0 (Main Process) - PBS Job"
echo "=========================================="
echo "Job ID: $PBS_JOBID"
echo "Job Name: $PBS_JOBNAME"
echo "Queue: $PBS_QUEUE"
echo "Node: $(hostname)"
echo "Start Time: $(date)"
echo "=========================================="

# --- Clean up old master info files ---
rm -f /tmp/sd35_master_addr.txt /tmp/sd35_master_port.txt

# --- Create output directories ---
mkdir -p "${OUTPUT_PATH}/logs"
mkdir -p "${OUTPUT_PATH}/py_output"

# --- Run GPU 0 (Main Process) ---
echo ""
echo "Starting GPU 0 (Main Process)..."
echo "This process will auto-detect master IP and port"
echo ""

cd "${PROJECT_PATH}/SDFinetune"

python train_sd35_lora.py \
    --gpu 0 \
    --world_size 2 \
    2>&1 | tee "${OUTPUT_PATH}/py_output/gpu0_output.txt"

EXIT_CODE=$?

# --- Completion message ---
echo ""
echo "=========================================="
echo "GPU 0 Job Completed"
echo "=========================================="
echo "End Time: $(date)"
echo "Exit Code: $EXIT_CODE"
echo "=========================================="

# Clean up shared files
rm -f /tmp/sd35_master_addr.txt /tmp/sd35_master_port.txt

exit $EXIT_CODE
