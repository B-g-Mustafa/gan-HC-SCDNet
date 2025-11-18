#!/bin/bash
#PBS -N sd35-lora-gpu1
#PBS -P personal-birul001
#PBS -q normal
#PBS -l select=1:ncpus=8:ngpus=1:mem=40GB
#PBS -l walltime=100:00:00
#PBS -j oe
#PBS -o logs/gpu1_job_output.log

# ==========================================
# PBS Job Script for GPU 1 (Worker Process)
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
echo "GPU 1 (Worker Process) - PBS Job"
echo "=========================================="
echo "Job ID: $PBS_JOBID"
echo "Job Name: $PBS_JOBNAME"
echo "Queue: $PBS_QUEUE"
echo "Node: $(hostname)"
echo "Start Time: $(date)"
echo "=========================================="

# --- Create output directories ---
mkdir -p "${OUTPUT_PATH}/logs"
mkdir -p "${OUTPUT_PATH}/py_output"

# --- Wait for GPU 0 to start and write master info ---
echo ""
echo "Waiting for GPU 0 to initialize..."
echo "Looking for master address file..."
echo ""

# Wait up to 60 seconds for master info files
for i in {1..60}; do
    if [ -f /tmp/sd35_master_addr.txt ] && [ -f /tmp/sd35_master_port.txt ]; then
        MASTER_ADDR=$(cat /tmp/sd35_master_addr.txt)
        MASTER_PORT=$(cat /tmp/sd35_master_port.txt)
        echo "✓ Found master info:"
        echo "  Address: $MASTER_ADDR"
        echo "  Port: $MASTER_PORT"
        break
    fi
    
    if [ $i -eq 1 ]; then
        echo "⏳ Waiting for GPU 0 to write master info..."
    fi
    
    sleep 1
    
    if [ $i -eq 60 ]; then
        echo "✗ Timeout waiting for master info files"
        echo "Make sure GPU 0 job is running first!"
        exit 1
    fi
done

# --- Run GPU 1 (Worker Process) ---
echo ""
echo "Starting GPU 1 (Worker Process)..."
echo ""

cd "${PROJECT_PATH}/SDFinetune"

python train_sd35_lora.py \
    --gpu 1 \
    --world_size 2 \
    2>&1 | tee "${OUTPUT_PATH}/py_output/gpu1_output.txt"

EXIT_CODE=$?

# --- Completion message ---
echo ""
echo "=========================================="
echo "GPU 1 Job Completed"
echo "=========================================="
echo "End Time: $(date)"
echo "Exit Code: $EXIT_CODE"
echo "=========================================="

exit $EXIT_CODE
