#!/bin/bash
#SBATCH --partition=MGPU-TC2          # GPU partition
#SBATCH --qos=normal                  # QoS level
#SBATCH --nodes=1                     # Use 1 node
#SBATCH --gres=gpu:1                  # Request 1 GPU
#SBATCH --mem=40G                     # Allocate 40 GB of memory
#SBATCH --time=05:55:00               # Max job time = 5 hours 55 minutes
#SBATCH --job-name=sd15-lora          # Job name
#SBATCH --output=/home/msai/birul001/gan-project/gan-HC-SCDNet/ccds-files/finetune/sd15/logs/finetune_sd15_%j.out   # Std output log
#SBATCH --error=/home/msai/birul001/gan-project/gan-HC-SCDNet/ccds-files/finetune/sd15/logs/finetune_sd15_%j.err     # Std error log

export WANDB_API_KEY="d9a74b72096b984643e4b3246a816e62be94d572"
export HF_TOKEN="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"


HOME_PATH="/home/msai/birul001/"   
FINETUNE_DIR="gan-project/ccds-files/finetune/sd15"
FINETUNE_NAME="sd15-20epochs"
source "${HOME_PATH}/birul001-gan/bin/activate"


PYTHON_SCRIPT="/home/msai/birul001/gan-project/gan-HC-SCDNet/SDFinetune/train_sd15_lora.py"

RUN_TS=$(date +"%Y%m%d_%H%M%S")
JOB_ID="${SLURM_JOB_ID:-local}"
CCDS_DIR="/home/msai/birul001/gan-project/gan-HC-SCDNet/ccds-files"
OUTPUT_TXT="${CCDS_DIR}/finetune/sd15/${FINETUNE_NAME}_${RUN_TS}_${JOB_ID}.txt"
# OUTPUT_TXT="${HOME_PATH}${FINETUNE_DIR}/${FINETUNE_NAME}_${JOB_ID}.txt"

{
  echo "==== SD15 Finetune Run ===="
  echo "Started: $(date)"
  echo "Job Name: ${SLURM_JOB_NAME:-SD15-Finetune}"
  echo "Job ID: ${SLURM_JOB_ID:-N/A}"
  echo "Node: $(hostname)"
  echo "GPU(s) requested: ${SLURM_GPUS:-1}"
  echo "Output File: $OUTPUT_TXT"
  echo "==========================="
  echo
} > "$OUTPUT_TXT"

echo "Running AI job on GPU..."

if [ -f "$PYTHON_SCRIPT" ]; then
    echo "Executing: python $PYTHON_SCRIPT" >> "$OUTPUT_TXT"
    python "$PYTHON_SCRIPT" >> "$OUTPUT_TXT" 2>&1
else
    echo "Python script not found at $PYTHON_SCRIPT" >> "$OUTPUT_TXT"
fi


echo "AI job completed at $(date)" | tee -a "$OUTPUT_TXT"
